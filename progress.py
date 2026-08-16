"""End-effector-to-target progress, read from predicted slot relations.

The scorer never touches the simulator or an observed graph. It asks the slot
relation decoder what facts it predicts between slot zero (the end effector) and
the slot flagged as the subtask target, and turns those distributions into one
potential in ``[0, 1]``.

Stages are compiled ahead of training -- a fixed table here, or a JSON file
naming the same relations and labels. Nothing is decided at runtime.

Two scorers share one table:

* ``hard`` takes the arg-max label of each relation. Discrete, readable, and
  what the tests assert against.
* ``soft`` takes the probability mass on the satisfying labels. Continuous, so
  the actor gets a gradient-friendly signal rather than a step function.

Both return ``sum_k w_k * p_k`` with positive weights summing to one, which is
what bounds the potential without a clamp. With
``progress_reward = (1 - discount) * potential`` the discounted return of a
permanently-solved episode is at most one, so the term cannot outgrow the task
return it is added to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


# Relation ids, fixed by ``scenegraph.core.relation_rules.RELATION_TYPES`` with
# index 0 reserved for padding.
REL_CONTACT = 1
REL_GRASP = 2
REL_SUPPORT = 3
REL_CONTAIN = 4
REL_PLANAR_DISTANCE = 5
REL_HEIGHT_OFFSET = 6
REL_GRASP_COMPAT = 7
REL_CONTACT_COMPAT = 8
REL_SUPPORT_COMPAT = 9
REL_CONTAIN_COMPAT = 10

# Absolute-label ids, from the shared sigma vocabulary.
ABS_NOT_HOLDS = 1
ABS_HOLDS = 2
ABS_VERY_NEAR, ABS_NEAR, ABS_MEDIUM, ABS_FAR, ABS_VERY_FAR = 3, 4, 5, 6, 7
ABS_FAR_BELOW, ABS_BELOW, ABS_LEVEL, ABS_ABOVE, ABS_FAR_ABOVE = 8, 9, 10, 11, 12
ABS_MATCH, ABS_PARTIAL_MATCH, ABS_POOR_MATCH, ABS_UNOBSERVED = 13, 14, 15, 16


@dataclass(frozen=True)
class ProgressStage:
    """One rung of the ladder: a relation and the labels that satisfy it."""

    name: str
    relation: int
    labels: tuple[int, ...]
    weight: float


# Pick. Every rung is a fact the graph already carries, ordered the way the
# subtask is actually solved: get near, get level, be a plausible grasp, touch,
# hold. Grasping implies all of the lower rungs, so it scores one.
PICK_STAGES: tuple[ProgressStage, ...] = (
    ProgressStage("planar", REL_PLANAR_DISTANCE, (ABS_VERY_NEAR, ABS_NEAR), 0.15),
    ProgressStage("height", REL_HEIGHT_OFFSET, (ABS_LEVEL,), 0.10),
    ProgressStage(
        "contact_compat",
        REL_CONTACT_COMPAT,
        (ABS_MATCH, ABS_PARTIAL_MATCH),
        0.10,
    ),
    ProgressStage(
        "grasp_compat", REL_GRASP_COMPAT, (ABS_MATCH, ABS_PARTIAL_MATCH), 0.15
    ),
    ProgressStage("contact", REL_CONTACT, (ABS_HOLDS,), 0.20),
    ProgressStage("grasp", REL_GRASP, (ABS_HOLDS,), 0.30),
)


def load_stages(path: str = "") -> tuple[ProgressStage, ...]:
    """Built-in Pick table, or a compiled JSON one with the same fields."""
    if not path:
        return PICK_STAGES
    with open(path, "r") as handle:
        raw = json.load(handle)
    entries = raw["stages"] if isinstance(raw, dict) else raw
    return tuple(
        ProgressStage(
            str(entry["name"]),
            int(entry["relation"]),
            tuple(int(label) for label in entry["labels"]),
            float(entry["weight"]),
        )
        for entry in entries
    )


class ProgressScorer(torch.nn.Module):
    """Predicted EE-to-target facts to one bounded potential.

    Parameter-free. It owns the stage table, the relation ids it needs decoded,
    and the label mask that turns those distributions into stage satisfaction.
    """

    def __init__(self, stages: Sequence[ProgressStage], n_abs: int):
        super().__init__()
        stages = tuple(stages)
        if not stages:
            raise ValueError("progress needs at least one stage")
        total = sum(stage.weight for stage in stages)
        if total <= 0:
            raise ValueError("progress stage weights must be positive")
        self.stages = stages
        self.names = tuple(stage.name for stage in stages)
        # One decode per distinct relation, reused by every stage naming it.
        relations = []
        for stage in stages:
            if stage.relation not in relations:
                relations.append(stage.relation)
        self.register_buffer(
            "relations", torch.tensor(relations, dtype=torch.long), persistent=False
        )
        select = torch.zeros(len(stages), dtype=torch.long)
        labels = torch.zeros(len(stages), n_abs, dtype=torch.bool)
        for index, stage in enumerate(stages):
            select[index] = relations.index(stage.relation)
            for label in stage.labels:
                if not 0 < label < n_abs:
                    raise ValueError(f"stage {stage.name!r} names label {label}")
                labels[index, label] = True
        self.register_buffer("select", select, persistent=False)
        self.register_buffer("labels", labels, persistent=False)
        self.register_buffer(
            "weights",
            torch.tensor([stage.weight / total for stage in stages], dtype=torch.float32),
            persistent=False,
        )

    def satisfaction(self, probs: torch.Tensor, hard: bool) -> torch.Tensor:
        """Per-stage satisfaction in ``[0, 1]`` from (..., R, n_abs) probs."""
        probs = probs.float().index_select(-2, self.select)
        if hard:
            chosen = probs.argmax(-1, keepdim=True)
            hit = torch.zeros_like(probs, dtype=torch.bool).scatter(-1, chosen, True)
            return (hit & self.labels).any(-1).float()
        return (probs * self.labels).sum(-1)

    def potential(self, probs: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """Weighted stage satisfaction. Bounded by construction, not by clamp."""
        return (self.satisfaction(probs, hard) * self.weights).sum(-1)

    def forward(self, probs: torch.Tensor, hard: bool = False) -> torch.Tensor:
        return self.potential(probs, hard)


def target_slot_index(slot_target: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
    """Which slot the subtask is acting on, or slot zero when none is flagged.

    Slot zero is the end effector, so pointing there is the honest "no target"
    encoding: the EE-to-EE pair is degenerate and ``has_target`` gates it out.
    """
    flag = slot_target.bool() & slot_mask.bool()
    return flag.long().argmax(-1)


def slot_pair(
    slots: torch.Tensor, slot_meta_target: torch.Tensor, slot_mask: torch.Tensor
):
    """Gather the (end effector, target) slot pair and a validity flag."""
    index = target_slot_index(slot_meta_target, slot_mask)
    has_target = (slot_meta_target.bool() & slot_mask.bool()).any(-1)
    source = slots[..., 0, :]
    gather = index[..., None, None].expand(*index.shape, 1, slots.shape[-1])
    target = slots.gather(-2, gather).squeeze(-2)
    return source, target, has_target & slot_mask[..., 0].bool()


class ProgressReward(torch.nn.Module):
    """Bounded shaping reward from imagined slots.

    Holds the scorer and the discount; the relation decoder is passed in so the
    frozen copy is used inside imagination and the live one inside tests.
    """

    def __init__(self, scorer: ProgressScorer, discount: float, soft: bool = True):
        super().__init__()
        self.scorer = scorer
        self.discount = float(discount)
        self.soft = bool(soft)

    def relation_probs(self, decoder, slots, slot_meta_target, slot_mask):
        source, target, valid = slot_pair(slots, slot_meta_target, slot_mask)
        probs = decoder.relation_probs(source, target, self.scorer.relations)
        return probs, valid

    def forward(self, decoder, slots, slot_meta_target, slot_mask, hard=None):
        probs, valid = self.relation_probs(decoder, slots, slot_meta_target, slot_mask)
        hard = (not self.soft) if hard is None else bool(hard)
        potential = self.scorer.potential(probs, hard=hard) * valid.float()
        # (1 - discount) keeps the discounted return of a permanently solved
        # episode at one, so beta means the same thing at any horizon.
        return (1.0 - self.discount) * potential, potential, probs
