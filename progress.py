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


# Pick. Each relation has cumulative rungs: the worst label scores zero and
# every improvement satisfies one more rung. The increments within a relation
# sum to its original budget (planar .15, height .10, contact compatibility
# .10, grasp compatibility .15, contact .20, grasp .30), so the best joint
# state still scores exactly one. Cumulative label sets are important here: an
# exact-label table would make mutually exclusive labels compete in the global
# weight normalisation and the maximum potential would fall below one.
PICK_STAGES: tuple[ProgressStage, ...] = (
    # Planar: very-far 0, far .25, medium .50, near .75, very-near 1.00
    # within the .15 planar budget.
    ProgressStage(
        "planar_far_or_better",
        REL_PLANAR_DISTANCE,
        (ABS_FAR, ABS_MEDIUM, ABS_NEAR, ABS_VERY_NEAR),
        0.0375,
    ),
    ProgressStage(
        "planar_medium_or_better",
        REL_PLANAR_DISTANCE,
        (ABS_MEDIUM, ABS_NEAR, ABS_VERY_NEAR),
        0.0375,
    ),
    ProgressStage(
        "planar_near_or_better",
        REL_PLANAR_DISTANCE,
        (ABS_NEAR, ABS_VERY_NEAR),
        0.0375,
    ),
    ProgressStage(
        "planar_very_near",
        REL_PLANAR_DISTANCE,
        (ABS_VERY_NEAR,),
        0.0375,
    ),
    # Height is symmetric around level: the two extreme bins score zero,
    # below/above score half credit, and level receives the full .10 budget.
    ProgressStage(
        "height_non_extreme",
        REL_HEIGHT_OFFSET,
        (ABS_BELOW, ABS_LEVEL, ABS_ABOVE),
        0.05,
    ),
    ProgressStage("height_level", REL_HEIGHT_OFFSET, (ABS_LEVEL,), 0.05),
    # Compatibility: unobserved 0, poor .25, partial .60, match 1.00 within
    # each relation's budget.
    ProgressStage(
        "contact_compat_poor_or_better",
        REL_CONTACT_COMPAT,
        (ABS_POOR_MATCH, ABS_PARTIAL_MATCH, ABS_MATCH),
        0.025,
    ),
    ProgressStage(
        "contact_compat_partial_or_better",
        REL_CONTACT_COMPAT,
        (ABS_PARTIAL_MATCH, ABS_MATCH),
        0.035,
    ),
    ProgressStage(
        "contact_compat_match", REL_CONTACT_COMPAT, (ABS_MATCH,), 0.04
    ),
    ProgressStage(
        "grasp_compat_poor_or_better",
        REL_GRASP_COMPAT,
        (ABS_POOR_MATCH, ABS_PARTIAL_MATCH, ABS_MATCH),
        0.0375,
    ),
    ProgressStage(
        "grasp_compat_partial_or_better",
        REL_GRASP_COMPAT,
        (ABS_PARTIAL_MATCH, ABS_MATCH),
        0.0525,
    ),
    ProgressStage(
        "grasp_compat_match", REL_GRASP_COMPAT, (ABS_MATCH,), 0.06
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

    def __init__(
        self, stages: Sequence[ProgressStage], n_abs: int, n_rel: int = 0
    ):
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
        # The soft potential is linear in the predicted probabilities:
        #     Phi = sum_k w_k sum_{l in A_k} p[r_k, l] = <p, W>.
        # Folding the stage table into one (R, n_abs) matrix makes it a single
        # multiply-and-reduce, so the stage axis is never materialised in the
        # hot path and overlapping cumulative rungs cost nothing extra. Exactly
        # equal to the stage sum; the hard scorer keeps the stage path because
        # its arg-max is not linear.
        potential = torch.zeros(len(relations), n_abs, dtype=torch.float32)
        for index, stage in enumerate(stages):
            potential[select[index]] += labels[index].float() * (
                stage.weight / total
            )
        self.register_buffer("potential_weights", potential, persistent=False)
        # Relation id -> row of ``potential_weights``, -1 for relations no stage
        # names. The id is not the row: the rows follow the stage table's order
        # of first appearance, not the relation vocabulary's.
        width = int(n_rel) if int(n_rel) > 0 else int(max(relations)) + 1
        if width <= int(max(relations)):
            raise ValueError(
                f"n_rel={n_rel} cannot address relation {max(relations)}"
            )
        row_of = torch.full((width,), -1, dtype=torch.long)
        row_of[self.relations] = torch.arange(len(relations), dtype=torch.long)
        self.register_buffer("row_of", row_of, persistent=False)

    @property
    def n_relations(self) -> int:
        return int(self.relations.numel())

    def replay_potential(
        self,
        edge_rel: torch.Tensor,
        edge_abs: torch.Tensor,
        edge_src_local: torch.Tensor,
        edge_dst_local: torch.Tensor,
        edge_graph: torch.Tensor,
        graph_count: int,
        target_row: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(potential, valid)`` per frame from *observed* relation labels.

        Reads the end-effector-to-target block straight out of the packed
        graph: row 0 to ``target_row``, one edge per scored relation. Labels
        are one-hot, so the same linear contraction the soft scorer uses over
        predicted probabilities returns exactly the hard stage sum here -- the
        two agree by construction rather than by a second table.

        Validity is strict and deliberately so. The stage weights sum to one
        only when all of the scorer's relations are present; a frame missing
        the grasp fact would score at most 0.70 through no fault of the robot,
        and regressing on that would teach the head that losing a fact is
        losing progress. Renormalising the present ones instead would make the
        potential mean something different from frame to frame. So an
        incomplete frame is masked, and a duplicated relation -- two edges
        claiming the same fact, which would double-count in the scatter -- is
        masked with it.
        """
        row = self.row_of.index_select(0, edge_rel)
        keep = (
            row.ge(0)
            & edge_src_local.eq(0)
            & edge_dst_local.eq(int(target_row))
        )
        frame = edge_graph[keep]
        row = row[keep]
        label = edge_abs[keep]
        shape = (int(graph_count), self.n_relations)
        device = self.potential_weights.device

        onehot = torch.zeros(
            (*shape, self.potential_weights.shape[-1]),
            device=device,
            dtype=torch.float32,
        )
        onehot[frame, row, label] = 1.0
        count = torch.zeros(shape, device=device, dtype=torch.float32)
        count.index_put_(
            (frame, row), torch.ones_like(frame, dtype=torch.float32), accumulate=True
        )
        # Exactly one edge per scored relation: catches both absence and the
        # duplicate that a one-hot scatter would otherwise hide.
        valid = count.eq(1.0).all(-1)
        potential = (onehot * self.potential_weights).sum((-2, -1))
        return potential * valid.float(), valid

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
        if not hard:
            return (probs.float() * self.potential_weights).sum((-2, -1))
        return (self.satisfaction(probs, hard) * self.weights).sum(-1)

    def forward(self, probs: torch.Tensor, hard: bool = False) -> torch.Tensor:
        return self.potential(probs, hard)


def target_distribution(target_logits, slot_alive, null_logit: float = 0.0):
    """Predicted target identity over object slots plus one null class.

    Slot zero is the end effector and is never a candidate. Without the null
    class a softmax has to name an object even when the scene has no target, and
    progress would then score a relation to an arbitrary slot.
    """
    objects = target_logits[..., 1:].float()
    objects = objects.masked_fill(
        ~(slot_alive[..., 1:] > 0.5), torch.finfo(torch.float32).min
    )
    null = objects.new_full((*objects.shape[:-1], 1), float(null_logit))
    probs = torch.softmax(torch.cat([objects, null], -1), -1)
    return probs[..., :-1], probs[..., -1]


class ProgressReward(torch.nn.Module):
    """Bounded shaping reward from imagined slots.

    Holds the scorer and the discount; the relation decoder is passed in so the
    frozen copy is used inside imagination and the live one inside tests.

    The target is *predicted*, never latched: a born object may be the target,
    and an imagined rollout has no observation to read a flag from. Each
    candidate object slot is decoded separately and the resulting relation
    distributions are mixed by the target weights -- the relation decoder is
    nonlinear, so decoding one averaged slot embedding would ask it about an
    object that does not exist.
    """

    def __init__(self, scorer: ProgressScorer, discount: float, soft: bool = True):
        super().__init__()
        self.scorer = scorer
        self.discount = float(discount)
        self.soft = bool(soft)

    def relation_probs(self, decoder, slots, target_logits, slot_alive, hard_target=False):
        weights, null = target_distribution(target_logits, slot_alive)
        if hard_target:
            choice = weights.argmax(-1, keepdim=True)
            weights = torch.zeros_like(weights).scatter(-1, choice, 1.0)
            weights = weights * (1.0 - null)[..., None]
        objects = slots[..., 1:, :]
        source = slots[..., :1, :].expand_as(objects)
        # (..., n_obj, R, n_abs): one decode per candidate, mixed afterwards.
        per_slot = decoder.relation_probs(source, objects, self.scorer.relations)
        probs = (weights[..., None, None] * per_slot).sum(-3)
        # Renormalise over the object mass so the result is a distribution
        # conditioned on "a target exists"; the null mass gates the potential.
        mass = weights.sum(-1).clamp_min(1e-6)
        return probs / mass[..., None, None], 1.0 - null

    def pooled(self, probs: torch.Tensor):
        """Shaping reward from a directly predicted EE-to-target block.

        The pooled graph state has no slots to enumerate and no target-null
        class: the subtask target is fixed for the episode and the recurrent
        state is what carries it through occlusion. So the head's own (R, n_abs)
        output is already conditioned on "the target", and the potential is one
        contraction over the whole ``B * H`` rollout at once.
        """
        potential = self.scorer.potential(probs, hard=not self.soft)
        return (1.0 - self.discount) * potential, potential

    def forward(self, decoder, slots, target_logits, slot_alive, hard=None):
        hard = (not self.soft) if hard is None else bool(hard)
        probs, has_target = self.relation_probs(
            decoder, slots, target_logits, slot_alive, hard_target=hard
        )
        alive_ee = (slot_alive[..., 0] > 0.5).float()
        potential = self.scorer.potential(probs, hard=hard) * has_target * alive_ee
        # (1 - discount) keeps the discounted return of a permanently solved
        # episode at one, so beta means the same thing at any horizon.
        return (1.0 - self.discount) * potential, potential, probs
