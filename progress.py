"""End-effector-to-target progress: one stage table, three ways in.

Stages are compiled ahead of training -- a fixed table here, or a JSON file
naming the same relations and labels. Nothing is decided at runtime. Every
reader turns end-effector-to-target facts into one potential in ``[0, 1]``:

* ``replay_potential`` reads *observed* labels straight out of a packed graph.
  One-hot in, so the result is the hard stage sum, and it is the regression
  target the world-model progress head is trained on. This is the only entry
  point that touches an observed graph.
* ``potential(probs, hard=True)`` takes the arg-max label of each predicted
  relation. Discrete, readable, and what the tests assert against.
* ``potential(probs, hard=False)`` takes the probability mass on the satisfying
  labels. Continuous, so a decoder-driven actor gets a gradient-friendly signal
  rather than a step function.

All three return ``sum_k w_k * p_k`` with positive weights summing to one,
which is what bounds the potential without a clamp -- and it is the same
contraction in each case, so the observed scalar and the predicted one are
comparable by construction rather than by a second table.

The potential becomes a reward through :func:`potential_shaping`, which is
potential-difference shaping, not occupancy. Occupancy paid for *being* in a
high-progress state, so standing still in one earned as much as improving,
and undoing progress cost nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from scenegraph.adapters.graph_vocab import (
    build_absolute_vocab,
    build_relation_vocab,
)

# Relation and label ids come from the shared vocabularies, never from literals.
# They are positions in a table the graph contract owns: adding the directional
# support/contain labels shifted every id from ``very-near`` upward by two, and
# a hardcoded table would have gone on scoring the wrong labels without
# erroring -- ``ProgressScorer`` only checks ``0 < label < n_abs``.
_REL = build_relation_vocab()
_ABS = build_absolute_vocab()

REL_CONTACT = _REL.encode("contact")
REL_GRASP = _REL.encode("grasp")
REL_SUPPORT = _REL.encode("support")
REL_CONTAIN = _REL.encode("contain")
REL_PLANAR_DISTANCE = _REL.encode("planar-distance")
REL_HEIGHT_OFFSET = _REL.encode("height-offset")
REL_GRASP_COMPAT = _REL.encode("grasp-compatibility")
REL_CONTACT_COMPAT = _REL.encode("contact-compatibility")
REL_SUPPORT_COMPAT = _REL.encode("support-compatibility")
REL_CONTAIN_COMPAT = _REL.encode("contain-compatibility")

ABS_NOT_HOLDS = _ABS.encode("not-holds")
ABS_HOLDS = _ABS.encode("holds")
ABS_SRC_HOLDS = _ABS.encode("src-holds")
ABS_DST_HOLDS = _ABS.encode("dst-holds")
ABS_VERY_NEAR = _ABS.encode("very-near")
ABS_NEAR = _ABS.encode("near")
ABS_MEDIUM = _ABS.encode("medium")
ABS_FAR = _ABS.encode("far")
ABS_VERY_FAR = _ABS.encode("very-far")
ABS_FAR_BELOW = _ABS.encode("far-below")
ABS_BELOW = _ABS.encode("below")
ABS_LEVEL = _ABS.encode("level")
ABS_ABOVE = _ABS.encode("above")
ABS_FAR_ABOVE = _ABS.encode("far-above")
ABS_MATCH = _ABS.encode("match")
ABS_PARTIAL_MATCH = _ABS.encode("partial-match")
ABS_POOR_MATCH = _ABS.encode("poor-match")
ABS_UNOBSERVED = _ABS.encode("unobserved")

# The width every stage table is validated against.
N_ABS = len(_ABS)


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


class TaskScheduleReplayPotential(torch.nn.Module):
    """Observed graph labels -> one bounded scalar, for a phase schedule.

    Separate from :class:`ProgressScorer` on purpose. That one is linear in the
    predicted probabilities because a decoder-driven actor needs a gradient
    through it. This one only ever sees *observed* one-hot labels in replay, so
    it is free to be non-linear -- which is what cumulative credit needs -- and
    it never runs during imagination. The head learns ``[z, g, h] -> Phi`` and
    imagination reads the head.

    Three things it does that the end-effector scorer does not:

    * **Roles, not rows.** A schedule names ``movable``; the packed graph has a
      row whose index moves between frames. Rows are resolved per frame by
      matching entity ids in ``node_ent``, and a role matching no row or several
      is unresolved rather than guessed.
    * **Phase-local validity.** A missing relation invalidates the phase that
      names it, not the frame. ``not-holds`` and ``unobserved`` are present
      observations with a false value, which is different from no edge at all.
    * **Cumulative credit.** A satisfied later phase grants every earlier phase
      its full weight. Without it StackCube's success state -- stacked and
      released -- would score below its own mid-episode peak, because the grasp
      it needed is over.
    """

    def __init__(self, schedule, n_abs: int):
        super().__init__()
        self.env_id = schedule.env_id
        self.n_abs = int(n_abs)
        slots = list(schedule.slots)
        entities = list(schedule.entity_ids)
        self.n_slots, self.n_phases = len(slots), len(schedule.phases)

        ent_row = {ent: i for i, ent in enumerate(entities)}
        long_buf = lambda values: torch.tensor(values, dtype=torch.long)
        self.register_buffer("entities", long_buf(entities), persistent=False)
        self.register_buffer("slot_rel", long_buf([s[0] for s in slots]),
                             persistent=False)
        self.register_buffer("slot_src", long_buf([ent_row[s[1]] for s in slots]),
                             persistent=False)
        self.register_buffer("slot_dst", long_buf([ent_row[s[2]] for s in slots]),
                             persistent=False)

        index = {slot: i for i, slot in enumerate(slots)}
        clause_slot, clause_phase, clause_weight, clause_mask = [], [], [], []
        done_slot, done_mask, phase_weight = [], [], []
        phase_uses = torch.zeros(self.n_phases, self.n_slots, dtype=torch.bool)
        for p, phase in enumerate(schedule.phases):
            phase_weight.append(phase.weight)
            for clause in phase.clauses:
                slot = index[clause.slot]
                clause_slot.append(slot)
                clause_phase.append(p)
                clause_weight.append(clause.weight)
                clause_mask.append(_label_mask(clause.label_ids, self.n_abs))
                phase_uses[p, slot] = True
            slot = index[phase.completion.slot]
            done_slot.append(slot)
            done_mask.append(_label_mask(phase.completion.label_ids, self.n_abs))
            phase_uses[p, slot] = True

        self.register_buffer("clause_slot", long_buf(clause_slot), persistent=False)
        self.register_buffer("clause_phase", long_buf(clause_phase), persistent=False)
        self.register_buffer(
            "clause_weight", torch.tensor(clause_weight, dtype=torch.float32),
            persistent=False)
        self.register_buffer("clause_mask", torch.stack(clause_mask), persistent=False)
        self.register_buffer("done_slot", long_buf(done_slot), persistent=False)
        self.register_buffer("done_mask", torch.stack(done_mask), persistent=False)
        self.register_buffer(
            "phase_weight", torch.tensor(phase_weight, dtype=torch.float32),
            persistent=False)
        self.register_buffer("phase_uses", phase_uses, persistent=False)

    def resolve_rows(self, node_ent: torch.Tensor):
        """``(rows, resolved)`` per graph and scheduled entity.

        Exactly one row must carry the entity id. None means the object has not
        been admitted yet; several means the scene holds indistinguishable
        instances, and picking one would be a guess.
        """
        match = node_ent[..., None].eq(self.entities)
        count = match.sum(1)
        rows = match.float().argmax(1)
        return rows, count.eq(1)

    def observe(self, node_ent, edge_rel, edge_abs, edge_src_local,
                edge_dst_local, edge_graph, graph_count: int):
        """``(onehot, present)`` per graph and slot.

        ``present`` is exactly-one-edge, so a missing fact and a duplicated one
        are both unresolved -- a duplicate would double-count in the scatter and
        read as a label neither edge carries.
        """
        device = self.phase_weight.device
        shape = (int(graph_count), self.n_slots)
        rows, resolved = self.resolve_rows(node_ent)
        want_src = rows.index_select(1, self.slot_src)
        want_dst = rows.index_select(1, self.slot_dst)
        ok = (resolved.index_select(1, self.slot_src)
              & resolved.index_select(1, self.slot_dst))

        onehot = torch.zeros((*shape, self.n_abs), device=device,
                             dtype=torch.float32)
        count = torch.zeros(shape, device=device, dtype=torch.float32)
        if edge_graph.numel():
            hit = (
                edge_rel[:, None].eq(self.slot_rel)
                & edge_src_local[:, None].eq(want_src[edge_graph])
                & edge_dst_local[:, None].eq(want_dst[edge_graph])
            )
            slot_ix = torch.arange(self.n_slots, device=device)
            g = edge_graph[:, None].expand_as(hit)[hit]
            s = slot_ix[None, :].expand_as(hit)[hit]
            a = edge_abs[:, None].expand_as(hit)[hit]
            onehot[g, s, a] = 1.0
            count.index_put_((g, s), torch.ones_like(g, dtype=torch.float32),
                             accumulate=True)
        return onehot, count.eq(1.0) & ok

    def forward(self, node_ent, edge_rel, edge_abs, edge_src_local,
                edge_dst_local, edge_graph, graph_count: int):
        """``(potential, valid)`` per frame from observed labels."""
        onehot, present = self.observe(
            node_ent, edge_rel, edge_abs, edge_src_local, edge_dst_local,
            edge_graph, graph_count)

        satisfied = (onehot.index_select(1, self.clause_slot)
                     * self.clause_mask).sum(-1)
        satisfied = satisfied * present.index_select(1, self.clause_slot).float()
        quality = torch.zeros(int(graph_count), self.n_phases,
                              device=satisfied.device, dtype=satisfied.dtype)
        quality.index_add_(1, self.clause_phase, satisfied * self.clause_weight)

        done = (onehot.index_select(1, self.done_slot) * self.done_mask).sum(-1)
        done = (done > 0) & present.index_select(1, self.done_slot)

        # OR over strictly later phases, so a completed settle carries the grasp
        # that is no longer held.
        back = done.flip(-1).cummax(-1).values.flip(-1)
        later = torch.zeros_like(done)
        later[:, :-1] = back[:, 1:]

        # A phase is readable when every slot it names resolved this frame.
        missing = (~present)[:, None, :] & self.phase_uses
        phase_valid = ~missing.any(-1)

        credit = torch.maximum(quality * phase_valid,
                               self.phase_weight * later.float())
        # An unreadable phase is still resolvable when a later one completed:
        # cumulative credit gives it its full weight regardless of its own
        # facts, so nothing about it is unknown.
        valid = (phase_valid | later).all(-1)
        return credit.sum(-1) * valid.float(), valid


def _label_mask(label_ids, n_abs: int) -> torch.Tensor:
    mask = torch.zeros(n_abs, dtype=torch.float32)
    for label in label_ids:
        if not 0 < int(label) < n_abs:
            raise ValueError(f"label id {label} outside [1, {n_abs})")
        mask[int(label)] = 1.0
    return mask


def potential_shaping(potential, cont, discount: float):
    """``F_t = gamma * c_{t+1} * Phi_{t+1} - Phi_t``, indexed for lambda return.

    ``potential`` and ``cont`` are ``(B, T, 1)`` over one imagined rollout.
    Index 0 is zero because ``_lambda_return`` consumes ``reward[:, 1:]``: the
    first entry is the reward for arriving at step 0, which no transition in
    this rollout produced.

    ``cont`` must be the same continuation the return uses, so a terminal
    transition keeps only ``-Phi_t`` and the telescoping stays exact. Nothing
    is clipped and negatives are kept -- a reward floored at zero would pay to
    undo and redo the same progress.
    """
    body = discount * cont[:, 1:] * potential[:, 1:] - potential[:, :-1]
    head = torch.zeros_like(potential[:, :1])
    return torch.cat([head, body], dim=1)
