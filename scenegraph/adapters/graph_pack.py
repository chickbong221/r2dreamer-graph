"""Hyper-relational per-frame packing.

Nodes fill a compact prefix, each carrying one bbox
per camera plus a flag marking the subtask's target; each fact is one row of
(relation, absolute, temporal). Row 0 is the end effector and row 1 the active
target, and every node carries a world-frame centroid. Arrays use the narrowest
dtype holding their vocabulary since these land in the replay buffer every step;
the encoder casts back on read.

Nothing derivable is stored. Index zero is padding in every vocabulary, so
validity, per-camera visibility, whether a camera ever observed a node, and
both counts all follow from the ids, the boxes and the embedding norms -- see
``graph_encoder`` for the exact derivations the model reads them back with.
"""

from __future__ import annotations

from typing import Dict, Tuple

from collections import Counter

import numpy as np

from ..core.relation_rules import (
    AFFORDANCE_RELATIONS,
    GOAL_RELATIONS,
    PHYSICAL_RELATIONS,
    TEMPORAL_RELATIONS,
)
from ..core.schema import Graph
from .graph_vocab import GraphVocab, entity_key_for


# Mirrors ``graph.py``; the tests assert the two agree. Duplicated rather than
# imported so this package never reaches into the model package.
_PHYSICAL = frozenset(PHYSICAL_RELATIONS)
_AFFORDANCE = frozenset(AFFORDANCE_RELATIONS)
_GOAL = frozenset(GOAL_RELATIONS)


def _edge_priority(edge) -> int:
    """Deterministic order: goal and physical state, then affordance, then
    spatial.

    Goal facts sort with the physical milestones rather than with the spatial
    family they resemble. Overflow raises rather than truncates, so this is
    ordering and not survival -- but a scheduled terminal milestone belongs
    beside the other things a phase completes on.
    """
    if edge.relation in _GOAL or edge.relation in _PHYSICAL:
        return 0
    if edge.relation in _AFFORDANCE:
        return 1
    return 2


def _row_assignment(graph, n_max: int, target_id, fixed_rows: bool,
                    site_id=None):
    """``([(row, node)], n_dropped)`` for one frame.

    With ``fixed_rows`` the pooled schema pins rows by meaning rather than by
    arrival order: row 0 is the end effector, row 1 is the active subtask
    target, row 2 is the protected site when the task declares one, and
    everything else fills upward from there. The vertex registry cannot supply
    this -- it hands each node whichever index happened to be free when it was
    first admitted -- and the encoder needs the target at a fixed row so an
    occluded one keeps the same identity frame to frame.

    Required nodes must have been admitted by the builder. MS-HAB Pick bounds
    its optional context before packing; a graph that still exceeds capacity
    here is a builder error, not an invitation to truncate required facts.
    """
    nodes = list(graph.nodes)
    if len(nodes) > n_max:
        raise RuntimeError(
            f"node budget exceeded: {len(nodes)} vertices against n_max="
            f"{n_max}. env={graph.env_id} frame={graph.frame} "
            f"nodes={[n.node_id for n in nodes]}. Apply the node budget before packing."
        )
    if not fixed_rows:
        return list(enumerate(nodes[:n_max])), max(0, len(nodes) - n_max)

    rows = []
    rest = []
    target_row_taken = False
    site_row_taken = False
    for node in nodes:
        if node.node_type == "ee":
            if not any(row == 0 for row, _ in rows):
                rows.append((0, node))
            continue
        if node.node_id == target_id and not target_row_taken and n_max > 1:
            rows.append((1, node))
            target_row_taken = True
            continue
        if (site_id and node.node_id == site_id and not site_row_taken
                and n_max > 2):
            rows.append((2, node))
            site_row_taken = True
            continue
        rest.append(node)

    # Row 2 is reserved only for a task that declares a protected site. A task
    # without one keeps the previous layout exactly, so nothing about the
    # ManiSkill packing moves.
    next_row = 3 if site_id and n_max > 2 else 2
    for node in rest:
        if next_row >= n_max:
            # Reserving row 1 for an unseen target costs one object row. Under
            # retention that is still a vertex nothing can hold, so it raises
            # like any other overflow rather than vanishing.
            reserved = f"row 1 reserved for target {target_id!r}"
            if site_id:
                reserved += f", row 2 for site {site_id!r}"
            raise RuntimeError(
                f"node budget exceeded: {node.node_id!r} has no row. "
                f"n_max={n_max} with {reserved}. env={graph.env_id} "
                f"frame={graph.frame} nodes={[n.node_id for n in nodes]}."
            )
        rows.append((next_row, node))
        next_row += 1
    return rows, 0


def verify_protected_rows(graph, node_ent, node_target, position,
                          target_id, site_id, vocab) -> None:
    """Check the rows that carry meaning actually carry it.

    The schedule resolves its active-target role to row 1 rather than by
    scanning entity ids, because two same-category instances share one id and
    scanning cannot tell them apart. That makes the row a contract, and a
    contract nothing checks is a silent wrong answer: the phase would score
    whichever object happened to land there.

    Bypassed only by a genuinely targetless run -- no target and no protected
    site. Everything else is under the protected-node contract, where an
    absent target is not a benign empty row: with the schedule's role
    resolving to row 1, an empty one makes every phase that names the target
    unreadable, on every frame, silently.
    """
    if target_id is None and site_id is None:
        return

    ee_row = position.get("ee")
    if ee_row != 0:
        raise RuntimeError(
            f"the end effector is at row {ee_row}, not the reserved row 0. "
            f"env={graph.env_id} frame={graph.frame}"
        )
    if vocab is not None and int(node_ent[0]) != int(vocab.entity.ee_id):
        raise RuntimeError(
            f"row 0 encodes entity id {int(node_ent[0])}, not the end "
            f"effector's {int(vocab.entity.ee_id)}. env={graph.env_id} "
            f"frame={graph.frame}"
        )

    if target_id is None:
        raise RuntimeError(
            f"this task protects the site {site_id!r} but named no active "
            "target. The schedule resolves its target role to row 1, so an "
            f"unnamed target leaves every phase unreadable. env={graph.env_id} "
            f"frame={graph.frame}"
        )
    row = position.get(target_id)
    if row is None:
        raise RuntimeError(
            f"the active target {target_id!r} is not in the packed graph. It "
            "is seeded from the simulator at reset and retained for the "
            "episode, so its absence means the seeding did not run -- and "
            "row 1 would be padding that the schedule reads as the target. "
            f"env={graph.env_id} frame={graph.frame}"
        )
    if row != 1:
        raise RuntimeError(
            f"active target {target_id!r} packed at row {row}, not the "
            f"reserved row 1. env={graph.env_id} frame={graph.frame}"
        )
    if not node_ent[1]:
        raise RuntimeError(
            f"row 1 holds target {target_id!r} but encodes to the pad entity "
            f"id. env={graph.env_id} frame={graph.frame}"
        )
    flagged = int(node_target.sum())
    if flagged != 1 or not node_target[1]:
        raise RuntimeError(
            f"the target flag names {flagged} row(s) and row 1 is "
            f"{'set' if node_target[1] else 'clear'}; exactly one row must be "
            f"flagged and it must be row 1. env={graph.env_id} "
            f"frame={graph.frame}"
        )
    if site_id is not None:
        site_row = position.get(site_id)
        if site_row is not None and not node_ent[site_row]:
            raise RuntimeError(
                f"row {site_row} holds the protected site {site_id!r} but "
                f"encodes to the pad entity id. env={graph.env_id} "
                f"frame={graph.frame}"
            )
        if site_row is None:
            raise RuntimeError(
                f"protected site {site_id!r} is not in the packed graph. It "
                "is derived from the robot every frame, so its absence means "
                f"the builder stopped producing it. env={graph.env_id} "
                f"frame={graph.frame}"
            )
        if site_row != 2:
            raise RuntimeError(
                f"protected site {site_id!r} packed at row {site_row}, not "
                f"the reserved row 2. env={graph.env_id} frame={graph.frame}"
            )
        if node_target[2]:
            raise RuntimeError(
                f"the protected site {site_id!r} carries the target flag. It "
                "is a place, not the object the subtask acts on. "
                f"env={graph.env_id} frame={graph.frame}"
            )


def pack_graph(
    graph: Graph,
    vocab: GraphVocab,
    *,
    n_max: int,
    e_max: int,
    n_cams: int,
    use_target_flag: bool = True,
) -> Dict[str, np.ndarray]:
    """Pack one frame. A node is addressed by the box it currently occupies."""
    # Targetless runs name no goal object, so row 1 is not reserved and
    # graph_node_target stays zero.
    fixed_rows = bool(use_target_flag)
    if n_max > 255:
        raise ValueError(
            f"n_max={n_max} exceeds 255; edge endpoints are packed as uint8"
        )

    node_ent = np.zeros(n_max, dtype=np.uint8)
    # Which vertex the current subtask is acting on. All-zero is the honest
    # encoding of "unknown": the target may be unresolved, not yet admitted, or
    # displaced by vertex overflow, and a bit on the wrong instance is worse
    # than no bit at all.
    node_target = np.zeros(n_max, dtype=np.uint8)
    node_bbox = np.zeros((n_max, n_cams, 4), dtype=np.float16)
    # float32: half precision would quantise a table-scale scene to millimetres.
    node_centroid = np.zeros((n_max, 3), dtype=np.float32)
    target_id = (graph.meta.get("active_target_node_id")
                 if use_target_flag else None)
    # Named by the builder, not inferred from the key prefix: PegInsertionSide
    # and PullCubeTool declare sites too, and reserving a row for those would
    # move the ManiSkill layout for no reason.
    site_id = (graph.meta.get("protected_site_node_id")
               if use_target_flag else None)

    position: Dict[str, int] = {}
    assigned, n_dropped = _row_assignment(
        graph, n_max, target_id, fixed_rows, site_id)
    n_nodes = 0
    for i, node in assigned:
        ent = vocab.entity.encode(entity_key_for(node))
        if ent == vocab.entity.pad_id:
            # Validity is read back as ``ent != pad``, so a real vertex landing
            # on the pad id would silently become padding that edges still
            # point at.
            raise ValueError(
                f"node {node.node_id!r} ({node.node_type}) encodes to the pad "
                "entity id; every packed vertex needs a whitelist key"
            )
        node_ent[i] = ent
        if node.bbox is not None:
            node_bbox[i] = node.bbox
        if node.pose_world is not None:
            node_centroid[i] = np.asarray(node.pose_world[:3], dtype=np.float32)
        if node.node_id == target_id:
            node_target[i] = 1
        position[node.node_id] = i
        n_nodes += 1

    candidates = [
        e for e in graph.edges
        if e.src in position and e.dst in position
    ]
    candidates.sort(key=_edge_priority)
    if len(candidates) > e_max:
        # Retention makes edge occupancy persistent, so truncation would drop
        # the same families every frame -- spatial first, by priority -- and
        # both reconstruction and the progress target would read a graph that
        # is systematically missing facts rather than occasionally short.
        by_family = Counter(e.relation for e in candidates)
        raise RuntimeError(
            f"edge budget exceeded: {len(candidates)} candidate facts against "
            f"e_max={e_max}. env={graph.env_id} frame={graph.frame} "
            f"nodes={n_nodes}. by relation: "
            f"{dict(sorted(by_family.items()))}. "
            "Raise model.graph.e_max -- truncation is not a safe fallback "
            "under unconditional retention."
        )
    kept = candidates

    edge_src = np.zeros(e_max, dtype=np.uint8)
    edge_dst = np.zeros(e_max, dtype=np.uint8)
    edge_rel = np.zeros(e_max, dtype=np.uint8)
    edge_abs = np.zeros(e_max, dtype=np.uint8)
    edge_temp = np.zeros(e_max, dtype=np.uint8)

    for i, e in enumerate(kept):
        rel = vocab.relation.encode(e.relation)
        if rel == vocab.relation.pad_id:
            raise ValueError(
                f"relation {e.relation!r} encodes to the pad id; fact validity "
                "is read back as a nonzero relation"
            )
        sig = vocab.absolute.encode(e.label)
        if sig == vocab.absolute.pad_id:
            raise ValueError(
                f"fact ({e.src!r}, {e.relation!r}, {e.dst!r}) has label "
                f"{e.label!r}, which encodes to the pad id"
            )
        edge_src[i] = position[e.src]
        edge_dst[i] = position[e.dst]
        edge_rel[i] = rel
        edge_abs[i] = sig
        if e.relation in TEMPORAL_RELATIONS and e.temp_label is not None:
            tau = vocab.temporal.encode(e.temp_label)
            if tau == vocab.temporal.pad_id:
                raise ValueError(
                    f"relation {e.relation!r} has temporal label "
                    f"{e.temp_label!r}, which encodes to the pad id"
                )
            edge_temp[i] = tau

    # Counts stay on the graph for logging and truncation warnings rather than
    # riding along in every replay transition.
    graph.meta["n_nodes_packed"] = n_nodes
    graph.meta["n_nodes_dropped"] = n_dropped + int(graph.meta.get("n_context_dropped", 0))
    graph.meta["n_edges_packed"] = len(kept)
    graph.meta["n_edges_dropped"] = len(candidates) - len(kept)
    graph.meta["target_packed"] = bool(node_target.any())
    # Separates the two ways the flag goes dark: the builder never named a
    # target, or it named one that is not a vertex.
    graph.meta["target_resolved"] = target_id is not None
    verify_protected_rows(graph, node_ent, node_target, position,
                          target_id, site_id, vocab)

    packed = {
        "graph_node_ent": node_ent,
        "graph_node_target": node_target,
        "graph_edge_src": edge_src,
        "graph_edge_dst": edge_dst,
        "graph_edge_rel": edge_rel,
        "graph_edge_abs": edge_abs,
        "graph_edge_temp": edge_temp,
    }
    packed["graph_node_bbox"] = node_bbox
    packed["graph_node_centroid"] = node_centroid
    return packed


# Mirrors ``graph.py``. The tests assert the two agree.
_EDGE_KEYS = (
    "graph_edge_src", "graph_edge_dst", "graph_edge_rel", "graph_edge_abs",
    "graph_edge_temp",
)

GRAPH_KEYS = (
    "graph_node_ent", "graph_node_bbox", "graph_node_centroid",
    "graph_node_target", *_EDGE_KEYS,
)


def graph_keys() -> Tuple[str, ...]:
    return GRAPH_KEYS
