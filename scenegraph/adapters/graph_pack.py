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


def _row_assignment(graph, n_max: int, target_id, fixed_rows: bool):
    """``([(row, node)], n_dropped)`` for one frame.

    With ``fixed_rows`` the pooled schema pins two rows by meaning rather than
    by arrival order: row 0 is the end effector, row 1 is the active subtask
    target, and everything else fills upward from row 2. The vertex registry
    cannot supply this -- it hands the target whichever index happened to be
    free when it was first admitted -- and the encoder now needs the target at
    a fixed row so an occluded one keeps the same identity frame to frame.

    Row 1 stays padding until the target is first observed. Reserving it costs
    one object row, so a scene that fills ``n_max`` without the target raises
    here -- a silently truncated vertex is a fact the model never learns to
    predict, and retention leaves no vertex that is safe to lose.
    """
    nodes = list(graph.nodes)
    if len(nodes) > n_max:
        raise RuntimeError(
            f"node budget exceeded: {len(nodes)} vertices against n_max="
            f"{n_max}. env={graph.env_id} frame={graph.frame} "
            f"nodes={[n.node_id for n in nodes]}. Retention never evicts."
        )
    if not fixed_rows:
        return list(enumerate(nodes[:n_max])), max(0, len(nodes) - n_max)

    rows = []
    rest = []
    target_row_taken = False
    for node in nodes:
        if node.node_type == "ee":
            if not any(row == 0 for row, _ in rows):
                rows.append((0, node))
            continue
        if node.node_id == target_id and not target_row_taken and n_max > 1:
            rows.append((1, node))
            target_row_taken = True
            continue
        rest.append(node)

    next_row = 2
    for node in rest:
        if next_row >= n_max:
            # Reserving row 1 for an unseen target costs one object row. Under
            # retention that is still a vertex nothing can hold, so it raises
            # like any other overflow rather than vanishing.
            raise RuntimeError(
                f"node budget exceeded: {node.node_id!r} has no row. "
                f"n_max={n_max} with row 1 reserved for target "
                f"{target_id!r}. env={graph.env_id} frame={graph.frame} "
                f"nodes={[n.node_id for n in nodes]}."
            )
        rows.append((next_row, node))
        next_row += 1
    return rows, 0


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

    position: Dict[str, int] = {}
    assigned, n_dropped = _row_assignment(graph, n_max, target_id, fixed_rows)
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
    graph.meta["n_nodes_dropped"] = n_dropped
    graph.meta["n_edges_packed"] = len(kept)
    graph.meta["n_edges_dropped"] = len(candidates) - len(kept)
    graph.meta["target_packed"] = bool(node_target.any())
    # Separates the two ways the flag goes dark: the builder never named a
    # target, or it named one that is not a vertex.
    graph.meta["target_resolved"] = target_id is not None

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
