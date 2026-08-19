"""Hyper-relational per-frame packing.

Nodes fill a compact prefix, each carrying one bbox and one appearance vector
per camera plus a flag marking the subtask's target; each fact is one row of
(relation, absolute, temporal). Ordering is vertex-index order under the full
and slot schemas; the pooled schema instead pins row 0 to the end effector and
row 1 to the active target, and adds a world-frame centroid per node. Arrays use the
narrowest dtype holding their vocabulary since these land in the replay buffer
every step; the encoder casts back on read.

Nothing derivable is stored. Index zero is padding in every vocabulary, so
validity, per-camera visibility, whether a camera ever observed a node, and
both counts all follow from the ids, the boxes and the embedding norms -- see
``graph_encoder`` for the exact derivations the model reads them back with.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..core.relation_rules import (
    AFFORDANCE_RELATIONS,
    PHYSICAL_RELATIONS,
    TEMPORAL_RELATIONS,
)
from ..core.schema import Graph
from .graph_vocab import GraphVocab, entity_key_for


# Mirrors ``graph.py``; the tests assert the two agree. Duplicated rather than
# imported so this package never reaches into the model package.
SCHEMA_FULL = "full"
SCHEMA_SIMPLE_POOLED = "simple_pooled_bbox"
SCHEMA_SIMPLE_SLOT = "simple_slot_uid"
GRAPH_SCHEMAS = (SCHEMA_FULL, SCHEMA_SIMPLE_POOLED, SCHEMA_SIMPLE_SLOT)


def graph_schema(simple: bool, state_mode: str = "pooled") -> str:
    if not simple:
        return SCHEMA_FULL
    if str(state_mode) == "slots":
        return SCHEMA_SIMPLE_SLOT
    return SCHEMA_SIMPLE_POOLED


_PHYSICAL = frozenset(PHYSICAL_RELATIONS)
_AFFORDANCE = frozenset(AFFORDANCE_RELATIONS)


def _edge_priority(edge) -> Tuple[int, int]:
    """Truncation order: physical state, then affordance, then spatial.
    Observed facts outrank retained ones within a family."""
    if edge.relation in _PHYSICAL:
        family = 0
    elif edge.relation in _AFFORDANCE:
        family = 1
    else:
        family = 2
    return (family, int(edge.stale))


def _row_assignment(graph, n_max: int, target_id, fixed_rows: bool):
    """``([(row, node)], n_dropped)`` for one frame.

    With ``fixed_rows`` the pooled schema pins two rows by meaning rather than
    by arrival order: row 0 is the end effector, row 1 is the active subtask
    target, and everything else fills upward from row 2. The vertex registry
    cannot supply this -- it hands the target whichever index happened to be
    free when it was first admitted -- and the encoder now needs the target at
    a fixed row so an occluded one keeps the same identity frame to frame.

    Row 1 stays padding until the target is first observed. Reserving it costs
    one object row in a frame that has more visible whitelisted objects than
    rows for them; the drop is returned rather than swallowed, because a
    silently truncated vertex is a fact the model never learns to predict.
    """
    nodes = list(graph.nodes)
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
    dropped = 0
    for node in rest:
        if next_row >= n_max:
            dropped += 1
            continue
        rows.append((next_row, node))
        next_row += 1
    return rows, dropped


def pack_graph(
    graph: Graph,
    vocab: GraphVocab,
    *,
    n_max: int,
    e_max: int,
    n_cams: int,
    app_dim: int,
    schema: str = SCHEMA_FULL,
    uid_vocab: int = 256,
) -> Dict[str, np.ndarray]:
    """Pack one frame under one of the three observation schemas.

    ``full`` carries appearance and boxes. ``simple_pooled_bbox`` keeps the
    boxes and drops appearance and identity: a node is addressed by where it
    currently is. ``simple_slot_uid`` is relation-only and carries
    ``graph_node_uid`` instead, which is what names a node across frames for
    slot alignment; the compact row cannot, because the registry reuses it.
    """
    if schema not in SCHEMA_KEYS:
        raise ValueError(
            f"unknown graph schema {schema!r}; expected one of {GRAPH_SCHEMAS}"
        )
    want_uid = schema == SCHEMA_SIMPLE_SLOT
    want_bbox = schema in (SCHEMA_FULL, SCHEMA_SIMPLE_POOLED)
    want_app = schema == SCHEMA_FULL
    # Object-frame position, packed only where the encoder reads it. Boxes say
    # where a node is on a screen; this says where it is in the world, which is
    # what survives a node going invisible.
    want_centroid = schema == SCHEMA_SIMPLE_POOLED
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
    uids = {}
    if want_uid:
        if uid_vocab > 256:
            raise ValueError(
                f"uid_vocab={uid_vocab} exceeds 256; graph_node_uid is packed "
                "as uint8 because the replay buffer has no uint16 index kernel"
            )
        node_uid = np.zeros(n_max, dtype=np.uint8)
        uids = graph.meta.get("node_uids") or {}
    if want_app:
        node_app = np.zeros((n_max, n_cams, app_dim), dtype=np.float16)
    if want_bbox:
        node_bbox = np.zeros((n_max, n_cams, 4), dtype=np.float16)
    if want_centroid:
        # float32, not float16: eight rows of three is 96 bytes a frame, and
        # half precision would quantise a table-scale scene to millimetres.
        node_centroid = np.zeros((n_max, 3), dtype=np.float32)
    target_id = graph.meta.get("active_target_node_id")

    position: Dict[str, int] = {}
    assigned, n_dropped = _row_assignment(graph, n_max, target_id, want_centroid)
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
        if want_uid:
            uid = uids.get(node.node_id)
            if uid is None:
                raise ValueError(
                    f"node {node.node_id!r} has no uid; the builder must assign "
                    "one to every emitted vertex in simple mode"
                )
            if not 1 <= int(uid) < uid_vocab:
                raise ValueError(
                    f"node {node.node_id!r} uid={uid} outside [1, {uid_vocab}); "
                    "zero is padding and wrapping would alias two objects"
                )
            node_uid[i] = int(uid)
        if want_bbox and node.bbox is not None:
            node_bbox[i] = node.bbox
        if want_centroid and node.pose_world is not None:
            node_centroid[i] = np.asarray(node.pose_world[:3], dtype=np.float32)
        if want_app and node.appearance is not None:
            node_app[i] = node.appearance
        if node.node_id == target_id:
            node_target[i] = 1
        position[node.node_id] = i
        n_nodes += 1

    candidates = [
        e for e in graph.edges
        if e.src in position and e.dst in position
    ]
    candidates.sort(key=_edge_priority)
    kept = candidates[:e_max]

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
    if want_uid:
        packed["graph_node_uid"] = node_uid
    if want_app:
        packed["graph_node_app"] = node_app
    if want_bbox:
        packed["graph_node_bbox"] = node_bbox
    if want_centroid:
        packed["graph_node_centroid"] = node_centroid
    return packed


# The three schemas, mirroring ``graph.py``. A field a schema does not name is
# absent everywhere -- the observation space, the transition, replay and the
# sampled batch -- rather than zeroed, so a run pays none of its bandwidth.
_EDGE_KEYS = (
    "graph_edge_src", "graph_edge_dst", "graph_edge_rel", "graph_edge_abs",
    "graph_edge_temp",
)

FULL_GRAPH_KEYS = (
    "graph_node_ent", "graph_node_app", "graph_node_bbox", "graph_node_target",
    *_EDGE_KEYS,
)

SIMPLE_POOLED_GRAPH_KEYS = (
    "graph_node_ent", "graph_node_bbox", "graph_node_centroid",
    "graph_node_target", *_EDGE_KEYS,
)

SIMPLE_SLOT_GRAPH_KEYS = (
    "graph_node_ent", "graph_node_uid", "graph_node_target", *_EDGE_KEYS,
)

SCHEMA_KEYS = {
    SCHEMA_FULL: FULL_GRAPH_KEYS,
    SCHEMA_SIMPLE_POOLED: SIMPLE_POOLED_GRAPH_KEYS,
    SCHEMA_SIMPLE_SLOT: SIMPLE_SLOT_GRAPH_KEYS,
}

GRAPH_KEYS = FULL_GRAPH_KEYS
SIMPLE_GRAPH_KEYS = SIMPLE_SLOT_GRAPH_KEYS


def graph_keys(schema: str) -> Tuple[str, ...]:
    try:
        return SCHEMA_KEYS[schema]
    except KeyError:
        raise ValueError(
            f"unknown graph schema {schema!r}; expected one of {GRAPH_SCHEMAS}"
        ) from None
