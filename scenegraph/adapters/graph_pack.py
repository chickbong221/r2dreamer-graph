"""Hyper-relational per-frame packing.

Nodes fill a compact prefix in vertex-index order, ee first, each carrying one
bbox and one appearance vector per camera plus a flag marking the subtask's
target; each fact is one row of (relation, absolute, temporal). Arrays use the
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


def pack_graph(
    graph: Graph,
    vocab: GraphVocab,
    *,
    n_max: int,
    e_max: int,
    n_cams: int,
    app_dim: int,
    simple: bool = False,
    uid_vocab: int = 256,
) -> Dict[str, np.ndarray]:
    """Pack one frame.

    ``simple`` emits the relation-only contract: no appearance and no boxes,
    plus ``graph_node_uid`` so a decoder can address a node across frames
    without relying on the compact slot, which the registry reuses.
    """
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
    if simple:
        node_uid = np.zeros(n_max, dtype=np.uint16)
        uids = graph.meta.get("node_uids") or {}
    else:
        node_app = np.zeros((n_max, n_cams, app_dim), dtype=np.float16)
        node_bbox = np.zeros((n_max, n_cams, 4), dtype=np.float16)
    target_id = graph.meta.get("active_target_node_id")

    position: Dict[str, int] = {}
    n_nodes = 0
    for node in graph.nodes:
        if n_nodes >= n_max:
            break
        i = n_nodes
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
        if simple:
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
        else:
            if node.bbox is not None:
                node_bbox[i] = node.bbox
            if node.appearance is not None:
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
    if simple:
        packed["graph_node_uid"] = node_uid
    else:
        packed["graph_node_app"] = node_app
        packed["graph_node_bbox"] = node_bbox
    return packed


FULL_GRAPH_KEYS = (
    "graph_node_ent", "graph_node_app", "graph_node_bbox", "graph_node_target",
    "graph_edge_src", "graph_edge_dst", "graph_edge_rel", "graph_edge_abs",
    "graph_edge_temp",
)

# Relation-only contract. Appearance and boxes are absent everywhere -- the
# observation space, the transition, replay and the sampled batch -- rather
# than zeroed, so a simple-mode run pays none of their memory or bandwidth.
SIMPLE_GRAPH_KEYS = (
    "graph_node_ent", "graph_node_uid", "graph_node_target",
    "graph_edge_src", "graph_edge_dst", "graph_edge_rel", "graph_edge_abs",
    "graph_edge_temp",
)

GRAPH_KEYS = FULL_GRAPH_KEYS


def graph_keys(simple: bool) -> Tuple[str, ...]:
    return SIMPLE_GRAPH_KEYS if simple else FULL_GRAPH_KEYS
