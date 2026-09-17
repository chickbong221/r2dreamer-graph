"""Annotated kitchen frames -> the packed arrays ``GraphEncoder`` consumes.

The packing itself is the repository's :func:`scenegraph.adapters.graph_pack.
pack_graph`, unchanged. This adapter only supplies what it expects: a
:class:`scenegraph.core.schema.Graph` whose nodes carry a whitelist key, one
``[x0, x1, y0, y1]`` box per camera and a world-frame centroid, whose edges are
already canonical, and whose meta names the active target. ``pack_graph`` then
enforces the row contract itself -- end effector at row 0 with the reserved
entity id, the flagged target at row 1 -- and raises rather than truncating
when a budget is exceeded.

Switching the target from banana to lid moves the lid into row 1 and the
banana into an ordinary row. The banana keeps its entity id and its facts; only
its row changes, exactly as the simulator path handles a new subtask.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from scenegraph.adapters.graph_pack import GRAPH_KEYS, pack_graph
from scenegraph.adapters.graph_vocab import GraphVocab
from scenegraph.core.schema import Edge, Graph, Node

from .schema import GraphSpec
from .validate import EpisodeAnnotation

ENV_ID = "real_robot/aloha_kitchen"


def build_frame_graph(
    spec: GraphSpec,
    annotation: EpisodeAnnotation,
    frame: int,
    boxes: np.ndarray,
    visible: np.ndarray,
    centroids: np.ndarray,
    centroid_known: np.ndarray,
) -> Graph:
    """One frame as a repository graph.

    ``boxes`` is ``(E, C, 4)`` normalised ``[x0, x1, y0, y1]``, ``visible``
    ``(E, C)``, ``centroids`` ``(E, 3)`` in the declared scene frame and
    ``centroid_known`` ``(E,)``, all in :attr:`GraphSpec.entities` order.
    """
    n_cams = len(spec.cameras)
    graph = Graph(frame=int(frame), env_id=ENV_ID, camera="+".join(spec.cameras))
    for index, entity in enumerate(spec.entities):
        seen = np.asarray(visible[index], dtype=bool).reshape(n_cams)
        box = np.asarray(boxes[index], dtype=np.float32).reshape(n_cams, 4) * seen[:, None]
        # A box that is not strictly positive in both extents reads back as
        # "not visible" in the encoder, so a degenerate tracked box is dropped
        # here rather than stored as a zero-area visible node.
        good = (box[:, 1] > box[:, 0]) & (box[:, 3] > box[:, 2])
        box = box * good[:, None]
        pose = None
        if bool(centroid_known[index]):
            xyz = np.asarray(centroids[index], dtype=np.float64).reshape(3)
            if np.all(np.isfinite(xyz)):
                pose = [float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0, 0.0, 0.0, 0.0]
        graph.nodes.append(Node(
            node_id=entity.node_id,
            node_type=entity.type,
            name=entity.name,
            visible=bool(good.any()),
            in_frame=True,
            pose_world=pose,
            bbox=box,
            source="annotation",
            attributes={} if entity.type == "ee" else {"whitelist_key": entity.key},
        ))
    for index, fact in enumerate(spec.facts):
        label = annotation.absolute[index][frame]
        if label is None:
            continue
        temp = annotation.temporal[index][frame] if fact.temporal else None
        graph.edges.append(Edge(
            src=spec.entity(fact.src).node_id,
            dst=spec.entity(fact.dst).node_id,
            relation=fact.relation,
            label=label,
            temp_label=temp,
        ))
    target = annotation.active_target[frame]
    graph.meta["active_target_node_id"] = spec.entity(target).node_id if target else None
    graph.meta["annotation_mode"] = annotation.mode
    graph.meta["active_subtask"] = f"target={target}" if target else ""
    return graph


def pack_frame(spec: GraphSpec, vocab: GraphVocab, graph: Graph) -> Dict[str, np.ndarray]:
    return pack_graph(graph, vocab, n_max=spec.n_max, e_max=spec.e_max,
                      n_cams=len(spec.cameras), use_target_flag=True)


def frame_is_complete(spec: GraphSpec, annotation: EpisodeAnnotation, frame: int) -> bool:
    """Every configured fact labelled and a target named on this frame."""
    if annotation.active_target[frame] is None:
        return False
    for index, fact in enumerate(spec.facts):
        if annotation.absolute[index][frame] is None:
            return False
        if fact.temporal and frame >= spec.temporal_window and annotation.temporal[index][frame] is None:
            return False
    return True


def pack_episode(
    spec: GraphSpec,
    vocab: GraphVocab,
    annotation: EpisodeAnnotation,
    boxes: np.ndarray,
    visible: np.ndarray,
    centroids: np.ndarray,
    centroid_known: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """``(arrays, graph_valid)`` with every array stacked over frames.

    ``boxes`` ``(T, E, C, 4)``, ``visible`` ``(T, E, C)``, ``centroids``
    ``(T, E, 3)``, ``centroid_known`` ``(T, E)``.
    """
    n = annotation.n_frames
    if boxes.shape[0] != n or centroids.shape[0] != n:
        raise ValueError(
            f"geometry covers {boxes.shape[0]}/{centroids.shape[0]} frames, annotation {n}"
        )
    packed = []
    valid = np.zeros(n, dtype=bool)
    for t in range(n):
        graph = build_frame_graph(spec, annotation, t, boxes[t], visible[t], centroids[t],
                                  centroid_known[t])
        packed.append(pack_frame(spec, vocab, graph))
        valid[t] = frame_is_complete(spec, annotation, t)
    arrays = {key: np.stack([frame[key] for frame in packed]) for key in GRAPH_KEYS}
    return arrays, valid


def empty_frame(spec: GraphSpec) -> Dict[str, np.ndarray]:
    """All-padding arrays with the packed dtypes, for frames with no graph."""
    n_cams = len(spec.cameras)
    return {
        "graph_node_ent": np.zeros(spec.n_max, dtype=np.uint8),
        "graph_node_target": np.zeros(spec.n_max, dtype=np.uint8),
        "graph_node_bbox": np.zeros((spec.n_max, n_cams, 4), dtype=np.float16),
        "graph_node_centroid": np.zeros((spec.n_max, 3), dtype=np.float32),
        "graph_edge_src": np.zeros(spec.e_max, dtype=np.uint8),
        "graph_edge_dst": np.zeros(spec.e_max, dtype=np.uint8),
        "graph_edge_rel": np.zeros(spec.e_max, dtype=np.uint8),
        "graph_edge_abs": np.zeros(spec.e_max, dtype=np.uint8),
        "graph_edge_temp": np.zeros(spec.e_max, dtype=np.uint8),
    }


def node_rows(spec: GraphSpec, packed: Dict[str, np.ndarray], vocab: GraphVocab) -> Dict[str, int]:
    """Entity id -> packed row for one frame, read back from the entity ids."""
    id_of_key = {e.key: e.id for e in spec.entities}
    key_of_token = {index: token for token, index in vocab.entity.token_to_id.items()}
    out = {}
    for row, token in enumerate(packed["graph_node_ent"].tolist()):
        if token == 0:
            continue
        out[id_of_key[key_of_token[token]]] = row
    return out
