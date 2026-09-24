"""A validated episode -> the packed arrays ``GraphEncoder`` consumes, one row per frame.

The packing is the repository's :func:`scenegraph.adapters.graph_pack.pack_graph`,
unchanged: end effector at row 0, the active target flagged at row 1, then the
task's other entities in vocabulary order. Boxes are linearly interpolated
between two visible keyframes, held from a visible keyframe up to a hidden one,
and absent while hidden. Centroids are unknown and stay zero.
"""

from __future__ import annotations

import bisect
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from scenegraph.adapters.graph_pack import GRAPH_KEYS, pack_graph
from scenegraph.adapters.graph_vocab import GraphVocab
from scenegraph.core.schema import Edge, Graph, Node

from .schema import GraphSpec
from .validate import EpisodeAnnotation

ENV_ID = "real_robot/so101"


def interpolate_track(keyframes: Sequence[Mapping], n_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(boxes (T, 4), visible (T,))`` from one entity's keyframes in one camera."""
    boxes = np.zeros((n_frames, 4), dtype=np.float32)
    visible = np.zeros(n_frames, dtype=bool)
    if not keyframes:
        return boxes, visible
    frames = [int(k["frame"]) for k in keyframes]
    for t in range(n_frames):
        i = max(bisect.bisect_right(frames, t) - 1, 0)
        a = keyframes[i]
        if not a["visible"]:
            continue
        box = np.asarray(a["box"], dtype=np.float32)
        b = keyframes[i + 1] if i + 1 < len(keyframes) else None
        if b is not None and b["visible"] and t > frames[i]:
            w = (t - frames[i]) / float(frames[i + 1] - frames[i])
            box = (1.0 - w) * box + w * np.asarray(b["box"], dtype=np.float32)
        boxes[t] = box
        visible[t] = True
    return boxes, visible


def episode_boxes(spec: GraphSpec, annotation: EpisodeAnnotation) -> Tuple[np.ndarray, np.ndarray]:
    """``(boxes (T, E, C, 4), visible (T, E, C))`` in :attr:`GraphSpec.entities` order."""
    n, n_ent, n_cam = annotation.n_frames, len(spec.entities), len(spec.cameras)
    boxes = np.zeros((n, n_ent, n_cam, 4), dtype=np.float32)
    visible = np.zeros((n, n_ent, n_cam), dtype=bool)
    for e, entity in enumerate(spec.entities):
        for c, camera in enumerate(spec.cameras):
            boxes[:, e, c], visible[:, e, c] = interpolate_track(annotation.boxes.get((entity.id, camera), []), n)
    return boxes, visible


def build_frame_graph(spec: GraphSpec, annotation: EpisodeAnnotation, frame: int,
                      boxes: np.ndarray, visible: np.ndarray) -> Graph:
    """One frame as a repository graph. ``boxes`` ``(E, C, 4)``, ``visible`` ``(E, C)``."""
    graph = Graph(frame=int(frame), env_id=ENV_ID, camera="+".join(spec.cameras))
    for index, entity in enumerate(spec.entities):
        box = np.asarray(boxes[index], dtype=np.float32) * np.asarray(visible[index], dtype=bool)[:, None]
        good = (box[:, 1] > box[:, 0]) & (box[:, 3] > box[:, 2])
        graph.nodes.append(Node(
            node_id=entity.node_id,
            node_type=entity.type,
            name=entity.name,
            visible=bool(good.any()),
            in_frame=True,
            bbox=box * good[:, None],
            source="annotation",
            attributes={} if entity.type == "ee" else {"whitelist_key": entity.key},
        ))
    for index, fact in enumerate(spec.facts):
        label = annotation.absolute[index][frame]
        if label is None:
            continue
        graph.edges.append(Edge(
            src=spec.entity(fact.src).node_id,
            dst=spec.entity(fact.dst).node_id,
            relation=fact.relation,
            label=label,
            temp_label=annotation.temporal[index][frame] if fact.temporal else None,
        ))
    target = annotation.active_target[frame]
    graph.meta["active_target_node_id"] = spec.entity(target).node_id if target else None
    graph.meta["active_subtask"] = f"target={target}" if target else ""
    return graph


def frame_is_complete(spec: GraphSpec, annotation: EpisodeAnnotation, frame: int) -> bool:
    """Every fact labelled and a target named on this frame."""
    if annotation.active_target[frame] is None:
        return False
    for index, fact in enumerate(spec.facts):
        if annotation.absolute[index][frame] is None:
            return False
        if fact.temporal and frame >= spec.temporal_window and annotation.temporal[index][frame] is None:
            return False
    return True


def pack_episode(spec: GraphSpec, vocab: GraphVocab, annotation: EpisodeAnnotation
                 ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """``(arrays, graph_valid)``; every array stacked over the episode's frames."""
    boxes, visible = episode_boxes(spec, annotation)
    frames: List[Dict[str, np.ndarray]] = []
    valid = np.zeros(annotation.n_frames, dtype=bool)
    for t in range(annotation.n_frames):
        graph = build_frame_graph(spec, annotation, t, boxes[t], visible[t])
        frames.append(pack_graph(graph, vocab, n_max=spec.n_max, e_max=spec.e_max,
                                 n_cams=len(spec.cameras), use_target_flag=True))
        valid[t] = frame_is_complete(spec, annotation, t)
    return {key: np.stack([frame[key] for frame in frames]) for key in GRAPH_KEYS}, valid
