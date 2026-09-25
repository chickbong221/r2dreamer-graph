"""Annotate the SO-101 blue-on-red task without an external vision model.

The cubes and gripper are tracked from their stable colours. Spatial and
compatibility labels come from current image measurements with past-only
smoothing; grasp/release comes from the current gripper command. Qualitative
height follows the currently observed grasp/placement state. The output uses
the normal annotation schema and can be rendered and packed unchanged.

    python -m real_robot.preprocessing.annotate_stack_rgb --episodes blue_on_red
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

from ..common import add_config_arguments, load_configs, stable_hash, utc_now
from ..graphs.validate import ValidationSettings, build_annotation, fact_ids
from .annotate_episode import EpisodeAnnotator
from .prepare_videos import prepared_status


def _components(rgb: np.ndarray, kind: str) -> List[Tuple[int, int, int, int, int]]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    if kind == "red_cube":
        mask = cv2.inRange(hsv, (0, 100, 70), (14, 255, 255)) | cv2.inRange(hsv, (170, 100, 70), (179, 255, 255))
    elif kind == "blue_cube":
        mask = cv2.inRange(hsv, (103, 90, 55), (138, 255, 255))
    elif kind == "ee":
        mask = cv2.inRange(hsv, (78, 55, 45), (102, 255, 255))
    else:
        return []
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = [tuple(int(v) for v in row) for row in stats[1:] if int(row[4]) >= 60]
    return sorted(out, key=lambda b: b[4], reverse=True)


def _pick(parts, kind: str, camera: str, previous=None, anchor=None):
    h, w = 480, 640
    candidates = []
    for x, y, bw, bh, area in parts:
        ratio = bw / max(bh, 1)
        if kind.endswith("cube"):
            if not (8 <= bw <= 240 and 8 <= bh <= 240 and 0.35 <= ratio <= 2.8):
                continue
            if camera == "top" and area > 7000:
                continue
        score = 0.0
        cx, cy = x + bw / 2, y + bh / 2
        if previous is not None:
            px, py = previous
            score += np.hypot(cx - px, cy - py)
        elif anchor is not None:
            ax, ay = anchor
            score += np.hypot(cx - ax, cy - ay)
        else:
            score -= np.sqrt(area)
        if camera == "top" and kind == "blue_cube" and previous is None:
            score -= 0.15 * cy
        candidates.append((score, (x, y, x + bw, y + bh), (cx, cy)))
    return min(candidates, default=(None, None, None), key=lambda item: item[0])[1:]


def _box1000(box, shape) -> List[int]:
    if box is None:
        return [0, 0, 0, 0]
    x0, y0, x1, y1 = box
    h, w = shape[:2]
    return [round(1000 * y0 / h), round(1000 * x0 / w),
            round(1000 * y1 / h), round(1000 * x1 / w)]


def track(source, episode: int, every: int = 15):
    n = source.lengths()[episode]
    frames = sorted(set(range(0, n, every)) | {n - 1})
    tracks: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    centers: Dict[Tuple[str, str], Dict[int, Tuple[float, float]]] = {}
    for camera in source.cameras():
        wanted = set(frames)
        previous = {"red_cube": None, "blue_cube": None, "ee": None}
        for index, rgb in source.frames(episode, camera):
            if index not in wanted:
                continue
            for entity in ("red_cube", "blue_cube", "ee"):
                box, center = _pick(_components(rgb, entity), entity, camera, previous[entity])
                if center is not None:
                    previous[entity] = center
                    centers.setdefault((entity, camera), {})[index] = center
                tracks.setdefault((entity, camera), []).append({
                    "frame": index, "visible": box is not None,
                    "box_2d": _box1000(box, rgb.shape)})
            tracks.setdefault(("table", camera), []).append({
                "frame": index, "visible": True, "box_2d": [0, 0, 1000, 1000]})
    return tracks, centers


def _runs(values: Sequence[Any]) -> List[Dict[str, Any]]:
    out, start = [], 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            out.append({"start": start, "end": i - 1, "label": values[start]})
            start = i
    return out


def _phase_boundaries(gripper: np.ndarray) -> Tuple[int, int]:
    n = len(gripper)
    lo, hi = float(np.quantile(gripper, .05)), float(np.quantile(gripper, .95))
    close_at = lo + .35 * max(hi - lo, 1e-6)
    open_at = lo + .55 * max(hi - lo, 1e-6)

    def persistent(mask, start, width=5):
        for i in range(start, max(start, n - width + 1)):
            if bool(np.all(mask[i:i + width])):
                return i
        return n - 1

    grasp = persistent(gripper <= close_at, max(5, n // 12))
    release = persistent(gripper >= open_at, min(grasp + 8, n - 1))
    if release <= grasp:
        release = min(n - 1, grasp + max(10, n // 4))
    return grasp, release


def track_geometry(tracks, n: int) -> Dict[Tuple[str, str], np.ndarray]:
    """Current-or-last-seen boxes, in the annotation's 0..1000 coordinates.

    Forward filling is deliberate: it is causal and is reproducible by a live
    extractor. No value is interpolated from a future frame.
    """
    result: Dict[Tuple[str, str], np.ndarray] = {}
    for key, keyframes in tracks.items():
        boxes = np.full((n, 4), np.nan, np.float32)
        last = None
        by_frame = {int(item["frame"]): item for item in keyframes}
        for frame in range(n):
            item = by_frame.get(frame)
            if item is not None and item["visible"]:
                last = np.asarray(item["box_2d"], np.float32)
            if last is not None:
                boxes[frame] = last
        result[key] = boxes
    return result


def _centre(box: np.ndarray) -> np.ndarray:
    return np.asarray([(box[1] + box[3]) / 2, (box[0] + box[2]) / 2])


def _edge(box: np.ndarray) -> float:
    return float(np.sqrt(max(1.0, (box[3] - box[1]) * (box[2] - box[0]))))


def measured_distance_cm(geometry, src: str, dst: str, frame: int) -> float:
    """Image-plane reference-point distance, scaled by the visible cube edge."""
    if "ee" in (src, dst):
        other = dst if src == "ee" else src
        box = geometry.get((other, "wrist"), np.empty((0, 4)))[frame]
        if not np.isfinite(box).all():
            return 1000.0
        # In the wrist camera the jaws are fixed near the lower centre.
        pixels = float(np.linalg.norm(_centre(box) - np.asarray([500.0, 820.0])))
        return pixels * 3.0 / max(_edge(box), 25.0)
    return measured_pair_distance_cm(geometry, src, dst, frame, "top")


def measured_pair_distance_cm(geometry, src: str, dst: str, frame: int,
                              camera: str) -> float:
    first = geometry.get((src, camera), np.empty((0, 4)))[frame]
    second = geometry.get((dst, camera), np.empty((0, 4)))[frame]
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        return 1000.0
    cube_edges = []
    for entity in ("red_cube", "blue_cube"):
        box = geometry.get((entity, camera), np.empty((0, 4)))[frame]
        if np.isfinite(box).all():
            cube_edges.append(_edge(box))
    scale = 3.0 / max(float(np.median(cube_edges)) if cube_edges else 50.0, 20.0)
    return float(np.linalg.norm(_centre(first) - _centre(second))) * scale


def distance_label(distance_cm: float, *, ee_object: bool) -> str:
    limits = ((2, 6, 15, 30) if ee_object else (3, 8, 20, 35))
    labels = ("very-near", "near", "medium", "far", "very-far")
    return labels[int(np.searchsorted(limits, distance_cm, side="right"))]


def causal_smooth(values: np.ndarray, window: int = 9,
                  alpha: float = .25) -> np.ndarray:
    """Past-only median plus EMA for stable labels from noisy colour boxes."""
    values = np.asarray(values, np.float32)
    out = np.empty_like(values)
    state = float(values[0])
    for t, value in enumerate(values):
        recent = values[max(0, t - int(window) + 1):t + 1]
        finite = recent[np.isfinite(recent) & (recent < 999.0)]
        observed = float(np.median(finite)) if finite.size else float(value)
        state = observed if t == 0 or not np.isfinite(state) else (
            (1.0 - alpha) * state + alpha * observed)
        out[t] = state
    return out


def compatibility(distance_cm: float, *, ee_object: bool) -> str:
    near, partial = ((2.0, 6.0) if ee_object else (3.0, 8.0))
    return "match" if distance_cm < near else "partial-match" if distance_cm < partial else "poor-match"


def decimate_tracks(tracks, every: int, n: int):
    wanted = set(range(0, n, every)) | {n - 1}
    return {key: [item for item in values if int(item["frame"]) in wanted]
            for key, values in tracks.items()}


def _temporal(labels: Sequence[str], k: int, relation: str) -> List[Optional[str]]:
    spatial = {
        "planar-distance": ["very-near", "near", "medium", "far", "very-far"],
        "height-offset": ["far-below", "below", "level", "above", "far-above"],
    }
    compat = ["match", "partial-match", "poor-match"]
    order = spatial.get(relation, compat)
    out: List[Optional[str]] = [None] * len(labels)
    for t in range(k, len(labels)):
        try:
            delta = order.index(labels[t]) - order.index(labels[t - k])
        except ValueError:
            out[t] = "stable"
            continue
        out[t] = ("decrease-fast" if delta <= -2 else "decrease-slow" if delta == -1 else
                  "increase-fast" if delta >= 2 else "increase-slow" if delta == 1 else "stable")
    return out


def make_answer(spec, n: int, grasp: int, release: int, tracks,
                box_every: int = 15) -> Dict[str, Any]:
    geometry = track_geometry(tracks, n)
    distances = {
        fact.key: causal_smooth(np.asarray([
            measured_distance_cm(geometry, fact.src, fact.dst, t)
            for t in range(n)]))
        for fact in spec.facts if fact.relation == "planar-distance"
    }
    blue_red = causal_smooth(np.asarray([
        measured_distance_cm(geometry, "blue_cube", "red_cube", t)
        for t in range(n)]))
    blue_red_wrist = causal_smooth(np.asarray([
        measured_pair_distance_cm(geometry, "blue_cube", "red_cube", t, "wrist")
        for t in range(n)]))
    placed_series = np.zeros(n, dtype=bool)
    settled = False
    for t in range(n):
        # The RGB boxes include perspective and partial occlusion error; the
        # destination cube can become only a small red rim under the blue one.
        # This post-release gate remains based on the current image.
        if t > release and min(blue_red[t], blue_red_wrist[t]) < 15.0:
            settled = True
        placed_series[t] = settled
    series: Dict[Tuple[str, str, str], List[str]] = {}
    for fact in spec.facts:
        values = []
        for t in range(n):
            before, holding, placed = t < grasp, grasp <= t < release, bool(placed_series[t])
            pair, rel = {fact.src, fact.dst}, fact.relation
            distance = distances.get(fact.key, np.full(n, 1000.0))[t]
            if rel == "grasp":
                label = "holds" if holding else "not-holds"
            elif rel == "contact":
                label = "holds" if ((pair == {"ee", "blue_cube"} and holding) or
                                    (pair == {"blue_cube", "red_cube"} and placed)) else "not-holds"
            elif rel == "support":
                if pair == {"red_cube", "table"}:
                    label = "dst-holds" if fact.dst == "table" else "src-holds"
                elif pair == {"blue_cube", "table"}:
                    supported = before
                    label = (("dst-holds" if fact.dst == "table" else "src-holds") if supported else "not-holds")
                else:
                    held = placed
                    holder = "red_cube"
                    label = (("src-holds" if fact.src == holder else "dst-holds") if held else "not-holds")
            elif rel == "planar-distance":
                known_contact = ((pair == {"ee", "blue_cube"} and holding) or
                                 (pair == {"blue_cube", "red_cube"} and placed))
                label = ("very-near" if known_contact else
                         distance_label(float(distance), ee_object="ee" in pair))
            elif rel == "height-offset":
                if pair == {"blue_cube", "red_cube"}:
                    label = "level" if before else "far-above" if holding else "above"
                elif pair == {"ee", "blue_cube"}:
                    label = "level" if holding else "above"
                else:
                    label = "above"
            elif rel == "grasp-compatibility":
                label = "match" if holding else compatibility(float(distance), ee_object=True)
            elif rel == "contact-compatibility":
                label = "match" if (holding or (pair == {"blue_cube", "red_cube"} and placed)) \
                    else compatibility(float(distance), ee_object="ee" in pair)
            elif rel == "support-compatibility":
                label = compatibility(float(blue_red[t]), ee_object=False)
            else:
                label = "unobserved"
            values.append(label)
        series[fact.key] = values

    facts = []
    for fid, fact in zip(fact_ids(spec), spec.facts):
        absolute = series[fact.key]
        temp = _temporal(absolute, spec.temporal_window, fact.relation) if fact.temporal else []
        facts.append({"fact": fid, "absolute": _runs(absolute),
                      "temporal": _runs(temp[spec.temporal_window:]) if temp else []})
        if temp:
            for interval in facts[-1]["temporal"]:
                interval["start"] += spec.temporal_window
                interval["end"] += spec.temporal_window
    return {
        "active_target": [{"start": 0, "end": n - 1, "object": "blue_cube"}],
        "facts": facts,
        "boxes": [{"entity": e, "camera": c, "keyframes": ks}
                  for (e, c), ks in sorted(decimate_tracks(tracks, box_every, n).items())],
        "notes": (f"Causal RGB/action-derived annotation; image-measured distances; "
                  f"gripper closed at {grasp}, opened at {release}.")
    }


def episode_actions(source, episode: int) -> np.ndarray:
    frame = pd.read_parquet(source.data_path(episode), columns=["episode_index", "action"])
    values = frame.loc[frame.episode_index == episode, "action"]
    return np.stack(values.to_numpy())


def main(argv: Optional[Sequence[str]] = None) -> None:
    from ..data.source import LeRobotSource

    parser = argparse.ArgumentParser(description="Annotate blue-on-red episodes from RGB and robot actions.")
    parser.add_argument("--episodes", default="blue_on_red")
    parser.add_argument("--force", action="store_true")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    configs = load_configs(["dataset", "annotation", "graph", "labels"], args.overrides)
    source = LeRobotSource(configs)
    annotator = EpisodeAnnotator(configs, source)
    settings = ValidationSettings.from_config(configs["annotation"], annotator.stride)
    for episode in source.select(args.episodes):
        spec = source.spec(episode)
        if spec.task != "blue_on_red":
            print(f"[stack-rgb] episode {episode}: skipped ({spec.task})", flush=True)
            continue
        path = annotator.path(episode)
        if os.path.isfile(path) and not args.force:
            print(f"[stack-rgb] episode {episode}: exists, use --force to replace", flush=True)
            continue
        prepared, reason = prepared_status(source, episode, configs["annotation"]["videos"])
        if prepared is None:
            print(f"[stack-rgb] episode {episode}: skipped, videos {reason}", flush=True)
            continue
        actions = episode_actions(source, episode)
        grasp, release = _phase_boundaries(actions[:, -1])
        tracks, _ = track(source, episode, 1)
        answer = make_answer(spec, len(actions), grasp, release, tracks, settings.box_every)
        annotation = build_annotation(spec, episode_index=episode, n_frames=len(actions),
                                      fps=source.fps(), answer=answer, settings=settings)
        annotation.input_identity = annotator.input_identity(spec, prepared)
        annotation.provenance = {
            "created": utc_now(), "producer": "annotate_stack_rgb",
            "method": "causal HSV measurements plus current recorded gripper action",
            "answer_hash": stable_hash(answer), "grasp_frame": grasp,
            "release_frame": release}
        annotator.save(annotation)
        print(f"[stack-rgb] episode {episode}: {'valid' if annotation.valid else 'INVALID'} "
              f"grasp={grasp} release={release} issues={len(annotation.issues)}", flush=True)


if __name__ == "__main__":
    main()
