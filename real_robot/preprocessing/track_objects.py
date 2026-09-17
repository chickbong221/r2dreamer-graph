"""Carry Gemini's boxes and points between its keyframes with optical flow.

    python -m real_robot.preprocessing.track_objects --episodes pilot

Gemini places every entity's box and named points at initialisation and at
correction frames. Between those, pyramidal Lucas-Kanade flow moves them:
corner features inside the box are tracked frame to frame, points whose
forward-backward round trip drifts more than ``forward_backward_px`` are
dropped, the box follows the median feature displacement and the median
change in feature spread, and each named point follows its own flow (or the
features' median displacement when its own round trip fails). A track with too
few surviving features is lost until the next keyframe, and every keyframe
resets its box and points outright. No segmentation is involved.

Tracking runs on the source frames, not on the labelled copies Gemini watched:
the burned-in text would otherwise attract features. The two share geometry
frame for frame, so Gemini's coordinates apply to both.

In ``full_episode`` mode each segment between keyframes is tracked forwards
from its start and backwards from its end, and the two are blended with
weights that favour the nearer keyframe. ``past_only`` tracks forwards only,
from the latest keyframe at or before each frame, so no position uses the
future.

A named point Gemini marked hidden at a keyframe is not carried past it: the
track of that point is lost until a keyframe places it again.

Tracks are reused only when they were made from the current annotation file,
tracking settings and source videos; otherwise they are made again. An episode
whose annotation is missing, invalid or itself stale -- made from another
prompt, other bins, other Gemini settings or other videos than the current ones
-- is not tracked at all.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..common import add_config_arguments, load_configs, stable_hash, utc_now, write_json


def _norm_to_px(box: Sequence[float], width: int, height: int) -> np.ndarray:
    x0, x1, y0, y1 = box
    return np.array([x0 * width, y0 * height, x1 * width, y1 * height], dtype=np.float64)


def _px_to_norm(box: np.ndarray, width: int, height: int) -> np.ndarray:
    x0, y0, x1, y1 = box
    out = np.array([x0 / width, x1 / width, y0 / height, y1 / height], dtype=np.float64)
    return np.clip(out, 0.0, 1.0)


def init_features(gray: np.ndarray, box: np.ndarray, settings: Mapping[str, Any]) -> np.ndarray:
    import cv2

    height, width = gray.shape
    x0, y0, x1, y1 = box
    # A slightly shrunken box keeps features off the background at its edges.
    dx, dy = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
    mask = np.zeros_like(gray, dtype=np.uint8)
    xa, xb = int(max(0, x0 + dx)), int(min(width, x1 - dx))
    ya, yb = int(max(0, y0 + dy)), int(min(height, y1 - dy))
    if xb - xa < 3 or yb - ya < 3:
        xa, xb, ya, yb = int(max(0, x0)), int(min(width, x1)), int(max(0, y0)), int(min(height, y1))
    if xb <= xa or yb <= ya:
        return np.zeros((0, 2), dtype=np.float32)
    mask[ya:yb, xa:xb] = 255
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=int(settings["max_corners"]),
                                      qualityLevel=float(settings["quality_level"]),
                                      minDistance=int(settings["min_distance_px"]), mask=mask)
    if corners is None:
        return np.zeros((0, 2), dtype=np.float32)
    return corners.reshape(-1, 2).astype(np.float32)


def flow(prev_gray: np.ndarray, gray: np.ndarray, points: np.ndarray, settings: Mapping[str, Any]
         ) -> Tuple[np.ndarray, np.ndarray]:
    """``(moved, good)`` with a forward-backward consistency check."""
    import cv2

    if points.shape[0] == 0:
        return points, np.zeros(0, dtype=bool)
    params = dict(winSize=(int(settings["lk_window_px"]), int(settings["lk_window_px"])),
                  maxLevel=int(settings["lk_levels"]),
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p0 = points.reshape(-1, 1, 2).astype(np.float32)
    p1, status1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **params)
    p0r, status2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **params)
    error = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
    good = (status1.reshape(-1) == 1) & (status2.reshape(-1) == 1) & (error <= float(settings["forward_backward_px"]))
    return p1.reshape(-1, 2), good


def move_box(box: np.ndarray, old: np.ndarray, new: np.ndarray, width: int, height: int) -> np.ndarray:
    """Median translation plus median change of spread about the feature centre."""
    shift = np.median(new - old, axis=0)
    c_old, c_new = np.median(old, axis=0), np.median(new, axis=0)
    d_old = np.linalg.norm(old - c_old, axis=1)
    d_new = np.linalg.norm(new - c_new, axis=1)
    keep = d_old > 1.0
    scale = float(np.clip(np.median(d_new[keep] / d_old[keep]), 0.8, 1.25)) if keep.sum() >= 2 else 1.0
    cx, cy = (box[0] + box[2]) / 2.0 + shift[0], (box[1] + box[3]) / 2.0 + shift[1]
    half_w, half_h = (box[2] - box[0]) * scale / 2.0, (box[3] - box[1]) * scale / 2.0
    out = np.array([cx - half_w, cy - half_h, cx + half_w, cy + half_h])
    out[[0, 2]] = np.clip(out[[0, 2]], 0, width)
    out[[1, 3]] = np.clip(out[[1, 3]], 0, height)
    return out


def track_segment(grays: Sequence[np.ndarray], keyframes: Mapping[int, Mapping[str, Any]],
                  point_names: Sequence[str], settings: Mapping[str, Any], backward: bool) -> Dict[str, np.ndarray]:
    """Track one (entity, camera) through the whole episode in one direction."""
    n = len(grays)
    height, width = grays[0].shape
    boxes = np.full((n, 4), np.nan)
    points = np.full((n, len(point_names), 2), np.nan)
    visible = np.zeros(n, dtype=bool)
    quality = np.zeros(n)
    since = np.full(n, np.inf)
    order = range(n - 1, -1, -1) if backward else range(n)

    box = None
    features = np.zeros((0, 2), dtype=np.float32)
    named: Dict[str, Optional[np.ndarray]] = {}
    initial_count = 1
    anchor = None
    previous = None
    held = 0
    max_hold = int(settings.get("max_hold_frames", 15))
    minimum = int(settings["min_points"])
    for t in order:
        gray = grays[t]
        key = keyframes.get(t)
        holding = False
        if key is not None:
            held = 0
            if key["visible"] and key.get("box") is not None:
                box = _norm_to_px(key["box"], width, height)
                features = init_features(gray, box, settings)
                initial_count = max(1, features.shape[0])
                named = {name: (np.array([key["points"][name][0] * width, key["points"][name][1] * height])
                                if name in key.get("points", {}) else None) for name in point_names}
                anchor = t
            else:
                box, features, named, anchor = None, np.zeros((0, 2), dtype=np.float32), {}, None
        elif box is not None and previous is not None:
            moved, good = flow(grays[previous], gray, features, settings)
            if good.sum() >= minimum:
                held = 0
                box = move_box(box, features[good], moved[good], width, height)
                shift = np.median(moved[good] - features[good], axis=0)
                features = moved[good]
                for name, value in list(named.items()):
                    if value is None:
                        continue
                    new_point, ok = flow(grays[previous], gray, value[None].astype(np.float32), settings)
                    named[name] = new_point[0] if ok[0] else value + shift
            else:
                # Too few features to measure motion. A shiny pot or a plain
                # lid can be that textureless while sitting perfectly still, so
                # the box is held where it was -- flagged with zero quality and
                # re-seeded -- for a bounded number of frames before the track
                # is given up until the next keyframe.
                held += 1
                if held > max_hold:
                    box, features, named, anchor = None, np.zeros((0, 2), dtype=np.float32), {}, None
                else:
                    holding = True
                    features = init_features(gray, box, settings)
        if box is not None and box[2] - box[0] > 1 and box[3] - box[1] > 1:
            boxes[t] = box
            visible[t] = True
            if key is not None:
                quality[t] = 1.0
            elif holding:
                quality[t] = 0.0
            else:
                quality[t] = min(1.0, features.shape[0] / initial_count)
            since[t] = abs(t - anchor) if anchor is not None else np.inf
            for index, name in enumerate(point_names):
                value = named.get(name)
                if value is not None and 0 <= value[0] <= width and 0 <= value[1] <= height:
                    points[t, index] = value
        previous = t
    return {"boxes": boxes, "points": points, "visible": visible, "quality": quality, "since": since}


def blend(forward: Dict[str, np.ndarray], backward: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Weight each direction by closeness to its own keyframe."""
    both = forward["visible"] & backward["visible"]
    df, db = forward["since"], backward["since"]
    wf = np.where(both, db / np.maximum(df + db, 1e-9), forward["visible"].astype(float))
    wb = 1.0 - wf
    out: Dict[str, np.ndarray] = {}
    out["visible"] = forward["visible"] | backward["visible"]
    fb, bb = np.nan_to_num(forward["boxes"]), np.nan_to_num(backward["boxes"])
    out["boxes"] = np.where(out["visible"][:, None], wf[:, None] * fb + wb[:, None] * bb, np.nan)
    fp, bp = forward["points"], backward["points"]
    fp_ok, bp_ok = np.isfinite(fp).all(-1), np.isfinite(bp).all(-1)
    wpf = np.where(fp_ok & bp_ok, wf[:, None], fp_ok.astype(float))
    out["points"] = np.where((fp_ok | bp_ok)[..., None],
                             wpf[..., None] * np.nan_to_num(fp) + (1 - wpf[..., None]) * np.nan_to_num(bp), np.nan)
    out["quality"] = wf * forward["quality"] + wb * backward["quality"]
    return out


def track_episode(frames: Mapping[str, Sequence[np.ndarray]], keyframes: Sequence[Mapping[str, Any]],
                  spec, settings: Mapping[str, Any], mode: str) -> Dict[str, np.ndarray]:
    """Arrays over ``(T, entities, cameras, ...)`` in graph-spec order, normalised coordinates."""
    import cv2

    cameras = list(spec.cameras)
    entities = list(spec.entity_ids)
    point_names = [f"{owner}:{name}" for owner in entities for name in spec.points.get(owner, ())]
    n = len(frames[cameras[0]])
    boxes = np.zeros((n, len(entities), len(cameras), 4), dtype=np.float32)
    visible = np.zeros((n, len(entities), len(cameras)), dtype=bool)
    quality = np.zeros((n, len(entities), len(cameras)), dtype=np.float32)
    points = np.full((n, len(point_names), len(cameras), 2), np.nan, dtype=np.float32)
    for c, camera in enumerate(cameras):
        grays = [cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY) for rgb in frames[camera]]
        if len(grays) != n:
            raise ValueError(f"camera {camera} has {len(grays)} frames, {cameras[0]} has {n}")
        height, width = grays[0].shape
        for e, entity in enumerate(entities):
            slot = {int(k["frame"]): k for k in keyframes if k["camera"] == camera and k["object"] == entity}
            if not slot:
                continue
            names = list(spec.points.get(entity, ()))
            forward = track_segment(grays, slot, names, settings, backward=False)
            if mode == "full_episode" and bool(settings.get("bidirectional", True)):
                result = blend(forward, track_segment(grays, slot, names, settings, backward=True))
            else:
                result = forward
            for t in range(n):
                if result["visible"][t] and np.all(np.isfinite(result["boxes"][t])):
                    boxes[t, e, c] = _px_to_norm(result["boxes"][t], width, height)
                    visible[t, e, c] = True
                    quality[t, e, c] = result["quality"][t]
            for index, name in enumerate(names):
                row = point_names.index(f"{entity}:{name}")
                pts = result["points"][:, index]
                ok = np.isfinite(pts).all(-1)
                points[ok, row, c, 0] = pts[ok, 0] / width
                points[ok, row, c, 1] = pts[ok, 1] / height
    return {"boxes": boxes, "visible": visible, "quality": quality, "points": points,
            "point_names": np.array(point_names)}


def tracks_inputs(source, episode: int, settings: Mapping[str, Any]) -> Dict[str, Any]:
    from .artifacts import file_digest, source_video_digest

    return {
        "annotation": file_digest(source.annotation_path(episode)),
        "tracking": stable_hash(dict(settings)),
        "videos": {camera: source_video_digest(source, episode, camera) for camera in source.spec.cameras},
        "mode": source.mode,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    from ..data.episode_dataset import RawEpisodeSource
    from .artifacts import reusable
    from .freshness import ArtifactChain, problems_text
    from .prepare_videos import read_frames

    parser = argparse.ArgumentParser(description="Track Gemini keyframes through each episode.")
    parser.add_argument("--episodes", default="pilot")
    parser.add_argument("--mode", choices=("full_episode", "past_only"), default=None)
    parser.add_argument("--force", action="store_true")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    configs = load_configs(["dataset", "annotation", "graph"], args.overrides)
    source = RawEpisodeSource(configs, mode=args.mode)
    settings = configs["annotation"]["tracking"]
    chain = ArtifactChain(configs, source)
    refused: Dict[int, Any] = {}
    for episode in source.select(args.episodes):
        path = source.tracks_path(episode)
        problems = chain.annotation(episode)
        if problems:
            refused[episode] = problems
            print(f"[track] episode {episode}: not tracked ({'; '.join(problems)})", flush=True)
            continue
        annotation = source.annotation(episode)
        inputs = tracks_inputs(source, episode, settings)
        current, reason = reusable(path, inputs)
        if current and not args.force:
            print(f"[track] episode {episode}: current, kept")
            continue
        if os.path.isfile(path):
            print(f"[track] episode {episode}: tracking again ({reason})", flush=True)
        frames = {camera: read_frames(source.video_path(episode, camera)) for camera in source.spec.cameras}
        result = track_episode(frames, annotation.keyframes, source.spec, settings, source.mode)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **result)
        coverage = {
            f"{entity}@{camera}": float(result["visible"][:, e, c].mean())
            for e, entity in enumerate(source.spec.entity_ids) for c, camera in enumerate(source.spec.cameras)
        }
        points = {name: float(np.isfinite(result["points"][:, p, 0, 0]).mean())
                  for p, name in enumerate(result["point_names"].tolist())}
        write_json(path[:-4] + ".json", {
            "episode_index": episode, "mode": source.mode, "created": utc_now(), "inputs": inputs,
            "visible_fraction": coverage, "point_known_fraction_primary_camera": points,
            "annotation_prompt_version": annotation.provenance.get("prompt_version"),
        })
        print(f"[track] episode {episode}: " + ", ".join(f"{k}={v:.2f}" for k, v in coverage.items()), flush=True)
    if refused:
        raise SystemExit(f"[track] {len(refused)} episode(s) without a current, valid annotation were not "
                         "tracked:\n  " + problems_text(refused))


if __name__ == "__main__":
    main()
