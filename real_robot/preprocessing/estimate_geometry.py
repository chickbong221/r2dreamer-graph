"""Metric geometry from Depth Pro, tracked points and the recorded arm.

    python -m real_robot.preprocessing.estimate_geometry all          --episodes pilot
    python -m real_robot.preprocessing.estimate_geometry check-camera --episodes all
    python -m real_robot.preprocessing.estimate_geometry depth        --episodes pilot
    python -m real_robot.preprocessing.estimate_geometry align        --episodes pilot
    python -m real_robot.preprocessing.estimate_geometry measure      --episodes pilot

1. **check-camera** -- whether the high camera stayed put across and within
   episodes (ORB features, RANSAC homography, median shift in pixels), against
   one reference episode. One focal length and one alignment are shared by
   every episode, which is only valid for a camera that did not move: ``all``
   runs this check first, and alignment and measurement refuse an episode
   whose camera moved or was never checked.
2. **depth** -- Depth Pro on every high-camera frame. The focal length is
   estimated once on a sample of frames and then fixed for every frame, so
   depth is metric on one consistent camera rather than on a per-frame guess.
3. **align** -- a similarity transform from the camera frame to the robot
   base, fitted with RANSAC between the gripper's closing point and the
   recorded end-effector position. The closing point is the 3D midpoint of the
   two fingertips, each lifted with the depth at its own surface -- depth read
   at the image midpoint between open fingertips would be the table or the
   object behind them. A fixed offset in the gripper frame is fitted jointly.
   Accepted only below a median residual; if it is refused, geometry is
   reported in a table-aligned camera frame and every artifact says so. Camera
   coordinates are never passed off as robot coordinates.
4. **measure** -- depth around each tracked point with background samples
   rejected, the table plane, positions in the declared scene frame, short
   gaps filled and everything smoothed as the annotation mode allows (centred
   for whole episodes, causal for past-only), node centroids and the
   continuous quantities the reward reads.

The table plane is fitted on the table keyframe whose surface points spread
widest -- the same keyframes validation accepts: three points far apart and not
in a line -- plus a grid inside their hull, and depth samples that still lie on
a line or a point are refused rather than fitted.

A point unmeasured for at most ``gaps.max_fill_frames`` frames is filled; a
longer gap stays unknown, so it can neither look like a measured position nor
a still object, and the reward counts the frames that need it. Which frames
were measured, and how long every gap was, is saved beside the positions. The
pot rim is the one exception, and a separate one: it is held through any gap
only when its own measurements verify that the pot does not move.

No quantity is rescaled to its own episode's range. Depth maps, the alignment
and the measurements are each reused only while the inputs they record --
video, weights, focal length, tracks, annotation, alignment, settings, camera
check -- are unchanged; otherwise they are made again. Alignment and
measurement also refuse inputs that are unchanged but were themselves made
from stale inputs (:mod:`real_robot.preprocessing.freshness`).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..common import (
    add_config_arguments,
    episode_name,
    load_configs,
    read_json,
    repo_path,
    stable_hash,
    utc_now,
    write_json,
)

SCENE_ROBOT = "robot_base"
SCENE_TABLE = "table_camera"
UP = np.array([0.0, 0.0, 1.0])


# --------------------------------------------------------------------------- #
# Geometry primitives (numpy only)
# --------------------------------------------------------------------------- #
def robust_depth(depth: np.ndarray, x: float, y: float, radius: int, background_tol: float) -> float:
    """Depth at normalised ``(x, y)``: the median of the nearest surface in a window.

    A thin object -- a rim, a handle, a banana's edge -- shares its window with
    whatever lies behind it and may cover only a few of its pixels. The nearest
    surface is referenced by the median of the few closest samples (robust to
    a single flying pixel), and anything more than ``background_tol`` beyond it
    is background and dropped.
    """
    height, width = depth.shape
    if not (np.isfinite(x) and np.isfinite(y)):
        return float("nan")
    col, row = int(round(x * (width - 1))), int(round(y * (height - 1)))
    if not (0 <= col < width and 0 <= row < height):
        return float("nan")
    window = depth[max(0, row - radius): row + radius + 1, max(0, col - radius): col + radius + 1]
    values = np.sort(window[np.isfinite(window) & (window > 0)].astype(np.float64))
    if values.size == 0:
        return float("nan")
    closest = values[: max(3, int(np.ceil(0.05 * values.size)))]
    near = float(np.median(closest))
    return float(np.median(values[values <= near + background_tol]))


def derive_tcp(names: Sequence[str], points: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """Append ``ee:tcp``: the midpoint of the two lifted fingertips, unknown unless both are known."""
    names = list(names)
    if "ee:fingertip_1" not in names or "ee:fingertip_2" not in names:
        return names, points
    a = points[:, names.index("ee:fingertip_1")]
    b = points[:, names.index("ee:fingertip_2")]
    both = np.all(np.isfinite(a), axis=-1) & np.all(np.isfinite(b), axis=-1)
    tcp = np.where(both[:, None], (a + b) / 2.0, np.nan)
    return names + ["ee:tcp"], np.concatenate([points, tcp[:, None]], axis=1)


def backproject(x: float, y: float, z: float, focal_px: float, width: int, height: int) -> np.ndarray:
    """Normalised image point and depth -> camera coordinates (x right, y down, z forward)."""
    u, v = x * width, y * height
    return np.array([(u - width / 2.0) * z / focal_px, (v - height / 2.0) * z / focal_px, z])


def fit_plane(points: np.ndarray) -> Tuple[np.ndarray, float]:
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid)
    normal = vt[-1]
    return normal / np.linalg.norm(normal), float(-normal @ centroid)


def plane_degeneracy(points: np.ndarray, min_extent: float) -> Optional[str]:
    """Why ``points`` cannot define a plane, or None.

    Too few points, too few distinct ones, or a spread along the second
    principal axis (a standard deviation, in the points' units) below
    ``min_extent``: points on a line fit every plane through that line.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < 3:
        return f"{points.shape[0]} sample(s); a plane needs three"
    distinct = np.unique(np.round(points, 6), axis=0).shape[0]
    if distinct < 3:
        return f"{points.shape[0]} samples but only {distinct} distinct point(s)"
    spread = np.linalg.svd(points - points.mean(axis=0), compute_uv=False) / np.sqrt(points.shape[0])
    if spread[1] < min_extent:
        return f"the samples lie along a line (second extent {spread[1]:.4f}, need {min_extent:g})"
    return None


def fit_plane_ransac(points: np.ndarray, iterations: int, inlier: float, rng: np.random.Generator
                     ) -> Tuple[np.ndarray, float, np.ndarray]:
    """``(normal, offset, inlier mask)`` with ``normal @ p + offset = 0``. Refuses degenerate points."""
    reason = plane_degeneracy(points, 1e-9)
    if reason:
        raise ValueError(f"cannot fit a plane: {reason}")
    best = None
    for _ in range(int(iterations)):
        sample = points[rng.choice(points.shape[0], 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        offset = -normal @ sample[0]
        mask = np.abs(points @ normal + offset) <= inlier
        if best is None or mask.sum() > best[2].sum():
            best = (normal, offset, mask)
    if best is None or best[2].sum() < 3:
        normal, offset = fit_plane(points)
        return normal, offset, np.ones(points.shape[0], dtype=bool)
    normal, offset = fit_plane(points[best[2]])
    return normal, offset, np.abs(points @ normal + offset) <= inlier


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    """``s, R, t`` minimising ``|dst - (s R src + t)|``."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / src.shape[0]
    u, sigma, vt = np.linalg.svd(cov)
    d = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[2, 2] = -1.0
    rotation = u @ d @ vt
    variance = (xs ** 2).sum() / src.shape[0]
    scale = float(np.trace(np.diag(sigma) @ d) / variance) if with_scale else 1.0
    translation = mu_d - scale * rotation @ mu_s
    return scale, rotation, translation


def apply_similarity(transform: Mapping[str, Any], points: np.ndarray) -> np.ndarray:
    rotation = np.asarray(transform["rotation"], dtype=np.float64)
    return float(transform["scale"]) * points @ rotation.T + np.asarray(transform["translation"], dtype=np.float64)


def euler_zyx(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """``R = Rz(yaw) Ry(pitch) Rx(roll)`` for arrays of angles, shape ``(N, 3, 3)``."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.stack([
        np.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], -1),
        np.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], -1),
        np.stack([-sp, cp * sr, cp * cr], -1),
    ], -2)


def _affine_with_offset(camera: np.ndarray, eef: np.ndarray, rotations: np.ndarray):
    """Least squares ``A p + t - R_ee o = eef``: linear in ``(A, t, o)``."""
    n = camera.shape[0]
    blocks = np.zeros((n, 3, 15))
    for row in range(3):
        blocks[:, row, 3 * row: 3 * row + 3] = camera
    blocks[:, :, 9:12] = np.eye(3)
    blocks[:, :, 12:15] = -rotations
    solution, *_ = np.linalg.lstsq(blocks.reshape(3 * n, 15), eef.reshape(-1), rcond=None)
    return solution[:9].reshape(3, 3), solution[9:12], solution[12:15]


def _translation_and_offset(scaled_rotation: np.ndarray, camera: np.ndarray, eef: np.ndarray,
                            rotations: np.ndarray):
    """With ``s R`` fixed, ``t - R_ee o = eef - s R p`` is linear in ``(t, o)``."""
    n = camera.shape[0]
    blocks = np.zeros((n, 3, 6))
    blocks[:, :, 0:3] = np.eye(3)
    blocks[:, :, 3:6] = -rotations
    target = eef - camera @ scaled_rotation.T
    solution, *_ = np.linalg.lstsq(blocks.reshape(3 * n, 6), target.reshape(-1), rcond=None)
    return solution[:3], solution[3:6]


def _nearest_similarity(affine: np.ndarray) -> Tuple[float, np.ndarray]:
    u, sigma, vt = np.linalg.svd(affine)
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))
    return float(np.mean(sigma)), u @ d @ vt


def fit_alignment(camera_points: np.ndarray, eef_xyz: np.ndarray, eef_rpy: np.ndarray,
                  iterations: int, inlier: float, rng: np.random.Generator) -> Dict[str, Any]:
    """Camera -> robot base similarity plus a fingertip offset in the gripper frame.

    The tracked point is between the fingertips; the recorded position is the
    end-effector frame's origin, and their offset ``o`` is fixed in the gripper
    frame: ``s R p + t = eef + R_ee o``. Fitting a similarity with ``o = 0``
    first would absorb part of the offset and throw out exactly the frames that
    reveal it, so the consensus set is found on the linear relaxation
    ``A p + t - R_ee o = eef`` instead, then projected to the nearest similarity
    and refined by alternating Umeyama with the linear ``(t, o)`` solve.
    """
    n = camera_points.shape[0]
    if n < 8:
        raise ValueError(f"alignment needs at least 8 point pairs, got {n}")
    rotations = euler_zyx(eef_rpy[:, 0], eef_rpy[:, 1], eef_rpy[:, 2])

    def residuals(s, R, t, o):
        return np.linalg.norm(s * camera_points @ R.T + t - eef_xyz - np.einsum("nij,j->ni", rotations, o), axis=1)

    best_mask, best_count = None, -1
    for _ in range(int(iterations)):
        sample = rng.choice(n, 8, replace=False)
        try:
            affine, t, o = _affine_with_offset(camera_points[sample], eef_xyz[sample], rotations[sample])
            s, R = _nearest_similarity(affine)
            t, o = _translation_and_offset(s * R, camera_points[sample], eef_xyz[sample], rotations[sample])
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(s) or s <= 0:
            continue
        mask = residuals(s, R, t, o) <= inlier
        if mask.sum() > best_count:
            best_mask, best_count = mask, int(mask.sum())
    if best_mask is None or best_count < 8:
        raise ValueError("RANSAC found no consistent camera-to-robot alignment")

    mask = best_mask
    affine, t, o = _affine_with_offset(camera_points[mask], eef_xyz[mask], rotations[mask])
    s, R = _nearest_similarity(affine)
    t, o = _translation_and_offset(s * R, camera_points[mask], eef_xyz[mask], rotations[mask])
    for _ in range(10):
        targets = eef_xyz + np.einsum("nij,j->ni", rotations, o)
        s, R, _ = umeyama(camera_points[mask], targets[mask])
        t, o = _translation_and_offset(s * R, camera_points[mask], eef_xyz[mask], rotations[mask])
        candidate = residuals(s, R, t, o) <= inlier
        if candidate.sum() < 8:
            break
        mask = candidate
    with_offset = residuals(s, R, t, o)

    s0, R0, t0 = umeyama(camera_points[mask], eef_xyz[mask])
    without = residuals(s0, R0, t0, np.zeros(3))
    # An offset the wrist's rotation does not reveal is not identifiable; keep
    # it only when it clearly explains the data better, and never an absurd one.
    use_offset = (np.median(with_offset[mask]) < 0.9 * np.median(without[mask])
                  and float(np.linalg.norm(o)) <= 0.25)
    if not use_offset:
        s, R, t, o = s0, R0, t0, np.zeros(3)
        with_offset = without
    return {
        "scale": float(s), "rotation": R.tolist(), "translation": t.tolist(),
        "tool_offset_gripper_frame": o.tolist(), "tool_offset_used": bool(use_offset),
        "pairs": int(n), "inliers": int(mask.sum()),
        "median_residual_m": float(np.median(with_offset[mask])),
        "p90_residual_m": float(np.percentile(with_offset[mask], 90)),
        "euler_convention": "R = Rz(yaw) Ry(pitch) Rx(roll)",
    }


def table_frame(normal: np.ndarray, offset: float) -> Dict[str, Any]:
    """Camera -> table-aligned frame: z along the table normal toward the camera,
    x the camera's x projected onto the table, origin below the camera."""
    normal = normal / np.linalg.norm(normal)
    # The camera sits at the origin of its own frame; "up" points to its side.
    if offset < 0:
        normal, offset = -normal, -offset
    z = normal
    x = np.array([1.0, 0.0, 0.0]) - z * z[0]
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rotation = np.stack([x, y, z])
    origin = -offset * normal
    return {"scale": 1.0, "rotation": rotation.tolist(), "translation": (-rotation @ origin).tolist()}


def fill_series(values: np.ndarray, valid: np.ndarray, mode: str, max_gap: int
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill short gaps; leave longer ones unknown. Returns ``(filled, known, gap)``.

    * ``full_episode``: a run of at most ``max_gap`` unmeasured frames between
      two measurements is interpolated linearly; after the last measurement
      its value holds for at most ``max_gap`` frames.
    * ``past_only``: the last measurement holds for at most ``max_gap`` frames.

    Nothing is extrapolated before the first measurement, and a longer gap is
    NaN and not known, whole. ``gap`` is 0 on measured frames; on every other
    frame it is the length of the unmeasured run the frame lies in (whole
    episodes) or the number of frames since the last measurement, this one
    included (past only, which may not know how long a gap will last). So
    ``gap == 0`` is the measurement mask, and every gap -- filled or not -- is
    on record for the quality limits downstream.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    flat = values.reshape(n, -1)
    measured = np.asarray(valid, dtype=bool) & np.all(np.isfinite(flat), axis=1)
    out = np.full_like(flat, np.nan)
    out[measured] = flat[measured]
    known = measured.copy()
    gap = np.zeros(n, dtype=np.int32)
    max_gap = int(max_gap)
    if mode == "full_episode":
        t = 0
        while t < n:
            if measured[t]:
                t += 1
                continue
            start = t
            while t < n and not measured[t]:
                t += 1
            length = t - start
            gap[start:t] = length
            if start == 0:
                continue                      # before the first measurement
            before = start - 1
            if t == n:                        # after the last measurement: hold, briefly
                hold = min(length, max_gap)
                out[start:start + hold] = flat[before]
                known[start:start + hold] = True
            elif length <= max_gap:
                weight = ((np.arange(start, t) - before) / (t - before))[:, None]
                out[start:t] = (1.0 - weight) * flat[before] + weight * flat[t]
                known[start:t] = True
    else:
        last, since = None, 0
        for t in range(n):
            if measured[t]:
                last, since = t, 0
                continue
            since += 1
            gap[t] = since
            if last is not None and since <= max_gap:
                out[t] = flat[last]
                known[t] = True
    return out.reshape(values.shape), known, gap


def stationary_position(values: np.ndarray, measured: np.ndarray, mode: str, tolerance: float,
                        min_samples: int) -> Tuple[Optional[np.ndarray], np.ndarray, Dict[str, Any]]:
    """A position held through every gap -- only when its measurements verify that it does not move.

    Verified means at least ``min_samples`` measurements whose per-axis
    standard deviation stays below ``tolerance``. Whole episodes: verified over
    all of them, and the median holds on every frame. Past only: verified at
    frame ``t`` from the measurements up to ``t`` alone, and the running median
    holds from then on until a measurement contradicts it. Returns
    ``(series or None, known, report)``.
    """
    values = np.asarray(values, dtype=np.float64)
    measured = np.asarray(measured, dtype=bool) & np.all(np.isfinite(values), axis=-1)
    n = len(values)
    samples = values[measured]
    spread = float(np.max(np.std(samples, axis=0))) if len(samples) else float("nan")
    report = {"samples": int(len(samples)), "std_m": spread, "tolerance_m": float(tolerance),
              "min_samples": int(min_samples)}
    if mode == "full_episode":
        verified = len(samples) >= int(min_samples) and spread < float(tolerance)
        report["verified"] = bool(verified)
        if not verified:
            return None, np.zeros(n, dtype=bool), report
        return np.tile(np.median(samples, axis=0), (n, 1)), np.ones(n, dtype=bool), report
    running = np.full((n, values.shape[-1]), np.nan)
    known = np.zeros(n, dtype=bool)
    seen: List[np.ndarray] = []
    for t in range(n):
        if measured[t]:
            seen.append(values[t])
        if len(seen) >= int(min_samples) and float(np.max(np.std(np.asarray(seen), axis=0))) < float(tolerance):
            running[t] = np.median(np.asarray(seen), axis=0)
            known[t] = True
    report["verified"] = bool(known.any())
    report["verified_from_frame"] = int(np.argmax(known)) if known.any() else -1
    return (running if known.any() else None), known, report


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise hull of 2D points (monotone chain)."""
    ordered = sorted(set(map(tuple, np.asarray(points, dtype=np.float64))))
    if len(ordered) <= 2:
        return np.asarray(ordered)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[tuple] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: List[tuple] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def inside_hull(hull: np.ndarray, x: float, y: float) -> bool:
    if len(hull) < 3:
        return False
    for (x0, y0), (x1, y1) in zip(hull, np.roll(hull, -1, axis=0)):
        if (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0) < -1e-12:
            return False
    return True


def smooth_series(values: np.ndarray, valid: np.ndarray, settings: Mapping[str, Any]) -> np.ndarray:
    """Savitzky-Golay over each valid run, or a causal exponential average."""
    values = np.asarray(values, dtype=np.float64).copy()
    method = settings["method"]
    if method == "savgol":
        from scipy.signal import savgol_filter

        window, order = int(settings["window"]), int(settings["order"])
        start = None
        for t in range(len(values) + 1):
            inside = t < len(values) and valid[t]
            if inside and start is None:
                start = t
            if not inside and start is not None:
                length = t - start
                w = min(window, length if length % 2 else length - 1)
                if w > order + 1:
                    values[start:t] = savgol_filter(values[start:t], w, order, axis=0)
                start = None
        return values
    if method == "ema":
        alpha = float(settings["alpha"])
        state = None
        for t in range(len(values)):
            if not valid[t]:
                state = None
                continue
            state = values[t] if state is None else alpha * values[t] + (1 - alpha) * state
            values[t] = state
        return values
    raise ValueError(f"unknown smoothing method {method!r}")


def planar(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a[..., :2] - b[..., :2], axis=-1)


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1)


def measurements(points: Mapping[str, np.ndarray], fps: float, entry_height: float) -> Dict[str, np.ndarray]:
    """The reward's continuous quantities from scene-frame positions (NaN where unknown)."""
    gripper = points["ee"]
    banana_grasp = np.where(np.isfinite(points["banana:grasp_region"]), points["banana:grasp_region"],
                            points["banana:center"])
    rim = points["pot:rim_center"]
    entry = rim + entry_height * UP
    handle = np.where(np.isfinite(points["lid:handle"]), points["lid:handle"], points["lid:center"])
    banana = points["banana:center"]
    lid = points["lid:center"]

    def speed(series):
        out = np.full(len(series), np.nan)
        out[1:] = np.linalg.norm(np.diff(series, axis=0), axis=1) * fps
        return out

    return {
        "d_gripper_banana": distance(gripper, banana_grasp),
        "d_banana_pot_entry": distance(banana, entry),
        "banana_placement_error": planar(banana, rim),
        "d_gripper_lid_handle": distance(gripper, handle),
        "lid_lateral_error": planar(lid, rim),
        "lid_height_above_rim": lid[:, 2] - rim[:, 2],
        "banana_speed": speed(banana),
        "lid_speed": speed(lid),
    }


# --------------------------------------------------------------------------- #
# Depth Pro
# --------------------------------------------------------------------------- #
class DepthProRunner:
    def __init__(self, cfg: Mapping[str, Any]):
        import torch
        import depth_pro
        from depth_pro import depth_pro as module

        checkpoint = repo_path(cfg["checkpoint"])
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Depth Pro weights not found at {checkpoint}; see real_robot/README.md")
        base = module.DEFAULT_MONODEPTH_CONFIG_DICT
        try:
            config = dataclasses.replace(base, checkpoint_uri=checkpoint)
        except TypeError:
            import copy
            config = copy.copy(base)
            config.checkpoint_uri = checkpoint
        device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
        precision = torch.half if bool(cfg["half_precision"]) and device.type == "cuda" else torch.float32
        self.torch = torch
        self.model, self.transform = depth_pro.create_model_and_transforms(
            config=config, device=device, precision=precision)
        self.model.eval()

    def infer(self, rgb: np.ndarray, focal_px: Optional[float] = None) -> Tuple[np.ndarray, float]:
        torch = self.torch
        with torch.no_grad():
            image = self.transform(np.ascontiguousarray(rgb))
            # infer() squeezes f_px, so a fixed focal length goes in as a tensor.
            f_px = None if focal_px is None else torch.tensor(float(focal_px), device=image.device,
                                                              dtype=torch.float32)
            prediction = self.model.infer(image, f_px=f_px)
        depth = prediction["depth"].detach().float().cpu().numpy()
        return depth, float(prediction["focallength_px"])


def _downsample(depth: np.ndarray, size: Sequence[int]) -> np.ndarray:
    from PIL import Image

    height, width = int(size[0]), int(size[1])
    return np.asarray(Image.fromarray(depth.astype(np.float32), mode="F").resize((width, height), Image.NEAREST))


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
class GeometryStage:
    def __init__(self, configs: Mapping[str, Mapping[str, Any]], mode: Optional[str] = None, source=None):
        from ..data.episode_dataset import RawEpisodeSource

        self.configs = configs
        self.cfg = configs["annotation"]["geometry"]
        self.source = source if source is not None else RawEpisodeSource(configs, mode=mode)
        self.spec = self.source.spec
        self.camera = self.spec.cameras[0]
        self.depth_root = repo_path(configs["dataset"]["paths"]["depth"])
        self.geometry_root = repo_path(configs["dataset"]["paths"]["geometry"])
        self.rng = np.random.default_rng(0)

    # ---------------------------------------------------------- files
    def camera_geometry_path(self) -> str:
        return os.path.join(self.depth_root, "camera_geometry.json")

    def depth_path(self, episode: int) -> str:
        return os.path.join(self.depth_root, episode_name(episode) + ".npz")

    def alignment_path(self) -> str:
        return os.path.join(self.geometry_root, "camera_alignment.json")

    def camera_check_path(self) -> str:
        return os.path.join(self.geometry_root, "camera_check.json")

    def settings_hash(self) -> str:
        return stable_hash({k: v for k, v in self.cfg.items() if k not in ("device",)})

    # ------------------------------------------------------- identities
    def depth_inputs(self, episode: int, focal_px: float) -> Dict[str, Any]:
        from .artifacts import file_digest, source_video_digest

        return {"video": source_video_digest(self.source, episode, self.camera),
                "checkpoint": file_digest(repo_path(self.cfg["checkpoint"])),
                "focal_px": round(float(focal_px), 3), "cache_resolution": list(self.cfg["cache_resolution"]),
                "frame_stride": int(self.cfg["frame_stride"])}

    def alignment_inputs(self, episodes: Sequence[int]) -> Dict[str, Any]:
        from .artifacts import file_digest

        check = read_json(self.camera_check_path()) if os.path.isfile(self.camera_check_path()) else {}
        return {"episodes": sorted(int(e) for e in episodes),
                "tracks": {str(e): file_digest(self.source.tracks_path(e)) for e in sorted(episodes)},
                "depth": {str(e): file_digest(self.depth_path(e)) for e in sorted(episodes)},
                "settings": self.settings_hash(),
                "camera_reference": check.get("reference_episode")}

    def alignment_current(self) -> Tuple[bool, str]:
        from .artifacts import stale_reason

        if not os.path.isfile(self.alignment_path()):
            return False, "no alignment yet"
        record = read_json(self.alignment_path())
        if "inputs" not in record:
            return False, "alignment written by an earlier version"
        reason = stale_reason(self.alignment_inputs(record["inputs"]["episodes"]), record["inputs"])
        return reason is None, reason or "current"

    def geometry_inputs(self, episode: int) -> Dict[str, Any]:
        from .artifacts import file_digest

        alignment = read_json(self.alignment_path()) if os.path.isfile(self.alignment_path()) else None
        return {"annotation": file_digest(self.source.annotation_path(episode)),
                "tracks": file_digest(self.source.tracks_path(episode)),
                "depth": file_digest(self.depth_path(episode)),
                "alignment": None if alignment is None else alignment.get("hash"),
                "settings": self.settings_hash(), "mode": self.source.mode,
                "camera_fixed": self.camera_status(episode)[0]}

    def settings_identity(self) -> Dict[str, Any]:
        camera = read_json(self.camera_geometry_path()) if os.path.isfile(self.camera_geometry_path()) else None
        alignment = read_json(self.alignment_path()) if os.path.isfile(self.alignment_path()) else None
        return {
            "settings": stable_hash({k: v for k, v in self.cfg.items() if k not in ("device",)}),
            "focal_px": None if camera is None else round(camera["focal_px"], 3),
            "alignment": None if alignment is None else alignment.get("hash"),
        }

    # --------------------------------------------------- check camera
    def check_camera(self, episodes: Sequence[int]) -> Dict[str, Any]:
        """Shift of each episode's first frame from the reference episode's, and of its last from its first.

        Results accumulate in ``camera_check.json`` against one reference
        episode; an episode is checked again only when its video changed.
        """
        import cv2
        from .artifacts import source_video_digest
        from .prepare_videos import read_frames

        def gray(episode, index):
            path = self.source.video_path(episode, self.camera)
            return cv2.cvtColor(read_frames(path, [index])[0], cv2.COLOR_RGB2GRAY)

        orb = cv2.ORB_create(1500)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        def shift(a, b) -> float:
            ka, da = orb.detectAndCompute(a, None)
            kb, db = orb.detectAndCompute(b, None)
            if da is None or db is None:
                return float("nan")
            matches = matcher.match(da, db)
            if len(matches) < 12:
                return float("nan")
            pa = np.float32([ka[m.queryIdx].pt for m in matches])
            pb = np.float32([kb[m.trainIdx].pt for m in matches])
            H, inliers = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
            if H is None:
                return float("nan")
            projected = cv2.perspectiveTransform(pa.reshape(-1, 1, 2), H).reshape(-1, 2)
            # The static background dominates the inliers; their shift is the camera's.
            return float(np.median(np.linalg.norm(projected - pa, axis=1)[inliers.reshape(-1) == 1]))

        path = self.camera_check_path()
        existing = read_json(path) if os.path.isfile(path) else {}
        entries: Dict[str, Any] = dict(existing.get("episodes") or {})
        reference_episode = int(existing.get("reference_episode", episodes[0]))
        reference = None
        lengths = self.source.lengths()
        for episode in episodes:
            video = source_video_digest(self.source, episode, self.camera)
            entry = entries.get(str(int(episode)))
            if entry and entry.get("video") == video and entry.get("reference") == reference_episode:
                continue
            if reference is None:
                reference = gray(reference_episode, 0)
            first = gray(episode, 0)
            entries[str(int(episode))] = {"across_px": shift(reference, first),
                                          "within_px": shift(first, gray(episode, lengths[episode] - 1)),
                                          "video": video, "reference": reference_episode, "checked": utc_now()}
        tolerance = float(self.cfg["camera_fixed_tolerance_px"])
        moved = sorted(int(e) for e, v in entries.items() if not self._fixed(v, tolerance))
        report = {"created": existing.get("created", utc_now()), "updated": utc_now(),
                  "reference_episode": reference_episode, "tolerance_px": tolerance, "episodes": entries,
                  "moved": moved, "camera_fixed": not moved}
        write_json(path, report)
        return report

    @staticmethod
    def _fixed(entry: Mapping[str, Any], tolerance: float) -> bool:
        return all(np.isfinite(entry.get(key, np.nan)) and float(entry[key]) <= tolerance
                   for key in ("across_px", "within_px"))

    def camera_status(self, episode: int) -> Tuple[bool, str]:
        """Whether this episode's camera is checked and fixed relative to the shared reference."""
        from .artifacts import source_video_digest

        path = self.camera_check_path()
        if not os.path.isfile(path):
            return False, "the camera has not been checked; run `estimate_geometry check-camera`"
        report = read_json(path)
        entry = (report.get("episodes") or {}).get(str(int(episode)))
        if entry is None:
            return False, "this episode is not in the camera check; run `estimate_geometry check-camera`"
        if entry.get("video") != source_video_digest(self.source, episode, self.camera):
            return False, "the video changed since the camera check"
        if not self._fixed(entry, float(self.cfg["camera_fixed_tolerance_px"])):
            return False, (f"the camera moved: {entry.get('across_px')} px from reference episode "
                           f"{report.get('reference_episode')}, {entry.get('within_px')} px within the episode")
        return True, "fixed"

    def require_fixed_camera(self, episodes: Sequence[int], what: str) -> None:
        problems = []
        for episode in episodes:
            fixed, reason = self.camera_status(episode)
            if not fixed:
                problems.append(f"episode {episode}: {reason}")
        if problems:
            raise SystemExit(f"[geometry] {what} shares one focal length and one camera alignment, which needs a "
                             "fixed camera:\n  " + "\n  ".join(problems[:30]))

    # ------------------------------------------------------------ depth
    def estimate_focal(self, runner: DepthProRunner, episodes: Sequence[int]) -> Dict[str, Any]:
        from .prepare_videos import read_frames

        count = int(self.cfg["focal_sample_frames"])
        per_episode = max(1, count // max(1, len(episodes)))
        focals = []
        for episode in episodes:
            n = self.source.lengths()[episode]
            indices = sorted(set(np.linspace(0, n - 1, per_episode).astype(int).tolist()))
            for rgb in read_frames(self.source.video_path(episode, self.camera), indices):
                focals.append(runner.infer(rgb)[1])
        from .artifacts import file_digest

        record = {"created": utc_now(), "focal_px": float(np.median(focals)), "focal_std_px": float(np.std(focals)),
                  "samples": len(focals), "episodes": [int(e) for e in episodes], "camera": self.camera,
                  "checkpoint": file_digest(repo_path(self.cfg["checkpoint"]))}
        write_json(self.camera_geometry_path(), record)
        return record

    def run_depth(self, episodes: Sequence[int], force: bool = False) -> None:
        from .artifacts import file_digest, reusable
        from .prepare_videos import iter_frames

        runner: Optional[DepthProRunner] = None

        def model() -> DepthProRunner:
            nonlocal runner
            if runner is None:
                runner = DepthProRunner(self.cfg)
            return runner

        focal_record = read_json(self.camera_geometry_path()) if os.path.isfile(self.camera_geometry_path()) else None
        if focal_record is None or focal_record.get("checkpoint") != file_digest(repo_path(self.cfg["checkpoint"])):
            record = self.estimate_focal(model(), list(episodes))
            print(f"[geometry] fixed focal length {record['focal_px']:.1f} px "
                  f"(std {record['focal_std_px']:.1f}, {record['samples']} frames)", flush=True)
        focal = float(read_json(self.camera_geometry_path())["focal_px"])
        stride = int(self.cfg["frame_stride"])
        for episode in episodes:
            path = self.depth_path(episode)
            inputs = self.depth_inputs(episode, focal)
            current, reason = reusable(path, inputs)
            if current and not force:
                continue
            if os.path.isfile(path):
                print(f"[geometry] depth for episode {episode}: again ({reason})", flush=True)
            runner = model()
            maps, frames, size = [], [], None
            for index, _, rgb in iter_frames(self.source.video_path(episode, self.camera)):
                if index % stride:
                    continue
                depth, _ = runner.infer(rgb, focal_px=focal)
                size = rgb.shape[:2]
                maps.append(_downsample(depth, self.cfg["cache_resolution"]).astype(np.float16))
                frames.append(index)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez_compressed(path, depth=np.stack(maps), frames=np.asarray(frames),
                                source_hw=np.asarray(size), focal_px=np.asarray(focal))
            write_json(path[:-4] + ".json", {"episode_index": int(episode), "created": utc_now(), "inputs": inputs})
            print(f"[geometry] depth for episode {episode}: {len(frames)} frames", flush=True)

    def load_depth(self, episode: int) -> Dict[str, np.ndarray]:
        path = self.depth_path(episode)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no depth at {path}; run `estimate_geometry depth` first")
        with np.load(path) as data:
            return {key: data[key] for key in data.files}

    # ----------------------------------------------------- camera points
    def camera_points(self, episode: int) -> Dict[str, Any]:
        """Named high-camera points lifted to camera coordinates, per frame."""
        tracks = self.source.tracks(episode)
        depth = self.load_depth(episode)
        n = self.source.lengths()[episode]
        height, width = (int(v) for v in depth["source_hw"])
        focal = float(depth["focal_px"])
        frame_to_map = {int(f): i for i, f in enumerate(depth["frames"])}
        names = [str(v) for v in tracks["point_names"]]
        c = self.spec.cameras.index(self.camera)
        radius = int(self.cfg["point_window_px"])
        tol = float(self.cfg["background_tolerance_m"])
        out = np.full((n, len(names), 3), np.nan)
        for t in range(n):
            index = frame_to_map.get(t)
            if index is None:
                continue
            dmap = depth["depth"][index].astype(np.float32)
            for p in range(len(names)):
                x, y = tracks["points"][t, p, c]
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                z = robust_depth(dmap, x, y, radius, tol)
                if np.isfinite(z):
                    out[t, p] = backproject(x, y, z, focal, width, height)
        names, out = derive_tcp(names, out)
        return {"names": names, "points": out, "focal": focal, "hw": (height, width), "tracks": tracks,
                "depth": depth}

    # ------------------------------------------------------------ align
    def align(self, episodes: Sequence[int]) -> Dict[str, Any]:
        self.require_fixed_camera(episodes, "alignment")
        spec_action = self.source.action_spec()
        cams, eefs, rpys = [], [], []
        for episode in episodes:
            lifted = self.camera_points(episode)
            if "ee:tcp" not in lifted["names"]:
                raise ValueError("graph.yaml declares no `ee` fingertip points; alignment needs both")
            p = lifted["names"].index("ee:tcp")
            state = self.source.table(episode)["state"]
            quality = lifted["tracks"]["quality"][:, self.spec.entity_ids.index("ee"), self.spec.cameras.index(self.camera)]
            for t in range(0, len(state), 2):
                point = lifted["points"][t, p]
                if np.all(np.isfinite(point)) and quality[t] > 0.3:
                    cams.append(point)
                    eefs.append(state[t, 7:10])
                    rpys.append(state[t, 10:13])
        result = fit_alignment(np.asarray(cams), np.asarray(eefs), np.asarray(rpys),
                               int(self.cfg["alignment_ransac_iters"]), float(self.cfg["alignment_inlier_m"]),
                               self.rng)
        accepted = result["median_residual_m"] <= float(self.cfg["alignment_max_median_residual_m"])
        inputs = self.alignment_inputs(episodes)
        record = {"created": utc_now(), "episodes": [int(e) for e in episodes], "accepted": bool(accepted),
                  "scene_frame": SCENE_ROBOT if accepted else SCENE_TABLE, "fit": result,
                  "action_spec_revision": spec_action.get("source_revision"), "inputs": inputs}
        record["hash"] = stable_hash({"inputs": inputs, "fit": result, "accepted": accepted})
        write_json(self.alignment_path(), record)
        return record

    # ----------------------------------------------------------- measure
    def measure(self, episode: int) -> str:
        self.require_fixed_camera([episode], "measurement")
        inputs = self.geometry_inputs(episode)
        alignment = read_json(self.alignment_path()) if os.path.isfile(self.alignment_path()) else None
        lifted = self.camera_points(episode)
        names, cam = lifted["names"], lifted["points"]
        n = cam.shape[0]
        fps = self.source.fps()
        mode = self.source.mode
        state = self.source.table(episode)["state"]
        annotation = self.source.annotation(episode)

        # Table plane from the bare-table points and a grid inside their hull.
        table_points, table_key = self._table_samples(episode, lifted, annotation)
        normal, offset, inliers = fit_plane_ransac(table_points, int(self.cfg["plane_ransac_iters"]),
                                                   float(self.cfg["plane_inlier_m"]), self.rng)

        if alignment is not None and alignment["accepted"]:
            frame = SCENE_ROBOT
            transform = alignment["fit"]
        else:
            frame = SCENE_TABLE
            transform = table_frame(normal, offset)
        scene = np.full_like(cam, np.nan)
        known = np.all(np.isfinite(cam), axis=-1)
        scene[known] = apply_similarity(transform, cam[known])

        positions: Dict[str, np.ndarray] = {}
        validity: Dict[str, np.ndarray] = {}

        def series(name: str) -> Tuple[np.ndarray, np.ndarray]:
            if name not in names:
                return np.full((n, 3), np.nan), np.zeros(n, dtype=bool)
            values = scene[:, names.index(name)]
            return values, np.all(np.isfinite(values), axis=-1)

        # The gripper: the recorded arm when the transform maps into its frame.
        if frame == SCENE_ROBOT:
            fit = alignment["fit"]
            rotations = euler_zyx(state[:, 10], state[:, 11], state[:, 12])
            tip = state[:, 7:10] + np.einsum("nij,j->ni", rotations, np.asarray(fit["tool_offset_gripper_frame"]))
            positions["ee"], validity["ee"] = tip, np.ones(n, dtype=bool)
        else:
            positions["ee"], validity["ee"] = series("ee:tcp")

        for name in ("banana:center", "banana:grasp_region", "lid:center", "lid:handle"):
            positions[name], validity[name] = series(name)
        rim_names = [f"pot:{p}" for p in ("rim_left", "rim_right", "rim_near", "rim_far")]
        rims = np.stack([series(r)[0] for r in rim_names], axis=1)
        count = np.sum(np.all(np.isfinite(rims), axis=-1), axis=1)
        with np.errstate(invalid="ignore"):
            rim_center = np.nanmean(rims, axis=1)
        rim_ok = count >= 2
        positions["pot:rim_center"], validity["pot:rim_center"] = np.where(rim_ok[:, None], rim_center, np.nan), rim_ok

        smoothing = self.cfg["smoothing"][mode]
        max_fill = int(self.cfg["gaps"]["max_fill_frames"])
        filled: Dict[str, np.ndarray] = {}
        filled_valid: Dict[str, np.ndarray] = {}
        gap_frames: Dict[str, np.ndarray] = {}
        for name, values in positions.items():
            f, v, gap = fill_series(values, validity[name], mode, max_fill)
            filled[name] = smooth_series(f, v, smoothing)
            filled_valid[name] = v
            gap_frames[name] = gap

        # The pot rim, and only the rim, is held through gaps of any length -- the
        # lid hides it -- when its own measurements verify that the pot does not move.
        stationary = self.cfg["stationary_pot"]
        rim, rim_known, rim_report = stationary_position(
            positions["pot:rim_center"], validity["pot:rim_center"], mode,
            float(stationary["tolerance_m"]), int(stationary["min_samples"]))
        if rim is not None:
            filled["pot:rim_center"], filled_valid["pot:rim_center"] = rim, rim_known

        values = measurements(filled, fps, float(self.cfg["pot_entry_height_m"]))

        # Node centroids in graph-spec order.
        centroids = np.zeros((n, len(self.spec.entities), 3), dtype=np.float32)
        centroid_known = np.zeros((n, len(self.spec.entities)), dtype=bool)
        table_scene = apply_similarity(transform, table_points[inliers]).mean(axis=0)
        sources = {"ee": "ee", "banana": "banana:center", "pot": "pot:rim_center", "lid": "lid:center"}
        for e, entity in enumerate(self.spec.entity_ids):
            if entity == "table":
                centroids[:, e] = table_scene
                centroid_known[:, e] = True
                continue
            key = sources.get(entity)
            if key is None:
                continue
            series_values = filled[key]
            ok = filled_valid[key] & np.all(np.isfinite(series_values), axis=-1)
            centroids[ok, e] = series_values[ok]
            centroid_known[:, e] = ok

        path = self.source.geometry_path(episode)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            centroids=centroids, centroid_known=centroid_known,
            point_names=np.array(list(filled)), points=np.stack([filled[k] for k in filled], axis=1).astype(np.float32),
            point_known=np.stack([filled_valid[k] for k in filled], axis=1),
            # What was measured, and how long each gap lasted, kept apart from what was filled.
            point_measured=np.stack([gap_frames[k] == 0 for k in filled], axis=1),
            point_gap_frames=np.stack([gap_frames[k] for k in filled], axis=1).astype(np.int32),
            **{key: value.astype(np.float32) for key, value in values.items()},
        )
        write_json(path[:-4] + ".json", {
            "episode_index": episode, "mode": mode, "scene_frame": frame, "created": utc_now(),
            "inputs": inputs, "settings": self.settings_identity(),
            "table_plane_camera": {"normal": normal.tolist(), "offset": float(offset),
                                   "inliers": int(inliers.sum()), "samples": int(len(inliers)),
                                   "keyframe": table_key},
            "known_fraction": {key: float(np.mean(v)) for key, v in filled_valid.items()},
            "gaps": {key: {"measured_fraction": float(np.mean(gap_frames[key] == 0)),
                           "filled_frames": int(np.sum(filled_valid[key] & (gap_frames[key] > 0))),
                           "unknown_frames": int(np.sum(~filled_valid[key])),
                           "longest_unmeasured_run": int(gap_frames[key].max(initial=0))}
                     for key in filled},
            "max_fill_frames": max_fill,
            "stationary_pot": rim_report,
            "note": ("positions are in the robot base frame from the fitted camera alignment"
                     if frame == SCENE_ROBOT else
                     "no accepted alignment: positions are in a table-aligned camera frame "
                     "(z along the table normal), not the robot frame"),
        })
        return path

    def _table_samples(self, episode: int, lifted: Mapping[str, Any], annotation) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Camera-frame points on the table, and which keyframe they came from.

        The keyframe is the one whose surface points spread widest among those
        validation accepts (three far apart, not in a line); the samples are its
        named points plus a grid inside their convex hull, off every object's box.
        """
        from ..graphs.validate import ValidationSettings, table_plane_keyframes

        depth = lifted["depth"]
        height, width = lifted["hw"]
        focal = lifted["focal"]
        radius, tol = int(self.cfg["point_window_px"]), float(self.cfg["background_tolerance_m"])
        c = self.spec.cameras.index(self.camera)
        settings = ValidationSettings.from_config(self.configs["annotation"], self.source.mode)
        candidates = table_plane_keyframes(annotation.keyframes, self.camera, settings)
        if not candidates:
            raise ValueError(f"episode {episode}: no {self.camera} table keyframe has three surface points far "
                             "enough apart and off a line to define a plane; annotate the episode again")
        key = candidates[0]
        frame = int(key["frame"])
        index = {int(f): i for i, f in enumerate(depth["frames"])}.get(frame)
        if index is None:
            index = int(np.argmin(np.abs(depth["frames"] - frame)))
        dmap = depth["depth"][index].astype(np.float32)
        corners = np.array([key["points"][name] for name in sorted(key["points"])], dtype=np.float64)
        hull = convex_hull(corners)
        tracks = lifted["tracks"]
        boxes = tracks["boxes"][frame, :, c]
        visible = tracks["visible"][frame, :, c]
        grid = []
        (xmin, ymin), (xmax, ymax) = corners.min(axis=0), corners.max(axis=0)
        for x in np.linspace(xmin, xmax, 12):
            for y in np.linspace(ymin, ymax, 12):
                if not inside_hull(hull, x, y):
                    continue
                inside_object = any(
                    visible[e] and boxes[e, 0] <= x <= boxes[e, 1] and boxes[e, 2] <= y <= boxes[e, 3]
                    for e in range(len(boxes)) if self.spec.entity_ids[e] != "table")
                if not inside_object:
                    grid.append((x, y))
        samples = []
        for x, y in list(map(tuple, corners)) + grid:
            z = robust_depth(dmap, x, y, radius, tol)
            if np.isfinite(z):
                samples.append(backproject(x, y, z, focal, width, height))
        samples = np.asarray(samples, dtype=np.float64).reshape(-1, 3)
        reason = plane_degeneracy(samples, float(self.cfg["plane_min_extent_m"]))
        if reason:
            raise ValueError(f"episode {episode}: the table depth samples at frame {frame} cannot define a plane "
                             f"({reason})")
        return samples, {"frame": frame, "points": sorted(key["points"]), "samples": int(len(samples))}


def main(argv: Optional[Sequence[str]] = None) -> None:
    from .artifacts import reusable
    from .freshness import ArtifactChain, problems_text

    parser = argparse.ArgumentParser(description="Depth, alignment and measurements.")
    parser.add_argument("command", choices=("check-camera", "depth", "align", "measure", "all"))
    parser.add_argument("--episodes", default="pilot")
    parser.add_argument("--mode", choices=("full_episode", "past_only"), default=None)
    parser.add_argument("--force", action="store_true")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    configs = load_configs(["dataset", "annotation", "graph"], args.overrides)
    stage = GeometryStage(configs, mode=args.mode)
    episodes = stage.source.select(args.episodes)

    if args.command in ("check-camera", "all"):
        report = stage.check_camera(episodes)
        print(f"[geometry] camera fixed across every checked episode: {report['camera_fixed']} "
              f"(reference episode {report['reference_episode']}, moved: {report['moved'][:10]})", flush=True)
        if args.command == "check-camera":
            return
        stage.require_fixed_camera(episodes, "estimate_geometry all")
    if args.command in ("depth", "all"):
        stage.run_depth(episodes, force=args.force)
    if args.command in ("align", "all"):
        current, reason = stage.alignment_current()
        # Unchanged tracks and depth are not enough: they must have been made from current inputs too.
        chain = ArtifactChain(configs, stage.source)
        stale = {e: chain.tracks(e) + chain.depth(e) for e in (
            episodes if (args.command == "align" or args.force or not current)
            else read_json(stage.alignment_path())["inputs"]["episodes"])}
        stale = {e: p for e, p in stale.items() if p}
        if stale:
            raise SystemExit("[geometry] the alignment needs current tracks and depth for its episodes:\n  "
                             + problems_text(stale))
        if args.command == "align" or args.force or not current:
            if args.command == "all" and os.path.isfile(stage.alignment_path()):
                print(f"[geometry] aligning again ({reason})", flush=True)
            record = stage.align(episodes)
            fit = record["fit"]
            print(f"[geometry] alignment {'accepted' if record['accepted'] else 'REFUSED'}: "
                  f"scale {fit['scale']:.3f}, median residual {fit['median_residual_m'] * 100:.1f} cm, "
                  f"{fit['inliers']}/{fit['pairs']} inliers -> scene frame {record['scene_frame']}")
    if args.command in ("measure", "all"):
        chain = ArtifactChain(configs, stage.source)
        refused: Dict[int, List[str]] = {}
        for episode in episodes:
            # Everything the measurement reads -- annotation, tracks, depth, the
            # alignment and what the alignment was fitted on -- has to be current.
            problems = chain.measurement_inputs(episode)
            if problems:
                refused[episode] = problems
                continue
            path = stage.source.geometry_path(episode)
            current, reason = reusable(path, stage.geometry_inputs(episode))
            if current and not args.force:
                print(f"[geometry] episode {episode}: current, kept")
                continue
            path = stage.measure(episode)
            meta = read_json(path[:-4] + ".json")
            print(f"[geometry] episode {episode}: frame {meta['scene_frame']}, known "
                  + ", ".join(f"{k}={v:.2f}" for k, v in meta["known_fraction"].items())
                  + "; longest unmeasured run "
                  + ", ".join(f"{k}={v['longest_unmeasured_run']}" for k, v in meta["gaps"].items()), flush=True)
        if refused:
            raise SystemExit(f"[geometry] {len(refused)} episode(s) not measured:\n  " + problems_text(refused))


if __name__ == "__main__":
    main()
