"""Pull RGB + segmentation out of a ManiSkill obs dict and build masks.

ManiSkill ``obs_mode="rgb+depth+segmentation"`` returns, per camera::

    obs["sensor_data"][cam]["rgb"]            uint8  [N, H, W, 3]
    obs["sensor_data"][cam]["depth"]                 [N, H, W, 1]
    obs["sensor_data"][cam]["segmentation"]   int    [N, H, W, 1]   (id 0 = bg)

All tensors are batched even when num_envs == 1.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def list_cameras(obs: dict) -> List[str]:
    sd = obs.get("sensor_data", {})
    return list(sd.keys())


def pick_camera(obs: dict, preferred: Optional[str] = None) -> str:
    cams = list_cameras(obs)
    if not cams:
        raise KeyError("obs has no sensor_data; is obs_mode a visual mode?")
    if preferred and preferred in cams:
        return preferred
    # Prefer a base/head camera if present, else first.
    for key in ("base_camera", "fetch_head", "render_camera"):
        if key in cams:
            return key
    return cams[0]


def extract_camera_obs(
    obs: dict, camera: str, env_idx: int = 0
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return (rgb[H,W,3] uint8, seg[H,W] int, depth[H,W] or None) for one env."""
    cam = obs["sensor_data"][camera]
    rgb = _to_np(cam["rgb"])[env_idx]                       # [H, W, 3]
    seg = _to_np(cam["segmentation"])[env_idx]              # [H, W, 1]
    seg = seg.squeeze(-1).astype(np.int64)                  # [H, W]
    depth = None
    if "depth" in cam:
        depth = _to_np(cam["depth"])[env_idx].squeeze(-1).astype(np.float32)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb, seg, depth


def unique_seg_ids(seg: np.ndarray, exclude_background: bool = True) -> List[int]:
    ids = [int(i) for i in np.unique(seg)]
    if exclude_background:
        ids = [i for i in ids if i != 0]               # ManiSkill: 0 == background
    return ids


def read_unwrapped_sensor(env, camera: str, env_idx: int = 0):
    """Read (seg[H,W], depth[H,W] or None) directly from the unwrapped env.

    Used when the observation wrappers (e.g. MS-HAB depth + framestack) have
    stripped segmentation from the policy obs. The unwrapped env still computes
    the full sensor_data when obs_mode includes segmentation.

    Returns (seg[H,W], depth[H,W] or None, rgb[H,W,3] uint8 or None).
    """
    u = env.unwrapped
    full = u.get_obs()
    cam = full["sensor_data"][camera]
    seg = _to_np(cam["segmentation"])[env_idx].squeeze(-1).astype(np.int64)
    depth = None
    if "depth" in cam:
        depth = _to_np(cam["depth"])[env_idx].squeeze(-1).astype(np.float32)
    rgb = None
    if "rgb" in cam:
        r = _to_np(cam["rgb"])[env_idx]
        if r.dtype != np.uint8:
            r = np.clip(r, 0, 255).astype(np.uint8)
        rgb = r
    return seg, depth, rgb


def read_unwrapped_rgbs(env, env_idx: int = 0) -> Dict[str, np.ndarray]:
    """Read every RGB sensor without requiring a segmentation texture.

    Observation wrappers used by MS-HAB remove the camera dictionaries before
    the policy sees them, so RGB export must read the unwrapped environment.
    Unlike :func:`read_unwrapped_sensor`, this helper also works when the env
    was created with ``obs_mode="rgb+depth"`` and segmentation is disabled.
    """
    full = env.unwrapped.get_obs()
    sensor_data = full.get("sensor_data", {})
    frames: Dict[str, np.ndarray] = {}
    for camera, camera_obs in sensor_data.items():
        if "rgb" not in camera_obs:
            continue
        rgb = _to_np(camera_obs["rgb"])[env_idx]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        frames[str(camera)] = rgb
    return frames


def depth_to_gray_rgb(depth: np.ndarray) -> np.ndarray:
    """Turn a [H,W] depth map into a [H,W,3] uint8 grayscale backdrop."""
    d = depth.astype(np.float32)
    finite = d[np.isfinite(d)]
    if finite.size == 0:
        g = np.zeros_like(d)
    else:
        lo, hi = float(finite.min()), float(finite.max())
        g = (d - lo) / (hi - lo + 1e-6)
        g = np.clip(1.0 - g, 0, 1)        # near = bright
    gray = (g * 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def depth_to_color_rgb(depth: np.ndarray, cmap_name: str = "turbo") -> np.ndarray:
    """Turn a [H,W] depth map into a vivid [H,W,3] uint8 colormapped backdrop.

    Near = warm, far = cool (turbo). Used when no RGB sensor is available so the
    depth-mode overlay still reads like a real scene rather than flat gray.
    """
    import matplotlib.cm as cm

    d = depth.astype(np.float32)
    finite = d[np.isfinite(d)]
    if finite.size == 0:
        norm = np.zeros_like(d)
    else:
        lo, hi = float(finite.min()), float(finite.max())
        norm = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)
        norm = 1.0 - norm            # near = high end of colormap
    colored = cm.get_cmap(cmap_name)(norm)[..., :3]   # [H,W,3] float
    return (colored * 255).astype(np.uint8)


def mask_for_id(seg: np.ndarray, seg_id: int) -> np.ndarray:
    return seg == seg_id


def pixel_area(mask: np.ndarray) -> int:
    return int(mask.sum())


class MaskAccumulator:
    """Collects binary masks keyed by node_id, OR-merging repeated writes."""

    def __init__(self, height: int, width: int):
        self.h, self.w = height, width
        self._masks: Dict[str, np.ndarray] = {}

    def add(self, node_id: str, mask: np.ndarray) -> None:
        if node_id in self._masks:
            self._masks[node_id] |= mask
        else:
            self._masks[node_id] = mask.copy()

    def get(self, node_id: str) -> Optional[np.ndarray]:
        return self._masks.get(node_id)

    def area(self, node_id: str) -> int:
        m = self._masks.get(node_id)
        return 0 if m is None else int(m.sum())

    def empty(self, node_id: str) -> np.ndarray:
        return np.zeros((self.h, self.w), dtype=bool)

    def keys(self) -> List[str]:
        return list(self._masks.keys())

    def as_dict(self) -> Dict[str, np.ndarray]:
        return self._masks
