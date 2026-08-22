"""Object-to-camera coverage without depth.

Relation eligibility under ``projected_camera`` asks one question: does a camera
cover the space this object occupies? Segmentation cannot answer it -- the robot
arm removes the pixels of exactly the objects a manipulation policy cares about
-- so the object's own collision geometry is projected instead.

Two caches with different lifetimes. Local AABB corners are geometry and change
only when the scene reconfigures. Extrinsics are pose and change every frame:
Fetch drives its base, pans its head and moves its arm, so both of its cameras
move in the world frame even though each is rigidly mounted to its own link.
Only the intrinsics are genuinely fixed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .contact_geometry import collision_shapes


def _corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """The eight corners of an axis-aligned box."""
    return np.array([[x, y, z] for x in (lo[0], hi[0])
                     for y in (lo[1], hi[1])
                     for z in (lo[2], hi[2])], dtype=float)


def _shape_corners(shape) -> Optional[np.ndarray]:
    """``(8, 3)`` corners of one collision shape's own local AABB.

    Eight corners rather than the two diagonal extremes: the caller rotates
    these into the body frame, and rotating a diagonal pair does not bound the
    rotated shape -- it bounds a line through it.
    """
    geom = getattr(shape, "geometry", None) or shape
    half = getattr(geom, "half_size", None)
    if half is not None:
        h = np.asarray(half, dtype=float).reshape(-1)[:3]
        return _corners(-h, h)
    verts = getattr(geom, "vertices", None)
    if verts is not None:
        v = np.asarray(verts, dtype=float).reshape(-1, 3)
        scale = getattr(geom, "scale", None)
        if scale is not None:
            v = v * np.asarray(scale, dtype=float).reshape(1, 3)
        return _corners(v.min(0), v.max(0))
    radius = getattr(geom, "radius", None)
    if radius is not None:
        r = float(radius)
        half_len = float(getattr(geom, "half_length", 0.0) or 0.0)
        e = np.array([r + half_len, r, r], dtype=float)
        return _corners(-e, e)
    return None


def _shape_local_pose(shape):
    """SAPIEN has exposed this as both a property and a getter. Take whichever
    is present -- silently missing it would drop every compound offset."""
    getter = getattr(shape, "get_local_pose", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            pass
    return getattr(shape, "local_pose", None)


def _quat_to_mat(q) -> np.ndarray:
    """SAPIEN wxyz quaternion to a rotation matrix."""
    w, x, y, z = [float(v) for v in np.asarray(q, dtype=float).reshape(-1)[:4]]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def local_aabb_corners(entity, row: int = 0) -> Optional[np.ndarray]:
    """``(8, 3)`` object-frame AABB corners, or None if nothing has geometry.

    Each shape's own local transform is applied first: a compound body's parts
    sit away from the body origin, and ignoring that would shrink the box to
    whichever part happens to be centred.
    """
    points: List[np.ndarray] = []
    for shape in collision_shapes(entity, row):
        pts = _shape_corners(shape)
        if pts is None:
            continue
        pose = _shape_local_pose(shape)
        if pose is not None:
            rot = _quat_to_mat(getattr(pose, "q", (1.0, 0.0, 0.0, 0.0)))
            off = np.asarray(getattr(pose, "p", (0.0, 0.0, 0.0)), dtype=float)
            pts = pts @ rot.T + off.reshape(1, 3)
        points.append(pts)
    if not points:
        return None
    allpts = np.concatenate(points, axis=0)
    return _corners(allpts.min(0), allpts.max(0))


def corners_world(corners_local: np.ndarray, pose_world) -> np.ndarray:
    """Object-frame corners into the world, given a ``[x,y,z,w,i,j,k]`` pose."""
    pose = np.asarray(pose_world, dtype=float).reshape(-1)
    rot = _quat_to_mat(pose[3:7])
    return corners_local @ rot.T + pose[:3].reshape(1, 3)


def box_covers_image(
    corners: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray,
    width: int, height: int, near: float = 1e-3,
) -> bool:
    """Does the projected box overlap the image rectangle?

    Corners behind the camera are dropped rather than projected -- a negative
    depth flips the sign of the projection and would place an object behind the
    robot in the middle of the frame. An object with no corner in front of the
    camera is not covered at all.
    """
    ext = np.asarray(extrinsic, dtype=float).reshape(3, 4)
    cam = corners @ ext[:, :3].T + ext[:, 3].reshape(1, 3)
    front = cam[cam[:, 2] > near]
    if front.size == 0:
        return False
    uvw = front @ np.asarray(intrinsic, dtype=float).reshape(3, 3).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    lo, hi = uv.min(0), uv.max(0)
    return bool(hi[0] >= 0.0 and lo[0] <= width
                and hi[1] >= 0.0 and lo[1] <= height)


class CameraCoverage:
    """Per-env camera matrices plus a per-entity local-AABB cache."""

    def __init__(self, env, camera_names: Optional[List[str]] = None):
        self.env = env
        self._names = list(camera_names) if camera_names else None
        self._intrinsics: Dict[str, np.ndarray] = {}
        self._shape: Dict[str, Tuple[int, int]] = {}
        self._aabb: Dict[int, Optional[np.ndarray]] = {}

    def invalidate(self) -> None:
        """Drop geometry caches. Reconfiguration destroys the actors they
        describe; intrinsics survive because the sensor config does."""
        self._aabb.clear()

    @property
    def _sensors(self) -> Dict[str, Any]:
        base = getattr(self.env, "unwrapped", self.env)
        sensors = getattr(getattr(base, "scene", None), "sensors", None) or {}
        if self._names is None:
            return dict(sensors)
        return {n: sensors[n] for n in self._names if n in sensors}

    def _matrices(self, name: str, sensor, env_idx: int):
        cam = getattr(sensor, "camera", sensor)
        if name not in self._intrinsics:
            k = _row(cam.get_intrinsic_matrix(), env_idx).reshape(3, 3)
            self._intrinsics[name] = k
            self._shape[name] = (_dim(cam, "width", k[0, 2]),
                                 _dim(cam, "height", k[1, 2]))
        ext = _row(cam.get_extrinsic_matrix(), env_idx).reshape(3, 4)
        return self._intrinsics[name], ext, self._shape[name]

    def corners_for(self, entity, row: int, pose_world) -> Optional[np.ndarray]:
        key = id(entity)
        if key not in self._aabb:
            self._aabb[key] = local_aabb_corners(entity, row)
        local = self._aabb[key]
        if local is None or pose_world is None:
            return None
        return corners_world(local, pose_world)

    def covers(self, entity, row: int, pose_world, env_idx: int) -> bool:
        """True when any camera's frame overlaps the entity's projected box.

        An entity with no collision geometry projects as its centroid alone.
        That under-covers a large object grazing the frame edge, but it answers
        the same question the same way for every node -- assuming ``True``
        would make "no geometry" mean "always relational", which is not a
        camera test at all.
        """
        corners = self.corners_for(entity, row, pose_world)
        if corners is None:
            if pose_world is None:
                return False
            corners = np.asarray(
                pose_world, dtype=float).reshape(-1)[:3][None, :]
        for name, sensor in self._sensors.items():
            try:
                k, ext, (w, h) = self._matrices(name, sensor, env_idx)
            except Exception:
                continue
            if box_covers_image(corners, ext, k, w, h):
                return True
        return False


def _dim(cam, name: str, principal: float) -> int:
    """Image size from the camera itself where it exposes one.

    ``2 * principal_point`` is only the size for a centred principal point, so
    it is the last resort rather than the rule.
    """
    getter = getattr(cam, f"get_{name}", None)
    if callable(getter):
        try:
            return int(getter())
        except Exception:
            pass
    value = getattr(cam, name, None)
    if value:
        return int(value)
    return int(round(float(principal) * 2))


def _row(value, env_idx: int) -> np.ndarray:
    arr = value.cpu() if hasattr(value, "cpu") else value
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 3:
        return arr[min(env_idx, arr.shape[0] - 1)]
    return arr
