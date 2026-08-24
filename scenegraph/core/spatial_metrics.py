"""One definition of the spatial measurement mining and runtime both use.

Two implementations drift: the miner calibrated bins on object origins while
the runtime measured surface anchors, so a token meant one distance when it
was mined and another when it was read. Everything that turns poses into a
planar distance or a height offset goes through here.

Pure numpy, no torch / maniskill.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

# Gravity is -z, so an object's lower surface is its centre minus radius * up.
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=float)

# Calibration scopes. The relation vocabulary is shared; the scales are not.
EE_OBJECT_SCOPE = "ee-object"
OBJECT_OBJECT_SCOPE = "object-object"
SPATIAL_SCOPES: Tuple[str, ...] = (EE_OBJECT_SCOPE, OBJECT_OBJECT_SCOPE)


def spatial_bin_key(scope: str, relation: str) -> str:
    """Asset key for a scoped absolute scale, e.g. ``ee-object-height-offset``."""
    return f"{scope}-{relation}"


def change_bin_key(bin_key: str) -> str:
    return f"{bin_key}-change"


def stat_key(scope: str, relation: str) -> str:
    """Miner statistic name for the same scale, underscored."""
    return spatial_bin_key(scope, relation).replace("-", "_")


def _quat_to_rot(pose7: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in np.asarray(pose7, dtype=float)[3:7])
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def anchor_world(
    pose7: Sequence[float],
    anchor_local: Optional[Sequence[float]],
    radial_offset: Optional[float] = None,
) -> Optional[np.ndarray]:
    """World point for a mined anchor on a body at ``pose7``.

    ``radial_offset`` selects the orientation-invariant form: a sphere's
    contact point is centre - r * up whatever its quaternion says, so a
    rotating but stationary ball keeps its spatial relations.
    """
    pose = np.asarray(pose7, dtype=float).reshape(-1)
    if pose.size < 7 or not np.all(np.isfinite(pose[:7])):
        return None
    centre = pose[:3]
    if radial_offset is not None:
        return centre - float(radial_offset) * WORLD_UP
    if anchor_local is None:
        return None
    local = np.asarray(anchor_local, dtype=float).reshape(-1)
    if local.size < 3 or not np.all(np.isfinite(local[:3])):
        return None
    return centre + _quat_to_rot(pose) @ local[:3]


def pair_points(
    pose_a: Sequence[float],
    pose_b: Sequence[float],
    anchor_a: Optional[Sequence[float]] = None,
    anchor_b: Optional[Sequence[float]] = None,
    radial_a: Optional[float] = None,
    radial_b: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """World points for a pair, origins when either anchor is unavailable.

    All-or-nothing on purpose: mixing one anchor with one origin would measure
    a quantity neither the miner nor the runtime intends.
    """
    pa = anchor_world(pose_a, anchor_a, radial_a)
    pb = anchor_world(pose_b, anchor_b, radial_b)
    if pa is None or pb is None:
        pa = np.asarray(pose_a, dtype=float)[:3]
        pb = np.asarray(pose_b, dtype=float)[:3]
    return pa, pb


def planar_distance(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a[:2] - b[:2]))


def height_offset(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.asarray(a, dtype=float)[2] - np.asarray(b, dtype=float)[2])


def measures(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """``(planar, signed height)`` for two world points, in ``a - b`` order."""
    return planar_distance(a, b), height_offset(a, b)
