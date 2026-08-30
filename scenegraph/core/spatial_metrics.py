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

# Object-to-region is deliberately NOT in SPATIAL_SCOPES. That tuple is a
# cross-product generator -- every consumer loops it against
# {planar-distance, height-offset} -- so appending here would silently create
# and then require an ``object-region-height-offset`` nothing measures.
# PullCubeTool's goal region is a disc around the robot base: it has a
# horizontal extent and no height target at all.
OBJECT_REGION_SCOPE = "object-region"
OBJECT_REGION_PLANAR_KEY = f"{OBJECT_REGION_SCOPE}-planar-distance"

# Object-to-site, for a virtual site that carries a distance ladder. Also
# registered by hand rather than added to SPATIAL_SCOPES, for the same reason.
# It gets both relations: PegInsertionSide's hole is offset up to 5cm in y and
# z against an aperture of under 3cm, so vertical alignment is real work and a
# planar-only scale would leave it unrewarded until ``reached`` fires.
#
# Deliberately separate from object-object. The peg head-to-mouth distance and
# the peg-to-box origin distance are both emitted, and they are not the same
# quantity: sharing a scale would let one pair's range set the other's bins.
# End-effector height families. The relation stays ``height-offset``; only the
# scale splits. One shared scale let the table -- a metre below the gripper
# because its origin is under its own top -- set a +/-0.21m deadband that swallowed
# every end-effector-to-manipuland height in every shipped task.
#
# Planar is deliberately not split: it is already a distance between two
# points the policy can act on, and a structural surface emits none at all.
FAMILY_STRUCTURAL = "structural-surface"
FAMILY_MANIPULAND = "manipuland"
FAMILY_RECEPTACLE = "receptacle"
FAMILY_GOAL_MARKER = "goal-marker"
EE_HEIGHT_FAMILIES: Tuple[str, ...] = (
    FAMILY_STRUCTURAL, FAMILY_MANIPULAND, FAMILY_RECEPTACLE,
    FAMILY_GOAL_MARKER,
)


def ee_family_bin_key(family: str) -> str:
    """``ee-manipuland-height-offset`` and friends."""
    return f"ee-{family}-height-offset"


OBJECT_SITE_SCOPE = "object-site"
OBJECT_SITE_PLANAR_KEY = f"{OBJECT_SITE_SCOPE}-planar-distance"
OBJECT_SITE_HEIGHT_KEY = f"{OBJECT_SITE_SCOPE}-height-offset"


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


def oriented_normal(normal: Sequence[float]) -> Optional[np.ndarray]:
    """A surface normal forced to point away from the surface, i.e. up.

    Support normals are mined from contact forces, and the force on a
    supporter from the thing it carries points *into* it -- every table in the
    shipped assets records ``[0, 0, -1]``. Reading that literally makes an
    object resting on the table report a negative height, and the sign error is
    invisible because ``level`` is symmetric around zero. Normalising here is
    what lets one formula serve both conventions.

    Returns None for a degenerate or horizontal-only normal, where "away from
    the surface" has no answer and a guess would be a silent half-metre error.
    """
    arr = np.asarray(normal, dtype=float).reshape(-1)
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    arr = arr[:3]
    length = float(np.linalg.norm(arr))
    if length <= 0.0:
        return None
    unit = arr / length
    vertical = float(np.dot(unit, WORLD_UP))
    if abs(vertical) < 1e-6:
        return None
    return unit if vertical > 0.0 else -unit


def surface_height(
    point: Sequence[float],
    surface_anchor: Sequence[float],
    surface_normal: Sequence[float],
) -> Optional[float]:
    """Signed distance from ``point`` to the plane through the anchor.

    ``dot(point - anchor, outward normal)``: positive above the surface,
    exactly zero on it. This is the whole of the table fix -- a table's link
    origin sits ~0.9m below its own top, so measuring against the origin
    reported every end-effector as a metre in the air and set the height scale
    for every other pair in the scene.
    """
    normal = oriented_normal(surface_normal)
    if normal is None:
        return None
    p = np.asarray(point, dtype=float).reshape(-1)
    a = np.asarray(surface_anchor, dtype=float).reshape(-1)
    if p.size < 3 or a.size < 3:
        return None
    if not (np.all(np.isfinite(p[:3])) and np.all(np.isfinite(a[:3]))):
        return None
    return float(np.dot(p[:3] - a[:3], normal))


def planar_distance(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a[:2] - b[:2]))


def height_offset(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.asarray(a, dtype=float)[2] - np.asarray(b, dtype=float)[2])


def measures(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """``(planar, signed height)`` for two world points, in ``a - b`` order."""
    return planar_distance(a, b), height_offset(a, b)
