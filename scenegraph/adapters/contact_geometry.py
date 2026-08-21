"""Contact points and collision shapes, read from the CPU sim.

Two things the GPU runtime cannot supply. ``px.get_contacts()`` exists only on
the CPU backend -- the GPU path returns one aggregate impulse per pair and no
per-point geometry -- and collision shapes are not exposed per-env there
either.

That asymmetry is fine because both are mined offline into object-local
descriptors. The runtime compares those against current poses and uses the
aggregate force only as a binary predicate. Nothing here may be needed live.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _entity_of(wrapper, row: int = 0):
    """Underlying SAPIEN entity for one env row of an Actor/Link wrapper."""
    objs = getattr(wrapper, "_objs", None)
    if not objs or row >= len(objs):
        return None
    obj = objs[row]
    return getattr(obj, "entity", obj)


def _rotmat(quat_wxyz) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat_wxyz]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def to_local(points: np.ndarray, pose_world) -> np.ndarray:
    """World positions -> the object's own frame."""
    pose = np.asarray(pose_world, dtype=float).reshape(-1)
    rot = _rotmat(pose[3:7])
    return (np.asarray(points, float) - pose[:3]) @ rot


def directions_to_local(vectors: np.ndarray, pose_world) -> np.ndarray:
    """World directions -> the object's own frame (rotation only)."""
    pose = np.asarray(pose_world, dtype=float).reshape(-1)
    return np.asarray(vectors, float) @ _rotmat(pose[3:7])


def pairwise_contact_points(
    scene, a, b, row_a: int = 0, row_b: int = 0,
) -> Optional[Dict[str, np.ndarray]]:
    """World-frame contact points between two entities, or None.

    Normals point from ``a`` toward ``b``; the sign is corrected when SAPIEN
    lists the pair the other way round, so the caller does not have to care
    which entity it passed first.
    """
    px = getattr(scene, "px", None)
    if px is None or not hasattr(px, "get_contacts"):
        return None
    ent_a, ent_b = _entity_of(a, row_a), _entity_of(b, row_b)
    if ent_a is None or ent_b is None:
        return None

    positions: List[np.ndarray] = []
    normals: List[np.ndarray] = []
    impulses: List[np.ndarray] = []
    for contact in px.get_contacts():
        bodies = getattr(contact, "bodies", None)
        if not bodies or len(bodies) < 2:
            continue
        e0 = getattr(bodies[0], "entity", None)
        e1 = getattr(bodies[1], "entity", None)
        if e0 is ent_a and e1 is ent_b:
            sign = 1.0
        elif e0 is ent_b and e1 is ent_a:
            sign = -1.0
        else:
            continue
        for point in getattr(contact, "points", ()) or ():
            positions.append(np.asarray(point.position, dtype=float))
            normals.append(sign * np.asarray(point.normal, dtype=float))
            impulses.append(sign * np.asarray(point.impulse, dtype=float))

    if not positions:
        return None
    return {
        "positions": np.asarray(positions),
        "normals": np.asarray(normals),
        "impulses": np.asarray(impulses),
    }


def contact_anchor(points: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Impulse-weighted contact position and unit normal for one frame.

    Weighted, not averaged: a light glancing point should not move the anchor
    as much as the one actually carrying the load.
    """
    positions = points["positions"]
    normals = points["normals"]
    weights = np.linalg.norm(points["impulses"], axis=1)
    if not np.any(weights > 0):
        weights = np.ones(len(positions))
    weights = weights / weights.sum()

    anchor = (positions * weights[:, None]).sum(axis=0)
    normal = (normals * weights[:, None]).sum(axis=0)
    norm = float(np.linalg.norm(normal))
    if norm > 0:
        normal = normal / norm
    return anchor, normal


def paired_contact_frame(
    scene, a, b, pose_a, pose_b, row_a: int = 0, row_b: int = 0,
) -> Optional[Dict[str, list]]:
    """One physical contact, described from both endpoints' frames.

    Both sides come from the same event, so the miner keeps them index-aligned
    and the runtime compares ``a[i]`` against ``b[i]`` -- N comparisons, never
    an N-by-N product.
    """
    points = pairwise_contact_points(scene, a, b, row_a, row_b)
    if points is None or pose_a is None or pose_b is None:
        return None
    anchor, normal = contact_anchor(points)
    return {
        "anchor_a_local": to_local(anchor[None, :], pose_a)[0].tolist(),
        "normal_a_local": directions_to_local(normal[None, :], pose_a)[0].tolist(),
        "anchor_b_local": to_local(anchor[None, :], pose_b)[0].tolist(),
        # Flipped: the outward normal of b points back toward a.
        "normal_b_local": directions_to_local(-normal[None, :], pose_b)[0].tolist(),
        "contact_position": anchor.tolist(),
        "contact_normal": normal.tolist(),
        "n_points": int(len(points["positions"])),
        "anchor_source": "contact_points",
    }


# --------------------------------------------------------------------------- #
# Collision shapes
# --------------------------------------------------------------------------- #
def collision_shapes(entity, row: int = 0) -> List[Any]:
    ent = _entity_of(entity, row)
    if ent is None:
        return []
    for comp in getattr(ent, "components", None) or ():
        getter = getattr(comp, "get_collision_shapes", None)
        if callable(getter):
            try:
                return list(getter())
            except Exception:
                return []
    getter = getattr(ent, "get_collision_shapes", None)
    if callable(getter):
        try:
            return list(getter())
        except Exception:
            return []
    return []


def spherical_radius(entity, row: int = 0) -> Optional[float]:
    """Radius if this actor's collision geometry is a single sphere.

    A sphere has no meaningful orientation, so grasp evidence for one must be
    stored as a radial offset rather than as a pose in the object frame --
    otherwise 300 samples record 300 arbitrary angular coordinates that mean
    nothing to a runtime comparison.
    """
    shapes = collision_shapes(entity, row)
    if len(shapes) != 1:
        return None
    shape = shapes[0]
    if type(shape).__name__ != "PhysxCollisionShapeSphere":
        return None
    radius = getattr(shape, "radius", None)
    return None if radius is None else float(radius)


def symmetry_of(entity, row: int = 0) -> Dict[str, Any]:
    radius = spherical_radius(entity, row)
    if radius is None:
        return {"symmetry": "none"}
    return {"symmetry": "spherical", "radius": radius,
            "orientation_invariant": True}
