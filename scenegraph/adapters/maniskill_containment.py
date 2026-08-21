"""Hole and slot features for normal ManiSkill containment.

The one place scene discovery is not enough. PegInsertionSide's hole and
PlugCharger's slot are geometric features of another actor, not actors in their
own right, so no segmentation id names them and no force query finds their
centre. The env exposes them directly and this reads them.

Detection is by capability, not by env id -- there is no task table here. Order
matters: PegInsertionSide also defines ``goal_pose``, so the peg capability is
tested first.

No virtual node is added. The graph keeps ``box -> peg`` and
``receptacle -> charger``; only the edge's spatial values are computed in the
hole/slot frame instead of the container actor's origin.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

CAPABILITY_PEG = "peg_hole"
CAPABILITY_SLOT = "charger_slot"

_PEG_ATTRS = ("box_hole_pose", "box_hole_radii", "peg_head_pose")
_SLOT_ATTRS = ("goal_pose", "charger", "receptacle")

# PegInsertionSide.has_peg_inserted: x is the hole axis and the head counts as
# in once it is 15mm short of the mouth.
PEG_AXIAL_MIN = -0.015
# PlugCharger.evaluate.
SLOT_POS_TOL = 5e-3
SLOT_ANGLE_TOL = 0.2


def _np(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value, dtype=float)


def _row(value, env_idx: int) -> np.ndarray:
    arr = _np(value)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr[min(env_idx, arr.shape[0] - 1)] if arr.ndim > 1 else arr


def _unwrap(env):
    return env.unwrapped if hasattr(env, "unwrapped") else env


def _quat_angle(q: np.ndarray) -> float:
    """Rotation magnitude of a wxyz quaternion, in radians."""
    w = float(np.clip(abs(q[0]), -1.0, 1.0))
    return float(2.0 * np.arccos(w))


def detect_capability(env) -> Optional[str]:
    """Which containment representation this env exposes, if any.

    Called after reset: PlugCharger assigns ``goal_pose`` during episode
    initialization, so it does not exist on a freshly constructed env.
    """
    base = _unwrap(env)
    if all(hasattr(base, a) for a in _PEG_ATTRS):
        return CAPABILITY_PEG
    if all(hasattr(base, a) for a in _SLOT_ATTRS):
        return CAPABILITY_SLOT
    return None


def peg_features(env, env_idx: int = 0) -> Dict[str, Any]:
    """Peg head in the hole frame.

    The hole offset varies per env but vanishes in this frame; the radius does
    not, so it is carried and the lateral error is expressed relative to it.
    Normalizing by ``max(|y|, |z|)`` reproduces the env's own square-opening
    test rather than approximating it with a circle.
    """
    base = _unwrap(env)
    rel = base.box_hole_pose.inv() * base.peg_head_pose
    p = _row(rel.p, env_idx)
    q = _row(rel.q, env_idx)
    radius = float(_np(base.box_hole_radii).reshape(-1)[
        min(env_idx, _np(base.box_hole_radii).size - 1)])

    axial = float(p[0])
    lateral = float(max(abs(p[1]), abs(p[2])))
    inserted = bool(axial >= PEG_AXIAL_MIN and lateral <= radius)
    return {
        "capability": CAPABILITY_PEG,
        "container_key": "actor:box_with_hole",
        "containee_key": "actor:peg",
        "hole_pose": _row(base.box_hole_pose.raw_pose, env_idx).tolist(),
        "hole_half_width": radius,
        "key_pose": _row(base.peg_head_pose.raw_pose, env_idx).tolist(),
        "rel_position": p.tolist(),
        "rel_quat": q.tolist(),
        "rel_angle": _quat_angle(q),
        "axial": axial,
        "lateral": lateral,
        # Scale-free, so a mined threshold survives a re-randomized radius.
        "lateral_norm": lateral / radius if radius > 0 else float("inf"),
        "containee_dims": _row(base.peg_half_sizes, env_idx).tolist()
        if hasattr(base, "peg_half_sizes") else None,
        "holds": inserted,
    }


def slot_features(env, env_idx: int = 0) -> Dict[str, Any]:
    """Charger in the slot-aligned goal frame.

    Dimensions are fixed in PlugCharger today but are recorded anyway, so the
    asset format survives the task gaining geometry randomization.
    """
    base = _unwrap(env)
    rel = base.goal_pose.inv() * base.charger.pose
    p = _row(rel.p, env_idx)
    q = _row(rel.q, env_idx)

    distance = float(np.linalg.norm(p))
    angle = _quat_angle(q)
    peg_size = getattr(base, "_peg_size", None)
    return {
        "capability": CAPABILITY_SLOT,
        "container_key": "actor:receptacle",
        "containee_key": "actor:charger",
        "hole_pose": _row(base.goal_pose.raw_pose, env_idx).tolist(),
        "hole_half_width": float(np.max(_np(peg_size)))
        if peg_size is not None else None,
        "key_pose": _row(base.charger.pose.raw_pose, env_idx).tolist(),
        "rel_position": p.tolist(),
        "rel_quat": q.tolist(),
        "rel_angle": angle,
        "axial": float(p[0]),
        "lateral": float(max(abs(p[1]), abs(p[2]))),
        "distance": distance,
        "containee_dims": _np(peg_size).tolist() if peg_size is not None
        else None,
        "container_dims": _np(getattr(base, "_base_size", None)).tolist()
        if getattr(base, "_base_size", None) is not None else None,
        "holds": bool(distance <= SLOT_POS_TOL and angle <= SLOT_ANGLE_TOL),
    }


def containment_features(env, env_idx: int = 0,
                         capability: Optional[str] = None
                         ) -> Optional[Dict[str, Any]]:
    """Features for whichever containment this env exposes, or None."""
    capability = capability or detect_capability(env)
    if capability == CAPABILITY_PEG:
        return peg_features(env, env_idx)
    if capability == CAPABILITY_SLOT:
        return slot_features(env, env_idx)
    return None
