"""Live goal geometry for spatial sites.

Every number here is read from the running environment, never mined. A task
that re-randomizes its goal each episode -- and PegInsertionSide re-randomizes
its hole radius and box depth on every reset -- would otherwise be scored
against a value frozen at mining time.

Which reader serves which site is named in the asset (``SiteDeclaration.
provider``), not inferred from the key here, so the per-task knowledge stays in
the mined file. This module only implements the readers.

The tolerances mirror each environment's own success predicate:

* PickCube: ``norm(cube - goal_site) <= goal_thresh`` (``is_obj_placed``). The
  static-robot half of ``success`` is a separate condition and is not our
  business.
* PullCubeTool: ``XY_distance(cube, robot_base) < 0.6`` in ``evaluate``. The
  unused ``goal_radius = 0.3`` attribute and the shifted dense-reward
  workspace centre are both deliberately ignored -- neither is the success
  geometry.
* PegInsertionSide: the hole mouth, which is ``box_hole_pose`` walked back
  along the entry axis by the box's half-depth.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.sites import (
    PROVIDER_PEG_HOLE_MOUTH,
    PROVIDER_PICK_CUBE_GOAL,
    PROVIDER_ROBOT_BASE_REGION,
    SiteDeclaration,
    SiteError,
    SiteSpec,
)

# PullCubeTool.evaluate: ``cube_to_base_dist < 0.6``. A literal upstream with
# no attribute behind it, so it is a literal here too, named to its source in
# the same way ``maniskill_containment.PEG_AXIAL_MIN`` is.
PULL_SUCCESS_RADIUS = 0.6


def _unwrap(env):
    return env.unwrapped if hasattr(env, "unwrapped") else env


def _np(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value, dtype=float)


def _row(value, env_idx: int) -> np.ndarray:
    arr = _np(value)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr[min(env_idx, arr.shape[0] - 1)] if arr.ndim > 1 else arr


def _scalar(value, env_idx: int) -> float:
    arr = _np(value).reshape(-1)
    return float(arr[min(env_idx, arr.size - 1)])


def _pose7(pose, env_idx: int) -> np.ndarray:
    """SAPIEN pose -> ``[x, y, z, qw, qx, qy, qz]`` for one sub-scene."""
    raw = getattr(pose, "raw_pose", None)
    if raw is not None:
        return _row(raw, env_idx).reshape(-1)[:7]
    p = _row(pose.p, env_idx).reshape(-1)[:3]
    q = _row(pose.q, env_idx).reshape(-1)[:4]
    return np.concatenate([p, q])


def _rot(pose7: np.ndarray) -> np.ndarray:
    w, x, y, z = pose7[3:7]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


# --------------------------------------------------------------------------- #
# PickCube
# --------------------------------------------------------------------------- #
def pick_cube_goal(env, env_idx: int, decl: SiteDeclaration) -> SiteSpec:
    """The goal marker's live pose and the task's own ``goal_thresh``."""
    base = _unwrap(env)
    for attr in ("goal_site", "goal_thresh"):
        if not hasattr(base, attr):
            raise SiteError(
                f"site {decl.key!r} wants provider {decl.provider!r} but the "
                f"environment exposes no {attr!r}"
            )
    return SiteSpec(
        declaration=decl,
        pose_world=_pose7(base.goal_site.pose, env_idx),
        tolerance=_scalar(base.goal_thresh, env_idx),
    )


# --------------------------------------------------------------------------- #
# PullCubeTool
# --------------------------------------------------------------------------- #
def robot_base_region(env, env_idx: int, decl: SiteDeclaration) -> SiteSpec:
    """A disc around the robot base link, the radius PullCubeTool succeeds at.

    ``get_links()[0]`` is the base, matching the environment's own
    ``evaluate``. Reading the agent rather than a stored pose matters less here
    than elsewhere -- the base does not move -- but it keeps the site defined
    the same way the predicate is.
    """
    base = _unwrap(env)
    robot = getattr(getattr(base, "agent", None), "robot", None)
    if robot is None:
        raise SiteError(
            f"site {decl.key!r} wants the robot base but the environment "
            "exposes no agent.robot"
        )
    links = robot.get_links()
    if not links:
        raise SiteError(f"site {decl.key!r}: the robot reports no links")
    return SiteSpec(
        declaration=decl,
        pose_world=_pose7(links[0].pose, env_idx),
        tolerance=PULL_SUCCESS_RADIUS,
    )


# --------------------------------------------------------------------------- #
# PegInsertionSide
# --------------------------------------------------------------------------- #
_PEG_ATTRS = ("box_hole_pose", "box_hole_radii", "peg_head_pose",
              "peg_half_sizes", "box")


class _BoxDepthCache:
    """The box's half-extent along the hole axis, per environment.

    Read from the box's own collision shapes: ``_build_box_with_hole`` gives
    all four blocks the same x half-size, which is the depth. It is *not* an
    attribute on the environment -- ``depth`` is a local in ``_load_scene`` --
    so the only alternatives are the geometry or an assumption.

    The assumption is available and currently true: upstream draws the box
    depth and the peg half-length from one ``lengths`` sample, so
    ``peg_half_sizes[:, 0]`` equals the depth. It is kept as a checked fallback
    and asserted against the geometry whenever both can be read, because
    nothing upstream promises the two stay tied.

    One entry per sub-scene, invalidated by the geometry that re-randomizes:
    a changed hole radius or peg length means a rebuilt box, so the depth is
    re-read without needing a reset hook. PegInsertionSide reconfigures on
    every reset at ``num_envs=1``, so this re-reads about as often as it is
    asked. Keying on the environment's identity instead would grow without
    bound and could hand a recycled id a stale answer.
    """

    # Two half-extents this close are the same sample, not two.
    TOLERANCE = 1e-6

    def __init__(self) -> None:
        self._key: Optional[Tuple] = None
        self._depth: Optional[float] = None
        self.provenance: str = "unread"

    def _signature(self, base, env_idx: int) -> Tuple:
        return (
            round(_scalar(base.box_hole_radii, env_idx), 9),
            round(float(_row(base.peg_half_sizes, env_idx).reshape(-1)[0]), 9),
        )

    def depth(self, base, env_idx: int) -> float:
        signature = self._signature(base, env_idx)
        if self._key == signature and self._depth is not None:
            return self._depth
        fallback = float(_row(base.peg_half_sizes, env_idx).reshape(-1)[0])
        measured = _collision_x_half_extent(base.box, env_idx)
        if measured is None:
            depth, provenance = fallback, "peg_half_sizes[0] (no collision geometry)"
        else:
            if abs(measured - fallback) > 1e-4:
                raise SiteError(
                    "PegInsertionSide box half-depth disagrees with the peg "
                    f"half-length: collision geometry says {measured:.6f}, "
                    f"peg_half_sizes[0] says {fallback:.6f}. Upstream draws "
                    "both from one 'lengths' sample; if that has changed, the "
                    "fallback is no longer safe and the mouth would be placed "
                    "on the wrong plane."
                )
            depth, provenance = measured, "box collision-shape x half-extent"
        self._key, self._depth, self.provenance = signature, depth, provenance
        return depth


def _collision_x_half_extent(actor, env_idx: int) -> Optional[float]:
    """Half-extent along x shared by the box's four blocks, or None.

    None rather than a raise: SAPIEN exposes collision shapes differently
    across versions and the checked fallback covers it. Disagreement between
    the blocks *is* an error -- that would mean this is not the box we think.
    """
    objs = getattr(actor, "_objs", None) or []
    if not objs:
        return None
    obj = objs[min(env_idx, len(objs) - 1)]
    getter = getattr(obj, "get_collision_shapes", None)
    shapes = getter() if callable(getter) else getattr(obj, "collision_shapes", None)
    if not shapes:
        return None
    extents: List[float] = []
    for shape in shapes:
        half = getattr(shape, "half_size", None)
        if half is None:
            geometry = getattr(shape, "geometry", None)
            half = getattr(geometry, "half_size", None) if geometry else None
        if half is None:
            return None
        extents.append(float(_np(half).reshape(-1)[0]))
    if not extents:
        return None
    if max(extents) - min(extents) > _BoxDepthCache.TOLERANCE:
        raise SiteError(
            "PegInsertionSide box blocks disagree on their x half-extent "
            f"({sorted(set(round(e, 6) for e in extents))}). The hole axis is "
            "x and all four blocks are built at the same depth, so this is "
            "not the geometry the mouth calculation assumes."
        )
    return extents[0]


# Keyed by sub-scene index: a builder owns one and walks its frames in order.
_DEPTH_CACHES: Dict[int, _BoxDepthCache] = {}


def peg_hole_mouth(env, env_idx: int, decl: SiteDeclaration) -> SiteSpec:
    """The hole's entrance plane, with the peg head as the source point.

    ``box_hole_pose`` sits on the box's centre plane -- its x offset is
    identically zero across every recorded episode -- so the mouth is that
    frame walked back along the hole axis by the box half-depth. Measuring to
    the hole frame instead would put the target inside the box and make the
    approach milestone fire only once the peg was already half inserted.
    """
    base = _unwrap(env)
    missing = [a for a in _PEG_ATTRS if not hasattr(base, a)]
    if missing:
        raise SiteError(
            f"site {decl.key!r} wants provider {decl.provider!r} but the "
            f"environment exposes no {missing!r}"
        )
    cache = _DEPTH_CACHES.setdefault(int(env_idx), _BoxDepthCache())
    depth = cache.depth(base, env_idx)

    hole = _pose7(base.box_hole_pose, env_idx)
    rot = _rot(hole)
    axis = rot @ np.array([1.0, 0.0, 0.0])
    mouth = np.concatenate([hole[:3] - depth * axis, hole[3:7]])

    return SiteSpec(
        declaration=decl,
        pose_world=mouth,
        # The opening's half-width, so ``reached`` means the head is within the
        # aperture at the mouth plane rather than merely near the box.
        tolerance=_scalar(base.box_hole_radii, env_idx),
        subject_point_world=_pose7(base.peg_head_pose, env_idx)[:3],
        axis_world=axis,
    )


def reset_depth_cache() -> None:
    """Drop cached box geometry. The cache self-invalidates on a changed
    signature; this exists for tests and for a torn-down environment."""
    _DEPTH_CACHES.clear()


def depth_provenance(env_idx: int = 0) -> Optional[str]:
    """Where the cached box depth came from, for diagnostics: the collision
    geometry or the checked ``peg_half_sizes`` fallback."""
    cache = _DEPTH_CACHES.get(int(env_idx))
    return None if cache is None else cache.provenance


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
_PROVIDERS = {
    PROVIDER_PICK_CUBE_GOAL: pick_cube_goal,
    PROVIDER_PEG_HOLE_MOUTH: peg_hole_mouth,
    PROVIDER_ROBOT_BASE_REGION: robot_base_region,
}


def site_specs(
    env, env_idx: int, declarations: Dict[str, SiteDeclaration],
) -> List[SiteSpec]:
    """One validated :class:`SiteSpec` per declaration, in stable key order.

    Raises rather than skipping. A declared site that cannot be read this frame
    would drop its ``reached`` edge, and a missing scored fact masks the whole
    frame's potential -- a much quieter failure than this one.
    """
    out: List[SiteSpec] = []
    for key in sorted(declarations):
        decl = declarations[key]
        provider = _PROVIDERS.get(decl.provider)
        if provider is None:
            raise SiteError(
                f"site {key!r} names provider {decl.provider!r}, which this "
                "build does not implement"
            )
        spec = provider(env, env_idx, decl)
        spec.validate(f"site {key!r}")
        out.append(spec)
    return out
