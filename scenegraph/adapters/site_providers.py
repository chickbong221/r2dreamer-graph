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
    PROVIDER_MSHAB_EE_REST,
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
    # Outside the shared frame cache, transfer only the requested tensor row.
    if hasattr(value, "detach") and value.ndim > 1:
        value = value[min(env_idx, value.shape[0] - 1)]
    arr = _np(value)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr[min(env_idx, arr.shape[0] - 1)] if arr.ndim > 1 else arr


def _scalar(value, env_idx: int) -> float:
    if hasattr(value, "detach"):
        value = value.reshape(-1)
        value = value[min(env_idx, value.numel() - 1)]
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
def robot_base_pose(env, env_idx: int) -> Optional[np.ndarray]:
    """The robot base link's world pose, or None if there is no robot.

    Exposed without a declaration because the calibration collector needs the
    region's centre before any asset exists to declare it.
    """
    base = _unwrap(env)
    robot = getattr(getattr(base, "agent", None), "robot", None)
    if robot is None:
        return None
    # A probe, so it answers "no base here" rather than raising: the
    # calibration collector asks every environment without knowing which of
    # them has a robot it can read. ``robot_base_region`` turns the same None
    # into a hard failure, because by then a schedule has asked for it.
    try:
        links = robot.get_links()
        return _pose7(links[0].pose, env_idx) if links else None
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def robot_base_region(env, env_idx: int, decl: SiteDeclaration) -> SiteSpec:
    """A disc around the robot base link, the radius PullCubeTool succeeds at.

    ``get_links()[0]`` is the base, matching the environment's own
    ``evaluate``. Reading the agent rather than a stored pose matters less here
    than elsewhere -- the base does not move -- but it keeps the site defined
    the same way the predicate is.
    """
    pose = robot_base_pose(env, env_idx)
    if pose is None:
        raise SiteError(
            f"site {decl.key!r} wants the robot base but the environment "
            "exposes no agent.robot with links"
        )
    return SiteSpec(
        declaration=decl, pose_world=pose, tolerance=PULL_SUCCESS_RADIUS,
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


def _shapes_of(obj):
    """Collision shapes for one SAPIEN entity, across the shapes of API it has.

    ``Actor._objs`` holds ``sapien.Entity`` objects, and an Entity owns no
    shapes directly -- they hang off its physx rigid-body component. Earlier
    SAPIEN versions and ManiSkill's own wrappers expose the list or a getter
    instead, so all three are tried before giving up.
    """
    shapes = getattr(obj, "collision_shapes", None)
    if shapes:
        return shapes
    getter = getattr(obj, "get_collision_shapes", None)
    if callable(getter):
        try:
            shapes = getter()
        except Exception:
            shapes = None
        if shapes:
            return shapes
    for component in getattr(obj, "components", None) or []:
        shapes = getattr(component, "collision_shapes", None)
        if shapes:
            return shapes
        getter = getattr(component, "get_collision_shapes", None)
        if callable(getter):
            try:
                shapes = getter()
            except Exception:
                shapes = None
            if shapes:
                return shapes
    return None


def _shape_half_size(shape) -> Optional[np.ndarray]:
    """Half-extents of one collision shape's own bounding box, in its frame.

    Only a box carries ``half_size``. A sphere reports a radius, a capsule a
    radius plus a half-length along x, and a convex mesh its vertices -- so a
    reader that only understood boxes silently returned nothing for
    PlaceSphere's sphere and for anything mesh-backed.
    """
    geometry = getattr(shape, "geometry", None)

    for holder in (shape, geometry):
        if holder is None:
            continue
        half = getattr(holder, "half_size", None)
        if half is not None:
            return _np(half).reshape(-1)[:3]

    for holder in (shape, geometry):
        if holder is None:
            continue
        radius = getattr(holder, "radius", None)
        if radius is None:
            continue
        radius = float(_np(radius).reshape(-1)[0])
        # A capsule's axis is x in SAPIEN; a sphere has no half-length.
        half_length = getattr(holder, "half_length", None)
        along = radius if half_length is None else (
            radius + float(_np(half_length).reshape(-1)[0]))
        return np.array([along, radius, radius], dtype=float)

    for holder in (shape, geometry):
        if holder is None:
            continue
        vertices = getattr(holder, "vertices", None)
        if vertices is None:
            continue
        verts = _np(vertices).reshape(-1, 3)
        if not verts.size:
            continue
        scale = getattr(holder, "scale", None)
        if scale is not None:
            verts = verts * _np(scale).reshape(-1)[:3]
        return (verts.max(axis=0) - verts.min(axis=0)) / 2.0
    return None


def collision_half_extents_status(
    actor, env_idx: int,
) -> Tuple[Optional[List[float]], str]:
    """``(half-extents, status)``: the measurement and why it failed if it did.

    The one measurement that separates an extended support plane from a
    localized receptacle. Roles cannot: in PlaceSphere the bin and the table
    carry byte-identical ``roles`` and ``interaction_types``, because both are
    kinematic and both support the sphere. Their sizes differ by an order of
    magnitude.

    Shape poses are included, so a surface built from several offset blocks
    reports the extent of the whole thing rather than of one block.

    The status exists for the collector. A member with no extent cannot be
    classified, and the difference between "this entity exposes no collision
    body" and "one of its shapes is a type we cannot measure" is what tells a
    pilot whether the gap is the scene or this reader.
    """
    objs = getattr(actor, "_objs", None) or []
    if not objs:
        return None, "no-sub-scene-objects"
    shapes = _shapes_of(objs[min(env_idx, len(objs) - 1)])
    if not shapes:
        return None, "no-collision-shapes"
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for shape in shapes:
        half = _shape_half_size(shape)
        if half is None:
            return None, f"unmeasurable-shape:{type(shape).__name__}"
        centre = np.zeros(3)
        pose = getattr(shape, "local_pose", None)
        if pose is not None:
            try:
                centre = _np(pose.p).reshape(-1)[:3]
            except Exception:
                centre = np.zeros(3)
        lo = np.minimum(lo, centre - half)
        hi = np.maximum(hi, centre + half)
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        return None, "non-finite-bounds"
    return [float(v) for v in (hi - lo) / 2.0], "ok"


def collision_half_extents(actor, env_idx: int) -> Optional[List[float]]:
    """Half-extents of the actor's collision shapes, or None when unreadable.

    None rather than a default: the miner treats it as "cannot classify"
    rather than "small", because a table quietly demoted to an ordinary object
    reinstates the metre of origin error the classification exists to remove.
    """
    return collision_half_extents_status(actor, env_idx)[0]


def _collision_x_half_extent(actor, env_idx: int) -> Optional[float]:
    """Half-extent along x shared by the box's four blocks, or None.

    None rather than a raise: SAPIEN exposes collision shapes differently
    across versions and the checked fallback covers it. Disagreement between
    the blocks *is* an error -- that would mean this is not the box we think.
    """
    objs = getattr(actor, "_objs", None) or []
    if not objs:
        return None
    shapes = _shapes_of(objs[min(env_idx, len(objs) - 1)])
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


def peg_mouth_geometry(env, env_idx: int) -> Optional[Dict[str, Any]]:
    """``mouth`` pose, entry ``axis``, ``aperture`` and live ``head`` point.

    ``box_hole_pose`` sits on the box's centre plane -- its x offset is
    identically zero across every recorded episode -- so the mouth is that
    frame walked back along the hole axis by the box half-depth. Measuring to
    the hole frame instead would put the target inside the box and make the
    approach milestone fire only once the peg was already half inserted.

    Returns None when the environment is not this task, so the calibration
    collector can ask every environment without knowing which it has.
    """
    base = _unwrap(env)
    if any(not hasattr(base, a) for a in _PEG_ATTRS):
        return None
    cache = _DEPTH_CACHES.setdefault(int(env_idx), _BoxDepthCache())
    depth = cache.depth(base, env_idx)

    hole = _pose7(base.box_hole_pose, env_idx)
    axis = _rot(hole) @ np.array([1.0, 0.0, 0.0])
    return {
        "mouth": np.concatenate([hole[:3] - depth * axis, hole[3:7]]),
        "axis": axis,
        # The opening's half-width, so ``reached`` means the head is within the
        # aperture at the mouth plane rather than merely near the box.
        "aperture": _scalar(base.box_hole_radii, env_idx),
        "head": _pose7(base.peg_head_pose, env_idx)[:3],
        "depth": depth,
    }


def peg_hole_mouth(env, env_idx: int, decl: SiteDeclaration) -> SiteSpec:
    """The hole's entrance plane, with the peg head as the source point."""
    geometry = peg_mouth_geometry(env, env_idx)
    if geometry is None:
        missing = [a for a in _PEG_ATTRS if not hasattr(_unwrap(env), a)]
        raise SiteError(
            f"site {decl.key!r} wants provider {decl.provider!r} but the "
            f"environment exposes no {missing!r}"
        )
    return SiteSpec(
        declaration=decl,
        pose_world=geometry["mouth"],
        tolerance=geometry["aperture"],
        subject_point_world=geometry["head"],
        axis_world=geometry["axis"],
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
# MS-HAB: the end-effector rest position
# --------------------------------------------------------------------------- #
# Where the two numbers come from. Named here rather than inlined so a rename
# upstream fails with the attribute in the message.
_EE_REST_OFFSET_ATTR = "ee_rest_pos_wrt_base"
_PICK_CFG_ATTR = "pick_cfg"
_EE_REST_THRESH_ATTR = "ee_rest_thresh"


def _ee_rest_snapshot(base):
    """Read the three batched inputs once; the owning graph frame scopes this."""
    agent = getattr(base, "agent", None)
    link = getattr(agent, "base_link", None)
    if link is None:
        raise SiteError("the MS-HAB rest site needs agent.base_link")
    offset = getattr(base, _EE_REST_OFFSET_ATTR, None)
    if offset is None:
        raise SiteError(f"the environment exposes no {_EE_REST_OFFSET_ATTR!r}")
    cfg = getattr(base, _PICK_CFG_ATTR, None)
    tolerance = getattr(cfg, _EE_REST_THRESH_ATTR, None)
    if tolerance is None:
        raise SiteError(f"the environment exposes no pick_cfg.{_EE_REST_THRESH_ATTR}")
    pose = link.pose
    raw = getattr(pose, "raw_pose", None)
    if raw is None:
        raw = np.concatenate([_np(pose.p), _np(pose.q)], axis=-1)
    else:
        raw = _np(raw)
    local = getattr(offset, "p", None)
    if local is None:
        local = getattr(offset, "raw_pose", offset)
    return raw, _np(local)[..., :3], _np(tolerance)


def ee_rest_geometry(env, env_idx: int) -> Tuple[np.ndarray, float]:
    """``(rest position in world, tolerance)`` for MS-HAB's EE rest pose.

    Declaration-free on purpose. The calibration collector needs these numbers
    before any asset declares the site, and making it fabricate a
    ``SiteDeclaration`` to read a position would mean two code paths deciding
    where the rest point is -- which is exactly how a mined scale comes to
    describe a different place than the runtime measures.

    Composed live from ``base_link.pose`` every time it is asked. MS-HAB
    re-places the robot on reset, so the rest position moves with it, and a
    pose held across an episode boundary names the previous episode's spot.
    The tolerance is read from the task config rather than written here: 0.05
    is its current value, not its definition.
    """
    from .privileged_state import frame_cached

    base = _unwrap(env)
    poses, offsets, tolerances = frame_cached(
        "ee_rest", id(base), lambda: _ee_rest_snapshot(base))
    pose = _row(poses, env_idx).reshape(-1)[:7]
    local = _row(offsets, env_idx).reshape(-1)[:3]
    if local.size < 3:
        raise SiteError(
            f"{_EE_REST_OFFSET_ATTR} has {local.size} component(s) for "
            f"sub-scene {env_idx}, not 3"
        )
    world = pose[:3] + _rot(pose) @ local
    tolerance = _scalar(tolerances, env_idx)
    if not np.all(np.isfinite(world)) or not np.isfinite(tolerance):
        raise SiteError(
            "the MS-HAB rest position is not finite: "
            f"base_pose={pose.tolist()} offset={local.tolist()} "
            f"tolerance={tolerance}"
        )
    return world, float(tolerance)


def mshab_ee_rest(env, env_idx: int, decl: SiteDeclaration) -> SiteSpec:
    """The rest position the gripper has to return to, still holding the
    object. Its subject is the end effector, not a manipuland."""
    world, tolerance = ee_rest_geometry(env, env_idx)
    # Position only. The site is a point in space with no orientation of its
    # own, and giving it the base's would imply an axis nothing measures.
    return SiteSpec(
        declaration=decl,
        pose_world=np.concatenate([world, [1.0, 0.0, 0.0, 0.0]]),
        tolerance=tolerance,
    )


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
_PROVIDERS = {
    PROVIDER_PICK_CUBE_GOAL: pick_cube_goal,
    PROVIDER_PEG_HOLE_MOUTH: peg_hole_mouth,
    PROVIDER_ROBOT_BASE_REGION: robot_base_region,
    PROVIDER_MSHAB_EE_REST: mshab_ee_rest,
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
