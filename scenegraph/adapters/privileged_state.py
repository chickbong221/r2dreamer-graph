"""``get_privileged_state`` -- the adapter the simulator does NOT give you.

There is no built-in ``env.get_privileged_state()`` in ManiSkill 3 or MS-HAB.
This module implements that name by gathering the primitives that *do* exist:

  * ``env.unwrapped.agent``                    (Fetch / Panda)
  * ``env.unwrapped.scene``                    -> get_pairwise_contact_forces(...)
  * ``env.unwrapped.segmentation_id_map``      seg-id int -> Actor / Link
  * ``agent.tcp`` / ``agent.tcp_pose``         end-effector pose
  * ``agent.finger1_link`` / ``finger2_link``  contact + grasp
  * ``agent.is_grasping(obj, max_angle=30)``   grasp predicate (MS-HAB convention)

For MS-HAB it also exposes the task internals used to decide which object must
persist as the active manipulation target:

  * ``env.unwrapped.subtask_objs``             (entries may be None)
  * ``env.unwrapped.subtask_goals``            (entries may be None)
  * ``env.unwrapped.subtask_articulations``    (entries may be None)
  * ``env.unwrapped.task_plan``                list of *Subtask dataclasses
  * ``env.unwrapped.subtask_pointer``          per-env current subtask index

Everything is torch-batched with a leading env dimension; helpers here index a
single ``env_idx`` and return plain python / numpy where convenient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Small typed snapshot returned to the rest of the pipeline.
# --------------------------------------------------------------------------- #
@dataclass
class PrivilegedState:
    env: Any                       # unwrapped env (kept for force queries)
    agent: Any
    scene: Any
    env_idx: int

    is_mshab: bool

    # End-effector.
    ee_links: List[Any] = field(default_factory=list)   # Link objs to merge into ee
    tcp_pose_world: Optional[np.ndarray] = None          # [7] xyz + wxyz

    # Gripper width in qpos-sum convention: qpos[-2] + qpos[-1]. Matches the
    # miner (tools/build_affordances.py). Correct for Fetch and Panda alike;
    # None only when the agent has no robot or qpos is unavailable.
    gripper_width: Optional[float] = None

    # Segmentation id -> Actor/Link (ManiSkill primitive).
    seg_id_map: Dict[int, Any] = field(default_factory=dict)

    # Robot link set (for "is this a robot link?" tests).
    robot_links: set = field(default_factory=set)
    # Scene-cached name sets mirroring ee_links / robot_links, so the node
    # builder's per-entity name-fallback checks stop rebuilding sets per call.
    ee_link_names: Optional[frozenset] = None
    robot_link_names: Optional[frozenset] = None

    # MS-HAB active manipulation handles for this env_idx (any may be None).
    active_obj: Optional[Any] = None
    # Pre-resolution merged MS-HAB handles (span all envs). Physics queries
    # can use these when the resolved per-env wrapper is not row-aligned.
    active_obj_merged: Optional[Any] = None
    active_handle_link_merged: Optional[Any] = None
    active_articulation: Optional[Any] = None
    active_handle_link: Optional[Any] = None
    active_subtask_type: Optional[str] = None
    # Original (pre-merge) obj_id for the current subtask -- e.g. "024_bowl-3".
    # MS-HAB rewrites task_plan[ptr].obj_id to "obj_<num>" during _merge_*; the
    # original lives in env.build_config_idx_to_task_plans[bci][tpi].subtasks[ptr].
    active_obj_id: Optional[str] = None

    # ----- queries the relation rules call ------------------------------- #
    def pairwise_force_vector(self, a: Any, b: Any) -> np.ndarray:
        """World-frame contact-force vector between two entities for this env.

        ManiSkill builds contact queries from ``zip(a._bodies, b._bodies)``.
        MS-HAB can mix full-span robot links with per-env actors/links, so row
        k is not necessarily env k. Slice to single-env views when needed.
        """
        if a is None or b is None:
            return np.zeros(3, dtype=float)
        ra = _obj_index_for_env(a, self.env_idx)
        rb = _obj_index_for_env(b, self.env_idx)
        if ra is None or rb is None:
            return np.zeros(3, dtype=float)
        if ra != rb:
            # Align the wider view to the narrower scene-idx subset when
            # possible. One multi-row query can then serve every env sharing
            # that pair, instead of creating one single-row query per env.
            sa = _scene_idxs_list(a)
            sb = _scene_idxs_list(b)
            a2 = b2 = None
            if sa is not None and sb is not None:
                if set(sb).issubset(sa):
                    a2, b2, row = _slice_view_rows(a, sb), b, rb
                elif set(sa).issubset(sb):
                    a2, b2, row = a, _slice_view_rows(b, sa), ra
            if a2 is None or b2 is None:
                a2 = _slice_view_for_env(a, self.env_idx)
                b2 = _slice_view_for_env(b, self.env_idx)
                row = 0
            a, b = a2, b2
            if a is None or b is None:
                return np.zeros(3, dtype=float)
        else:
            row = ra

        if _FRAME_CACHE is not None:
            fkey = (id(a), id(b))
            forces = _FRAME_CACHE["forces"].get(fkey)
            if forces is None:
                forces = _to_np(self.scene.get_pairwise_contact_forces(a, b))
                _FRAME_CACHE["forces"][fkey] = forces
        else:
            forces = _to_np(self.scene.get_pairwise_contact_forces(a, b))
        if forces.ndim == 1:
            return forces.astype(float) if row == 0 else np.zeros(3, dtype=float)
        if row >= forces.shape[0]:
            return np.zeros(3, dtype=float)
        return np.asarray(forces[row], dtype=float)

    def pairwise_force(self, a: Any, b: Any) -> float:
        """Scalar contact-force magnitude between two entities for this env."""
        return float(np.linalg.norm(self.pairwise_force_vector(a, b)))

    def ee_object_contact_force(self, obj: Any) -> float:
        """Sum of both finger contact forces against ``obj`` (MS-HAB style)."""
        if obj is None:
            return 0.0
        f1 = self.pairwise_force(self.agent.finger1_link, obj)
        f2 = self.pairwise_force(self.agent.finger2_link, obj)
        return f1 + f2

    def _finger_open_dir(self, finger_link, sign: float) -> Optional[np.ndarray]:
        """World-frame Fetch gripper-opening direction for this env."""
        arr = entity_pose_world_array(finger_link, self.env_idx)
        if arr is None:
            return None
        w, x, y, z = arr[3], arr[4], arr[5], arr[6]
        ydir = np.array(
            [
                2.0 * (x * y - w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z + w * x),
            ],
            dtype=float,
        )
        return sign * ydir

    def _is_grasping_manual(
        self, obj: Any, max_angle: int, min_force: float = 0.5
    ) -> bool:
        """Per-env reimplementation of Fetch.is_grasping for sliced objects."""
        f1 = getattr(self.agent, "finger1_link", None)
        f2 = getattr(self.agent, "finger2_link", None)
        if f1 is None or f2 is None:
            return False
        lf = self.pairwise_force_vector(f1, obj)
        rf = self.pairwise_force_vector(f2, obj)
        lmag = float(np.linalg.norm(lf))
        rmag = float(np.linalg.norm(rf))
        if lmag < min_force or rmag < min_force:
            return False
        ld = self._finger_open_dir(f1, -1.0)
        rd = self._finger_open_dir(f2, +1.0)
        if ld is None or rd is None:
            return False

        def _angle_deg(d: np.ndarray, f: np.ndarray, fmag: float) -> float:
            denom = float(np.linalg.norm(d)) * fmag + 1e-12
            c = float(np.dot(d, f)) / denom
            return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

        return (
            _angle_deg(ld, lf, lmag) <= max_angle
            and _angle_deg(rd, rf, rmag) <= max_angle
        )

    def is_grasping(self, obj: Any, max_angle: int = 30) -> bool:
        """Env-consistent grasp predicate for full-span and per-env objects."""
        if obj is None:
            return False
        if _obj_index_for_env(obj, self.env_idx) is None:
            return False

        num_envs = getattr(self.scene, "num_envs", None)
        objs = getattr(obj, "_objs", None)
        n_rows = len(objs) if objs is not None else 1

        query = None
        if num_envs is not None and n_rows == num_envs:
            query = obj
        elif self.active_obj_merged is not None:
            try:
                same = _entity_for_env(obj, self.env_idx) is _entity_for_env(
                    self.active_obj_merged, self.env_idx
                )
            except Exception:
                same = False
            if same:
                query = self.active_obj_merged

        if query is None or not hasattr(self.agent, "is_grasping"):
            return self._is_grasping_manual(obj, max_angle)

        def _run_grasp_query():
            try:
                gg = self.agent.is_grasping(query, max_angle=max_angle)
            except TypeError:
                gg = self.agent.is_grasping(query)
            return _to_np(gg).reshape(-1)

        if _FRAME_CACHE is not None:
            gkey = (id(query), int(max_angle))
            g = _FRAME_CACHE["grasp"].get(gkey)
            if g is None:
                g = _run_grasp_query()
                _FRAME_CACHE["grasp"][gkey] = g
        else:
            g = _run_grasp_query()
        if self.env_idx >= g.shape[0]:
            return False
        return bool(g[self.env_idx])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


_FRAME_CACHE: Optional[Dict[str, Any]] = None

# Sentinel used to detect "attribute missing" from a single getattr call, so
# that hot-path lookups do not have to call the property twice via hasattr()
# + getattr(). Fetch's ``tcp_pose`` is a computed property; probing it with
# hasattr triggers the entire ManiSkill Pose-wrapper chain once for the
# existence test and again for the actual read.
_MISSING = object()

_SCENE_CACHE_KEYS = (
    "_teemo_sidxs_cache",
    "_teemo_ee_links",
    "_teemo_ee_link_names",
    "_teemo_robot_links",
    "_teemo_robot_link_names",
    "_teemo_per_env_seg_maps",
    "_teemo_sliced_views",
    "_teemo_row_sliced_views",
    "_teemo_resolve_cache",
)


def begin_frame_cache(scene: Any = None) -> None:
    """Start one-frame memoization for the graph env loop.

    When ``scene`` is provided and supports the GPU rigid-body buffer, the
    whole ``cuda_rigid_body_data`` tensor is pulled to CPU once here so every
    downstream pose read can index into a plain numpy array instead of firing
    ManiSkill's per-call Pose + tensor-wrapper chain (the top allocator on
    the graph hot path). Falls back silently when: no scene, no GPU sim,
    parallel-in-single-scene mode (needs per-entity scene-offset subtraction
    that can't be replicated from the raw buffer alone), or any exception --
    downstream ``entity.pose`` fallback then behaves as before.
    """
    global _FRAME_CACHE
    _FRAME_CACHE = {"np": {}, "pose": {}, "grasp": {}, "forces": {}}
    if scene is None:
        return
    if getattr(scene, "parallel_in_single_scene", False):
        return
    px = getattr(scene, "px", None)
    if px is None:
        return
    getter = getattr(px, "cuda_rigid_body_data", None)
    if getter is None:
        return
    try:
        buf_np = getter.torch().detach().cpu().numpy()
    except Exception:
        return
    _FRAME_CACHE["rigid_body_data"] = buf_np


def end_frame_cache() -> None:
    """Clear one-frame memoization."""
    global _FRAME_CACHE
    _FRAME_CACHE = None


def _cache_dicts(env_or_scene) -> List[dict]:
    """The ``__dict__``s carrying the scene-attached caches."""
    candidates = []
    base = getattr(env_or_scene, "unwrapped", env_or_scene)
    scene = getattr(base, "scene", None)
    if scene is None and any(k in getattr(base, "__dict__", {}) for k in _SCENE_CACHE_KEYS):
        scene = base
    if scene is not None:
        candidates.append(scene)

    agent = getattr(base, "agent", None)
    robot = getattr(agent, "robot", None) if agent is not None else None
    robot_scene = getattr(robot, "scene", None) if robot is not None else None
    if robot_scene is not None and robot_scene not in candidates:
        candidates.append(robot_scene)

    return [
        d for d in (getattr(c, "__dict__", None) for c in candidates)
        if d is not None
    ]


def clear_privileged_state_caches(env_or_scene) -> None:
    """Drop scene-attached TEEMO caches after a simulator reconfiguration."""
    for d in _cache_dicts(env_or_scene):
        for key in _SCENE_CACHE_KEYS:
            d.pop(key, None)


def clear_resolve_cache(env_or_scene, env_idx: int) -> None:
    """Drop one env's merged-handle resolutions at an episode boundary.

    ``_resolve_actual_entity`` keys on the MS-HAB handle name (``obj_0``) and
    guards the hit by identity on that same handle. Both outlive the episode:
    the handle is a task-level actor, and a reset re-randomizes the task plan
    without reconfiguring the scene. So without this the next episode resolves
    ``obj_0`` to the previous episode's object and flags a vertex that is no
    longer there.
    """
    for d in _cache_dicts(env_or_scene):
        cache = d.get("_teemo_resolve_cache")
        if not cache:
            continue
        for key in [
            k for k in list(cache)
            if isinstance(k, tuple) and k and k[-1] == env_idx
        ]:
            cache.pop(key, None)


def scene_cache_size(env_or_scene) -> int:
    """Entries in the largest scene-attached cache."""
    return max(
        (
            len(d[key])
            for d in _cache_dicts(env_or_scene)
            for key in _SCENE_CACHE_KEYS
            if key in d
        ),
        default=0,
    )


def purge_scene_caches(env_or_scene, cap: int) -> int:
    """Drop every scene cache once the largest passes ``cap``; return its size.

    These key on ``id(entity)`` and store the entity beside the value, so each
    stale key pins a dead actor alive. MS-HAB recreates merged actors on
    partial reset, and those views never enter ``scene.actors`` -- which is all
    the reconfiguration signature watches -- so nothing else ever evicts them.
    Every one is memoization behind an identity check on read, so dropping an
    entry costs one recompute and nothing else.
    """
    size = scene_cache_size(env_or_scene)
    if size > cap:
        clear_privileged_state_caches(env_or_scene)
    return size


def _frame_np(key, fn):
    if _FRAME_CACHE is None:
        return fn()
    bucket = _FRAME_CACHE["np"]
    if key not in bucket:
        bucket[key] = fn()
    return bucket[key]


def _scene_idxs_list(entity, pin: bool = False) -> Optional[List[int]]:
    try:
        s = entity._scene_idxs
    except AttributeError:
        return None
    scene = getattr(entity, "scene", None)
    durable = (
        scene.__dict__.setdefault("_teemo_sidxs_cache", {})
        if scene is not None else None
    )
    if durable is not None:
        hit = durable.get(id(entity))
        if hit is not None and hit[0] is entity:
            return hit[1]

    frame = _FRAME_CACHE["np"] if _FRAME_CACHE is not None else None
    fkey = ("sidxs", id(entity))
    if frame is not None and fkey in frame:
        return frame[fkey]

    out = s.tolist() if hasattr(s, "tolist") else list(s)
    if pin and durable is not None:
        durable[id(entity)] = (entity, out)
    elif frame is not None:
        frame[fkey] = out
    return out


def _obj_index_for_env(entity, env_idx: int) -> Optional[int]:
    """Row of ``entity._objs`` that lives in parallel env ``env_idx``."""
    if entity is None:
        return None
    s = _scene_idxs_list(entity)
    if s is None:
        objs = getattr(entity, "_objs", None)
        if objs is None:
            return 0
        return env_idx if env_idx < len(objs) else None
    try:
        return s.index(env_idx)
    except ValueError:
        return None


def get_tcp_pose(agent) -> Any:
    """Fetch defines ``tcp_pose`` (computed); Panda exposes ``tcp.pose``."""
    pose = getattr(agent, "tcp_pose", _MISSING)
    if pose is not _MISSING:
        return pose
    return agent.tcp.pose


def _tcp_pose_pq(agent):
    """Fast path for TCP pose: index ``agent.tcp`` in the raw rigid-body buffer
    when both are available; otherwise fall back to ``get_tcp_pose(agent)``.

    Fetch's ``agent.tcp_pose`` is a computed property that reads TCP through
    the same Link.pose chain the entity fast path bypasses; hitting it once
    per frame is small compared to the ``entity.pose`` fanout, but still
    worth eliminating since we already have the buffer cached."""
    buf = _FRAME_CACHE.get("rigid_body_data") if _FRAME_CACHE is not None else None
    if buf is not None:
        tcp_link = getattr(agent, "tcp", None)
        if tcp_link is not None:
            hit = _pose_pq_from_buffer(tcp_link, buf)
            if hit is not None:
                return hit
    pose = get_tcp_pose(agent)
    return _to_np(pose.p), _to_np(pose.q)


def _tcp_pose_world_cached(agent, env_idx: int) -> np.ndarray:
    pq = _frame_np("tcp_pq", lambda: _tcp_pose_pq(agent))
    p, q = pq
    if p.ndim == 2:
        p = p[env_idx]
    if q.ndim == 2:
        q = q[env_idx]
    return np.concatenate([p, q]).astype(float)


def compute_gripper_width(agent, env_idx: int) -> Optional[float]:
    """Width = qpos[-2] + qpos[-1] (matches MS-HAB miner / collect_data.py).

    Fetch qpos is 15-D ([base 3 | head 2 | torso 1 | arm 7 | gripper 2]) and
    Panda 9-D ([arm 7 | gripper 2]); both end in the two finger prismatic
    joints, so qpos[-2:] is the right slice for either. We deliberately do NOT
    use ``||finger1.pose.p - finger2.pose.p||`` here -- the URDF joint origins
    add a constant +0.03085 m to that distance, which would systematically bias
    every ``gripper-width-alignment`` reading relative to the mined preferred
    widths.
    """
    if agent is None or not hasattr(agent, "robot"):
        return None
    try:
        qpos = _frame_np("qpos", lambda: _to_np(agent.robot.qpos))
    except Exception:
        return None
    if qpos.ndim == 0:
        return None
    if qpos.ndim == 1:
        row = qpos
    elif qpos.ndim == 2:
        if env_idx < 0 or env_idx >= qpos.shape[0]:
            return None
        row = qpos[env_idx]
    else:
        return None
    if row.shape[0] < 2:
        return None
    w = float(row[-2] + row[-1])
    if not np.isfinite(w):
        return None
    return w


def pose_to_world_array(pose, env_idx: int) -> np.ndarray:
    """SAPIEN Pose -> [x, y, z, qw, qx, qy, qz] for one env."""
    p = _to_np(pose.p)
    q = _to_np(pose.q)
    if p.ndim == 2:
        p = p[env_idx]
    if q.ndim == 2:
        q = q[env_idx]
    return np.concatenate([p, q]).astype(float)


def _scene_scoped(agent, key: str, fn):
    robot = getattr(agent, "robot", None)
    scene = getattr(robot, "scene", None) if robot is not None else None
    if scene is None:
        return fn()
    hit = scene.__dict__.get(key)
    if hit is not None:
        return hit
    out = fn()
    scene.__dict__[key] = out
    return out


def get_ee_links(agent) -> List[Any]:
    """Links to fold into the single ``ee`` node.

    Fetch / Panda both cache ``tcp``, ``finger1_link``, ``finger2_link`` in
    ``_after_init``. We start with those; the run scripts can print the seg map
    and extend this set (e.g. wrist/hand links) once empirically observed.
    """
    def _build():
        links = []
        for attr in ("tcp", "finger1_link", "finger2_link"):
            link = getattr(agent, attr, None)
            if link is not None:
                links.append(link)
        return links

    return _scene_scoped(agent, "_teemo_ee_links", _build)


def _robot_link_set(agent) -> set:
    def _build():
        try:
            return set(agent.robot.get_links())
        except Exception:
            return set()

    return _scene_scoped(agent, "_teemo_robot_links", _build)


def get_ee_link_names(agent) -> frozenset:
    def _build():
        return frozenset(
            getattr(l, "name", None) for l in get_ee_links(agent)
        )

    return _scene_scoped(agent, "_teemo_ee_link_names", _build)


def _robot_link_names(agent) -> frozenset:
    def _build():
        return frozenset(
            getattr(l, "name", None) for l in _robot_link_set(agent)
        )

    return _scene_scoped(agent, "_teemo_robot_link_names", _build)


def _looks_like_mshab(env) -> bool:
    return all(
        hasattr(env, a)
        for a in ("subtask_objs", "task_plan", "subtask_pointer")
    )


def looks_like_mshab(env_or_wrapped) -> bool:
    """Public form, tolerant of wrappers.

    Callers use it to pick an identity convention: MS-HAB resolves a merged
    handle to the per-env actual, ordinary ManiSkill needs the reverse.
    """
    env = getattr(env_or_wrapped, "unwrapped", env_or_wrapped)
    return _looks_like_mshab(env)


def _subtask_type(subtask) -> Optional[str]:
    # *Subtask dataclasses expose ``.type`` ("pick"/"place"/"open"/"close"/...).
    return getattr(subtask, "type", None)


def _entity_for_env(entity, env_idx: int):
    """Return the underlying simulator entity represented at ``env_idx``."""
    if entity is None:
        return None
    objs = getattr(entity, "_objs", None)
    if objs is None:
        return entity

    try:
        scene_idxs = _scene_idxs_list(entity)
        if scene_idxs is not None:
            return objs[scene_idxs.index(env_idx)]
    except (ValueError, IndexError, TypeError):
        pass

    try:
        if len(objs) == 1:
            return objs[0]
        return objs[env_idx]
    except (IndexError, TypeError):
        return entity


# --------------------------------------------------------------------------- #
# Merged-view aliasing (normal ManiSkill)
# --------------------------------------------------------------------------- #
_ALIAS_FLAG = "_teemo_alias_merged_views"
_ALIAS_MAP = "_teemo_merged_view_alias"


def _scene_of(env_or_scene):
    if hasattr(env_or_scene, "unwrapped"):
        return env_or_scene.unwrapped.scene
    return getattr(env_or_scene, "scene", env_or_scene)


def set_merged_view_aliasing(env_or_scene, enabled: bool = True) -> None:
    """Opt one scene into resolving per-sub-scene actors to their merged view.

    Off by default and not auto-detected: MS-HAB resolves the other way
    (merged handle -> per-env actual, see ``_resolve_actual_entity``), so a
    global default would rewrite MS-HAB identities.
    """
    scene = _scene_of(env_or_scene)
    scene.__dict__[_ALIAS_FLAG] = bool(enabled)
    # Both derived caches are keyed on the old answer.
    scene.__dict__.pop(_ALIAS_MAP, None)
    scene.__dict__.pop("_teemo_per_env_seg_maps", None)


def merged_view_aliasing_enabled(env_or_scene) -> bool:
    return bool(_scene_of(env_or_scene).__dict__.get(_ALIAS_FLAG, False))


# Everything derived from the actor set. All of it is invalid once a scene
# reconfigures, which some tasks do on every reset at num_envs=1.
_SCENE_CACHES = (
    _ALIAS_MAP, "_teemo_per_env_seg_maps", "_teemo_sidxs_cache",
    "_teemo_sliced_views", "_teemo_row_sliced_views", "_teemo_resolve_cache",
)


def invalidate_scene_caches(env_or_scene) -> None:
    """Drop caches that name actors. The aliasing flag is a setting, not a
    cache, so it survives."""
    d = _scene_of(env_or_scene).__dict__
    for key in _SCENE_CACHES:
        d.pop(key, None)


def merged_view_alias_map(scene) -> Dict[int, Any]:
    """``id(underlying body) -> merged Actor view``, for one scene.

    ``Actor.merge`` registers in ``scene.actor_views``, but
    ``segmentation_id_map`` iterates ``scene.actors``, so PegInsertionSide's
    per-env ``peg_0 ... peg_{n-1}`` reach identity resolution instead of the
    logical ``peg``. ``canonical_actor_key`` strips ``-<i>`` but not ``_<i>``,
    so num_envs=1 mining records ``actor:peg_0`` and a 126-env run emits keys
    the entity vocab rejects.

    Widening that regex is not the fix -- ``_\\d+$`` would also collapse
    ``frl_apartment_table_01`` and ``_02``. Aliasing to the merged view touches
    no MS-HAB key. Links are not aliased; merge is actor-only.
    """
    cached = scene.__dict__.get(_ALIAS_MAP)
    if cached is not None:
        return cached

    alias: Dict[int, Any] = {}
    owner: Dict[int, Any] = {}
    for view in getattr(scene, "actor_views", {}).values():
        for obj in getattr(view, "_objs", None) or ():
            prior = owner.get(id(obj))
            if prior is not None and prior is not view:
                raise ValueError(
                    f"merged-view aliasing is ambiguous: body {id(obj)} belongs "
                    f"to both {getattr(prior, 'name', '?')!r} and "
                    f"{getattr(view, 'name', '?')!r}"
                )
            owner[id(obj)] = view
            alias[id(obj)] = view

    scene.__dict__[_ALIAS_MAP] = alias
    return alias


def per_env_segmentation_id_map(
    env, env_idx: int, *, alias_merged_views: Optional[bool] = None,
) -> Dict[int, Any]:
    """Segmentation-id -> Actor/Link map valid for one parallel env.

    ManiSkill's global ``env.segmentation_id_map`` keys wrappers by
    ``_objs[0].per_scene_id``. MS-HAB can load heterogeneous actors and
    articulations across vector envs, so the same integer id can refer to
    different entities in different sub-scenes.

    ``alias_merged_views`` overrides the scene flag set by
    ``set_merged_view_aliasing``; it exists for tests, which need both answers
    from one scene.
    """
    scene = env.unwrapped.scene if hasattr(env, "unwrapped") else env.scene
    alias_on = (
        merged_view_aliasing_enabled(scene)
        if alias_merged_views is None
        else bool(alias_merged_views)
    )
    cache = scene.__dict__.setdefault("_teemo_per_env_seg_maps", {})
    # Keyed by the flag too: a map built before aliasing was switched on
    # describes different identities and must not be served afterwards.
    cache_key = (env_idx, alias_on)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    alias = merged_view_alias_map(scene) if alias_on else {}
    seen_views: Dict[int, Any] = {}

    res: Dict[int, Any] = {}
    for actor in scene.actors.values():
        if getattr(actor, "merged", False):
            continue
        _scene_idxs_list(actor, pin=True)
        row = _obj_index_for_env(actor, env_idx)
        objs = getattr(actor, "_objs", None)
        if row is None or not objs or row >= len(objs):
            continue
        resolved = alias.get(id(objs[row]), actor)
        if resolved is not actor:
            # Two same-env actors behind one view would share a node. No task
            # in scope does this, so raise rather than collapse them.
            prior = seen_views.get(id(resolved))
            if prior is not None:
                raise ValueError(
                    f"merged view {getattr(resolved, 'name', '?')!r} covers "
                    f"both {getattr(prior, 'name', '?')!r} and "
                    f"{getattr(actor, 'name', '?')!r} in env {env_idx}"
                )
            seen_views[id(resolved)] = actor
        res[int(objs[row].per_scene_id)] = resolved

    for art in scene.articulations.values():
        if getattr(art, "merged", False):
            continue
        _scene_idxs_list(art, pin=True)
        if _obj_index_for_env(art, env_idx) is None:
            continue
        for link in art.links:
            _scene_idxs_list(link, pin=True)
            row = _obj_index_for_env(link, env_idx)
            objs = getattr(link, "_objs", None)
            if row is None or not objs or row >= len(objs):
                continue
            res[int(objs[row].entity.per_scene_id)] = link

    cache[cache_key] = res
    return res


def _pose_pq_from_buffer(entity, buf: np.ndarray):
    """Fast path: read ``(p, q)`` for every env row of ``entity`` directly out
    of the pre-cached ``cuda_rigid_body_data`` numpy buffer.

    Returns ``None`` -- caller must fall back to ``.pose`` -- when the entity
    is not represented in that buffer: static / hidden actors, entities that
    do not carry a ``_body_data_index`` (e.g. some Articulation views), out-
    of-range indices, or malformed buffers.
    """
    if getattr(entity, "px_body_type", None) == "static":
        return None
    if getattr(entity, "hidden", False):
        return None
    idx = getattr(entity, "_body_data_index", None)
    if idx is None:
        return None
    try:
        idx_np = _to_np(idx).astype(np.int64, copy=False).reshape(-1)
    except Exception:
        return None
    if idx_np.size == 0:
        return None
    if buf.ndim != 2 or buf.shape[1] < 7:
        return None
    max_idx = int(idx_np.max())
    min_idx = int(idx_np.min())
    if min_idx < 0 or max_idx >= buf.shape[0]:
        return None
    rows = buf[idx_np]  # (N, D) numpy view; downstream indexing is free.
    return rows[:, :3], rows[:, 3:7]


def _resolve_pose_pq(entity):
    """Return ``(p_np, q_np)`` for ``entity``, preferring the raw-buffer fast
    path when available and falling back to ``.pose`` for everything else."""
    buf = _FRAME_CACHE.get("rigid_body_data") if _FRAME_CACHE is not None else None
    if buf is not None:
        hit = _pose_pq_from_buffer(entity, buf)
        if hit is not None:
            return hit
    pose = getattr(entity, "pose", None)
    if pose is None:
        return None
    return _to_np(pose.p), _to_np(pose.q)


def entity_pose_world_array(entity, env_idx: int) -> Optional[np.ndarray]:
    """Pose row of ``entity`` for parallel env ``env_idx``.

    ``pose.p`` rows follow ``entity._objs`` order, not global env order. The
    fast path indexes a pre-fetched ``cuda_rigid_body_data`` numpy buffer via
    ``entity._body_data_index`` and never touches ManiSkill's ``.pose``
    accessor, which otherwise allocates a fresh Pose + tensor slice + torch
    dispatch context per call and drives the graph-only RAM slope. Static /
    hidden / CPU-sim / index-less entities fall back to the accessor.
    """
    if entity is None:
        return None
    row = _obj_index_for_env(entity, env_idx)
    if row is None:
        return None
    if _FRAME_CACHE is not None:
        hit = _FRAME_CACHE["pose"].get(id(entity))
        if hit is None:
            hit = _resolve_pose_pq(entity)
            if hit is None:
                return None
            _FRAME_CACHE["pose"][id(entity)] = hit
        p, q = hit
    else:
        hit = _resolve_pose_pq(entity)
        if hit is None:
            return None
        p, q = hit
    if p.ndim == 2:
        if row >= p.shape[0]:
            return None
        p = p[row]
    if q.ndim == 2:
        q = q[row] if row < q.shape[0] else q[0]
    return np.concatenate([p, q]).astype(float)


def _slice_view_for_env(entity, env_idx: int):
    """Return a single-row Actor/Link view of ``entity`` scoped to ``env_idx``."""
    objs = getattr(entity, "_objs", None)
    if objs is None:
        return entity
    if len(objs) == 1:
        return entity if _obj_index_for_env(entity, env_idx) is not None else None
    row = _obj_index_for_env(entity, env_idx)
    if row is None:
        return None

    scene = getattr(entity, "scene", None)
    cache = scene.__dict__.setdefault("_teemo_sliced_views", {}) if scene else {}
    key = (id(entity), env_idx)
    hit = cache.get(key)
    if hit is not None and hit[0] is entity:
        return hit[1]

    import torch
    from mani_skill.utils.structs.actor import Actor
    from mani_skill.utils.structs.link import Link

    sidx = torch.tensor([env_idx], dtype=torch.int64)
    if type(entity).__name__ == "Link":
        view = Link.create([objs[row]], scene, sidx)
        view.articulation = getattr(entity, "articulation", None)
    else:
        view = Actor.create_from_entities([objs[row]], scene, sidx)
    try:
        view.name = f"{getattr(entity, 'name', 'entity')}@{id(entity)}@env{env_idx}"
    except Exception:
        pass
    cache[key] = (entity, view)
    return view


def _slice_view_rows(entity, target_scene_idxs: List[int]):
    """Return a multi-row Actor/Link view restricted to target scene indices."""
    objs = getattr(entity, "_objs", None)
    sidxs = _scene_idxs_list(entity)
    if objs is None or sidxs is None:
        return None
    target_scene_idxs = list(target_scene_idxs)
    if sidxs == target_scene_idxs:
        return entity

    scene = getattr(entity, "scene", None)
    cache = (
        scene.__dict__.setdefault("_teemo_row_sliced_views", {})
        if scene is not None else {}
    )
    key = (id(entity), tuple(target_scene_idxs))
    hit = cache.get(key)
    if hit is not None and hit[0] is entity:
        return hit[1]

    try:
        rows = [sidxs.index(s) for s in target_scene_idxs]
    except ValueError:
        return None

    import torch
    from mani_skill.utils.structs.actor import Actor
    from mani_skill.utils.structs.link import Link

    sub = torch.tensor(target_scene_idxs, dtype=torch.int64)
    picked = [objs[r] for r in rows]
    if type(entity).__name__ == "Link":
        view = Link.create(picked, scene, sub)
        view.articulation = getattr(entity, "articulation", None)
    else:
        view = Actor.create_from_entities(picked, scene, sub)
    try:
        view.name = (
            f"{getattr(entity, 'name', 'entity')}@{id(entity)}"
            f"@rows{hash(tuple(target_scene_idxs)) & 0xFFFFFFFF:08x}"
        )
    except Exception:
        pass
    cache[key] = (entity, view)
    return view


def _resolve_actual_entity(entity, seg_id_map: Dict[int, Any], env_idx: int):
    """Map an MS-HAB merged handle to its per-env segmentation wrapper.

    MS-HAB names task-level merged actors ``obj_0``, ``obj_1``, etc. The
    segmentation map contains another ManiSkill wrapper around the same SAPIEN
    entity, but with its actual scene name (for example
    ``env-0_024_bowl-3``). Returning that wrapper keeps pose/contact APIs valid
    and lets the persistent target merge with its visible segmentation node.
    """
    if entity is None:
        return None
    scene = getattr(entity, "scene", None)
    handle_name = getattr(entity, "name", None)
    cache = None
    if scene is not None and handle_name is not None:
        cache = scene.__dict__.setdefault("_teemo_resolve_cache", {})
        hit = cache.get((handle_name, env_idx))
        if hit is not None and hit[0] is entity:
            return hit[1]

    target = _entity_for_env(entity, env_idx)
    target_name = getattr(target, "name", None)

    result = None
    for candidate in seg_id_map.values():
        candidate_target = _entity_for_env(candidate, env_idx)
        if candidate_target is target:
            result = candidate
            break

    # Identity is the reliable path, but matching the concrete SAPIEN name is
    # a useful fallback across wrapper/proxy implementations.
    if result is None and target_name is not None:
        for candidate in seg_id_map.values():
            candidate_target = _entity_for_env(candidate, env_idx)
            if getattr(candidate_target, "name", None) == target_name:
                result = candidate
                break

    if result is None:
        result = entity
    if cache is not None:
        cache[(handle_name, env_idx)] = (entity, result)
    return result


def _alias_segmentation_entity(
    seg_id_map: Dict[int, Any], alias, env_idx: int
) -> Dict[int, Any]:
    """Use a merged MS-HAB handle for matching segmentation entries.

    This is the inverse of ``_resolve_actual_entity``. It preserves every
    segmentation id and mask while ensuring merged-name mode creates only the
    ``obj_x`` node, rather than both ``obj_x`` and the actual-name node.
    """
    if alias is None:
        return seg_id_map
    target = _entity_for_env(alias, env_idx)
    target_name = getattr(target, "name", None)
    aliased = dict(seg_id_map)

    for seg_id, candidate in seg_id_map.items():
        candidate_target = _entity_for_env(candidate, env_idx)
        same_entity = candidate_target is target
        same_name = (
            target_name is not None
            and getattr(candidate_target, "name", None) == target_name
        )
        if same_entity or same_name:
            aliased[seg_id] = alias
    return aliased


def _active_mshab_handles(
    env,
    env_idx: int,
    seg_id_map: Optional[Dict[int, Any]] = None,
    object_name: str = "actual",
) -> Dict[str, Any]:
    """Resolve the current subtask's object / articulation / handle link.

    Robust to None entries: close & navigate subtasks have ``subtask_objs[i] is
    None``; only open/close populate articulations. Never raises.
    """
    out = dict(
        active_obj=None,
        active_obj_merged=None,
        active_articulation=None,
        active_handle_link=None,
        active_handle_link_merged=None,
        active_subtask_type=None,
        active_obj_id=None,
    )
    try:
        ptr_all = _frame_np(
            "subtask_ptr", lambda: _to_np(env.subtask_pointer).reshape(-1)
        )
        ptr = int(ptr_all[env_idx])
    except Exception:
        return out

    task_plan = getattr(env, "task_plan", [])
    if not task_plan:
        return out
    ptr = min(ptr, len(task_plan) - 1)            # clip past-end pointer
    subtask = task_plan[ptr]
    out["active_subtask_type"] = _subtask_type(subtask)

    # Resolve the ORIGINAL obj_id from the un-merged task plan. MS-HAB rewrites
    # task_plan[ptr].obj_id to "obj_<num>" during _merge_pick_subtasks /
    # _merge_place_subtasks (mshab/envs/sequential_task.py:219, 274), so reading
    # subtask.obj_id directly would yield the merged name. The original plans
    # live in env.build_config_idx_to_task_plans, indexed per env.
    try:
        bcis = getattr(env, "build_config_idxs", None)
        tpis = getattr(env, "task_plan_idxs", None)
        bcitp = getattr(env, "build_config_idx_to_task_plans", None)
        if bcis is not None and tpis is not None and bcitp is not None:
            bci = int(bcis[env_idx])
            tpi_all = _frame_np("task_plan_idxs", lambda: _to_np(tpis).reshape(-1))
            tpi = int(tpi_all[env_idx])
            tp_list = bcitp.get(bci) if hasattr(bcitp, "get") else None
            if tp_list is not None and 0 <= tpi < len(tp_list):
                original_plan = tp_list[tpi]
                subtasks = getattr(original_plan, "subtasks", None)
                if subtasks is not None and 0 <= ptr < len(subtasks):
                    out["active_obj_id"] = getattr(subtasks[ptr], "obj_id", None)
    except Exception:
        pass  # active_obj_id stays None; affordance lookup falls back to name.

    objs = getattr(env, "subtask_objs", [])
    if ptr < len(objs):
        out["active_obj"] = objs[ptr]             # may legitimately be None

    arts = getattr(env, "subtask_articulations", [])
    art = arts[ptr] if ptr < len(arts) else None
    out["active_articulation"] = art

    handle_idx = getattr(subtask, "articulation_handle_link_idx", None)
    if art is not None and handle_idx is not None:
        try:
            out["active_handle_link"] = art.links[handle_idx]
        except Exception:
            out["active_handle_link"] = None

    out["active_obj_merged"] = out["active_obj"]
    out["active_handle_link_merged"] = out["active_handle_link"]
    if object_name == "actual":
        seg_id_map = seg_id_map or {}
        out["active_obj"] = _resolve_actual_entity(
            out["active_obj"], seg_id_map, env_idx
        )
        out["active_handle_link"] = _resolve_actual_entity(
            out["active_handle_link"], seg_id_map, env_idx
        )
    return out


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def get_privileged_state(
    env,
    env_idx: int = 0,
    *,
    mshab_object_name: str = "actual",
) -> PrivilegedState:
    """Gather a typed privileged snapshot from a (possibly wrapped) env."""
    if mshab_object_name not in ("actual", "merged"):
        raise ValueError(
            "mshab_object_name must be 'actual' or 'merged', got "
            f"{mshab_object_name!r}"
        )
    e = env.unwrapped
    agent = e.agent
    scene = e.scene

    state = PrivilegedState(
        env=e,
        agent=agent,
        scene=scene,
        env_idx=env_idx,
        is_mshab=_looks_like_mshab(e),
        ee_links=get_ee_links(agent),
        tcp_pose_world=_tcp_pose_world_cached(agent, env_idx),
        gripper_width=compute_gripper_width(agent, env_idx),
        seg_id_map=per_env_segmentation_id_map(e, env_idx),
        robot_links=_robot_link_set(agent),
        ee_link_names=get_ee_link_names(agent),
        robot_link_names=_robot_link_names(agent),
    )

    if state.is_mshab:
        handles = _active_mshab_handles(
            e,
            env_idx,
            seg_id_map=state.seg_id_map,
            object_name=mshab_object_name,
        )
        state.active_obj = handles["active_obj"]
        state.active_obj_merged = handles["active_obj_merged"]
        state.active_articulation = handles["active_articulation"]
        state.active_handle_link = handles["active_handle_link"]
        state.active_handle_link_merged = handles["active_handle_link_merged"]
        state.active_subtask_type = handles["active_subtask_type"]
        state.active_obj_id = handles["active_obj_id"]
        if mshab_object_name == "merged":
            state.seg_id_map = _alias_segmentation_entity(
                state.seg_id_map, state.active_obj, env_idx
            )
            state.seg_id_map = _alias_segmentation_entity(
                state.seg_id_map, state.active_handle_link, env_idx
            )

    return state
