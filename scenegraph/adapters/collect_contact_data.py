"""Local successful-rollout interaction collector (schema version 6).

MS-HAB is treated as read-only.  This wrapper observes the simulator through
public environment state, buffers each vector environment independently, and
commits a rollout only when that environment succeeds.

For each successful rollout it records:

* every non-robot entity contacted by an **ee link** (tcp, finger1, finger2);
  contact by any other robot link does not count as evidence -- the runtime
  graph only cares about ee-driven interactions.
* the task target key, used to name/select offline assets;
* direct supporters of contacted entities (one hop only).  Each support event
  carries a pose snapshot (supporter + supported) so the affordance miner can
  derive ``support_components`` and ``bottom_components``.
* obj-obj contact events (force > eps, neither side a robot link, neither
  classified as the support pair).  Pose snapshot + contact-force vector are
  stored so the affordance miner can derive ``contact_components`` (anchor +
  outward normal) on both sides.
* a per-rollout ``bin_samples`` block: raw per-tick samples of ee->object
  planar distance, |height offset|, and their K-window absolute changes, used
  by the whitelist miner to derive per-(subtask, target) bin edges via a
  robust quantile across rollouts.
* a per-rollout ``pose_samples`` trace for the same restricted subject set.
  The whitelist miner uses this after ``affordances.json`` exists to derive
  K-window compatibility-change thresholds.

The active target is not injected as a whitelist member unless an ee link
actually contacts it. Robot links other than tcp/finger1/finger2 are evidence
of nothing.  Pose arrays are stored for affordance mining, including
schema-v4 ``tcp_pose_wrt_base`` to avoid rebuilding TCP poses with a separate
FK chain.
"""

from __future__ import annotations

import os
import pickle
from collections import defaultdict, deque
from pathlib import Path
from typing import (
    Any, Deque, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING,
)

import gymnasium as gym
import numpy as np
import sapien.physx as physx

from mani_skill import ASSET_DIR
from mani_skill.agents.robots import Fetch

from scenegraph.configs.loader import default_temporal_k
from scenegraph.core.affordance import canonical_affordance_key
from scenegraph.core.spatial_metrics import (
    EE_OBJECT_SCOPE,
    EE_SITE_SCOPE,
    stat_key,
)
from scenegraph.core.entity_identity import (
    _articulation,
    entity_kind,
    entity_name,
    stable_entity_key,
)
from scenegraph.adapters.site_providers import (
    collision_half_extents_status,
    ee_rest_geometry,
)
from scenegraph.adapters.privileged_state import (
    entity_pose_world_array,
    get_privileged_state,
    per_env_segmentation_id_map,
)

if TYPE_CHECKING:
    from mshab.envs.sequential_task import SequentialTaskEnv


# v9 adds, per rollout: the build configuration and task plan it came from,
# the collision extents of every entity that can enter ``members``, and the
# end-effector-to-rest-site calibration. None of it can be recovered from a v8
# pickle -- the numbers were never read -- so the miner rejects v8 rather than
# defaulting them.
_SCHEMA_VERSION = 9
# Per-rollout cap on stored obj-obj contact event poses (keeps pickles small).
_OBJ_CONTACT_SAMPLE_CAP = 1024
_EPS_FORCE_DEFAULT = 0.05
_MIN_VERTICAL_FORCE_RATIO_DEFAULT = 0.5
_OBSERVE_STRIDE_DEFAULT = 1
# Runtime temporal.K. Differencing over a different horizon here would
# compress the mined stable band relative to the labels it calibrates.
_DEFAULT_TEMPORAL_K = default_temporal_k()
# Reject the first few ticks after a reset so we never sample the spawned-but-
# settling state or, worse, the just-autoreset state of an env whose previous
# step returned done=True (MS-HAB autoresets inside super().step()).
_RESET_WARMUP_TICKS = 3
# Per-rollout sample buffer cap (per relation, per env). The miner takes a
# robust quantile across these.
_BIN_SAMPLE_CAP = 4096
_POSE_SAMPLE_CAP = 4096
# Per-rollout cap on stored end-effector-to-rest-site diagnostic frames. The
# planar and height streams feed the bins through the ordinary sample buffers;
# these carry the exact distance, tolerance and predicate alongside, which is
# what later checks that a successful state implies every final-phase rung.
_EE_REST_SAMPLE_CAP = 4096


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _entity_xyz(ent, env_idx: int) -> Optional[np.ndarray]:
    arr = entity_pose_world_array(ent, env_idx)
    if arr is None:
        return None
    p = np.asarray(arr[:3], dtype=float).reshape(-1)
    return p[:3] if p.size >= 3 and np.all(np.isfinite(p[:3])) else None


def _entity_pose7(ent, env_idx: int) -> Optional[List[float]]:
    """Return ``[x, y, z, qw, qx, qy, qz]`` for ``ent`` at ``env_idx`` or None."""
    try:
        arr = entity_pose_world_array(ent, env_idx)
    except Exception:
        return None
    if arr is None:
        return None
    if arr.shape[0] != 7 or not np.all(np.isfinite(arr)):
        return None
    return arr.tolist()


def _record(ent, *, key: Optional[str] = None) -> Optional[Dict[str, str]]:
    """One entity's identity: canonical key plus the raw names behind it.

    ``key`` is already canonicalised -- actors lose their instance suffix,
    links keep theirs -- and that canonicalisation is not reversible. The raw
    names are kept beside it because a cross-scene audit has to ask whether
    one logical counter appears as ``kitchen_counter-0`` here and
    ``kitchen_counter-1`` in another build configuration, and the canonical
    key alone cannot answer it.
    """
    stable = key or stable_entity_key(ent)
    if not stable:
        return None
    out = {
        "key": stable,
        "name": entity_name(ent),
        "kind": entity_kind(ent),
    }
    art = _articulation(ent)
    if art is not None:
        out["raw_articulation"] = entity_name(art)
    return out


def _plan_identity(plan, index: int) -> Dict[str, Any]:
    """Which scene and starting arrangement one vector environment is running.

    Recorded per rollout rather than once per pickle: MS-HAB fills the
    environments from a list of plans and autoresets them independently, so
    "which build configuration produced this evidence" is a per-environment
    question and a payload-level answer would be an average of several.
    """
    return {
        "task_plan_index": int(index),
        "build_config_name": str(
            getattr(plan, "build_config_name", "") or ""),
        "init_config_name": str(getattr(plan, "init_config_name", "") or ""),
    }


def _live_plan(base, env_idx: int):
    """``(plan, build_config_idx, task_plan_idx)`` the sub-scene is running now.

    Read from the environment, not from the list handed to ``gym.make``. MS-HAB
    resamples ``task_plan_idxs`` on every reset and looks the active plan up
    through ``build_config_idx_to_task_plans``, so the plan a vector slot
    started with stops describing it after the first episode. Recording the
    assignment instead would attribute each rollout to whichever arrangement
    happened to be dealt at construction -- wrong for every episode but the
    first, and wrong silently.

    Returns ``(None, None, None)`` when the environment does not expose the
    indices, so the caller can say the provenance is unavailable rather than
    fall back to a value it knows is stale.
    """
    try:
        build_idxs = _to_np(base.build_config_idxs).reshape(-1)
        plan_idxs = _to_np(base.task_plan_idxs).reshape(-1)
        bci = int(build_idxs[min(env_idx, build_idxs.size - 1)])
        tpi = int(plan_idxs[min(env_idx, plan_idxs.size - 1)])
        plans = base.build_config_idx_to_task_plans[bci]
        return plans[tpi], bci, tpi
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None, None, None


def _subtask_target_id(subtask) -> str:
    # CloseSubtask carries only ``articulation_id``; pick / place / open / navigate
    # all carry ``obj_id`` (open also has ``articulation_id`` via
    # ArticulationConfig, and we prefer obj_id there to keep pickle naming
    # consistent with the existing pick/open pipeline).
    obj_id = getattr(subtask, "obj_id", None)
    if obj_id:
        return obj_id
    art_id = getattr(subtask, "articulation_id", None)
    if art_id:
        return art_id
    raise AttributeError(
        f"{type(subtask).__name__} has neither obj_id nor articulation_id; "
        "cannot derive a canonical target key"
    )


class FetchCollectContactDataWrapper(gym.Wrapper):
    """Collect successful-rollout interactions without modifying MS-HAB."""

    def __init__(
        self,
        env,
        *,
        eps_force: float = _EPS_FORCE_DEFAULT,
        min_vertical_force_ratio: float = _MIN_VERTICAL_FORCE_RATIO_DEFAULT,
        observe_stride: int = _OBSERVE_STRIDE_DEFAULT,
        temporal_k: int = _DEFAULT_TEMPORAL_K,
        out_root: Optional[str] = None,
        task_group: Optional[str] = None,
        split: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        task_plan_path: Optional[str] = None,
        env_plans: Optional[Sequence[Any]] = None,
        require_build_config: Optional[str] = None,
    ) -> None:
        super().__init__(env)
        base: "SequentialTaskEnv" = env.unwrapped
        self._base_env = base
        self.agent: Fetch = base.agent
        if not isinstance(self.agent, Fetch):
            raise TypeError(f"{self.__class__.__name__} currently supports Fetch only")

        plans = [tp for values in base.bc_to_task_plans.values() for tp in values]
        if not plans or not all(len(tp.subtasks) == 1 for tp in plans):
            raise ValueError("interaction collection requires single-subtask plans")
        canonical_ids = {
            canonical_affordance_key(_subtask_target_id(tp.subtasks[0]))
            for tp in plans
        }
        if len(canonical_ids) != 1:
            raise ValueError(
                "all task plans must reference the same canonical target; "
                f"got {sorted(canonical_ids)}"
            )
        self.obj_id = next(iter(canonical_ids))
        self.subtask_type = plans[0].subtasks[0].type
        self.num_envs = int(getattr(base, "num_envs", 1))

        # Which scenes this environment can actually produce. ``bc_to_task_plans``
        # is the env's own map, so this is what was built rather than what was
        # requested -- the two differ if the caller's filter missed a plan.
        self.build_configs = sorted(str(bc) for bc in base.bc_to_task_plans)
        if require_build_config:
            wanted = str(require_build_config)
            if self.build_configs != [wanted]:
                # Refused at construction, not recorded and mined later. A
                # scene-generalisation experiment whose training evidence came
                # from several build configurations is not the experiment, and
                # provenance would only tell us so after the sim-hours were
                # spent.
                raise ValueError(
                    f"collection was pinned to build config {wanted!r} but the "
                    f"environment holds {self.build_configs}. Filter the task "
                    "plans before constructing the environment."
                )
        self.require_build_config = str(require_build_config or "")
        # Index is the vector-environment index, matching the ``task_plans``
        # list the caller passed to ``gym.make``.
        self.env_plan_identity: List[Dict[str, Any]] = [
            _plan_identity(plan, i) for i, plan in enumerate(env_plans or [])
        ]

        self.eps_force = float(eps_force)
        self.min_vertical_force_ratio = float(min_vertical_force_ratio)
        self.observe_stride = max(1, int(observe_stride))
        self.temporal_k = max(1, int(temporal_k))
        self._step_count = 0
        self._last_observed_step = -1
        # Per-tick force samples, keyed by (id(a), id(b)). Cleared each step
        # after the physics tick so we never serve stale GPU contact data.
        self._force_cache: Dict[Tuple[int, int], Optional[np.ndarray]] = {}
        # Persistent GPU impulse queries with intra-env pairs, keyed the
        # same way. Each entry is ``(query_object, env_to_row)`` where
        # ``env_to_row[env_idx]`` is the row index inside the query's
        # impulse tensor for the intra-env pair in sub-scene env_idx.
        # See ``_pairwise_force`` for why we can't reuse ManiSkill's
        # own scene-level batched query directly.
        self._pair_query_cache: Dict[
            Tuple[int, int], Tuple[Optional[Any], Dict[int, int]]
        ] = {}

        self.success_robot_qpos: List[List[float]] = []
        self.success_obj_pose_wrt_base: List[List[float]] = []
        self.success_tcp_pose_wrt_base: List[List[float]] = []
        self.interaction_rollouts: List[Dict[str, Any]] = []

        # Per-env transient buffers.
        self._episode_interacted: List[Dict[str, Dict[str, Any]]] = []
        self._episode_entities: List[Dict[str, Any]] = []
        self._episode_supports: List[Dict[Tuple[str, str], Dict[str, Any]]] = []
        # Per (env, relation) list of sampled values for the current rollout.
        # Replaces a running-max so the miner can take a robust quantile.
        self._episode_bin_samples: List[Dict[str, List[float]]] = []
        # Per-env compact per-tick pose trace for offline compatibility mining.
        self._episode_pose_samples: List[List[Dict[str, Any]]] = []
        # (entity_key, relation_name) -> deque of last K+1 values
        self._episode_history: List[Dict[Tuple[str, str], Deque[float]]] = []
        # Per-env tick count since last reset (used to skip warmup samples).
        self._episode_ticks: List[int] = []
        # Per-env raw obj-obj contact event samples. Each item is a dict with
        # ``a_key``, ``b_key``, ``a_pose``, ``b_pose``, ``force_vector``.
        self._episode_obj_contacts: List[List[Dict[str, Any]]] = []
        # Per-env entity geometry, keyed by canonical key. Read once per
        # entity per episode rather than per frame: collision extents are a
        # property of the built scene, and re-reading them every tick would
        # pay for the shape walk thousands of times for one unchanging number.
        self._episode_extents: List[Dict[str, Dict[str, Any]]] = []
        # Per-env end-effector-to-rest-site frames: distance, tolerance and
        # the exact predicate, alongside the streams that feed the bins.
        self._episode_ee_rest: List[List[Dict[str, Any]]] = []
        # Whether active-object resolution ever fell back to the merged MS-HAB
        # handle this episode. The protected-target path wants to make that
        # fatal, and this is the evidence for whether it ever happens.
        self._episode_merged_fallback: List[bool] = []
        self._reset_buffers()

        # The task group is part of the path, not just of the payload. The same
        # object is a different mining problem in each MS-HAB task -- set_table's
        # bowl rests in a counter drawer, prepare_groceries' on the counter --
        # and a shared <robot>/<subtask>/<obj>.pkl let whichever task ran last
        # overwrite the others with no trace.
        if not task_group:
            raise ValueError(
                f"{self.__class__.__name__} needs task_group: rollouts are "
                "stored and mined per MS-HAB task, so an unlabelled collection "
                "cannot be told apart from another task's"
            )
        self.task_group = str(task_group)
        self.split = str(split or "train")
        self.checkpoint_path = str(checkpoint_path or "")
        self.task_plan_path = str(task_plan_path or "")

        root = Path(out_root) if out_root else Path(ASSET_DIR) / "robot_success_states"
        save_dir = root / self.agent.uid / self.task_group / self.subtask_type
        os.makedirs(save_dir, exist_ok=True)
        self.save_path = save_dir / f"{self.obj_id}.pkl"

    def _reset_buffers(self, env_indices=None) -> None:
        if not self._episode_interacted:
            self._episode_interacted = [dict() for _ in range(self.num_envs)]
            self._episode_entities = [dict() for _ in range(self.num_envs)]
            self._episode_supports = [dict() for _ in range(self.num_envs)]
            self._episode_bin_samples = [
                defaultdict(list) for _ in range(self.num_envs)
            ]
            self._episode_pose_samples = [[] for _ in range(self.num_envs)]
            self._episode_history = [dict() for _ in range(self.num_envs)]
            self._episode_ticks = [0 for _ in range(self.num_envs)]
            self._episode_obj_contacts = [[] for _ in range(self.num_envs)]
            self._episode_extents = [dict() for _ in range(self.num_envs)]
            self._episode_ee_rest = [[] for _ in range(self.num_envs)]
            self._episode_merged_fallback = [False] * self.num_envs
        indices = range(self.num_envs) if env_indices is None else env_indices
        for idx in indices:
            i = int(idx)
            self._episode_interacted[i].clear()
            self._episode_entities[i].clear()
            self._episode_supports[i].clear()
            self._episode_bin_samples[i].clear()
            self._episode_pose_samples[i].clear()
            self._episode_history[i].clear()
            self._episode_ticks[i] = 0
            self._episode_obj_contacts[i].clear()
            # Extents included: the robot is re-placed and the scene may be
            # rebuilt across a boundary, so a geometry read held over is a
            # measurement of the previous episode's scene.
            self._episode_extents[i].clear()
            self._episode_ee_rest[i].clear()
            self._episode_merged_fallback[i] = False

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        self._reset_buffers()
        # Reconfigure-style resets rebuild the scene; the cached queries
        # still reference the previous SAPIEN body ids, so drop them.
        self._pair_query_cache.clear()
        return result

    def step(self, action, *args, **kwargs):
        obs, rew, term, trunc, info = super().step(action, *args, **kwargs)
        if not physx.is_gpu_enabled():
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support CPU simulation"
            )

        # Forces are valid only for the just-stepped tick. Drop the cache
        # before any observation so we never read stale GPU contact data.
        self._force_cache.clear()
        self._step_count += 1
        for i in range(self.num_envs):
            self._episode_ticks[i] += 1

        # Episode-boundary detection. The collector is always wrapped by
        # ManiSkillVectorEnv(ignore_terminations=True), so MS-HAB's task-success
        # `term` signal does NOT trigger an autoreset -- the env keeps running
        # in the post-success state (bowl still held) until `trunc` fires at
        # max_episode_steps. Using `term|trunc` here would clear the per-env
        # buffers the instant MS-HAB reported success, wiping every drawer /
        # counter supporter we accumulated during the pick, right before the
        # outer script calls `commit_success`. Only trunc is a real boundary.
        done = _to_np(trunc).astype(bool).reshape(-1)
        done_envs = set(np.where(done[:self.num_envs])[0].tolist())

        if self._step_count % self.observe_stride == 0:
            for env_idx in range(self.num_envs):
                if env_idx in done_envs:
                    continue
                if self._episode_ticks[env_idx] < _RESET_WARMUP_TICKS:
                    continue
                self._observe_step(env_idx)
            self._last_observed_step = self._step_count

        self._reset_buffers(sorted(done_envs))
        return obs, rew, term, trunc, info

    def _pairwise_force(self, scene, a, b, env_idx: int) -> Optional[np.ndarray]:
        """Contact-force 3-vector between ``a`` and ``b`` in sub-scene ``env_idx``.

        ManiSkill's ``scene.get_pairwise_contact_forces`` builds its GPU
        query via ``zip(a._bodies, b._bodies)``
        (mani_skill/envs/scene.py:771). MS-HAB's merged pick target always
        spans ``arange(num_envs)`` -- but scene-config-specific supporters
        (e.g. ``kitchen_counter-0/drawer3``, or any articulation link whose
        parent isn't in every build_config) have ``_scene_idxs`` equal to
        just the subset of sub-scenes whose build_config contains them.
        The zip then pairs their bodies position-for-position with the
        merged target's bodies, so unless positions coincide the pair
        crosses sub-scenes and the impulse is always zero. Every such
        supporter is silently dropped from the collected rollouts.

        This helper builds a targeted per-``(a, b)`` GPU impulse query
        containing exactly one pair per sub-scene where both entities
        have a body, and returns the row for ``env_idx``. Query objects
        are persistent across ticks (they reference SAPIEN body ids that
        stay stable until a reconfigure-style reset); the raw impulse
        tensor is cached per tick alongside the rest of ``_force_cache``.
        """
        key = (id(a), id(b))

        query_entry = self._pair_query_cache.get(key)
        if query_entry is None:
            query_entry = self._build_pair_query(scene, a, b)
            self._pair_query_cache[key] = query_entry
        query, env_to_row = query_entry
        if query is None or env_idx not in env_to_row:
            return None

        if key not in self._force_cache:
            try:
                scene.px.gpu_query_contact_pair_impulses(query)
                impulses = _to_np(query.cuda_impulses.torch().clone())
                self._force_cache[key] = impulses / float(scene.px.timestep)
            except Exception:
                self._force_cache[key] = None
        forces = self._force_cache[key]
        if forces is None:
            return None

        return self._pair_row_lookup(
            forces, env_to_row[env_idx], env_idx, len(env_to_row),
        )

    def _build_pair_query(
        self, scene, a, b,
    ) -> Tuple[Optional[Any], Dict[int, int]]:
        """Return ``(query, env_to_row)`` for a GPU impulse query holding
        one ``(a, b)`` pair per sub-scene where both entities have a body.

        Falls back to the classic ``zip(a._bodies, b._bodies)`` pairing when
        either side lacks ``_scene_idxs`` -- that keeps the single-env / EE
        cases working unchanged, since there the two lists always align.
        """
        try:
            a_bodies = list(a._bodies)
            b_bodies = list(b._bodies)
        except AttributeError:
            return None, {}
        a_idxs = self._scene_idxs_list(a)
        b_idxs = self._scene_idxs_list(b)
        pairs: List[Tuple[Any, Any]] = []
        env_to_row: Dict[int, int] = {}
        if a_idxs is None or b_idxs is None:
            n = min(len(a_bodies), len(b_bodies))
            for i in range(n):
                env_to_row[i] = i
                pairs.append((a_bodies[i], b_bodies[i]))
        else:
            for env_idx in sorted(set(a_idxs) & set(b_idxs)):
                try:
                    pair = (
                        a_bodies[a_idxs.index(env_idx)],
                        b_bodies[b_idxs.index(env_idx)],
                    )
                except (IndexError, ValueError):
                    continue
                env_to_row[env_idx] = len(pairs)
                pairs.append(pair)
        if not pairs:
            return None, {}
        try:
            query = scene.px.gpu_create_contact_pair_impulse_query(pairs)
        except Exception:
            return None, {}
        return query, env_to_row

    @staticmethod
    def _scene_idxs_list(entity) -> Optional[List[int]]:
        idxs_attr = getattr(entity, "_scene_idxs", None)
        if idxs_attr is None:
            return None
        try:
            return _to_np(idxs_attr).reshape(-1).tolist()
        except Exception:
            return None

    def _pair_row_lookup(
        self, forces: np.ndarray, row: int, env_idx: int, n_pairs: int,
    ) -> Optional[np.ndarray]:
        """Extract the row for a specific pair from a GPU impulse tensor.

        The comment at ``mani_skill/envs/scene.py:779`` claims
        ``cuda_impulses`` has shape ``(num_unique_pairs * num_envs, 3)``,
        but real observations show single-pair queries returning either
        ``(1, 3)`` or ``(num_envs, 3)``. We resolve the ambiguity
        empirically: try direct ``(num_pairs, 3)`` indexing first, then
        the pair-major ``(num_pairs * num_envs, 3)`` layout, then a plain
        row index as last resort.
        """
        if forces.ndim == 1:
            return forces.astype(float) if n_pairs == 1 and row == 0 else None
        if forces.ndim != 2 or forces.shape[1] != 3:
            return None
        n_rows = forces.shape[0]
        if n_rows == n_pairs:
            return forces[row].astype(float)
        num_envs = int(self.num_envs)
        if n_rows == n_pairs * num_envs:
            candidate = row * num_envs + env_idx
            if 0 <= candidate < n_rows:
                return forces[candidate].astype(float)
        if 0 <= row < n_rows:
            return forces[row].astype(float)
        return None

    def _robot_links(self) -> Tuple[List[Any], set]:
        try:
            links = list(self.agent.robot.get_links())
        except Exception:
            links = []
        names = {entity_name(link) for link in links}
        return links, names

    def _scene_entities(
        self, env_idx: Optional[int] = None, seg_id_map: Optional[Dict[int, Any]] = None
    ) -> List[Any]:
        if seg_id_map is None:
            if env_idx is not None:
                seg_id_map = per_env_segmentation_id_map(self._base_env, env_idx)
            else:
                seg_id_map = dict(getattr(self._base_env, "segmentation_id_map", {}))
        robot_links, robot_names = self._robot_links()
        robot_ids = {id(link) for link in robot_links}
        entities: List[Any] = []
        seen = set()
        for seg_id, ent in seg_id_map.items():
            if not seg_id or ent is None:
                continue
            if id(ent) in robot_ids or entity_name(ent) in robot_names:
                continue
            key = stable_entity_key(ent)
            if not key or key in seen:
                continue
            seen.add(key)
            entities.append(ent)
        return entities

    def _target(self, env_idx: int, state=None) -> Tuple[Optional[Any], str]:
        state = state or get_privileged_state(
            self, env_idx, mshab_object_name="actual"
        )
        if self.subtask_type == "pick":
            # MS-HAB may expose pick targets through a merged runtime actor
            # named ``obj_0``.  The offline assets and runtime whitelist are
            # keyed by the original task object id instead.
            return state.active_obj, f"actor:{self.obj_id}"

        target = state.active_handle_link or state.active_obj
        if target is not None:
            key = stable_entity_key(target)
            if key:
                return target, key
        return target, f"actor:{self.obj_id}"

    def _mark_interacted(
        self,
        env_idx: int,
        key: str,
        ent: Any,
        *,
        max_ee_force: float = 0.0,
        grasped: bool = False,
    ) -> None:
        rec = _record(ent, key=key)
        if rec is None:
            return
        rec["max_ee_force"] = float(max_ee_force)
        if grasped:
            rec["grasped"] = True

        interacted = self._episode_interacted[env_idx]
        prior = interacted.get(key)
        prior_force = 0.0 if prior is None else float(
            prior.get("max_ee_force", 0.0)
        )
        if prior is not None and prior.get("grasped"):
            rec["grasped"] = True
        if prior is None or grasped or max_ee_force > prior_force:
            interacted[key] = rec
        self._episode_entities[env_idx][key] = ent

    def _update_bin_stats(
        self,
        env_idx: int,
        entity_by_key: Dict[str, Any],
        tcp_pose_world: Optional[np.ndarray],
        gripper_width: Optional[float],
    ) -> None:
        """Buffer per-rollout samples of ee<->object spatial values and their
        K-window absolute changes for a *restricted* set of entities (the
        target plus already-known interacted/supporter keys). The miner takes a
        robust quantile across these samples instead of trusting a single max,
        so a transient outlier (autoreset jump, physics blow-up) cannot poison
        the per-(subtask, target) bin edges.

        Compatibility metrics are not sampled here (they require the affordance
        asset) -- defaults are applied in the miner.
        """
        if tcp_pose_world is None or not entity_by_key:
            return
        samples = self._episode_bin_samples[env_idx]
        history = self._episode_history[env_idx]
        tcp_pose = np.asarray(tcp_pose_world, dtype=float).reshape(-1)
        if tcp_pose.size < 3 or not np.all(np.isfinite(tcp_pose[:3])):
            return
        ee_xyz = tcp_pose[:3]
        ee_xy = ee_xyz[:2]
        ee_z = float(ee_xyz[2])
        pose_entities: Dict[str, Dict[str, Any]] = {}
        for key, ent in entity_by_key.items():
            obj_xyz = _entity_xyz(ent, env_idx)
            if obj_xyz is None:
                continue
            pose = _entity_pose7(ent, env_idx)
            if pose is not None:
                pose_entities[str(key)] = {
                    "pose": pose,
                    "kind": entity_kind(ent),
                    "name": entity_name(ent),
                }
            pd = float(np.linalg.norm(ee_xy - obj_xyz[:2]))
            ho_signed = ee_z - float(obj_xyz[2])
            self._push_sample(
                samples, stat_key(EE_OBJECT_SCOPE, "planar-distance"), pd)
            self._push_sample(
                samples, stat_key(EE_OBJECT_SCOPE, "height-offset"),
                abs(ho_signed))
            for relation, value in (
                (stat_key(EE_OBJECT_SCOPE, "planar-distance"), pd),
                (stat_key(EE_OBJECT_SCOPE, "height-offset"), ho_signed),
            ):
                hk = (key, relation)
                buf = history.get(hk)
                if buf is None:
                    buf = deque(maxlen=self.temporal_k + 1)
                    history[hk] = buf
                buf.append(value)
                if len(buf) > self.temporal_k:
                    change = abs(buf[-1] - buf[0])
                    self._push_sample(samples, f"{relation}_change", change)
        # Independent of ``entity_by_key``: the gripper has a distance to its
        # rest position whether or not any object was near enough to sample.
        self._observe_ee_rest(env_idx, ee_xyz)
        if pose_entities and len(self._episode_pose_samples[env_idx]) < _POSE_SAMPLE_CAP:
            pose_sample: Dict[str, Any] = {
                "tcp_pose": tcp_pose[:7].astype(float).tolist(),
                "entities": pose_entities,
            }
            if gripper_width is not None and np.isfinite(gripper_width):
                pose_sample["gripper_width"] = float(gripper_width)
            self._episode_pose_samples[env_idx].append(pose_sample)

    def _observe_extents(
        self, env_idx: int, entity_by_key: Dict[str, Any],
    ) -> None:
        """Record collision half-extents once per entity per episode.

        The miner cannot derive this from anything else in the pickle, and it
        is what separates an extended support plane from a localized
        receptacle -- roles cannot, because a bin and a tabletop carry
        identical ones. A failed read is stored with its reason rather than
        omitted: "no extent" and "small" have to stay distinguishable, since
        assuming small is how a metre-wide counter gets measured from its own
        origin.
        """
        store = self._episode_extents[env_idx]
        for key, ent in entity_by_key.items():
            if key in store:
                continue
            half, status = collision_half_extents_status(ent, env_idx)
            record = _record(ent, key=key) or {"key": key}
            record["extent_status"] = status
            if half is not None:
                record["half_extents"] = [float(v) for v in half]
            store[key] = record

    def _observe_ee_rest(self, env_idx: int, ee_xyz: np.ndarray) -> None:
        """Sample the end effector against MS-HAB's rest position.

        Pick only: ``pick_cfg.ee_rest_thresh`` is that subtask's own success
        geometry and no other subtask defines one. Read through the same
        helper the runtime provider uses, so the mined scale and the labelled
        distance are measurements of the same point.

        The planar and height streams get dedicated keys. Pooling them into
        ``ee-object-*`` would stretch the band a two-centimetre approach has
        to register against with a distance to the robot's own base.
        """
        if self.subtask_type != "pick":
            return
        world, tolerance = ee_rest_geometry(self, env_idx)
        delta = ee_xyz - world
        planar = float(np.linalg.norm(delta[:2]))
        height = float(delta[2])
        euclidean = float(np.linalg.norm(delta))
        samples = self._episode_bin_samples[env_idx]
        self._push_sample(
            samples, stat_key(EE_SITE_SCOPE, "planar-distance"), planar)
        self._push_sample(
            samples, stat_key(EE_SITE_SCOPE, "height-offset"), abs(height))
        if len(self._episode_ee_rest[env_idx]) >= _EE_REST_SAMPLE_CAP:
            return
        self._episode_ee_rest[env_idx].append({
            "rest_pos_world": [float(v) for v in world],
            "planar_distance": planar,
            "height_offset": height,
            "euclidean_distance": euclidean,
            "tolerance": float(tolerance),
            # The environment's own predicate, recorded rather than inferred
            # later from the distance and a remembered threshold.
            "reached": bool(euclidean <= tolerance),
        })

    @staticmethod
    def _push_sample(samples: Dict[str, List[float]], key: str, value: float) -> None:
        if not np.isfinite(value):
            return
        bucket = samples[key]
        if len(bucket) >= _BIN_SAMPLE_CAP:
            return
        bucket.append(float(value))

    def _observe_step(self, env_idx: int) -> None:
        scene = self._base_env.scene
        actual_state = get_privileged_state(
            self, env_idx, mshab_object_name="actual"
        )
        entities = self._scene_entities(
            env_idx=env_idx, seg_id_map=actual_state.seg_id_map
        )
        target_ent, target_key = self._target(env_idx, actual_state)
        # Did ``_resolve_actual_entity`` map the MS-HAB handle to this
        # sub-scene's actor, or give the merged handle back? The protected
        # target path wants to treat the fallback as fatal, and this records
        # whether it ever fires rather than assuming either way.
        resolved = getattr(actual_state, "active_obj", None)
        if (resolved is not None
                and resolved is getattr(actual_state, "active_obj_merged", None)):
            self._episode_merged_fallback[env_idx] = True
        physics_target_ent = target_ent
        ee_links = list(actual_state.ee_links)

        # For pick, MS-HAB's grasp predicate and contact forces are most
        # reliable on the merged runtime actor.  We still record the canonical
        # task key, so the whitelist never learns ``actor:obj_0``.
        merged_state = None
        if self.subtask_type == "pick":
            merged_state = get_privileged_state(
                self, env_idx, mshab_object_name="merged"
            )
            if getattr(merged_state, "active_obj", None) is not None:
                physics_target_ent = merged_state.active_obj

        entity_by_key: Dict[str, Any] = {}
        for ent in entities:
            key = stable_entity_key(ent)
            if key:
                entity_by_key[key] = ent
        if target_ent is not None:
            raw_key = stable_entity_key(target_ent)
            if raw_key and raw_key != target_key:
                entity_by_key.pop(raw_key, None)
        if physics_target_ent is not None:
            raw_key = stable_entity_key(physics_target_ent)
            if raw_key and raw_key != target_key:
                entity_by_key.pop(raw_key, None)
            entity_by_key[target_key] = physics_target_ent

        # Contact evidence is restricted to the ee link set so the whitelist
        # ignores incidental robot-body bumps.
        for key, ent in entity_by_key.items():
            total_force = 0.0
            for ee_link in ee_links:
                vector = self._pairwise_force(scene, ee_link, ent, env_idx)
                if vector is not None:
                    total_force += float(np.linalg.norm(vector))
            if total_force <= self.eps_force:
                continue
            self._mark_interacted(
                env_idx, key, ent, max_ee_force=total_force
            )

        if self.subtask_type == "pick" and physics_target_ent is not None:
            grasped = False
            if merged_state is not None:
                grasped = merged_state.is_grasping(physics_target_ent)
            if not grasped:
                grasped = actual_state.is_grasping(target_ent)
            if grasped:
                force = actual_state.ee_object_contact_force(target_ent)
                if merged_state is not None:
                    force = max(
                        force,
                        merged_state.ee_object_contact_force(physics_target_ent),
                    )
                self._mark_interacted(
                    env_idx, target_key, physics_target_ent,
                    max_ee_force=force, grasped=True,
                )

        # Obj-obj contact event sampling. Restrict the expensive contact
        # queries to pairs touching a real EE source; direct target/supporter
        # contacts are still handled by the support pass below.
        self._sample_obj_obj_contacts(env_idx, entity_by_key)

        # Support evidence may be visible before robot-object contact. Buffer
        # interacted entities (including one-hop-elevated) plus the active
        # target; the whitelist miner later admits only supporters of
        # interacted roots.
        roots = dict(self._episode_entities[env_idx])
        if physics_target_ent is not None:
            roots[target_key] = physics_target_ent
        for supported_key, supported in roots.items():
            self._observe_direct_supporters(
                env_idx, supported_key, supported, entity_by_key,
            )

        # Spatial sampling is intentionally restricted to the target plus
        # already-known interacted/supporter entities. Sampling against every
        # non-robot scene entity (counters, walls, far cabinet links) used to
        # learn the EE's distance to the *farthest* thing in the kitchen and
        # produced 9 m planar-distance maxes for a pick-bowl rollout, making
        # the runtime label always 'near'. The runtime only evaluates spatial
        # relations against whitelisted objects, so this is the right subject
        # set.
        subjects: Dict[str, Any] = {}
        if physics_target_ent is not None:
            subjects[target_key] = physics_target_ent
        for k, ent in self._episode_entities[env_idx].items():
            subjects.setdefault(k, ent)
        for (supporter_key, _supported_key), rec in (
            self._episode_supports[env_idx].items()
        ):
            ent = entity_by_key.get(supporter_key)
            if ent is not None:
                subjects.setdefault(supporter_key, ent)
        # The same subject set the bins are sampled over: the target, the
        # interacted entities and their direct supporters -- which is exactly
        # what can end up in ``members``.
        self._observe_extents(env_idx, subjects)
        self._update_bin_stats(
            env_idx, subjects,
            actual_state.tcp_pose_world,
            actual_state.gripper_width,
        )

    def _sample_obj_obj_contacts(
        self, env_idx: int, entity_by_key: Dict[str, Any],
    ) -> None:
        """Record EE-local obj-obj contacts and one-hop lift the other endpoint.

        The old implementation scanned every object-object pair in the visible
        scene. ReplicaCAD kitchens contain many static links, so that becomes
        O(N^2) GPU contact queries per env per tick. The whitelist only uses
        obj-obj contacts to propagate one hop from an EE-touched source (for
        example knife -> onion), while direct support evidence is collected in
        ``_observe_direct_supporters``. Query only pairs that touch a true EE
        source and let the support pass cover target/supporter relations.

        Elevation runs even when the sample bucket is full so a late-in-episode
        contact still promotes membership; the bucket cap only bounds the
        affordance-mining sample pool, not the whitelist role graph.
        """
        bucket = self._episode_obj_contacts[env_idx]
        support_pairs = {
            frozenset({supporter, supported})
            for (supporter, supported) in self._episode_supports[env_idx]
        }
        scene = self._base_env.scene
        interacted = self._episode_interacted[env_idx]
        source_keys = [
            key for key, rec in interacted.items()
            if key in entity_by_key and self._is_ee_source(rec)
        ]
        if not source_keys:
            return

        seen_pairs = set()
        for source_key in source_keys:
            source = entity_by_key[source_key]
            for other_key, other in entity_by_key.items():
                if other_key == source_key:
                    continue
                pair_key = frozenset({source_key, other_key})
                if pair_key in seen_pairs or pair_key in support_pairs:
                    continue
                seen_pairs.add(pair_key)

                vector = self._pairwise_force(scene, source, other, env_idx)
                if vector is None:
                    continue
                force = float(np.linalg.norm(vector))
                if force <= self.eps_force:
                    continue
                source_pose = _entity_pose7(source, env_idx)
                other_pose = _entity_pose7(other, env_idx)
                if source_pose is None or other_pose is None:
                    continue
                if len(bucket) < _OBJ_CONTACT_SAMPLE_CAP:
                    bucket.append({
                        "a_key": source_key,
                        "b_key": other_key,
                        "a_pose": source_pose,
                        "b_pose": other_pose,
                        "force_vector": [
                            float(vector[0]), float(vector[1]), float(vector[2]),
                        ],
                        "force": force,
                    })
                # One-hop elevation. The source side has real EE evidence;
                # the other side is admitted with zero EE force and therefore
                # cannot propagate further in this or later mining.
                if other_key not in interacted:
                    self._mark_interacted(
                        env_idx, other_key, other, max_ee_force=0.0
                    )

    @staticmethod
    def _is_ee_source(rec: Optional[Dict[str, Any]]) -> bool:
        if rec is None:
            return False
        return (
            float(rec.get("max_ee_force", 0.0)) > 0.0
            or bool(rec.get("grasped"))
        )

    def _observe_direct_supporters(
        self,
        env_idx: int,
        supported_key: str,
        supported: Any,
        candidates: Dict[str, Any],
    ) -> None:
        supported_xyz = _entity_xyz(supported, env_idx)
        scene = self._base_env.scene
        for supporter_key, supporter in candidates.items():
            if supporter_key == supported_key:
                continue
            vector = self._pairwise_force(scene, supporter, supported, env_idx)
            if vector is None:
                continue
            force = float(np.linalg.norm(vector))
            if force <= self.eps_force:
                continue
            supporter_xyz = _entity_xyz(supporter, env_idx)
            vertical_ratio = abs(float(vector[2])) / force
            vertical_support = bool(
                vertical_ratio >= self.min_vertical_force_ratio
                # ``_pairwise_force(supporter, supported)`` returns force on
                # the candidate supporter due to the supported object. A real
                # support contact pushes the supporter downward. Without this
                # sign check we record both directions for the same contact,
                # which teaches the assets that e.g. a bowl supports a drawer.
                and float(vector[2]) < 0.0
            )
            if not vertical_support:
                continue
            dz = (
                float(supported_xyz[2] - supporter_xyz[2])
                if supporter_xyz is not None and supported_xyz is not None
                else None
            )
            rec = {
                "supporter": _record(supporter, key=supporter_key),
                "supported_key": supported_key,
                "force": force,
                "vertical_force_ratio": vertical_ratio,
                "dz": dz,
                "vertical_support": vertical_support,
                "evidence": "force",
                "supporter_pose": _entity_pose7(supporter, env_idx),
                "supported_pose": _entity_pose7(supported, env_idx),
                "force_vector": [
                    float(vector[0]), float(vector[1]), float(vector[2]),
                ],
            }
            self._merge_support(env_idx, supporter_key, supported_key, rec)

    def _merge_support(
        self,
        env_idx: int,
        supporter_key: str,
        supported_key: str,
        rec: Dict[str, Any],
    ) -> None:
        pair = (supporter_key, supported_key)
        prior = self._episode_supports[env_idx].get(pair)
        if prior is None:
            self._episode_supports[env_idx][pair] = rec
            return
        prior_rank = (
            bool(prior.get("vertical_support", False)),
            float(prior.get("force", 0.0)),
        )
        new_rank = (
            bool(rec.get("vertical_support", False)),
            float(rec.get("force", 0.0)),
        )
        if new_rank > prior_rank:
            self._episode_supports[env_idx][pair] = rec

    def commit_success(
        self,
        env_idx: int,
        qpos: np.ndarray,
        obj_pose_wrt_base: np.ndarray,
        tcp_pose_wrt_base: np.ndarray,
    ) -> None:
        """Append one success sample. Called by the collection script after
        venv.step returns, with state read at the script level.
        """
        # Guarantee the grasp moment is captured: if stride skipped this tick,
        # observe this single env now while sim forces are still valid.
        if self._last_observed_step != self._step_count:
            self._observe_step(env_idx)

        self.success_robot_qpos.append(np.asarray(qpos, dtype=float).tolist())
        self.success_obj_pose_wrt_base.append(
            np.asarray(obj_pose_wrt_base, dtype=float).tolist()
        )
        self.success_tcp_pose_wrt_base.append(
            np.asarray(tcp_pose_wrt_base, dtype=float).tolist()
        )

        _target_ent, target_key = self._target(env_idx)
        interacted = list(self._episode_interacted[env_idx].values())
        root_keys = {item["key"] for item in interacted}
        root_keys.add(target_key)
        supports = [
            value for (_supporter, supported), value
            in self._episode_supports[env_idx].items()
            if supported in root_keys
            and value.get("supporter") is not None
        ]
        rollout_bin_samples: Dict[str, List[float]] = {
            k: list(values)
            for k, values in self._episode_bin_samples[env_idx].items()
            if values
        }
        rollout_pose_samples = list(self._episode_pose_samples[env_idx])
        self.interaction_rollouts.append({
            "target_key": target_key,
            "interacted": interacted,
            "supports": supports,
            "bin_samples": rollout_bin_samples,
            "pose_samples": rollout_pose_samples,
            "obj_contacts": list(self._episode_obj_contacts[env_idx]),
            # Which scene this evidence came from. Per rollout because the
            # environments are filled from a list of plans and autoreset
            # independently, so one pickle can hold several -- and a
            # scene-generalisation split needs to be able to prove it did not.
            "provenance": self._rollout_provenance(env_idx),
            # Geometry and rest-site calibration: neither is derivable from
            # anything else stored here.
            "extents": dict(self._episode_extents[env_idx]),
            "ee_rest_samples": list(self._episode_ee_rest[env_idx]),
        })

    def _rollout_provenance(self, env_idx: int) -> Dict[str, Any]:
        """Where one rollout came from: the live plan, uncanonicalised.

        Raw on purpose. A later audit has to be able to ask whether one
        logical articulation is numbered differently across build
        configurations, and a normalised record cannot answer that -- the
        normalisation is what erases the difference.
        """
        plan, build_idx, plan_idx = _live_plan(self._base_env, env_idx)
        if plan is not None:
            identity = _plan_identity(plan, plan_idx)
            identity["build_config_index"] = int(build_idx)
            identity["plan_source"] = "live"
        else:
            # Named, not substituted. The construction-time assignment is
            # what this slot *started* with, and after the first autoreset
            # that is a different arrangement -- so it travels under its own
            # key rather than posing as the plan that produced the rollout.
            identity = {"plan_source": "unavailable"}
            if env_idx < len(self.env_plan_identity):
                identity["assigned_at_construction"] = dict(
                    self.env_plan_identity[env_idx])
        identity.update({
            "env_idx": int(env_idx),
            "split": self.split,
            "task_plan_path": self.task_plan_path,
            "task_group": self.task_group,
            # Every configuration the environment can build, so a rollout
            # whose own plan identity is missing is still bounded by this.
            "env_build_configs": list(self.build_configs),
            "requested_build_config": self.require_build_config,
            # ``merged`` means active-object resolution never found this
            # sub-scene's own actor and handed back the shared handle.
            "target_resolution": (
                "merged" if self._episode_merged_fallback[env_idx] else "actual"
            ),
        })
        return identity

    def close(self):
        # Never overwrite an existing pkl with an empty payload. Diagnostic
        # wrappers (which never call commit_success) previously wiped the
        # production pkl on venv.close() and turned every downstream whitelist
        # build into a silent no-op.
        if not self.interaction_rollouts and self.save_path.exists():
            return super().close()
        entity_key = (
            self.interaction_rollouts[0].get("target_key")
            if self.interaction_rollouts else f"actor:{self.obj_id}"
        )
        payload = {
            "_schema_version": _SCHEMA_VERSION,
            "obj_id": self.obj_id,
            "entity_key": entity_key,
            "subtask_type": self.subtask_type,
            # Provenance travels with the evidence. The miner cross-checks it
            # against the directory it found the file under, so a pkl moved or
            # copied between task trees fails loudly instead of contributing
            # another task's scene to this group's whitelist.
            "provenance": {
                "task_group": self.task_group,
                "subtask_type": self.subtask_type,
                "target": entity_key,
                "checkpoint_path": self.checkpoint_path,
                "task_plan_path": self.task_plan_path,
                "split": self.split,
            },
            "temporal_k": self.temporal_k,
            # Every build configuration this collection could have produced.
            # A one-element list is the evidence a scene-pinned run needs.
            "build_configs": list(self.build_configs),
            "requested_build_config": self.require_build_config,
            "robot_qpos": self.success_robot_qpos,
            "obj_pose_wrt_base": self.success_obj_pose_wrt_base,
            "tcp_pose_wrt_base": self.success_tcp_pose_wrt_base,
            "interaction_rollouts": self.interaction_rollouts,
        }
        with open(self.save_path, "wb") as stream:
            pickle.dump(payload, stream)
        return super().close()
