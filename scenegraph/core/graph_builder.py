"""Per-frame vertex maintenance and fact orchestration.

Pipeline: build_nodes -> apply_whitelist -> merge_retained -> live pose
refresh -> visibility policy -> registry.assign -> absolute facts -> temporal
labels.

Retention is unconditional: a whitelisted object seen once stays a vertex until
episode reset. Capacity is therefore a configuration error, not a runtime
decision, and overflow raises rather than evicting.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .affordance import canonical_affordance_key
from .entity_identity import stable_entity_key, stable_node_id
from .schema import Edge, Graph, Node
from .node_builder import build_nodes
from .relation_rules import REQUIRED_BIN_RELATIONS, build_absolute_edges
from .temporal_buffer import TemporalBuffer
from .mask_extractor import MaskAccumulator
from .selector import EntityRegistry, NodeSelector
from .whitelist import entity_match_key, load_whitelist, resolve_whitelist_path
from ..adapters.camera_projection import CameraCoverage
from ..adapters.privileged_state import (
    entity_pose_world_array,
    get_privileged_state,
)

# Packed-UID reservations. Zero is padding so an unfilled slot decodes as
# "no node"; one is the end effector, whose identity never varies.
VISIBILITY_PROJECTED = "projected_camera"
VISIBILITY_KEEP = "keep_tabletop"
VISIBILITY_POLICIES = frozenset({VISIBILITY_PROJECTED, VISIBILITY_KEEP})

UID_PAD = 0
UID_EE = 1
_UID_FIRST_OBJECT = 2
# UIDs are packed as uint8, the narrowest dtype holding the vocabulary and the
# one the replay buffer can index. Wider integer types are not universally
# supported there -- torchrl's index_put has no uint16 kernel.
UID_VOCAB_MAX = 256

# Targetless (normal ManiSkill) whitelist: <dir>/task_all.json.
TASK_LEVEL_SUBTASK = "task"
TASK_LEVEL_TARGET = "all"


class EpisodeUIDs:
    """Episode-scoped object identity, independent of the compact slot.

    The vertex registry releases and reuses slots, so a slot cannot name an
    object across frames. A UID can: it is allocated on first sight, survives
    the object leaving the view, and is never handed to a second object before
    the episode ends.

    Codes are drawn in a per-episode random order so a UID carries no meaning
    beyond "the same object as before" -- a fixed mapping would let the model
    memorise simulator identities across episodes.
    """

    def __init__(self, uid_vocab: int, seed: int = 0):
        self.uid_vocab = int(uid_vocab)
        if self.uid_vocab <= _UID_FIRST_OBJECT:
            raise ValueError(
                f"uid_vocab={self.uid_vocab} leaves no object codes; "
                f"codes {UID_PAD} and {UID_EE} are reserved"
            )
        if self.uid_vocab > UID_VOCAB_MAX:
            raise ValueError(
                f"uid_vocab={self.uid_vocab} exceeds {UID_VOCAB_MAX}; "
                "graph_node_uid is packed as uint8. Widening it means "
                "changing the packed dtype, and the replay buffer has no "
                "uint16 index_put kernel -- use int32 if you truly need more."
            )
        self._rng = np.random.default_rng(seed)
        self._codes: Dict[str, int] = {}
        self._free: List[int] = []
        self.reset()

    def reset(self) -> None:
        self._codes.clear()
        pool = np.arange(_UID_FIRST_OBJECT, self.uid_vocab)
        self._free = [int(code) for code in self._rng.permutation(pool)]

    def __len__(self) -> int:
        return len(self._codes)

    def uid_for(self, node_id: str, *, is_ee: bool = False) -> int:
        if is_ee:
            return UID_EE
        uid = self._codes.get(node_id)
        if uid is not None:
            return uid
        if not self._free:
            raise RuntimeError(
                f"episode exhausted uid_vocab={self.uid_vocab}; raise "
                "model.graph.uid_vocab above the peak "
                "episode/graph_episode_entities. Reusing a code would "
                "silently merge two objects."
            )
        uid = self._free.pop()
        self._codes[node_id] = uid
        return uid


class GraphBuilder:
    def __init__(
        self,
        env,
        cfg: dict,
        *,
        env_idx: int = 0,
        env_id: str = "env",
        camera: Optional[str] = None,
        camera_order: Optional[List[str]] = None,
        visibility_policy: str = VISIBILITY_KEEP,
        uid_vocab: int = 256,
        appearance_enabled: bool = True,
        bbox_enabled: bool = True,
        uids_enabled: bool = True,
        use_target_flag: bool = True,
    ):
        self.env = env
        self.cfg = cfg
        self.env_idx = env_idx
        self.env_id = env_id
        self.camera = camera
        self.camera_order = list(camera_order) if camera_order else None
        if visibility_policy not in VISIBILITY_POLICIES:
            raise ValueError(
                f"unknown visibility_policy {visibility_policy!r}; "
                f"expected one of {sorted(VISIBILITY_POLICIES)}"
            )
        self.visibility_policy = visibility_policy
        # Two switches, not one. The pooled relation contract wants boxes and
        # no patch coverage; appearance implies both.
        self.appearance_enabled = bool(appearance_enabled)
        self.bbox_enabled = bool(bbox_enabled)
        # Episode-random identity codes exist for slot alignment only. The
        # pooled contract addresses nodes by their box, so nothing assigns or
        # packs a UID there.
        self.uids_enabled = bool(uids_enabled)
        # False: no subtask target exists, so no whitelist is
        # bound per target and no row is reserved.
        self.use_target_flag = bool(use_target_flag)
        self.uids = EpisodeUIDs(uid_vocab, seed=1000 + int(env_idx))

        self.temporal = TemporalBuffer(K=cfg["temporal"]["K"])
        self.selector = NodeSelector(cfg)
        self.registry = EntityRegistry(n_max=int(cfg["selection"]["n_max"]))
        self.cfg.setdefault("_affordance_selection_cache", {})
        # node_id -> simulator entity. A retained node has no segmentation
        # ids to look one up with, and physics queries still need it.
        # Shared with relation_rules through cfg, which is per-env.
        self._entities: Dict[str, Any] = self.cfg.setdefault(
            "_entity_cache", {})
        self._coverage = (
            CameraCoverage(env, self.camera_order)
            if self.visibility_policy == VISIBILITY_PROJECTED
            else None
        )

        self._whitelist_dir: Optional[str] = cfg.get("whitelist_dir")
        self._task_group: str = str(cfg.get("task_group") or "")
        self._whitelist_key: Optional[Tuple[str, str]] = None
        self._bin_edges_subtask: Optional[str] = None

        self._last_seen: Dict[str, int] = {}
        # entity -> whitelist match key, identity-guarded (ids recycle).
        self._match_key_cache: Dict[int, Tuple[Any, Optional[str]]] = {}
        # Last frame's relation-eligible vertex count, for logging only.
        self.last_in_frame: int = 0

    def _uid_for(self, node: Node) -> int:
        return self.uids.uid_for(node.node_id, is_ee=node.node_type == "ee")

    def reset_episode(self) -> None:
        self.selector.reset_episode()
        self.registry.reset_episode()
        self.temporal = TemporalBuffer(K=self.cfg["temporal"]["K"])
        self._last_seen.clear()
        self._match_key_cache.clear()
        self._entities.clear()
        if self._coverage is not None:
            # Reconfiguration destroys the actors the AABB cache describes.
            self._coverage.invalidate()
        self.cfg.setdefault("_affordance_selection_cache", {}).clear()
        self._whitelist_key = None
        self.uids.reset()

    def _bind_global_bin_edges(self, subtask: str) -> None:
        """One relation-bin set per subtask, taken from the union asset.

        Each per-target file calibrates its bins against the scenes that target
        appeared in, so binding them per episode would leave the same relation
        token meaning a different metric distance from one episode to the next.
        The union file is the elementwise maximum of those statistics, so it
        never clips and holds one interpretation for the whole run.

        The asset is the only source of bins -- there is no rule-based
        fallback -- so an asset that does not calibrate an absolute relation
        raises here rather than letting that relation quietly go unlabelled
        for the whole run. cfg["compat_norm"] is untouched.
        """
        if self._bin_edges_subtask == subtask:
            return
        path = resolve_whitelist_path(self._whitelist_dir, subtask, "all")
        if path is None:
            raise FileNotFoundError(
                f"union whitelist not found for subtask={subtask!r} under "
                f"whitelist_dir={self._whitelist_dir!r}; it supplies the global "
                "relation bins. Build it with tools/build_union_whitelist.py "
                "(prepare_assets runs it for you)."
            )
        union = load_whitelist(path)
        self._check_task_group(union, path)
        bin_edges = dict(union.bin_edges or {})
        missing = [r for r in REQUIRED_BIN_RELATIONS if not bin_edges.get(r)]
        if missing:
            raise ValueError(
                f"union whitelist {path!r} calibrates no bins for "
                f"{', '.join(missing)}; those relations would emit nothing for "
                "the whole run. Re-mine the whitelists against the task being "
                "run with tools/prepare_assets.py."
            )
        self.cfg["bin_edges"] = bin_edges
        self._bin_edges_subtask = subtask

    def _check_task_group(self, whitelist, path: str) -> None:
        """Refuse an asset mined against a different MS-HAB task.

        The whitelist directory is already selected by group, so this only
        fires on a file copied into the wrong tree -- which is exactly the case
        no other check catches: it parses, validates and names plausible
        furniture, just not the furniture this task has.
        """
        if not self._task_group:
            return
        if whitelist.task_group != self._task_group:
            raise ValueError(
                f"whitelist {path!r} was mined for task group "
                f"{whitelist.task_group or '<none>'!r} but the run is "
                f"{self._task_group!r}. Re-mine the group with "
                "tools/prepare_assets.py rather than copying files between "
                "task trees."
            )

    def _bind_task_whitelist(self) -> None:
        """Targetless binding: one ``task_all.json`` for the whole env.

        Normal ManiSkill names no subtask target, so membership is task-level
        and the same file supplies the global relation bins.
        """
        self._bind_global_bin_edges(TASK_LEVEL_SUBTASK)
        key = (TASK_LEVEL_SUBTASK, TASK_LEVEL_TARGET)
        if self._whitelist_key == key and self.selector.whitelist is not None:
            return
        path = resolve_whitelist_path(
            self._whitelist_dir, TASK_LEVEL_SUBTASK, TASK_LEVEL_TARGET,
        )
        if path is None:
            raise FileNotFoundError(
                f"task-level whitelist not found under "
                f"{self._whitelist_dir!r}; expected "
                f"{TASK_LEVEL_SUBTASK}_{TASK_LEVEL_TARGET}.json. Mine it with "
                "tools/collect_maniskill_interactions.py."
            )
        whitelist = load_whitelist(path)
        self._check_task_group(whitelist, path)
        self.selector.set_whitelist(whitelist)
        self._whitelist_key = key

    def _resolve_and_bind_whitelist(self, state) -> None:
        """Bind the whitelist for this episode's (subtask, target).

        Bound once per episode and pinned: the vertex set, the appearance cache
        and the edge history all carry one target's membership, so a change of
        target has no meaning to transfer and raises instead.
        """
        subtask = state.active_subtask_type
        if subtask is None:
            raise RuntimeError(
                "whitelist selection requires an active subtask type; got "
                "None. Probe must run inside an MS-HAB-like env."
            )
        self._bind_global_bin_edges(subtask)
        if state.active_handle_link is not None:
            target = stable_entity_key(state.active_handle_link)
        else:
            canonical = (
                canonical_affordance_key(state.active_obj_id)
                if state.active_obj_id else None
            )
            target = f"actor:{canonical}" if canonical else None
        if target is None:
            raise RuntimeError(
                "whitelist selection requires a target key; got "
                f"active_obj_id={state.active_obj_id!r}, "
                f"active_handle_link={state.active_handle_link!r}. Probe "
                "must run inside an MS-HAB-like env."
            )
        key = (subtask, target)
        if self._whitelist_key is not None and self._whitelist_key != key:
            raise RuntimeError(
                f"whitelist target changed mid-episode: {self._whitelist_key} "
                f"-> {key}. One episode binds one target."
            )
        if self._whitelist_key == key and self.selector.whitelist is not None:
            return
        self.cfg.setdefault("_affordance_selection_cache", {}).clear()
        path = resolve_whitelist_path(self._whitelist_dir, subtask, target)
        if path is None:
            raise FileNotFoundError(
                f"per-subtask whitelist not found for subtask={subtask!r}, "
                f"target={target!r} under whitelist_dir={self._whitelist_dir!r}. "
                "Mine this task group with tools/prepare_assets.py."
            )
        whitelist = load_whitelist(path)
        self._check_task_group(whitelist, path)
        self.selector.set_whitelist(whitelist)
        self._whitelist_key = key

    def _entity_admitted(self, entity) -> bool:
        """Early whitelist gate for build_nodes, matching apply_whitelist.

        Skips entities whose match key is absent from the whitelist, which is
        exactly what apply_whitelist drops.
        """
        wl = self.selector.whitelist
        if wl is None:
            return True
        hit = self._match_key_cache.get(id(entity))
        if hit is not None and hit[0] is entity:
            key = hit[1]
        else:
            key = entity_match_key(entity)
            self._match_key_cache[id(entity)] = (entity, key)
        return wl.contains(key)

    @property
    def _seed_gate(self):
        """Whitelist gate for scene seeding, or None where seeding is wrong.

        Tabletop only: a whitelisted actor is a vertex whether or not it
        rendered, so a role bound to a marker the task hides before sensor
        capture still resolves. Under ``projected_camera`` the scene is a whole
        apartment and coverage, not admissibility, is the question.
        """
        if self.visibility_policy != VISIBILITY_KEEP:
            return None
        return self._entity_admitted

    def _check_capacity(self, nodes: Dict[str, Node], frame: int, state) -> None:
        """Retention makes capacity a configuration fact, not a runtime choice.

        Nothing is evicted any more, so an overflowing scene has no correct
        behaviour left: dropping a vertex would silently delete facts the
        progress target reads. Raise instead, naming what would have to grow.
        """
        objects = [n for n in nodes.values() if n.node_type == "object"]
        capacity = self.registry.n_max - 1
        if len(objects) <= capacity:
            return
        raise RuntimeError(
            f"graph capacity exceeded: {len(objects)} retained objects need "
            f"{len(objects) + 1} rows but n_max={self.registry.n_max} allows "
            f"{capacity} objects plus the end effector. "
            f"env={self.env_id} task={self._task_group or '?'} "
            f"subtask={state.active_subtask_type or '?'} frame={frame}. "
            f"nodes={sorted(n.node_id for n in objects)}. "
            "Raise model.graph.n_max (and e_max with it) or tighten the "
            "whitelist -- retention never evicts."
        )

    def _entity_for(self, node: Node, state):
        """Cached simulator entity for one node, resolved on first sight.

        For a rendered node the association is made while it is still visible
        and kept afterwards; a node seeded from the scene has no pixels ever and
        falls back to matching on its node id. Either way it has to be found:
        without an entity a node's force queries read zero, which would be
        emitted as a confident ``not-holds``.
        """
        ent = self._entities.get(node.node_id)
        if ent is not None:
            return ent
        named = None
        for seg_id in node.segmentation_ids:
            candidate = state.seg_id_map.get(seg_id)
            if candidate is None:
                continue
            if getattr(candidate, "name", None) == node.name:
                self._entities[node.node_id] = candidate
                return candidate
            named = named or candidate
        if named is not None:
            self._entities[node.node_id] = named
            return named
        # A node seeded from the scene never rendered, so it has no
        # segmentation ids to look one up with. Its node id is derived from the
        # entity, so match on that instead. One scan per node per episode: the
        # result is cached, and a node that was seen resolved above.
        for candidate in state.seg_id_map.values():
            if candidate is None:
                continue
            if stable_node_id(candidate) == node.node_id:
                self._entities[node.node_id] = candidate
                return candidate
        return None

    def _refresh_live_state(self, nodes: Dict[str, Node], state) -> None:
        """Current simulator pose for every object node, seen or not.

        A retained node's snapshot pose is from the frame it was last seen. For
        anything the gripper is carrying that is simply wrong, so the pose is
        re-read every frame and the snapshot is never trusted for geometry.
        """
        for node in nodes.values():
            if node.node_type != "object":
                continue
            ent = self._entity_for(node, state)
            if ent is None:
                continue
            pose = entity_pose_world_array(ent, self.env_idx)
            if pose is None:
                continue
            node.pose_world = [float(v) for v in np.asarray(pose).reshape(-1)]

    def _apply_visibility(self, nodes: Dict[str, Node], state) -> None:
        """Write ``in_frame`` per the environment's policy.

        ``keep_tabletop``: every retained node stays relational. Tabletop scenes
        are small and fully covered, and an object hidden inside a hole is the
        state the task is about.

        ``projected_camera``: a node is relational when a camera covers the
        space it occupies, whether or not any pixel survived the robot.
        """
        keep = self.visibility_policy == VISIBILITY_KEEP
        for node in nodes.values():
            if keep or node.node_type == "ee" or node.visible:
                node.in_frame = True
            else:
                node.in_frame = self._projects(node, state)

    def _projects(self, node: Node, state) -> bool:
        if self._coverage is None:
            return False
        ent = self._entity_for(node, state)
        if ent is None:
            return False
        return self._coverage.covers(
            ent, self.env_idx, node.pose_world, self.env_idx)

    def step(
        self, obs: dict, frame: int,
        *,
        episode_boundary: bool = False,
        seg_override=None, seg_overrides=None,
        rgb_override=None, camera_override=None, record_camera=None,
        need_masks: bool = True, patch_grid: int = 8,
    ) -> Tuple[Graph, MaskAccumulator, str, np.ndarray]:
        if episode_boundary:
            self.reset_episode()

        state = get_privileged_state(self.env, self.env_idx)

        if self.use_target_flag:
            self._resolve_and_bind_whitelist(state)
        else:
            self._bind_task_whitelist()

        nodes, masks, cam, rgb = build_nodes(
            obs, state,
            camera=self.camera,
            seg_override=seg_override,
            seg_overrides=seg_overrides,
            rgb_override=rgb_override,
            camera_override=camera_override,
            record_camera=record_camera,
            camera_order=self.camera_order,
            need_masks=need_masks,
            patch_grid=patch_grid,
            appearance=self.appearance_enabled,
            bbox=self.bbox_enabled,
            # Recording paths keep full masks/nodes for overlays; the training
            # hot path skips node construction for never-admissible entities.
            admit=None if need_masks else self._entity_admitted,
            seed_scene=self._seed_gate,
        )

        # Whitelist admission comes first. Optional episode history can then
        # reinsert a previously seen node; with history disabled, only nodes
        # observed by at least one camera this frame continue below.
        active_target_node_id: Optional[str] = None
        if self.use_target_flag and state.active_obj is not None:
            # Leave the goal unflagged if active-object resolution fell back to
            # the merged MS-HAB handle itself. Its node id is like
            # ``actor:obj_0`` and matches no visible segmentation node, so
            # flagging it would name a vertex that does not exist.
            active_obj_merged = getattr(state, "active_obj_merged", None)
            resolution_fell_back = (
                active_obj_merged is not None
                and state.active_obj is active_obj_merged
            )
            if not resolution_fell_back:
                try:
                    active_target_node_id = stable_node_id(state.active_obj)
                except Exception:
                    active_target_node_id = None
        nodes = self.selector.apply_whitelist(nodes)
        nodes = self.selector.merge_retained(nodes, frame)
        self._refresh_live_state(nodes, state)
        self._apply_visibility(nodes, state)

        for nid, n in nodes.items():
            if n.node_type != "ee" and n.visible:
                self._last_seen[nid] = frame

        self._check_capacity(nodes, frame, state)
        nodes = self.registry.assign(nodes)

        ordered = sorted(nodes.values(), key=lambda n: n.index)

        graph = Graph(
            frame=frame,
            env_id=self.env_id,
            camera=cam,
            nodes=ordered,
            meta=dict(
                is_mshab=state.is_mshab,
                active_subtask=state.active_subtask_type,
                active_obj_id=state.active_obj_id,
                active_target_node_id=active_target_node_id,
                node_uids=(
                    {n.node_id: self._uid_for(n) for n in ordered}
                    if self.uids_enabled
                    else {}
                ),
                n_objects=sum(1 for n in ordered if n.node_type == "object"),
                n_visible=sum(1 for n in ordered if n.visible),
                n_in_frame=sum(1 for n in ordered if n.in_frame),
            ),
        )

        self.last_in_frame = graph.meta["n_in_frame"]
        build_absolute_edges(graph, state, self.cfg)
        self.temporal.annotate(graph, self.cfg)
        self.selector.commit(nodes, frame)
        return graph, masks, cam, rgb

