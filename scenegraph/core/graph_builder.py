"""Per-frame vertex maintenance and fact orchestration.

Pipeline: build_nodes -> apply_whitelist -> merge_persistent
-> registry.assign -> absolute facts -> retained facts -> temporal labels.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .affordance import canonical_affordance_key
from .entity_identity import stable_entity_key, stable_node_id
from .schema import Edge, Graph, Node
from .node_builder import build_nodes
from .relation_rules import build_absolute_edges
from .temporal_buffer import TemporalBuffer
from .mask_extractor import MaskAccumulator
from .selector import EntityRegistry, NodeSelector
from .whitelist import entity_match_key, load_whitelist, resolve_whitelist_path
from ..adapters.privileged_state import get_privileged_state

# A node that left the view keeps only its most recently observed
# object--object physical state. Spatial and affordance facts need current
# perceptual evidence and are omitted instead.
_STALE_REPLAY_RELATIONS = frozenset({"contact", "support", "contain"})


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
        staleness_enabled: bool = False,
    ):
        self.env = env
        self.cfg = cfg
        self.env_idx = env_idx
        self.env_id = env_id
        self.camera = camera
        self.camera_order = list(camera_order) if camera_order else None
        self.staleness_enabled = bool(staleness_enabled)

        self.temporal = TemporalBuffer(K=cfg["temporal"]["K"])
        self.selector = NodeSelector(cfg)
        self.registry = EntityRegistry(n_max=int(cfg["selection"]["n_max"]))
        self.cfg.setdefault("_affordance_selection_cache", {})

        self._whitelist_dir: Optional[str] = cfg.get("whitelist_dir")
        self._whitelist_key: Optional[Tuple[str, str]] = None
        self._bin_edges_subtask: Optional[str] = None

        self._last_seen: Dict[str, int] = {}
        self._first_unseen: Dict[str, int] = {}
        # Last observed fact per (src,dst,relation) -- replayed while an
        # endpoint is out of view.
        self._edge_history: Dict[Tuple[str, str, str], Edge] = {}
        # entity -> whitelist match key, identity-guarded (ids recycle).
        self._match_key_cache: Dict[int, Tuple[Any, Optional[str]]] = {}

    def reset_episode(self) -> None:
        self.selector.reset_episode()
        self.registry.reset_episode()
        self.temporal = TemporalBuffer(K=self.cfg["temporal"]["K"])
        self._last_seen.clear()
        self._first_unseen.clear()
        self._edge_history.clear()
        self._match_key_cache.clear()
        self.cfg.setdefault("_affordance_selection_cache", {}).clear()
        self._whitelist_key = None

    def _bind_global_bin_edges(self, subtask: str) -> None:
        """One relation-bin set per subtask, taken from the union asset.

        Each per-target file calibrates its bins against the scenes that target
        appeared in, so binding them per episode would leave the same relation
        token meaning a different metric distance from one episode to the next.
        The union file is the elementwise maximum of those statistics, so it
        never clips and holds one interpretation for the whole run.

        cfg["profile"] stays the fallback for relations the asset omits, and
        cfg["compat_norm"] is untouched.
        """
        if self._bin_edges_subtask == subtask:
            return
        path = resolve_whitelist_path(self._whitelist_dir, subtask, "all")
        if path is None:
            raise FileNotFoundError(
                f"union whitelist not found for subtask={subtask!r} under "
                f"whitelist_dir={self._whitelist_dir!r}; it supplies the global "
                "relation bins. Build it with tools/build_union_whitelist.py."
            )
        self.cfg["bin_edges"] = dict(load_whitelist(path).bin_edges or {})
        self._bin_edges_subtask = subtask

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
                "Mine assets with tools/build_subtask_whitelists.py."
            )
        self.selector.set_whitelist(load_whitelist(path))
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

        self._resolve_and_bind_whitelist(state)

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
            # Recording paths keep full masks/nodes for overlays; the training
            # hot path skips node construction for never-admissible entities.
            admit=None if need_masks else self._entity_admitted,
        )

        # Whitelist admission comes first. Optional episode history can then
        # reinsert a previously seen node; with history disabled, only nodes
        # observed by at least one camera this frame continue below.
        active_target_node_id: Optional[str] = None
        if state.active_obj is not None:
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
        if self.staleness_enabled:
            nodes = self.selector.merge_persistent(nodes, frame)

        for nid, n in nodes.items():
            if n.node_type == "ee":
                continue
            if n.visible:
                self._last_seen[nid] = frame
                n.steps_since_seen = 0
            elif nid not in self._last_seen:
                first = self._first_unseen.setdefault(nid, frame)
                n.steps_since_seen = max(1, frame - first + 1)
            else:
                n.steps_since_seen = frame - self._last_seen[nid]

        # With history off the vertex set is exactly this frame, so slots held
        # for absent objects are slots nothing can use -- and nothing else
        # frees them, because commit/evict_expired are both skipped below.
        # Capacity has to describe what the cameras can see.
        if not self.staleness_enabled:
            self.registry.retain(nodes.keys())

        # History-off graphs protect the target only while its exact instance
        # is present in the current observation. If it later reappears, the
        # registry force-admits it by evicting a non-target.
        protected_target = (
            active_target_node_id
            if active_target_node_id is not None and active_target_node_id in nodes
            else None
        )
        nodes = self.registry.assign(nodes, protected_id=protected_target)

        # k_persist=-1 must not inject an overflow-evicted old instance again
        # on the next frame and displace one of the newer residents.
        overflow_evicted = list(self.registry.evicted_ids)
        if overflow_evicted:
            self.selector.evict(overflow_evicted)

        expired = self.selector.evict_expired(frame)
        purged = list(dict.fromkeys([*overflow_evicted, *expired]))
        if purged:
            self.temporal.purge(purged)
        for nid in purged:
            if nid in expired:
                self.registry.release(nid)
            self._last_seen.pop(nid, None)
            self._first_unseen.pop(nid, None)
            for key in [k for k in self._edge_history if nid in k[:2]]:
                del self._edge_history[key]

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
                n_objects=sum(1 for n in ordered if n.node_type == "object"),
                n_visible=sum(1 for n in ordered if n.visible),
            ),
        )

        build_absolute_edges(graph, state, self.cfg)
        if self.staleness_enabled:
            self._attach_stale_edges(graph, frame)
        self.temporal.annotate(graph, self.cfg)

        if self.staleness_enabled:
            self.selector.commit(nodes, frame)
        return graph, masks, cam, rgb

    def _attach_stale_edges(self, graph: Graph, frame: int) -> None:
        """Cache observed object--object physical facts and replay the last one
        for pairs whose endpoint left the view. Both polarities are retained so
        a later negative-to-positive transition stays legible."""
        by_id = {n.node_id: n for n in graph.nodes}
        visible_objects = {
            nid for nid, n in by_id.items()
            if n.node_type == "object" and n.visible
        }

        for edge in graph.edges:
            if edge.stale or edge.relation not in _STALE_REPLAY_RELATIONS:
                continue
            if edge.src in visible_objects and edge.dst in visible_objects:
                key = (edge.src, edge.dst, edge.relation)
                self._edge_history[key] = replace(
                    edge, stale=False, observed_frame=frame, age=0,
                )

        existing = {(e.src, e.dst, e.relation) for e in graph.edges}
        for key, cached in self._edge_history.items():
            if key in existing:
                continue
            if cached.src not in by_id or cached.dst not in by_id:
                continue
            if cached.src in visible_objects and cached.dst in visible_objects:
                continue
            observed = cached.observed_frame
            age = max(1, frame - observed) if observed is not None else 1
            graph.edges.append(replace(cached, stale=True, age=age))
