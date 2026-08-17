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

# Packed-UID reservations. Zero is padding so an unfilled slot decodes as
# "no node"; one is the end effector, whose identity never varies.
UID_PAD = 0
UID_EE = 1
_UID_FIRST_OBJECT = 2
# UIDs are packed as uint8, the narrowest dtype holding the vocabulary and the
# one the replay buffer can index. Wider integer types are not universally
# supported there -- torchrl's index_put has no uint16 kernel.
UID_VOCAB_MAX = 256


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
        staleness_enabled: bool = False,
        uid_vocab: int = 256,
        appearance_enabled: bool = True,
    ):
        self.env = env
        self.cfg = cfg
        self.env_idx = env_idx
        self.env_id = env_id
        self.camera = camera
        self.camera_order = list(camera_order) if camera_order else None
        self.staleness_enabled = bool(staleness_enabled)
        self.appearance_enabled = bool(appearance_enabled)
        self.uids = EpisodeUIDs(uid_vocab, seed=1000 + int(env_idx))

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

    def _uid_for(self, node: Node) -> int:
        return self.uids.uid_for(node.node_id, is_ee=node.node_type == "ee")

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
        self.uids.reset()

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
            appearance=self.appearance_enabled,
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
        # Capacity has to describe what the cameras can see, with one exception:
        # the subtask target keeps its registry position once admitted. It is
        # fixed for the episode, the world model has to keep predicting it while
        # it is occluded, and releasing it would let the next arriving object
        # take its slot.
        if not self.staleness_enabled:
            retain_ids = set(nodes.keys())
            if (
                active_target_node_id is not None
                and self.registry.index_of(active_target_node_id) is not None
            ):
                retain_ids.add(active_target_node_id)
            self.registry.retain(retain_ids)

        # Protection is unconditional, not "while visible": an absent retained
        # target must also be ineligible for category-balanced eviction. Before
        # the target is ever admitted this is a no-op, so no slot sits reserved
        # for it; when it first appears it force-admits by evicting a non-target.
        nodes = self.registry.assign(nodes, protected_id=active_target_node_id)

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
                node_uids={n.node_id: self._uid_for(n) for n in ordered},
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
