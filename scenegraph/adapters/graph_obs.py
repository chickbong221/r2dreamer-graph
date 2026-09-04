"""Online graph plumbing shared by every trainer.

Owns one ``GraphBuilder`` per parallel env, one frozen DINO encoder, and one
per-camera appearance cache, and turns each per-env ``Graph`` into a
fixed-shape batched tensor dict.

Segmentation and RGB are sliced from the raw observation that
``NamedCameraRGBWrapper`` stashes on its way past, rather than re-fetched per
env. MS-HAB's ``BaseEnv._last_obs`` carries only the state half, and calling
``env.unwrapped.get_obs()`` would rerun ``get_info`` -> MS-HAB ``evaluate``,
which mutates ``subtask_pointer`` / ``subtask_steps_left`` / cumulative force,
and would also re-render + CUDA-sync once per env.
"""

from __future__ import annotations

from copy import copy as _shallow_copy
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from .privileged_state import (
    begin_frame_cache,
    clear_privileged_state_caches,
    clear_resolve_cache,
    end_frame_cache,
    looks_like_mshab,
    purge_scene_caches,
    set_merged_view_aliasing,
)
from .graph_pack import graph_keys, pack_graph
from .graph_vocab import GraphVocab, build_graph_vocab
from ..configs.loader import load_config as load_teemo_config
from ..core.graph_builder import GraphBuilder

# ManiSkill caches one native PhysX GPU query per (obj1.name + obj2.name) pair
# and never evicts. MS-HAB partial resets recreate merged actors under fresh
# names, so stale entries accumulate (~122 KB each). Queries rebuild lazily on
# the next contact lookup, so clearing past this cap is always safe.
_CONTACT_QUERY_CAP = 2048

# The privileged-state caches key on id(entity) and hold the entity, so a stale
# key pins a dead actor. Their legitimate size is entities x envs; the scene
# signature only watches scene.actors, which recreated merged actors never
# enter, so this cap is the only thing that bounds them.
_SCENE_CACHE_CAP = 8192

_DTYPES: Dict[str, np.dtype] = {
    "graph_node_ent": np.uint8,
    "graph_node_bbox": np.float16,
    "graph_node_centroid": np.float32,
    "graph_node_target": np.uint8,
    "graph_edge_src": np.uint8,
    "graph_edge_dst": np.uint8,
    "graph_edge_rel": np.uint8,
    "graph_edge_abs": np.uint8,
    "graph_edge_temp": np.uint8,
}


def _verify_whitelist_coverage(
    env, whitelist_dir: str, task_group: str,
) -> None:
    """Fail at startup if any object-target plan lacks a mined whitelist.

    Catches the common split mismatch early, for example training with
    train-mined whitelists while eval uses val task plans. Only pick/place are
    checked because their runtime target is exactly actor:<obj>; open and close
    bind through live handle links and fail loudly at runtime instead.

    Every file found is also read back and its recorded ``task_group`` checked
    against the run's. Selecting the directory by group already isolates the
    groups, so this catches the remaining case: a file hand-copied into the
    wrong tree, which would otherwise load cleanly and describe furniture the
    run never sees.
    """
    from ..core.affordance import canonical_affordance_key
    from ..core.whitelist import load_whitelist, resolve_whitelist_path

    base = getattr(env, "unwrapped", env)
    plans_by_bci = getattr(base, "build_config_idx_to_task_plans", None)
    if plans_by_bci is None:
        return

    groups = (
        plans_by_bci.values() if hasattr(plans_by_bci, "values") else plans_by_bci
    )
    missing = set()
    mislabelled = []
    checked = set()
    for plans in groups:
        for plan in plans:
            for subtask in getattr(plan, "subtasks", []) or []:
                st_type = getattr(subtask, "type", None)
                if st_type not in {"pick", "place"}:
                    continue
                obj_id = getattr(subtask, "obj_id", None)
                if not obj_id:
                    continue
                key = canonical_affordance_key(str(obj_id))
                if not key:
                    continue
                pair = (str(st_type), key)
                if pair in checked:
                    continue
                checked.add(pair)
                target = f"actor:{key}"
                path = resolve_whitelist_path(whitelist_dir, str(st_type), target)
                if path is None:
                    missing.add(pair)
                    continue
                recorded = load_whitelist(path).task_group
                if recorded != task_group:
                    mislabelled.append((path, recorded))

    if mislabelled:
        listing = ", ".join(
            f"{path} records {recorded or '<none>'!r}"
            for path, recorded in sorted(mislabelled)
        )
        raise ValueError(
            f"graph: {len(mislabelled)} whitelist(s) under {whitelist_dir!r} do "
            f"not belong to task group {task_group!r}: {listing}. Re-mine the "
            "group instead of copying files between task trees."
        )

    if missing:
        listing = ", ".join(f"{st}:{key}" for st, key in sorted(missing))
        raise FileNotFoundError(
            f"graph: {len(missing)} object-target whitelist(s) missing under "
            f"{whitelist_dir!r}: {listing}. Mine them with "
            f"'python -m scenegraph.tools.prepare_assets --mshab-task "
            f"{task_group} --subtask pick' for the active mshab_split/"
            "mshab_eval_split before training."
        )


class GraphObsBuilder:
    """One GraphBuilder per env. Emits packed batched arrays per frame."""

    def __init__(
        self,
        env,
        *,
        num_envs: int,
        teemo_cfg: dict,
        vocab: GraphVocab,
        use_target_flag: bool = True,
        n_max: int,
        e_max: int,
        cameras: List[str],
        sensor_source=None,
        bypass_teemo: bool = False,
        visibility_policy: str = "keep_tabletop",
    ):
        self.env = env
        self.sensor_source = sensor_source
        self.num_envs = int(num_envs)
        self.vocab = vocab
        # Where this run's mined assets came from. A task schedule is compiled
        # against the same entity vocabulary the graph packs, so it has to read
        # the directory that was actually resolved, not re-derive it.
        self.whitelist_dir = str(teemo_cfg.get("whitelist_dir") or "")
        self.task_group = str(teemo_cfg.get("task_group") or "")
        self.use_target_flag = bool(use_target_flag)
        self.n_max = int(n_max)
        self.e_max = int(e_max)
        self.bypass_teemo = bool(bypass_teemo)
        self.visibility_policy = str(visibility_policy)
        self.graph_keys = graph_keys()
        self.cameras = list(cameras)
        if not self.cameras:
            raise ValueError("graph: cameras is empty")
        # Overlay rendering and the offline tools need one frame to draw on.
        # The model reads every camera independently and never sees this.
        self.record_camera = self.cameras[0]
        # Ordinary ManiSkill needs per-sub-scene actors resolved to their
        # merged view or the whitelist rejects every one of them by name.
        # Decided once here, re-applied whenever the scene is rebuilt.
        self._merged_view_aliasing = not looks_like_mshab(env)
        self._apply_merged_view_aliasing()

        self.builders = []
        for i in range(self.num_envs):
            cfg_i = _shallow_copy(teemo_cfg)
            cfg_i["_affordance_selection_cache"] = {}
            self.builders.append(
                GraphBuilder(env, cfg_i, env_idx=i, env_id=f"env{i}",
                             camera=self.record_camera,
                             camera_order=self.cameras,
                             visibility_policy=self.visibility_policy,
                             use_target_flag=self.use_target_flag)
            )
        self._frames = np.zeros(self.num_envs, dtype=np.int64)
        # Last packed arrays per env, re-emitted on terminal frames whose
        # sensors already belong to the next episode.
        self._last_packed: List[Optional[Dict[str, np.ndarray]]] = [
            None for _ in range(self.num_envs)
        ]
        # Env indices whose latest graph + record-cam masks are cached for
        # offline overlays. Graph-only recording is used by trainer videos and
        # avoids constructing masks that their node-link panel never reads.
        self.record_env_indices: Set[int] = set()
        self.record_graph_env_indices: Set[int] = set()
        self.last_graph_by_env: Dict[int, Any] = {}
        self.last_masks_by_env: Dict[int, Any] = {}
        self._cams_checked = False
        # Facts the packer could not seat this frame. Vertex overflow has its
        # own counter; without this one an e_max that is too small drops
        # spatial edges silently for a whole run.
        self._reset_counters()
        self._scene_cache_signature = None
        self._cpu_buffers: Dict[Tuple[str, str], torch.Tensor] = {}

    def _reset_counters(self) -> None:
        """Per-env diagnostics, allocated in one place.

        Each of these silently degrades a run rather than failing it: too small
        an e_max drops spatial edges, and a graph that names no target vertex
        trains ungrounded and looks fine. Kept together so a caller that builds
        the object without ``__init__`` cannot miss one.
        """
        self._fact_drops = np.zeros(self.num_envs, dtype=np.float32)
        self._node_drops = np.zeros(self.num_envs, dtype=np.float32)
        self._target_missing = np.zeros(self.num_envs, dtype=np.float32)
        self._target_unresolved = np.zeros(self.num_envs, dtype=np.float32)

    @property
    def n_cams(self) -> int:
        return len(self.cameras)

    @property
    def obs_spec_shapes(self) -> Dict[str, tuple]:
        """Per-env shapes for each graph key of the active contract."""
        shapes = {
            "graph_node_ent":  (self.n_max,),
            "graph_node_bbox": (self.n_max, self.n_cams, 4),
            "graph_node_centroid": (self.n_max, 3),
            "graph_node_target": (self.n_max,),
            "graph_edge_src":  (self.e_max,),
            "graph_edge_dst":  (self.e_max,),
            "graph_edge_rel":  (self.e_max,),
            "graph_edge_abs":  (self.e_max,),
            "graph_edge_temp": (self.e_max,),
        }
        return {key: shapes[key] for key in self.graph_keys}

    @property
    def obs_spec_dtypes(self) -> Dict[str, np.dtype]:
        return {key: _DTYPES[key] for key in self.graph_keys}

    @property
    def in_frame_nodes(self) -> np.ndarray:
        """Per-env count of nodes relations may be emitted for.

        Under ``keep_tabletop`` this equals the vertex count. Under
        ``projected_camera`` the gap between this and the vertex count is what
        the projection is actually excluding, which is the only way to notice a
        camera matrix that has gone wrong.
        """
        return np.array(
            [float(b.last_in_frame) for b in self.builders], np.float32)

    @property
    def episode_entities(self) -> np.ndarray:
        """Per-env distinct object instances presented this episode.

        Under retention this equals live occupancy. Reaching ``n_max - 1`` is
        the last reading before the run raises on capacity, so it is the number
        to size ``n_max`` against.
        """
        return np.array(
            [b.registry.episode_entities for b in self.builders], np.float32)

    @property
    def fact_drops(self) -> np.ndarray:
        """Per-env facts the packer could not seat in the last packed frame."""
        return self._fact_drops.copy()

    @property
    def node_drops(self) -> np.ndarray:
        """Per-env vertices the packer could not seat in the last frame.

        The pooled schema reserves row 1 for the subtask target, so a frame
        with more visible whitelisted objects than remaining rows loses one.
        Reserving the row is deliberate; losing a vertex without saying so is
        not, which is the whole reason this counter exists.
        """
        return self._node_drops.copy()

    @property
    def target_missing(self) -> np.ndarray:
        """Per-env 1.0 where the last packed frame flagged no target vertex."""
        return self._target_missing.copy()

    @property
    def target_unresolved(self) -> np.ndarray:
        """Per-env 1.0 where the builder never named a target at all.

        The complement of ``target_missing`` is the other failure: a target was
        named but no vertex carried its node id.
        """
        return self._target_unresolved.copy()

    def cache_stats(self) -> Dict[str, int]:
        """Sizes of every container that could grow without bound, for leak
        triage: a linear counter here names the leak directly."""
        stats: Dict[str, int] = {}
        scene = getattr(self.env.unwrapped, "scene", None)
        if scene is not None:
            d = getattr(scene, "__dict__", {})
            for key in (
                "_teemo_sidxs_cache", "_teemo_sliced_views",
                "_teemo_row_sliced_views", "_teemo_resolve_cache",
                "_teemo_per_env_seg_maps",
            ):
                v = d.get(key)
                if v is not None:
                    stats[key.replace("_teemo_", "")] = len(v)
            for key in ("pairwise_contact_queries", "actor_views"):
                v = getattr(scene, key, None)
                if v is not None:
                    stats[key] = len(v)
        stats["match_key"] = sum(len(b._match_key_cache) for b in self.builders)
        stats["registry"] = sum(len(b.registry) for b in self.builders)
        stats["temporal_values"] = sum(
            len(b.temporal._values) for b in self.builders
        )
        return stats

    @property
    def cache_entries(self) -> int:
        """Live entries across every container that outlives an episode.

        A rise here that does not level off is a leak; ``cache_stats`` then
        names which container.
        """
        return int(sum(self.cache_stats().values()))

    def _zero_pack(self) -> Dict[str, np.ndarray]:
        return {
            k: np.zeros(shape, dtype=_DTYPES[k])
            for k, shape in self.obs_spec_shapes.items()
        }

    def _build_one(
        self, env_idx: int, episode_boundary: bool,
        seg_by_cam: Dict[str, np.ndarray],
    ):
        need_masks = env_idx in self.record_env_indices
        need_graph = need_masks or env_idx in self.record_graph_env_indices
        if episode_boundary:
            self._frames[env_idx] = 0
            clear_resolve_cache(self.env, env_idx)
        graph, masks, _, _ = self.builders[env_idx].step(
            {},
            int(self._frames[env_idx]),
            episode_boundary=episode_boundary,
            seg_overrides=seg_by_cam,
            rgb_override=None,
            record_camera=self.record_camera,
            need_masks=need_masks,
        )
        if need_graph:
            self.last_graph_by_env[env_idx] = graph
        if need_masks:
            self.last_masks_by_env[env_idx] = masks
        self._frames[env_idx] += 1
        return graph

    def _sensor_data(self):
        source = self.sensor_source
        raw = getattr(source, "raw_obs", None)
        if not isinstance(raw, dict) or "sensor_data" not in raw:
            raise RuntimeError(
                "graph: no stashed observation carrying sensor_data; the "
                "graph path needs NamedCameraRGBWrapper passed as "
                f"sensor_source (source={type(source).__name__}, "
                f"stash={type(raw).__name__})"
            )
        sensor_data = raw["sensor_data"]
        if not self._cams_checked:
            for cam in self.cameras:
                if cam not in sensor_data:
                    raise KeyError(
                        f"graph: camera {cam!r} not in sensor_data "
                        f"(available: {list(sensor_data)}). Check obs_mode and "
                        "sensor configs render this camera."
                    )
                for field in ("segmentation", "rgb"):
                    if field not in sensor_data[cam]:
                        raise KeyError(
                            f"graph: camera {cam!r} has no {field!r} in "
                            f"the observation; obs_mode must include it."
                        )
            self._cams_checked = True
        return sensor_data

    def _read_segmentation(self) -> Dict[str, np.ndarray]:
        """Return ``{cam: [N, H, W]}`` staged through a reused CPU buffer."""
        sensor_data = self._sensor_data()
        out: Dict[str, np.ndarray] = {}
        for cam in self.cameras:
            value = sensor_data[cam]["segmentation"].squeeze(-1).detach()
            key = (cam, "segmentation")
            buf = self._cpu_buffers.get(key)
            if (
                buf is None
                or tuple(buf.shape) != tuple(value.shape)
                or buf.dtype != value.dtype
            ):
                buf = torch.empty(tuple(value.shape), dtype=value.dtype, device="cpu")
                self._cpu_buffers[key] = buf
            buf.copy_(value, non_blocking=False)
            out[cam] = buf.numpy()
        return out

    def _current_scene_signature(self):
        base = self.env.unwrapped
        scene = getattr(base, "scene", None)
        if scene is None:
            return None
        actors = getattr(scene, "actors", {}) or {}
        articulations = getattr(scene, "articulations", {}) or {}
        actor_ids = tuple(sorted(id(a) for a in actors.values()))
        link_ids = []
        for art in articulations.values():
            link_ids.extend(id(link) for link in getattr(art, "links", []) or [])
        return (id(scene), actor_ids, tuple(sorted(link_ids)))

    def _apply_merged_view_aliasing(self) -> None:
        """Resolve per-sub-scene actors to their merged view.

        Without it PegInsertionSide names its actors ``peg_0`` and
        ``box_with_hole_0`` while the mined whitelist says ``actor:peg``
        and ``actor:box_with_hole``, so the hard gate drops both and the
        graph keeps only the end effector and the table.

        MS-HAB is excluded: it resolves the other way, merged handle to
        per-env actual, and a global default would rewrite its
        identities. Re-applied after a reconfigure, because the flag
        lives on the scene object and a rebuilt scene never saw it.
        """
        if not self._merged_view_aliasing:
            return
        set_merged_view_aliasing(self.env, True)

    def _refresh_scene_caches_if_needed(self) -> None:
        sig = self._current_scene_signature()
        if sig is None:
            return
        if self._scene_cache_signature is None:
            self._scene_cache_signature = sig
            return
        if sig == self._scene_cache_signature:
            return
        clear_privileged_state_caches(self.env)
        self._apply_merged_view_aliasing()
        self._scene_cache_signature = sig

    def _purge_caches(self) -> None:
        """Cap the scene-attached containers that outlive an episode.

        Nothing in the builder survives ``reset_episode``; these live on the
        simulator's scene and only a size cap bounds them. Both rebuild lazily
        on the next lookup.
        """
        purge_scene_caches(self.env, _SCENE_CACHE_CAP)
        scene = getattr(self.env.unwrapped, "scene", None)
        queries = getattr(scene, "pairwise_contact_queries", None)
        if queries is None or len(queries) <= _CONTACT_QUERY_CAP:
            return
        queries.clear()
        hashes = getattr(scene, "_pairwise_contact_query_unique_hashes", None)
        if hashes is not None:
            hashes.clear()

    def step(
        self,
        *,
        is_first: Optional[Sequence[bool]] = None,
        is_last: Optional[Sequence[bool]] = None,
    ) -> Dict[str, np.ndarray]:
        """Pack one frame for every env.

        ``is_first`` drives the per-env episode reset. ``is_last`` marks envs
        whose sensors already belong to the next episode because the vector env
        auto-reset inside ``step``; those re-emit the previous frame's arrays
        rather than a graph built from the wrong episode, and neither pool DINO
        features nor touch their appearance cache.
        """
        first = (
            np.asarray(is_first, dtype=bool).reshape(-1)
            if is_first is not None else np.zeros(self.num_envs, dtype=bool)
        )
        last = (
            np.asarray(is_last, dtype=bool).reshape(-1)
            if is_last is not None else np.zeros(self.num_envs, dtype=bool)
        )
        if first.any():
            self._refresh_scene_caches_if_needed()

        if self.bypass_teemo:
            return self._stack([self._zero_pack() for _ in range(self.num_envs)])

        active = [i for i in range(self.num_envs) if not last[i]]
        graphs: Dict[int, Any] = {}
        if active:
            segs = self._read_segmentation()
            self._purge_caches()
            begin_frame_cache(getattr(self.env.unwrapped, "scene", None))
            try:
                for i in active:
                    graphs[i] = self._build_one(
                        i, bool(first[i]),
                        {cam: segs[cam][i] for cam in self.cameras},
                    )
            finally:
                end_frame_cache()

        packed: List[Dict[str, np.ndarray]] = []
        for i in range(self.num_envs):
            if i in graphs:
                out = pack_graph(
                    graphs[i], self.vocab,
                    n_max=self.n_max, e_max=self.e_max,
                    n_cams=self.n_cams,
                    use_target_flag=self.use_target_flag,
                )
                self._last_packed[i] = out
                self._fact_drops[i] = float(
                    graphs[i].meta.get("n_edges_dropped", 0))
                self._node_drops[i] = float(
                    graphs[i].meta.get("n_nodes_dropped", 0))
                self._target_missing[i] = float(
                    not graphs[i].meta.get("target_packed", False))
                self._target_unresolved[i] = float(
                    not graphs[i].meta.get("target_resolved", False))
            else:
                out = self._last_packed[i] or self._zero_pack()
            packed.append(out)
        return self._stack(packed)

    def _stack(self, packed: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        return {
            k: np.stack([p[k] for p in packed], axis=0).astype(_DTYPES[k], copy=False)
            for k in self.graph_keys
        }

    def reset(self) -> Dict[str, np.ndarray]:
        self._frames[:] = 0
        self._last_packed = [None for _ in range(self.num_envs)]
        return self.step(is_first=np.ones(self.num_envs, dtype=bool))


def build_graph_obs(
    env,
    graph_cfg: dict,
    *,
    num_envs: int,
    sensor_source=None,
    builder_cls: type = GraphObsBuilder,
) -> Optional[GraphObsBuilder]:
    """Return a builder or None when graph obs is disabled."""
    if not bool(graph_cfg.get("enabled", False)):
        return None

    # Unset paths arrive from elements.Config as "", which load_config reads as
    # "use the packaged thresholds".
    task_group = str(graph_cfg.get("mshab_task") or "")
    if not task_group:
        raise ValueError(
            "graph: mshab_task is empty; the affordance and whitelist assets "
            "are mined per MS-HAB task and the adapter cannot pick a group "
            "for the run"
        )
    teemo_cfg = load_teemo_config(
        path=graph_cfg.get("thresholds_path"), task_group=task_group,
    )
    if "n_max" in graph_cfg:
        teemo_cfg["selection"]["n_max"] = int(graph_cfg["n_max"])
    if graph_cfg.get("whitelist_dir"):
        teemo_cfg["whitelist_dir"] = graph_cfg["whitelist_dir"]
    use_target_flag = bool(graph_cfg.get("use_target_flag", True))
    teemo_cfg["use_target_flag"] = use_target_flag
    # Named per environment, not inferred: MS-HAB and normal ManiSkill want
    # different graphs here and neither should get the other's by default.
    teemo_cfg["object_object_spatial"] = bool(
        graph_cfg.get("object_object_spatial", False))
    teemo_cfg["disable_object_object_relations"] = bool(
        graph_cfg.get("disable_object_object_relations", False))
    if teemo_cfg.get("whitelist_dir") is None:
        raise ValueError(
            "graph: whitelist_dir is not set in the loaded config; set "
            "graph.whitelist_dir or configure scenegraph/configs/thresholds.yaml."
        )

    _verify_whitelist_coverage(env, teemo_cfg["whitelist_dir"], task_group)
    vocab = build_graph_vocab(teemo_cfg["whitelist_dir"])
    if vocab.sizes["entity"] > np.iinfo(np.uint8).max + 1:
        raise ValueError(
            f"graph: entity vocabulary has {vocab.sizes['entity']} entries; "
            "the compact PyTorch runtime supports at most 256"
        )

    # The model sizes nn.Embedding from model.graph.entity_vocab, which is a
    # static config value. Re-mining a task group changes how many entities
    # exist, and an id past the table's end is an out-of-range CUDA gather:
    # it surfaces as a device-side assert inside the RSSM, several calls after
    # the real fault. Compare the two here, where the numbers are still named.
    declared = int(graph_cfg.get("entity_vocab", 0) or 0)
    mined = int(vocab.sizes["entity"])
    if declared and declared < mined:
        raise ValueError(
            f"graph: model.graph.entity_vocab={declared} is smaller than the "
            f"{mined} entities mined for task group {task_group!r} under "
            f"{teemo_cfg['whitelist_dir']!r}. Set model.graph.entity_vocab="
            f"{mined} (or re-mine with a narrower membership policy); leaving "
            "it short makes the entity embedding fail on the first act."
        )

    n_max = int(teemo_cfg["selection"]["n_max"])
    e_max = int(graph_cfg.get("e_max", 256))

    cameras = graph_cfg.get("cameras")
    if not cameras:
        cameras = [graph_cfg.get("camera", "fetch_head")]

    return builder_cls(
        env,
        num_envs=num_envs,
        teemo_cfg=teemo_cfg,
        vocab=vocab,
        use_target_flag=use_target_flag,
        n_max=n_max,
        e_max=e_max,
        cameras=list(cameras),
        sensor_source=sensor_source,
        bypass_teemo=bool(graph_cfg.get("bypass_teemo", False)),
        visibility_policy=str(graph_cfg["visibility_policy"]),
    )
