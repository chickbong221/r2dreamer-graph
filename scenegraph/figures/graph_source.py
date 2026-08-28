"""One ``GraphBuilder`` driven by hand, outside the training env stack.

``envs/maniskill.py`` builds the same graph, but only as one part of a
vectorised DreamerV3 observation: it packs to fixed-shape tensors, needs the
whole Hydra config tree, and runs a GPU batch. A figure wants the ``Graph``
object itself, from the single CPU env the motion planner is allowed to drive,
so this is the short path to it.

A graph is built on *every* step even when only a few are exported. Temporal
relation labels difference over the last ``K`` frames, so a builder fed one step
in five would report a change spanning five times the horizon it was mined for,
and the exported chips would be quietly wrong.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..adapters.privileged_state import (
    begin_frame_cache, end_frame_cache, invalidate_scene_caches,
    per_env_segmentation_id_map, set_merged_view_aliasing,
)
from ..configs.loader import load_config
from ..core.entity_identity import stable_node_id
from ..core.graph_builder import GraphBuilder, VISIBILITY_KEEP
from ..core.mask_extractor import extract_camera_obs
from ..core.schema import Graph


class FigureGraphSource:
    """Per-step ``Graph`` for one env, plus the actors its nodes stand for.

    The entity map is what lets a label find its object: the graph carries node
    ids and poses but not simulator handles, and a label needs the handle to ask
    for a collision AABB to project.
    """

    def __init__(
        self,
        env,
        *,
        env_id: str,
        cameras: Optional[Sequence[str]] = None,
        thresholds_path: str = "",
        whitelist_dir: str = "",
        use_target_flag: bool = False,
        object_object_spatial: bool = True,
        visibility_policy: str = VISIBILITY_KEEP,
        env_idx: int = 0,
    ):
        self.env = env
        self.env_id = str(env_id)
        self.env_idx = int(env_idx)
        self.visibility_policy = str(visibility_policy)
        self.use_target_flag = bool(use_target_flag)
        self._cameras = [str(c) for c in (cameras or [])]

        # Mined assets are namespaced by task group, and for an ordinary
        # ManiSkill task the group is the gym id itself -- the same mapping
        # ``envs/maniskill.py`` makes when it passes ``task_group=task``.
        cfg = load_config(thresholds_path or None, task_group=self.env_id)
        cfg["use_target_flag"] = self.use_target_flag
        # Normal ManiSkill emits object-object spatial edges; MS-HAB does not.
        # Named here rather than inferred, matching configs/env/maniskill.yaml.
        cfg["object_object_spatial"] = bool(object_object_spatial)
        cfg["_affordance_selection_cache"] = {}
        if whitelist_dir:
            cfg["whitelist_dir"] = str(whitelist_dir)
        self.cfg = cfg

        self.builder: Optional[GraphBuilder] = None
        self.entities: Dict[str, Any] = {}
        self._frame = 0
        self._boundary = True

    @property
    def whitelist_dir(self) -> str:
        return str(self.cfg.get("whitelist_dir") or "")

    @property
    def cameras(self) -> List[str]:
        """Segmentation cameras in builder order, empty until the first step."""
        return list(self._cameras)

    def on_reset(self) -> None:
        """Re-establish scene-scoped state after every ``env.reset``.

        Aliasing is a scene flag rather than a cache, and a reset that
        reconfigures builds a new scene, so it has to be set again here; the
        caches keyed on the old actors have to go for the same reason.
        """
        set_merged_view_aliasing(self.env, True)
        invalidate_scene_caches(self.env)
        self.entities = {}
        self._frame = 0
        self._boundary = True

    def step(self, obs: dict) -> Graph:
        """Build this frame's graph from the observation's segmentation."""
        cameras = self._resolve_cameras(obs)
        if self.builder is None:
            self.builder = GraphBuilder(
                self.env, self.cfg,
                env_idx=self.env_idx,
                env_id=self.env_id,
                camera=cameras[0],
                camera_order=cameras,
                visibility_policy=self.visibility_policy,
                use_target_flag=self.use_target_flag,
            )
        segmentation = {
            cam: extract_camera_obs(obs, cam, self.env_idx)[1] for cam in cameras
        }
        begin_frame_cache(getattr(self.env.unwrapped, "scene", None))
        try:
            graph, _masks, _cam, _rgb = self.builder.step(
                {}, self._frame,
                episode_boundary=self._boundary,
                seg_overrides=segmentation,
                record_camera=cameras[0],
                # No overlay is drawn on the camera frame -- labels are placed
                # from projected geometry, which does not go dark when the arm
                # occludes the object it names.
                need_masks=False,
            )
        finally:
            end_frame_cache()
        if not self.entities:
            self.entities = self._resolve_entities()
        self._boundary = False
        self._frame += 1
        return graph

    # ------------------------------------------------------------- internals
    def _resolve_cameras(self, obs: dict) -> List[str]:
        """Camera order for this run, fixed on the first observation."""
        if self._cameras:
            return self._cameras
        sensor_data = obs.get("sensor_data") if isinstance(obs, dict) else None
        if not sensor_data:
            raise KeyError(
                "graph: the observation carries no sensor_data; the env has to "
                "be built with an obs_mode that includes segmentation"
            )
        self._cameras = [str(name) for name in sensor_data]
        return self._cameras

    def _resolve_entities(self) -> Dict[str, Any]:
        """``node_id -> actor`` for the current scene.

        Built from the segmentation id map rather than from rendered pixels, so
        an object the arm is hiding this frame still resolves -- the same reason
        ``GraphBuilder`` falls back to matching on node id for a seeded node.
        """
        out: Dict[str, Any] = {}
        seg_map = per_env_segmentation_id_map(self.env, self.env_idx)
        for entity in seg_map.values():
            if entity is None:
                continue
            try:
                node_id = stable_node_id(entity)
            except Exception:                              # noqa: BLE001
                continue
            out.setdefault(node_id, entity)
        return out
