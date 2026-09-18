"""The online environment, built to match the demonstrations exactly.

Every setting that differs between collection and rollout is a difference the
policy has to absorb, so the ones that matter are read from the dataset's own
metadata rather than restated here: the robot, the controller, the camera set
and resolution, the proprioception field order and the control frequency. A
horizon of 150 and the recorded reward mode come from the config.

``ignore_terminations=True``, matching ``envs/maniskill.py`` and matching the
demonstration loader. Nothing terminates; episodes end at the horizon and the
value function bootstraps there.

The final observation is captured before the reset that follows it. A vector
env that resets on the same step it reports ``done`` hands back the *next*
episode's first observation in the slot where the last one belongs, and a world
model trained on that learns a transition between two unrelated scenes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..data.schema import flatten_proprio, unbatch


class SimVlaEnv:
    """One CPU ManiSkill env producing the dataset's observation contract."""

    def __init__(self, metadata: Mapping[str, Any], *, graph_enabled: bool,
                 max_steps: int = 150, seed: int = 0,
                 record_graphs: bool = False):
        self.metadata = dict(metadata)
        self.graph_enabled = bool(graph_enabled)
        # Recording graphs is not learning from them. A baseline run may set
        # this for diagnostics; the objects stay out of the learning batch.
        self.record_graphs = bool(record_graphs) or self.graph_enabled
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self.env_id = str(metadata["env_id"])
        self.camera_keys = dict(metadata.get("camera_keys") or {})
        self.proprio_spec = [tuple(f) for f in metadata.get("proprio_fields") or []]
        self.proprio_names = list(metadata.get("proprio_names") or [])
        self.image_size = tuple(metadata.get("image_size") or (112, 112))
        controller = dict(metadata.get("controller") or {})
        self.control_mode = str(controller.get("control_mode") or "pd_joint_pos")
        self.action_dim = int(controller.get("action_dim") or 0)
        self._env = None
        self._graphs = None
        self._vocab = None
        self._steps = 0

    # ------------------------------------------------------------------ build
    def build(self):
        import gymnasium as gym                            # noqa: F401
        import mani_skill.envs                             # noqa: F401

        from envs.maniskill import _make_with_supported_reward

        graph_meta = dict(self.metadata.get("graph") or {})
        kwargs: Dict[str, Any] = dict(
            id=self.env_id,
            obs_mode="rgb+segmentation" if self.record_graphs else "rgb",
            control_mode=self.control_mode,
            render_mode="rgb_array",
            sensor_configs=dict(width=int(self.image_size[1]),
                                height=int(self.image_size[0])),
            sim_backend="cpu",
            reward_mode=str(self.metadata.get("reward_mode") or "normalized_dense"),
            max_episode_steps=self.max_steps,
        )
        self._env = _make_with_supported_reward(kwargs, ["sparse"])
        if self.record_graphs:
            from scenegraph.adapters.graph_vocab import build_graph_vocab
            from scenegraph.figures.graph_source import FigureGraphSource

            self._graphs = FigureGraphSource(
                self._env, env_id=self.env_id,
                thresholds_path=str(graph_meta.get("thresholds_path") or ""),
                whitelist_dir=str(graph_meta.get("whitelist_dir") or ""),
                use_target_flag=bool(graph_meta.get("use_target_flag", False)),
                object_object_spatial=bool(
                    graph_meta.get("object_object_spatial", True)),
                visibility_policy=str(
                    graph_meta.get("visibility_policy") or "keep_tabletop"),
            )
            self._vocab = build_graph_vocab(self._graphs.whitelist_dir)
        return self

    # ------------------------------------------------------------ observation
    def _observation(self, raw: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        sensors = raw.get("sensor_data") or {}
        for camera, key in self.camera_keys.items():
            out[key] = np.asarray(unbatch(sensors[camera]["rgb"]), dtype=np.uint8)
        vector, _ = flatten_proprio(raw, self.proprio_spec)
        out["proprio"] = vector
        if self.graph_enabled:
            from scenegraph.adapters.graph_pack import pack_graph

            graph_meta = dict(self.metadata.get("graph") or {})
            packed = pack_graph(
                self._graphs.step(raw), self._vocab,
                n_max=int(graph_meta.get("n_max", 8)),
                e_max=int(graph_meta.get("e_max", 168)),
                n_cams=int(graph_meta.get("n_cams", len(self.camera_keys))),
                use_target_flag=bool(graph_meta.get("use_target_flag", False)))
            out |= packed
        elif self.record_graphs:
            # Built for diagnostics and deliberately not returned: an object in
            # the observation dict is an object that reaches a batch.
            self._graphs.step(raw)
        return out

    # ----------------------------------------------------------------- driving
    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        if self._env is None:
            self.build()
        if self._graphs is not None:
            # Per environment, and independent of the recurrent state reset:
            # temporal edges difference over the last K frames and a builder
            # carried across a reset describes the previous episode.
            self._graphs.on_reset()
        raw, _info = self._env.reset(seed=self.seed if seed is None else int(seed))
        self._steps = 0
        obs = self._observation(raw)
        obs["is_first"] = np.array(True)
        return obs

    def step(self, action: np.ndarray) -> Dict[str, Any]:
        raw, reward, terminated, truncated, info = self._env.step(
            np.asarray(action, dtype=np.float32))
        self._steps += 1
        obs = self._observation(raw)
        obs["is_first"] = np.array(False)
        done = bool(self._steps >= self.max_steps) or bool(
            np.asarray(unbatch(truncated)).reshape(-1)[0])
        return {
            "obs": obs,
            "reward": float(np.asarray(unbatch(reward)).reshape(-1)[0]),
            # ignore_terminations: the recorded flag is kept for diagnostics
            # and is not what ends an episode here.
            "terminated_recorded": bool(
                np.asarray(unbatch(terminated)).reshape(-1)[0]),
            "is_terminal": False,
            "is_last": done,
            "success": bool(np.asarray(
                unbatch(info.get("success", False))).reshape(-1)[0])
            if isinstance(info, dict) else False,
        }

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


def load_metadata(dataset: str | Path) -> Dict[str, Any]:
    """The contract the online env has to match, from the dataset itself."""
    sidecar = Path(dataset).with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    return dict(payload.get("metadata") or {})
