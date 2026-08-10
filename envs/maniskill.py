"""GPU-vectorized MS-HAB adapter for the independent PyTorch Dreamer.

Only the PickSubtask runtime is included: named RGB cameras, non-privileged
state, frozen instructions, and the nine compact scene-graph observations.
MS-HAB and ManiSkill remain external simulator dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path.resolve()


def _select_build_configs(task_plans, count: int):
    if count <= 0:
        return list(task_plans)
    names = sorted({plan.build_config_name for plan in task_plans})
    keep = set(names[:count])
    selected = [plan for plan in task_plans if plan.build_config_name in keep]
    print(
        f"[env] using {min(count, len(names))}/{len(names)} build configs "
        f"({len(selected)}/{len(task_plans)} plans)",
        flush=True,
    )
    return selected


class ManiSkillVecEnv:
    """OnlineTrainer-compatible wrapper around one ManiSkill GPU vector env."""

    _nonfinite_seen = False

    def __init__(self, config: Any):
        import mani_skill.envs  # noqa: F401 - register ManiSkill tasks
        import mshab.envs  # noqa: F401 - register MS-HAB tasks
        from mani_skill import ASSET_DIR
        from mani_skill.utils import gym_utils
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
        from mshab.envs.planner import plan_data_from_file
        from mshab.envs.wrappers import FetchActionWrapper

        from envs.instruction import InstructionReader, InstructionTable
        from envs.maniskill_obs import NamedCameraRGBWrapper, NonPrivilegedObsWrapper
        from scenegraph.adapters.graph_obs import build_graph_obs

        self._num_envs = int(config.env_num)
        self._device = torch.device(config.device)
        self._seed = int(config.seed)
        self._camera_names = list(config.cameras)
        if not self._camera_names:
            raise ValueError("MS-HAB needs at least one named camera")
        self._camera_keys = {
            "image_" + camera.rsplit("_", 1)[-1]: camera
            for camera in self._camera_names
        }

        task = str(config.task).split("_", 1)[1]
        subtask = task.split("SubtaskTrain", 1)[0].lower()
        split = str(config.split)
        rearrange = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
        plan_path = (
            rearrange
            / "task_plans"
            / str(config.mshab_task)
            / subtask
            / split
            / f"{config.mshab_obj}.json"
        )
        plan_data = plan_data_from_file(plan_path)
        task_plans = _select_build_configs(
            plan_data.plans, int(config.num_build_configs)
        )
        if not task_plans:
            raise ValueError(f"MS-HAB task selection produced no plans: {plan_path}")

        size = tuple(map(int, config.size))
        make_kwargs = dict(
            id=task,
            obs_mode=str(config.obs_mode),
            render_mode="rgb_array",
            sensor_configs=dict(width=size[1], height=size[0]),
            # Allocated per scene whatever the render mode is; nothing here renders.
            human_render_camera_configs=dict(width=128, height=128),
            num_envs=self._num_envs,
            sim_backend=str(config.sim_backend),
            task_plans=task_plans,
            scene_builder_cls=plan_data.dataset,
            spawn_data_fp=(
                rearrange
                / "spawn_data"
                / str(config.mshab_task)
                / subtask
                / split
                / "spawn_data.pt"
            ),
            require_build_configs_repeated_equally_across_envs=False,
            reward_mode=str(config.reward_mode),
            max_episode_steps=int(config.time_limit),
            shader_dir=str(config.shader_dir),
        )
        control_mode = str(config.control_mode)
        if control_mode:
            make_kwargs["control_mode"] = control_mode
        sim_config = OmegaConf.to_container(config.sim_config, resolve=True)
        if sim_config:
            make_kwargs["sim_config"] = sim_config
        env = gym.make(**make_kwargs)
        if bool(config.nonprivileged_obs):
            env = NonPrivilegedObsWrapper(env)
        named = NamedCameraRGBWrapper(env, self._camera_keys)
        env = named
        env = FetchActionWrapper(
            env,
            stationary_base=False,
            stationary_torso=False,
            stationary_head=True,
        )
        self._max_episode_steps = gym_utils.find_max_episode_steps_value(env)
        self._env = ManiSkillVectorEnv(
            env, ignore_terminations=True, record_metrics=True
        )

        graph_cfg = {
            "enabled": bool(config.graph.enabled),
            "profile": str(config.graph.profile),
            "thresholds_path": str(config.graph.thresholds_path),
            "whitelist_dir": str(config.graph.whitelist_dir),
            "n_max": int(config.graph.n_max),
            "e_max": int(config.graph.e_max),
            "k_persist": int(config.graph.k_persist),
            "cameras": self._camera_names,
            "app_dim": int(config.graph.app_dim),
            "dino_model": str(config.graph.dino_model),
            "dino_res": int(config.graph.dino_res),
            "dino_weights": str(config.graph.dino_weights),
            "staleness_enabled": bool(config.graph.staleness_enabled),
            "bypass_teemo": bool(config.graph.bypass_teemo),
        }
        for key in ("thresholds_path", "whitelist_dir", "dino_weights"):
            if graph_cfg[key]:
                graph_cfg[key] = str(_repo_path(graph_cfg[key]))
        self._graph = build_graph_obs(
            self._env,
            graph_cfg,
            num_envs=self._num_envs,
            sensor_source=named,
        )

        instruction_path = _repo_path(str(config.instruction_table))
        self._instruction = InstructionReader(
            self._env, InstructionTable(instruction_path), self._num_envs
        )

        obs, _ = self._env.reset(seed=self._seed)
        obs = self._obs_to_dict(obs)
        self._graph_obs = self._graph.reset() if self._graph is not None else {}
        self._instruction_obs = self._instruction.step()
        self._observation_space = self._build_observation_space(obs)
        action_dim = int(self._env.action_space.shape[-1])
        self._action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(action_dim,), dtype=np.float32
        )

    @property
    def env_num(self):
        return self._num_envs

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def _build_observation_space(self, obs):
        state = self._extract_state(obs)
        spaces = {
            "reward": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, shape=(1,), dtype=np.bool_),
            "is_last": gym.spaces.Box(0, 1, shape=(1,), dtype=np.bool_),
            "is_terminal": gym.spaces.Box(0, 1, shape=(1,), dtype=np.bool_),
            "state": gym.spaces.Box(
                -np.inf, np.inf, shape=(state.shape[-1],), dtype=np.float32
            ),
            "instruction": gym.spaces.Box(
                -np.inf,
                np.inf,
                shape=(self._instruction.table.dim,),
                dtype=np.float32,
            ),
        }
        for key in self._camera_keys:
            shape = tuple(obs[key].shape[1:])
            spaces[key] = gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)
        for key in (
            "log_success_once",
            "log_success_at_end",
            "log_fail_once",
            "log_nonfinite_obs",
        ):
            spaces[key] = gym.spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float32
            )
        if self._graph is not None:
            for key in (
                "log_graph_overflow_drops",
                "log_graph_fact_drops",
                "log_graph_target_missing",
                "log_graph_cache_entries",
            ):
                spaces[key] = gym.spaces.Box(
                    -np.inf, np.inf, shape=(1,), dtype=np.float32
                )
            for key, shape in self._graph.obs_spec_shapes.items():
                dtype = self._graph.obs_spec_dtypes[key]
                info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else np.finfo(dtype)
                spaces[key] = gym.spaces.Box(
                    info.min, info.max, shape=shape, dtype=dtype
                )
        return gym.spaces.Dict(spaces)

    @staticmethod
    def _obs_to_dict(obs):
        return {"state": obs} if isinstance(obs, torch.Tensor) else obs

    @staticmethod
    def _extract_state(obs):
        if "state" in obs:
            return obs["state"].float()
        parts = []
        for group in (obs.get("agent", {}), obs.get("extra", {})):
            for value in group.values():
                parts.append(value.reshape(value.shape[0], -1))
        if not parts:
            raise ValueError(f"No state tensor in observation keys {list(obs)}")
        return torch.cat(parts, -1).float()

    @staticmethod
    def _replace_terminal_observation(obs, final, done):
        mask = torch.as_tensor(done, dtype=torch.bool, device=next(iter(obs.values())).device)
        for key, value in list(obs.items()):
            if key not in final or not isinstance(value, torch.Tensor):
                continue
            other = final[key]
            if not isinstance(other, torch.Tensor) or other.shape != value.shape:
                continue
            expand = mask.view(-1, *([1] * (value.ndim - 1))).expand_as(value)
            obs[key] = torch.where(expand, other, value)

    def _episode_logs(self, info, done):
        logs = {}
        final_info = info.get("final_info", {}) if isinstance(info, dict) else {}
        episode = final_info.get("episode", {}) if isinstance(final_info, dict) else {}
        for key in ("success_once", "success_at_end", "fail_once"):
            value = episode.get(key)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().to(self._device, torch.float32)
            else:
                value = torch.as_tensor(value, device=self._device, dtype=torch.float32)
            value = value.reshape(self._num_envs, -1)[:, 0]
            logs[f"log_{key}"] = torch.where(
                torch.as_tensor(done, device=self._device), value, torch.zeros_like(value)
            )
        return logs

    def _clip_nonfinite(self, data):
        """Clip inf/NaN from a diverged scene and count the scenes it hits."""
        checked = {key: ~torch.isfinite(data[key]) for key in ("state", "reward")}
        flags = torch.zeros(self._num_envs, 1, dtype=torch.bool, device=self._device)
        for mask in checked.values():
            flags |= mask.reshape(self._num_envs, -1).any(-1, keepdim=True)
        if not ManiSkillVecEnv._nonfinite_seen and bool(flags.any()):
            ManiSkillVecEnv._nonfinite_seen = True
            for key, mask in checked.items():
                where = torch.nonzero(mask.reshape(self._num_envs, -1))
                if not len(where):
                    continue
                print(
                    f"[env] non-finite {key}: {len(where)} value(s), first in "
                    f"env {int(where[0, 0])} column {int(where[0, 1])}; clipped "
                    "and counted as log_nonfinite_obs",
                    flush=True,
                )
        for key in checked:
            data[key] = torch.nan_to_num(data[key], nan=0.0, posinf=1e6, neginf=-1e6)
        data["log_nonfinite_obs"] = flags.to(torch.float32)
        return data

    def _transition(self, obs, reward, terminated, truncated, is_first, logs=None):
        done = np.asarray(terminated, bool) | np.asarray(truncated, bool)
        data = {
            "reward": torch.as_tensor(reward, device=self._device, dtype=torch.float32).reshape(-1, 1),
            "is_first": torch.as_tensor(is_first, device=self._device, dtype=torch.bool).reshape(-1, 1),
            "is_last": torch.as_tensor(done, device=self._device, dtype=torch.bool).reshape(-1, 1),
            "is_terminal": torch.as_tensor(terminated, device=self._device, dtype=torch.bool).reshape(-1, 1),
            "state": self._extract_state(obs).to(self._device),
            "instruction": torch.as_tensor(self._instruction_obs, device=self._device),
        }
        for key in self._camera_keys:
            data[key] = obs[key].to(self._device, torch.uint8)
        zeros = torch.zeros(self._num_envs, 1, device=self._device)
        for key in ("log_success_once", "log_success_at_end", "log_fail_once"):
            data[key] = zeros.clone()
        if logs:
            for key, value in logs.items():
                data[key] = value.reshape(-1, 1)
        if self._graph is not None:
            for key, value in self._graph_obs.items():
                data[key] = torch.as_tensor(value, device=self._device)
            data["log_graph_overflow_drops"] = torch.as_tensor(
                self._graph.overflow_drops, device=self._device
            ).reshape(-1, 1)
            data["log_graph_fact_drops"] = torch.as_tensor(
                self._graph.fact_drops, device=self._device
            ).reshape(-1, 1)
            data["log_graph_target_missing"] = torch.as_tensor(
                self._graph.target_missing, device=self._device
            ).reshape(-1, 1)
            data["log_graph_cache_entries"] = torch.full(
                (self._num_envs, 1),
                float(self._graph.cache_entries),
                device=self._device,
            )
        self._clip_nonfinite(data)
        return TensorDict(data, batch_size=(self._num_envs,), device=self._device)

    def step(self, action: torch.Tensor, reset: torch.Tensor):
        reset_np = reset.detach().cpu().numpy().astype(bool)
        if reset_np.any():
            if not reset_np.all():
                raise RuntimeError(
                    "MS-HAB vector resets became asynchronous. This compact "
                    "adapter assumes ignore_terminations=True and one shared horizon."
                )
            indices = torch.arange(self._num_envs, device=self._device)
            obs, _ = self._env.reset(options={"env_idx": indices})
            obs = self._obs_to_dict(obs)
            self._graph_obs = (
                self._graph.step(is_first=reset_np) if self._graph is not None else {}
            )
            self._instruction_obs = self._instruction.step()
            zeros = np.zeros(self._num_envs)
            trans = self._transition(obs, zeros, reset_np & False, reset_np & False, reset_np)
            return trans, torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)

        obs, reward, terminated, truncated, info = self._env.step(action.to(self._device))
        obs = self._obs_to_dict(obs)
        done_t = terminated | truncated
        done = done_t.detach().cpu().numpy().astype(bool)
        if done.any() and isinstance(info, dict) and "final_observation" in info:
            final = self._obs_to_dict(info["final_observation"])
            self._replace_terminal_observation(obs, final, done)
        if self._graph is not None:
            self._graph_obs = self._graph.step(is_last=done)
        self._instruction_obs = self._instruction.step(is_last=done)
        trans = self._transition(
            obs,
            reward.detach().cpu().numpy(),
            terminated.detach().cpu().numpy(),
            truncated.detach().cpu().numpy(),
            np.zeros(self._num_envs, bool),
            self._episode_logs(info, done),
        )
        return trans, done_t.to(self._device)

    def close(self):
        self._env.close()
