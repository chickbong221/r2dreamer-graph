"""GPU-vectorized MS-HAB adapter for the independent PyTorch Dreamer.

Only the PickSubtask runtime is included: named RGB cameras, non-privileged
state, frozen instructions, and the nine compact scene-graph observations.
MS-HAB and ManiSkill remain external simulator dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

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


# Every key ``build_graph_obs`` reads. Kept explicit and asserted in tests: a
# key added to the env config but forgotten here silently downgrades the run
# to the previous behaviour instead of failing.
_GRAPH_CONFIG_KEYS = (
    # ``simple`` alone no longer picks a contract: pooled graph-simple emits
    # boxes and no identity code, slot graph-simple the reverse, so the env has
    # to know the state mode too.
    "enabled", "simple", "state_mode", "uid_vocab", "mshab_task", "entity_vocab",
    "use_target_flag", "object_object_spatial",
    "thresholds_path", "whitelist_dir", "n_max", "e_max",
    "app_dim", "dino_model", "dino_res", "dino_weights", "visibility_policy",
    "bypass_teemo",
)

_GRAPH_CONFIG_CASTS = {
    "enabled": bool,
    "use_target_flag": bool,
    "object_object_spatial": bool,
    "simple": bool,
    "visibility_policy": str,
    "bypass_teemo": bool,
    "uid_vocab": int,
    "entity_vocab": int,
    "n_max": int,
    "e_max": int,
    "app_dim": int,
    "dino_res": int,
}


def is_mshab_task(task_id: str) -> bool:
    """MS-HAB subtask envs carry their scene through a ReplicaCAD task plan.

    Ordinary ManiSkill tasks build their own scene from the gym id alone, so
    everything plan-shaped -- spawn data, scene builder, object split -- is
    meaningless for them.
    """
    return "SubtaskTrain" in str(task_id)


def graph_observation_config(graph_config, camera_names,
                             task_group: str = "") -> dict:
    """Flatten the env's graph config into the dict ``build_graph_obs`` reads.

    ``task_group`` overrides ``mshab_task``, which names the mined asset tree.
    For MS-HAB that is the task group (``set_table``) and it comes from config;
    for ordinary ManiSkill it is the gym id, which the config can only spell as
    ``maniskill_PickCube-v1`` and the assets are stored under ``PickCube-v1``.
    """
    out = {
        key: _GRAPH_CONFIG_CASTS.get(key, str)(getattr(graph_config, key))
        for key in _GRAPH_CONFIG_KEYS
    }
    if task_group:
        out["mshab_task"] = str(task_group)
    out["cameras"] = list(camera_names)
    for key in ("thresholds_path", "whitelist_dir", "dino_weights"):
        if out[key]:
            out[key] = str(_repo_path(out[key]))
    return out


def _select_build_configs(task_plans, count: int, label: str = "train"):
    """Keep whole build configs, as a prefix of the sorted names.

    A prefix, not a sample, so the same count always names the same scenes and
    two runs are comparable. It also means there is no way to ask for the
    *complement* of a training subset -- held-out scenes come from a different
    split, not from a different slice of this one.
    """
    if count <= 0:
        print(
            f"[env] using all build configs ({len(task_plans)} plans, {label})",
            flush=True,
        )
        return list(task_plans)
    names = sorted({plan.build_config_name for plan in task_plans})
    keep = set(names[:count])
    selected = [plan for plan in task_plans if plan.build_config_name in keep]
    print(
        f"[env] using {min(count, len(names))}/{len(names)} build configs "
        f"({len(selected)}/{len(task_plans)} plans, {label})",
        flush=True,
    )
    return selected


class ManiSkillVecEnv:
    """OnlineTrainer-compatible wrapper around one ManiSkill GPU vector env."""

    def __init__(self, config: Any, eval: bool = False):
        """``eval=True`` builds the held-out arm of the same task.

        It differs from training in three ways and nothing else: it reads
        ``eval_split`` rather than ``split``, it takes every build config in
        that split rather than a prefix, and it runs to ``eval_time_limit``.
        The observation contract, the graph adapter and the action space are
        constructed identically, so one agent drives both.
        """
        import mani_skill.envs  # noqa: F401 - register ManiSkill tasks
        from mani_skill.utils import gym_utils
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

        from envs.instruction import InstructionReader, InstructionTable
        from envs.maniskill_obs import NamedCameraRGBWrapper, NonPrivilegedObsWrapper
        from scenegraph.adapters.graph_obs import build_graph_obs

        self._eval = bool(eval)
        # The eval loop runs exactly one episode per env, so the env count is
        # the episode count.
        self._num_envs = int(
            config.eval_episode_num if self._eval else config.env_num
        )
        if self._num_envs <= 0:
            raise ValueError(
                "MS-HAB eval was requested with eval_episode_num<=0; that is "
                "the switch that disables it, so no env should be built"
            )
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
        self._task_id = task
        self._is_mshab = is_mshab_task(task)
        size = tuple(map(int, config.size))
        make_kwargs = dict(
            id=task,
            obs_mode=str(config.obs_mode),
            render_mode="rgb_array",
            sensor_configs=dict(width=size[1], height=size[0]),
            num_envs=self._num_envs,
            sim_backend=str(config.sim_backend),
            reward_mode=str(config.reward_mode),
            max_episode_steps=int(
                (getattr(config, "eval_time_limit", 0) or config.time_limit)
                if self._eval
                else config.time_limit
            ),
            shader_dir=str(config.shader_dir),
        )

        if self._is_mshab:
            # Imported here, not at the top: an ordinary ManiSkill run should
            # not need the MS-HAB package installed to build its scene.
            import mshab.envs  # noqa: F401 - register MS-HAB tasks
            from mani_skill import ASSET_DIR
            from mshab.envs.planner import plan_data_from_file

            subtask = task.split("SubtaskTrain", 1)[0].lower()
            # An empty eval_split evaluates on the training scenes, which
            # measures fit rather than generalisation. Left possible on
            # purpose, but it is not the default.
            eval_split = str(getattr(config, "eval_split", "") or "")
            split = eval_split if (self._eval and eval_split) else str(config.split)
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
            # Training subsets its split; evaluation takes the whole of its own
            # by default, so a held-out score is not reported over a sliver.
            task_plans = _select_build_configs(
                plan_data.plans,
                int(
                    getattr(config, "eval_num_build_configs", 0)
                    if self._eval
                    else config.num_build_configs
                ),
                label="eval" if self._eval else "train",
            )
            if not task_plans:
                raise ValueError(
                    f"MS-HAB task selection produced no plans: {plan_path}")
            make_kwargs.update(
                task_plans=task_plans,
                scene_builder_cls=plan_data.dataset,
                spawn_data_fp=(
                    rearrange / "spawn_data" / str(config.mshab_task) / subtask
                    / split / "spawn_data.pt"
                ),
                require_build_configs_repeated_equally_across_envs=False,
            )

        control_mode = str(config.control_mode)
        if control_mode:
            make_kwargs["control_mode"] = control_mode
        sim_config = OmegaConf.to_container(config.sim_config, resolve=True)
        if sim_config:
            make_kwargs["sim_config"] = sim_config
        if self._eval:
            # Eval only, matching ReLDreamer/TD-MPC2. 0 pins the scene set for
            # the whole run instead of rebuilding it per reset, so two
            # evaluations differ by the policy and nothing else.
            make_kwargs["reconfiguration_freq"] = int(
                getattr(config, "eval_reconfiguration_frequency", 0)
            )
        env = gym.make(**make_kwargs)
        if bool(config.nonprivileged_obs):
            env = NonPrivilegedObsWrapper(env)
        named = NamedCameraRGBWrapper(env, self._camera_keys)
        env = named
        if self._is_mshab:
            # Fetch-specific: it pins the mobile base, torso and head joints,
            # none of which a Panda arm has.
            from mshab.envs.wrappers import FetchActionWrapper

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

        self._graph = build_graph_obs(
            self._env,
            graph_observation_config(
                config.graph, self._camera_names,
                task_group="" if self._is_mshab else task,
            ),
            num_envs=self._num_envs,
            sensor_source=named,
        )

        # Optional. An ordinary ManiSkill task has one goal and no language to
        # disambiguate it, so the key is absent from the observation rather
        # than present and constant.
        instruction_path = str(getattr(config, "instruction_table", "") or "")
        self._instruction = (
            InstructionReader(
                self._env,
                InstructionTable(_repo_path(instruction_path)),
                self._num_envs,
            )
            if instruction_path
            else None
        )

        obs, _ = self._env.reset(seed=self._seed)
        obs = self._obs_to_dict(obs)
        self._graph_obs = self._graph.reset() if self._graph is not None else {}
        self._graph_panel_env: Optional[int] = None
        self._instruction_obs = (
            self._instruction.step() if self._instruction is not None else None
        )
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
        }
        if self._instruction is not None:
            spaces["instruction"] = gym.spaces.Box(
                -np.inf, np.inf,
                shape=(self._instruction.table.dim,), dtype=np.float32,
            )
        for key in self._camera_keys:
            shape = tuple(obs[key].shape[1:])
            spaces[key] = gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)
        for key in (
            "log_success_once",
            "log_success_at_end",
            "log_fail_once",
        ):
            spaces[key] = gym.spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float32
            )
        if self._graph is not None:
            for key in (
                "log_graph_in_frame_nodes",
                "log_graph_episode_entities",
                "log_graph_fact_drops",
                "log_graph_node_drops",
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

    def _transition(self, obs, reward, terminated, truncated, is_first, logs=None):
        done = np.asarray(terminated, bool) | np.asarray(truncated, bool)
        data = {
            "reward": torch.as_tensor(reward, device=self._device, dtype=torch.float32).reshape(-1, 1),
            "is_first": torch.as_tensor(is_first, device=self._device, dtype=torch.bool).reshape(-1, 1),
            "is_last": torch.as_tensor(done, device=self._device, dtype=torch.bool).reshape(-1, 1),
            "is_terminal": torch.as_tensor(terminated, device=self._device, dtype=torch.bool).reshape(-1, 1),
            "state": self._extract_state(obs).to(self._device),
        }
        if self._instruction_obs is not None:
            data["instruction"] = torch.as_tensor(
                self._instruction_obs, device=self._device)
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
            # Both counters are cumulative within an episode, so the trainer's
            # per-episode maximum is the episode total.
            data["log_graph_in_frame_nodes"] = torch.as_tensor(
                self._graph.in_frame_nodes, device=self._device
            ).reshape(-1, 1)
            data["log_graph_episode_entities"] = torch.as_tensor(
                self._graph.episode_entities, device=self._device
            ).reshape(-1, 1)
            data["log_graph_fact_drops"] = torch.as_tensor(
                self._graph.fact_drops, device=self._device
            ).reshape(-1, 1)
            # Reserving row 1 for the target costs a vertex in a frame with
            # more visible objects than rows. Logged rather than swallowed.
            data["log_graph_node_drops"] = torch.as_tensor(
                self._graph.node_drops, device=self._device
            ).reshape(-1, 1)
            data["log_graph_target_missing"] = torch.as_tensor(
                self._graph.target_missing, device=self._device
            ).reshape(-1, 1)
            data["log_graph_cache_entries"] = torch.full(
                (self._num_envs, 1),
                float(self._graph.cache_entries),
                device=self._device,
            )
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
            if self._instruction is not None:
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
        if self._instruction is not None:
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


def task_schedule_source(envs):
    """``(env_id, whitelist_dir)`` for compiling a task schedule, or None.

    The env id is the gym id (``PickCube-v1``), never the config's
    ``maniskill_`` form, and the whitelist directory is the one the graph
    adapter actually resolved -- a schedule compiled against a different
    entity vocabulary than the graph packs would resolve roles to wrong rows.
    """
    task_id = getattr(envs, "_task_id", None)
    graph = getattr(envs, "_graph", None)
    if task_id and graph is not None and getattr(graph, "whitelist_dir", ""):
        return str(task_id), str(graph.whitelist_dir)
    inner = getattr(envs, "env", None) or getattr(envs, "_env", None)
    return task_schedule_source(inner) if inner is not None else None


def graph_panel_source(envs):
    """The graph builder behind an env stack, or None.

    Eval video only. Returns the object holding ``last_graph_by_env``, which
    is populated only for envs listed in ``record_env_indices``.
    """
    for attr in ("_graph", "graph"):
        builder = getattr(envs, attr, None)
        if builder is not None and hasattr(builder, "last_graph_by_env"):
            return builder
    inner = getattr(envs, "env", None) or getattr(envs, "_env", None)
    return graph_panel_source(inner) if inner is not None else None
