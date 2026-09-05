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
    "enabled", "mshab_task", "entity_vocab",
    "use_target_flag", "object_object_spatial",
    "disable_object_object_relations",
    "thresholds_path", "whitelist_dir", "n_max", "e_max",
    "visibility_policy", "bypass_teemo",
)

_GRAPH_CONFIG_CASTS = {
    "enabled": bool,
    "use_target_flag": bool,
    "object_object_spatial": bool,
    "disable_object_object_relations": bool,
    "visibility_policy": str,
    "bypass_teemo": bool,
    "uid_vocab": int,
    "entity_vocab": int,
    "n_max": int,
    "e_max": int,
    "app_dim": int,
    "dino_res": int,
}


def camera_obs_key(camera: str) -> str:
    """Observation key for a named sensor.

    Two naming conventions meet here. ManiSkill names cameras by position with
    a ``_camera`` suffix (``base_camera``, ``hand_camera``); MS-HAB names them
    ``<robot>_<position>`` (``fetch_head``). Stripping the suffix where there is
    one and taking the last segment otherwise gives ``image_base`` /
    ``image_hand`` / ``image_head`` from either -- and, more to the point, keeps
    ``base_camera`` and ``hand_camera`` distinct, which the last-segment rule
    alone does not.
    """
    suffix = "_camera"
    short = (camera[: -len(suffix)] if camera.endswith(suffix)
             else camera.rsplit("_", 1)[-1])
    return "image_" + short


def rendered_cameras(env) -> list:
    """Sensor names this task actually renders, in the task's own order.

    The camera set is a property of the task and the robot it registers -- a
    bare ``panda`` renders ``base_camera`` alone, ``panda_wristcam`` adds
    ``hand_camera`` -- so pinning it in config means one number per task to
    keep in step with ManiSkill, and a silent mismatch when it drifts.

    Read from the initial observation rather than from the sensor registry:
    that is the same dict ``NamedCameraRGBWrapper`` validates against, so
    discovery and validation cannot disagree.
    """
    base = getattr(env, "unwrapped", env)
    obs = getattr(base, "_init_raw_obs", None)
    sensors = (obs or {}).get("sensor_data", None) if obs is not None else None
    if not sensors:
        sensors = getattr(base, "_sensors", None)
    names = [str(name) for name in (sensors or {})]
    if not names:
        raise RuntimeError(
            f"{getattr(base, 'spec', None) and base.spec.id!r} renders no "
            "sensors, so the graph has no pixels to build nodes from. Name "
            "cameras explicitly in env.cameras if this task is unusual"
        )
    return names


def _camera_keys(cameras) -> dict:
    """``observation key -> camera``, rejecting a set that would lose one."""
    keys = {camera_obs_key(camera): camera for camera in cameras}
    if len(keys) != len(cameras):
        raise ValueError(
            f"cameras {list(cameras)} collapse to {sorted(keys)}; two cameras "
            "sharing an observation key would silently drop one"
        )
    return keys


def _make_with_supported_reward(make_kwargs: dict, fallback) -> "gym.Env":
    """Build the env, stepping down `env.reward_fallback` until one is implemented.

    `normalized_dense` needs the task to supply a normaliser and `dense` needs a
    shaped reward at all; not every ManiSkill task has either -- PlugCharger-v1
    raises on the first. `reward_fallback` names what to try instead, in order.
    An empty list is strict: the run fails rather than quietly training on a
    different reward.

    Announced, never silent. A task on `sparse` is a different learning problem
    from one on `normalized_dense`, so it has to be visible in the log rather
    than inferred later from a flat return curve. Both arms read the same env
    config, so whichever applies, the within-task comparison is unaffected.
    """
    wanted = str(make_kwargs["reward_mode"])
    chain = [wanted] + [str(m) for m in (fallback or []) if str(m) != wanted]
    for mode in chain:
        make_kwargs["reward_mode"] = mode
        try:
            env = gym.make(**make_kwargs)
        except NotImplementedError as exc:
            if "reward mode" not in str(exc).lower():
                raise
            continue
        if mode != wanted:
            print(
                f"[env] {make_kwargs['id']} does not implement {wanted!r}; "
                f"using {mode!r}. Returns are on a different scale from tasks "
                "that do -- compare arms within a task, never across tasks",
                flush=True,
            )
        return env
    raise NotImplementedError(
        f"{make_kwargs['id']} implements none of {chain}. Add a mode to "
        "env.reward_fallback, or set env.reward_mode to one it supports"
    )


def is_mshab_task(task_id: str) -> bool:
    """MS-HAB subtask envs carry their scene through a ReplicaCAD task plan.

    Ordinary ManiSkill tasks build their own scene from the gym id alone, so
    everything plan-shaped -- spawn data, scene builder, object split -- is
    meaningless for them.
    """
    return "SubtaskTrain" in str(task_id)


def mshab_subtask(task_id: str) -> str:
    """``PickSubtaskTrain-v0`` -> ``pick``; empty for ordinary ManiSkill.

    MS-HAB mines its assets per subtask inside a task group, so this is half
    of the pair that names them. Ordinary tasks have no subtask -- their gym
    id already is the whole task.
    """
    text = str(task_id)
    return text.split("SubtaskTrain", 1)[0].lower() if is_mshab_task(text) else ""


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
    for key in ("thresholds_path", "whitelist_dir"):
        if out[key]:
            out[key] = str(_repo_path(out[key]))
    return out


def select_named_build_configs(task_plans, names, label: str = "train"):
    """Keep only plans from build configurations named outright.

    ``num_build_configs`` takes a sorted *prefix*, so which scene it means
    moves the moment the build list changes -- and "the first one" is not a
    scene identity anyone can check a result against. An experiment pinned to
    a scene names it.

    Raises rather than falling back: a misspelled configuration silently
    training on a different apartment is the failure this exists to remove.
    """
    wanted = [str(n) for n in names if str(n)]
    if not wanted:
        return list(task_plans)
    available = sorted({str(p.build_config_name) for p in task_plans})
    missing = [n for n in wanted if n not in available]
    if missing:
        raise ValueError(
            f"build configuration(s) {missing} are not in this task plan "
            f"({len(available)} available, e.g. {available[:3]}). A pinned "
            "scene that does not exist would otherwise train somewhere else."
        )
    keep = set(wanted)
    selected = [p for p in task_plans if str(p.build_config_name) in keep]
    print(f"[env] pinned {len(keep)} named build config(s) "
          f"({len(selected)}/{len(task_plans)} plans, {label}): "
          f"{sorted(keep)}", flush=True)
    return selected


def balance_objects(plans_by_object, label: str = "train"):
    """Equal plan counts per object, deterministically.

    Concatenating the five objects' plan files samples them in proportion to
    how many plans each happens to have -- 5,115 to 5,823 across tidy_house's
    pick targets, so one object would get 14% more episodes than another for
    no reason anybody chose. Truncating each to the smallest count removes
    that, and taking a prefix of an already-ordered list keeps two runs
    comparable.
    """
    if not plans_by_object:
        return []
    counts = {key: len(plans) for key, plans in plans_by_object.items()}
    smallest = min(counts.values())
    out = []
    for key in sorted(plans_by_object):
        out.extend(list(plans_by_object[key])[:smallest])
    print(f"[env] balanced {len(plans_by_object)} object(s) to {smallest} "
          f"plan(s) each ({len(out)} total, {label}); before: {counts}",
          flush=True)
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
        that split rather than a prefix, and it runs to ``eval_time_limit``
        when that is set -- 0 leaves the task's registered horizon alone, so
        training and evaluation share it. The observation contract, the graph
        adapter and the action space are constructed identically, so one agent
        drives both.
        """
        import mani_skill.envs  # noqa: F401 - register ManiSkill tasks
        from mani_skill.utils import gym_utils
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

        from envs.instruction import InstructionReader, InstructionTable
        from envs.maniskill_obs import NamedCameraRGBWrapper, NonPrivilegedObsWrapper
        from scenegraph.adapters.graph_obs import build_graph_obs

        self._eval = bool(eval)
        self.eval_cases = []
        self.training_scenes = list(getattr(config, "train_build_config_ids", []) or [])
        self._eval_seed = int(getattr(config, "eval_seed", config.seed))
        self._lighting_controller = None
        self.eval_reset_metrics = {}
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
        # Empty means "whatever this task renders", resolved from the built env
        # below. Naming cameras here is the override, and MS-HAB uses it: its
        # scenes carry sensors the graph has no use for.
        self._camera_names = [str(c) for c in (config.cameras or [])]

        task = str(config.task).split("_", 1)[1]
        self._task_id = task
        self._is_mshab = is_mshab_task(task)
        # What names this run's mined assets. For MS-HAB that is the group and
        # the subtask, not the gym id -- ``PickSubtaskTrain-v0`` appears in no
        # asset path. Both stay empty for ordinary ManiSkill.
        self._mshab_subtask = mshab_subtask(task)
        self._mshab_task_group = (
            str(config.mshab_task) if self._is_mshab else "")
        size = tuple(map(int, config.size))
        make_kwargs = dict(
            id=task,
            obs_mode=str(config.obs_mode),
            render_mode="rgb_array",
            sensor_configs=dict(width=size[1], height=size[0]),
            num_envs=self._num_envs,
            sim_backend=str(config.sim_backend),
            reward_mode=str(config.reward_mode),
            shader_dir=str(config.shader_dir),
        )
        # 0 leaves the horizon to the task's own registration -- PickCube's 50,
        # PegInsertionSide's 100 -- which is what a comparison against a
        # published ManiSkill number needs. MS-HAB sets both explicitly instead,
        # because its 100/200 come from mshab's own configs and not from the
        # registration. `find_max_episode_steps_value` below reads back whichever
        # applied, so nothing downstream has to know which way it went.
        horizon = int(
            (getattr(config, "eval_time_limit", 0) or config.time_limit)
            if self._eval
            else config.time_limit
        )
        if horizon > 0:
            make_kwargs["max_episode_steps"] = horizon
        # Likewise the robot: empty takes the task's default agent, which for
        # the tabletop tasks is a bare `panda` that renders base_camera only.
        # A wrist view is a different robot, not a sensor config, so asking for
        # one here also changes the action space.
        robot_uids = str(getattr(config, "robot_uids", "") or "")
        if robot_uids:
            make_kwargs["robot_uids"] = robot_uids

        if self._is_mshab:
            # Imported here, not at the top: an ordinary ManiSkill run should
            # not need the MS-HAB package installed to build its scene.
            import mshab.envs  # noqa: F401 - register MS-HAB tasks
            from mani_skill import ASSET_DIR
            from mshab.envs.planner import plan_data_from_file

            subtask = self._mshab_subtask
            # An empty eval_split evaluates on the training scenes, which
            # measures fit rather than generalisation. Left possible on
            # purpose, but it is not the default.
            eval_split = str(getattr(config, "eval_split", "") or "")
            split = eval_split if (self._eval and eval_split) else str(config.split)
            rearrange = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
            plan_dir = (rearrange / "task_plans" / str(config.mshab_task)
                        / subtask / split)
            # Named objects win over the single ``mshab_obj``: Experiment A
            # trains on five of tidy_house's nine, and "all" is a different
            # policy's plan file rather than a subset of theirs.
            objects = [str(o) for o in
                       (getattr(config, "mshab_objects", None) or []) if o]
            label = "eval" if self._eval else "train"
            if objects:
                by_object, plan_data = {}, None
                for obj in objects:
                    path = plan_dir / f"{obj}.json"
                    if not path.exists():
                        raise FileNotFoundError(
                            f"MS-HAB task plan not found for {obj!r}: {path}")
                    data = plan_data_from_file(path)
                    plan_data = plan_data or data
                    by_object[obj] = data.plans
                plan_path = plan_dir
                raw_plans = [plan for plans in by_object.values() for plan in plans]
            else:
                plan_path = plan_dir / f"{config.mshab_obj}.json"
                plan_data = plan_data_from_file(plan_path)
                raw_plans = plan_data.plans
                by_object = {str(config.mshab_obj): raw_plans}

            # A named scene beats a count. See ``select_named_build_configs``.
            named = [str(n) for n in (
                getattr(config, "eval_build_config_ids", None) if self._eval
                else getattr(config, "train_build_config_ids", None)) or []
                if str(n) and str(n) != "all"]
            if named:
                task_plans = select_named_build_configs(
                    raw_plans, named, label=label)
            else:
                # Training subsets its split; evaluation takes the whole of
                # its own by default, so a held-out score is not reported over
                # a sliver.
                task_plans = _select_build_configs(
                    raw_plans,
                    int(
                        getattr(config, "eval_num_build_configs", 0)
                        if self._eval
                        else config.num_build_configs
                    ),
                    label=label,
                )
            if not task_plans:
                raise ValueError(
                    f"MS-HAB task selection produced no plans: {plan_path}")
            # Filter BEFORE balancing. A global balance can become skewed
            # when a named scene is selected afterwards.
            selected_ids = {id(plan) for plan in task_plans}
            selected = {
                obj: [p for p in plans if id(p) in selected_ids]
                for obj, plans in by_object.items()
            }
            if any(not plans for plans in selected.values()):
                raise ValueError("a requested object has no plans in the selected scenes")
            if not self._eval and objects:
                task_plans = balance_objects(selected, label=label)
            else:
                task_plans = [p for plans in selected.values() for p in plans]
            if self._eval and getattr(config, "eval_panel", ""):
                from envs.evaluation import build_panel, LightingController
                self.eval_cases = build_panel(selected, config)
                self._num_envs = len(self.eval_cases)
                make_kwargs["num_envs"] = self._num_envs
                self._lighting_controller = LightingController()
                if any(c.group == "light" for c in self.eval_cases) and str(config.shader_dir) != "minimal":
                    raise ValueError("lighting evaluation currently requires shader_dir=minimal")
                print(f"[eval] fixed panel: {config.eval_episode_num} primary + "
                      f"{self._num_envs - int(config.eval_episode_num)} lighting environments", flush=True)
            make_kwargs.update(
                task_plans=task_plans,
                scene_builder_cls=plan_data.dataset,
                spawn_data_fp=(
                    rearrange / "spawn_data" / str(config.mshab_task) / subtask
                    / split / "spawn_data.pt"
                ),
                # Evaluation over every scene needs each sub-scene pinned to
                # a different one. ``reconfiguration_freq=0`` fixes a
                # sub-scene's configuration for the whole run, so the number
                # of *distinct* scenes evaluated is bounded by the environment
                # count, not the episode count -- and without this flag
                # nothing makes the spread even. MS-HAB then requires
                # divisibility, which is the check that 63 envs over 63
                # configurations is one each.
                require_build_configs_repeated_equally_across_envs=bool(
                    self._eval
                    and not self.eval_cases
                    and getattr(config, "eval_even_build_configs", False)),
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
            # The third-person "render_camera" that `env.render()` returns, not
            # an observation sensor: the policy never sees it, so it is free to
            # be far larger than the 112x112 the encoder reads. Empty keeps the
            # task's own default.
            self._render_size = tuple(
                int(v) for v in (getattr(config, "eval_render_size", None) or ())
            )
            if self._render_size:
                make_kwargs["human_render_camera_configs"] = dict(
                    width=self._render_size[1], height=self._render_size[0]
                )
        env = _make_with_supported_reward(
            make_kwargs, getattr(config, "reward_fallback", None))
        if not self._camera_names:
            self._camera_names = rendered_cameras(env)
        self._camera_keys = _camera_keys(self._camera_names)
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

        obs, _ = self._reset_simulator(initial=True)
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

    def _reset_simulator(self, initial=False):
        """Pin actual build, task and spawn indices on every evaluation reset."""
        if not self.eval_cases:
            if initial:
                return self._env.reset(seed=self._seed)
            return self._env.reset(options={
                "env_idx": torch.arange(self._num_envs, device=self._device)})
        base = self._env.unwrapped
        names = base.scene_builder.build_config_names_to_idxs
        bcis = [int(names[c.scene]) for c in self.eval_cases]
        spawns = []
        for case, bci in zip(self.eval_cases, bcis):
            plan = base.build_config_idx_to_task_plans[bci][case.plan_index]
            subtask = plan.subtasks[0]
            data = base.spawn_data[subtask.composite_subtask_uids[0]]
            count = len(next(iter(data.values())))
            if count <= 0:
                raise ValueError("evaluation task has no premade spawns")
            spawns.append(case.repetition % count)
        options = {
            "task_plan_idxs": torch.tensor([c.plan_index for c in self.eval_cases], device=self._device),
            "spawn_selection_idxs": spawns,
        }
        if initial:
            options.update(reconfigure=True, build_config_idxs=bcis)
        elif list(base.build_config_idxs) != bcis:
            raise RuntimeError("fixed evaluation scene assignment drifted")
        # Apply illumination before reset renders the observation. Reconfigure
        # rebuilds lights; apply afterwards and refresh sensors in that case.
        if not initial:
            self._lighting_controller.apply(base.scene, self.eval_cases)
        obs, info = self._env.reset(seed=self._eval_seed, options=options)
        if initial and any(c.group == "light" for c in self.eval_cases):
            self._lighting_controller.apply(base.scene, self.eval_cases)
            # A second reset refreshes the wrapper's raw segmentation cache,
            # with identical plans/spawns and without another reconfiguration.
            return self._reset_simulator(initial=False)
        from envs.evaluation import check_lighting_reset, lighting_pixel_metrics
        self.eval_reset_metrics = check_lighting_reset(base, self.eval_cases)
        self.eval_reset_metrics.update(lighting_pixel_metrics(obs, self.eval_cases))
        return obs, info

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
            obs, _ = self._reset_simulator()
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

    def render(self):
        """Human-render camera for every env, ``[N, H, W, 3]`` uint8 on CPU.

        Not an observation: this is the separate third-person camera the eval
        video is made from, so it carries no obligation to match the encoder's
        resolution. CPU because the frames are only ever stacked and encoded,
        and a 200-step eval at 512x512 is a third of a gigabyte the training
        step would otherwise be sharing a device with.
        """
        frames = self._env.render()
        if frames is None:
            return None
        if not torch.is_tensor(frames):
            frames = torch.as_tensor(np.asarray(frames))
        if frames.ndim == 3:
            frames = frames[None]
        if frames.dtype != torch.uint8:
            # A float render is in [0, 1]; anything already 0-255 stays put.
            scale = 255.0 if float(frames.max()) <= 1.0 else 1.0
            frames = (frames * scale).clamp(0, 255).to(torch.uint8)
        return frames.detach().cpu()

    def close(self):
        self._env.close()


def task_schedule_source(envs, schedule_dir: str = ""):
    """The :class:`ScheduleSource` this env compiles against, or None.

    Two naming schemes, kept apart. An ordinary ManiSkill task is named once
    by its gym id (``PickCube-v1``), which is also its schedule, its
    affordance file and its whitelist directory. An MS-HAB run is named by the
    task group and the subtask (``set_table``, ``pick``); its gym id
    (``PickSubtaskTrain-v0``) appears in no asset path at all, and asking for
    it by that id found nothing.

    The whitelist directory is always the one the graph adapter itself
    resolved. A schedule compiled against any other would resolve roles
    against a different entity vocabulary than the packer writes rows with,
    and the mismatch would show up as phases that silently never score.
    """
    from scenegraph.core.schedule import (
        maniskill_schedule_source,
        mshab_schedule_source,
    )

    task_id = getattr(envs, "_task_id", None)
    graph = getattr(envs, "_graph", None)
    whitelist_dir = str(getattr(graph, "whitelist_dir", "") or "")
    if task_id and whitelist_dir:
        # <configs>/subtask_whitelists/<asset tree> -> <configs>
        configs = str(Path(whitelist_dir).parent.parent)
        if getattr(envs, "_is_mshab", False):
            return mshab_schedule_source(
                str(getattr(envs, "_mshab_task_group", "") or ""),
                str(getattr(envs, "_mshab_subtask", "") or ""),
                configs, schedule_dir, whitelist_dir,
            )
        return maniskill_schedule_source(
            str(task_id), configs, schedule_dir, whitelist_dir)
    inner = getattr(envs, "env", None) or getattr(envs, "_env", None)
    return (task_schedule_source(inner, schedule_dir)
            if inner is not None else None)


def graph_panel_source(envs):
    """The graph builder behind an env stack, or None.

    Diagnostic videos only. Returns the object holding ``last_graph_by_env``,
    which is populated for envs listed in ``record_graph_env_indices`` (or the
    mask-producing ``record_env_indices`` used by offline overlays).
    """
    for attr in ("_graph", "graph"):
        builder = getattr(envs, attr, None)
        if builder is not None and hasattr(builder, "last_graph_by_env"):
            return builder
    inner = getattr(envs, "env", None) or getattr(envs, "_env", None)
    return graph_panel_source(inner) if inner is not None else None
