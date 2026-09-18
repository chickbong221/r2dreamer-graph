import gymnasium as gym
import numpy as np
from common.logger import Logger
from envs.wrappers.pixels import PixelWrapper
from envs.wrappers.tensor import TensorWrapper
from envs.wrappers.record_episode import RecordEpisodeWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
from mani_skill.utils.wrappers import FlattenRGBDObservationWrapper
from mani_skill.utils import gym_utils
from functools import partial
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv

import mani_skill.envs

def cpu_env_factory(env_make_fn, idx: int, wrappers=[], record_video_path: str = None, record_episode_kwargs=dict(), logger: Logger = None):
	def _init():
		env = env_make_fn()
		for wrapper in wrappers:
			env = wrapper(env)
		env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
		if record_video_path is not None and (not record_episode_kwargs["record_single"] or idx == 0):
			env = RecordEpisodeWrapper(
                env,
                record_video_path,
                trajectory_name=f"trajectory_{idx}",
                save_video=record_episode_kwargs["save_video"],
                save_trajectory=record_episode_kwargs["save_trajectory"],
                info_on_video=record_episode_kwargs["info_on_video"],
                logger=logger,
            )
		return env

	return _init

def mshab_enabled(cfg):
	return str(cfg.get('mshab_task', 'none')).lower() not in ('none', '')

def mshab_make_kwargs(cfg, is_eval):
	"""
	Extra gym.make kwargs for an MS-HAB subtask env.

	Mirrors mshab/mshab/envs/make.py: task plans, scene builder and spawn data
	come from the ReplicaCAD rearrange directory, keyed by task group, subtask
	and split.
	"""
	import mshab.envs # noqa: F401 registers the SubtaskTrain envs
	from mani_skill import ASSET_DIR
	from mshab.envs.planner import plan_data_from_file

	subtask = cfg.env_id.split('SubtaskTrain')[0].lower()
	rearrange_dir = ASSET_DIR / 'scene_datasets/replica_cad_dataset/rearrange'
	split = cfg.mshab_split
	if is_eval and str(cfg.mshab_eval_split).lower() != 'none':
		split = cfg.mshab_eval_split
	plan_data = plan_data_from_file(
		rearrange_dir / 'task_plans' / cfg.mshab_task / subtask / split / f'{cfg.mshab_obj}.json')

	# A plan is one (build_config, init_config) pair, so there are many plans per
	# build config: keep whole build configs. The prefix of the sorted names is
	# stable across runs, so the same count always names the same scenes.
	task_plans = plan_data.plans
	n_bc = int(cfg.mshab_eval_num_build_configs if is_eval else cfg.mshab_num_build_configs)
	if n_bc > 0:
		names = sorted({p.build_config_name for p in task_plans})
		if n_bc < len(names):
			keep = set(names[:n_bc])
			task_plans = [p for p in task_plans if p.build_config_name in keep]
		print(f'[mshab] {"eval" if is_eval else "train"} split={split}: '
			f'{min(n_bc, len(names))}/{len(names)} build configs, '
			f'{len(task_plans)}/{len(plan_data.plans)} plans')

	kwargs = dict(
		task_plans=task_plans,
		scene_builder_cls=plan_data.dataset,
		spawn_data_fp=rearrange_dir / 'spawn_data' / cfg.mshab_task / subtask / split / 'spawn_data.pt',
		# Otherwise num_envs must divide the scene count (63 train / 21 val).
		require_build_configs_repeated_equally_across_envs=False,
		# Bounded reward, as mshab's own training code uses. TD-MPC2 regresses
		# reward into bins spanning [vmin, vmax], which raw dense reward overflows.
		reward_mode='normalized_dense',
		shader_dir='minimal',
	)
	if int(cfg.mshab_max_episode_steps) > 0:
		kwargs['max_episode_steps'] = int(cfg.mshab_max_episode_steps)
	return kwargs

def make_envs(cfg, num_envs, record_video_path, is_eval, logger):
	"""
	Make ManiSkill3 environment.
	"""
	mshab = mshab_enabled(cfg)
	record_episode_kwargs = dict(save_video=True, save_trajectory=False, record_single=True, info_on_video=False)

	# Set up env make fn for consistency
	env_make_fn = partial(
		gym.make, 
		disable_env_checker=True,
		id=cfg.env_id, 
		obs_mode=cfg.obs, 
		render_mode=cfg.render_mode, 
		sensor_configs=dict(width=cfg.render_size, height=cfg.render_size)
		)
	if cfg.control_mode != 'default':
		env_make_fn = partial(env_make_fn, control_mode=cfg.control_mode)
	if is_eval: # https://maniskill.readthedocs.io/en/latest/user_guide/reinforcement_learning/setup.html#evaluation
		env_make_fn = partial(env_make_fn, reconfiguration_freq=cfg.eval_reconfiguration_frequency)
	if mshab:
		assert cfg.env_type == 'gpu', 'MS-HAB scenes are GPU-only; set env_type=gpu'
		env_make_fn = partial(env_make_fn, **mshab_make_kwargs(cfg, is_eval))

	if cfg.env_type == 'cpu':
		# Get default control_mode and max_episode_steps values
		dummy_env = env_make_fn()
		control_mode = dummy_env.unwrapped.control_mode
		max_episode_steps = gym_utils.find_max_episode_steps_value(dummy_env)
		dummy_env.close()
		del dummy_env
		# Create cpu async vectorized env
		vector_env_cls = partial(AsyncVectorEnv, context="forkserver")
		if num_envs == 1:
			vector_env_cls = SyncVectorEnv
		wrappers = []
		if cfg['obs'] == 'rgb':
			wrappers.append(partial(PixelWrapper(cfg=cfg, num_envs=num_envs)))
		env: VectorEnv = vector_env_cls(
			[
				cpu_env_factory(env_make_fn, i, wrappers, record_video_path, record_episode_kwargs, logger)
				for i in range(num_envs)
			]
		)
		env = TensorWrapper(env)
	elif cfg.env_type == 'gpu':
		env = env_make_fn(num_envs=num_envs)
		control_mode = env.unwrapped.control_mode
		max_episode_steps = gym_utils.find_max_episode_steps_value(env)
		if cfg['obs'] == 'rgb':
			env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=cfg.include_state)
			env = PixelWrapper(cfg, env, num_envs)
		if record_video_path is not None:
			env = RecordEpisodeWrapper(
					env,
					record_video_path,
					trajectory_name=f"trajectory",
					max_steps_per_video=max_episode_steps,
					save_video=record_episode_kwargs["save_video"],
					save_trajectory=record_episode_kwargs["save_trajectory"],
					logger=logger,
				)
		if mshab:
			# Same placement and settings as mshab/mshab/envs/make.py: masks the
			# head joints out of Fetch's action space so the cameras hold still.
			from mshab.envs.wrappers import FetchActionWrapper
			env = FetchActionWrapper(
				env, stationary_base=False, stationary_torso=False, stationary_head=True)
		env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
	else:
		raise Exception('env_type must be cpu or gpu')
	cfg.env_cfg.control_mode = cfg.eval_env_cfg.control_mode = control_mode
	cfg.env_cfg.env_horizon = cfg.eval_env_cfg.env_horizon = env.max_episode_steps = max_episode_steps
	
	return env