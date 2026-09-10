"""Reward logging for an MS-HAB checkpoint rollout.

No simulator and no checkpoint. The fixture is the released
``set_table/close/fridge`` config verbatim, because what this tool mostly does
is read one -- what can go wrong is reading the wrong half of it (the training
horizon where the evaluation one was meant), dropping the env kwargs that are
terms in the reward, running an episode of the wrong length, or labelling a
random-action rollout with a checkpoint's name.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scenegraph.tools import demo_mshab_checkpoint_reward as demo
from scenegraph.tools.demo_mshab_checkpoint_reward import (
    env_section, load_ckpt_config, make_dropping_unknown, resolve_path,
    run_episode, write_episode,
)
from scenegraph.tools.demo_motionplanning_reward import RewardTrace

# The released config for rl/set_table/close/fridge, trimmed to the fields this
# tool reads and keeping their exact values -- including the two different
# max_episode_steps, which is the whole reason --config-section exists.
CONFIG = """
seed: 2337
env:
  env_id: CloseSubtaskTrain-v0
  num_envs: 189
  max_episode_steps: 100
  continuous_task: true
  cat_state: true
  cat_pixels: false
  frame_stack: 3
  stationary_base: false
  stationary_torso: false
  stationary_head: true
  task_plan_fp: ~/.maniskill/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/set_table/close/train/fridge.json
  spawn_data_fp: ~/.maniskill/data/scene_datasets/replica_cad_dataset/rearrange/spawn_data/set_table/close/train/spawn_data.pt
  env_kwargs:
    robot_force_mult: 0.001
    robot_force_penalty_min: 0.2
    target_randomization: false
eval_env:
  env_id: CloseSubtaskTrain-v0
  num_envs: 63
  max_episode_steps: 200
  continuous_task: true
  cat_state: true
  cat_pixels: false
  frame_stack: 3
  stationary_base: false
  stationary_torso: false
  stationary_head: true
  task_plan_fp: ~/.maniskill/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/set_table/close/train/fridge.json
  spawn_data_fp: ~/.maniskill/data/scene_datasets/replica_cad_dataset/rearrange/spawn_data/set_table/close/train/spawn_data.pt
  env_kwargs:
    robot_force_mult: 0.001
    robot_force_penalty_min: 0.2
    target_randomization: false
algo:
  name: ppo
  gamma: 0.9
model_ckpt: "mshab_checkpoints/rl/set_table/close/fridge/policy.pt"
"""


def _ckpt_dir(tmp: str) -> Path:
    """A checkpoint directory: the config, and a policy file with no weights
    in it -- nothing under test here ever loads one."""
    path = Path(tmp)
    (path / "config.yml").write_text(CONFIG, encoding="utf-8")
    (path / "policy.pt").write_bytes(b"")
    return path


class _Policy:
    """Stands in for a PolicyHandle: one action per observation."""

    def __init__(self, kind="ppo"):
        self.kind = kind
        self.seen = 0

    def act(self, obs):
        self.seen += 1
        return np.zeros(4)


class _Venv:
    """A vector env of one, paying a scripted reward per step."""

    def __init__(self, rewards, success_from=None, truncate_at=None):
        self.rewards = list(rewards)
        self.success_from = success_from
        self.truncate_at = truncate_at
        self.index = 0
        self.seeds = []
        self.closed = False

    def reset(self, seed=None):
        self.seeds.append(seed)
        self.index = 0
        return {"state": np.zeros(3)}, {}

    def step(self, action):
        reward = self.rewards[min(self.index, len(self.rewards) - 1)]
        self.index += 1
        success = (self.success_from is not None
                   and self.index >= self.success_from)
        truncated = (self.truncate_at is not None
                     and self.index >= self.truncate_at)
        return ({"state": np.zeros(3)}, np.array([reward]),
                np.array([False]), np.array([truncated]),
                {"success": np.array([success])})

    def close(self):
        self.closed = True


# --------------------------------------------------------------------------- #
# Reading the checkpoint's config
# --------------------------------------------------------------------------- #
class TestConfig(unittest.TestCase):
    def test_the_two_sections_carry_different_horizons(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_ckpt_config(_ckpt_dir(tmp))
        self.assertEqual(env_section(config, "eval_env")["max_episode_steps"],
                         200)
        self.assertEqual(env_section(config, "env")["max_episode_steps"], 100)
        # The default is the evaluation block: a trained policy is being rolled
        # out, not trained.
        self.assertEqual(demo.CONFIG_SECTIONS[0], "eval_env")

    def test_the_reward_shaping_kwargs_survive_the_read(self):
        # robot_force_mult is a term in the reward, so an env rebuilt without
        # it reports a different number for the same behaviour.
        with tempfile.TemporaryDirectory() as tmp:
            section = env_section(load_ckpt_config(_ckpt_dir(tmp)), "eval_env")
        self.assertEqual(section["env_kwargs"], {
            "robot_force_mult": 0.001,
            "robot_force_penalty_min": 0.2,
            "target_randomization": False,
        })
        self.assertEqual(section["env_id"], "CloseSubtaskTrain-v0")
        self.assertEqual(section["frame_stack"], 3)
        self.assertTrue(section["stationary_head"])
        self.assertFalse(section["stationary_base"])

    def test_a_missing_section_names_what_is_there(self):
        with self.assertRaises(SystemExit) as caught:
            env_section({"env": {}}, "eval_env")
        self.assertIn("env", str(caught.exception))

    def test_a_missing_config_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                load_ckpt_config(Path(tmp))

    def test_config_paths_expand_the_home_shorthand(self):
        with tempfile.TemporaryDirectory() as tmp:
            section = env_section(load_ckpt_config(_ckpt_dir(tmp)), "eval_env")
        plan = resolve_path(section["task_plan_fp"])
        self.assertNotIn("~", str(plan))
        self.assertEqual(plan.name, "fridge.json")
        self.assertEqual(resolve_path(section["spawn_data_fp"]).name,
                         "spawn_data.pt")


class TestUnknownKwargs(unittest.TestCase):
    """Different mshab builds accept different kwargs; a GPU run should not
    die on one of them."""

    def test_an_unaccepted_kwarg_is_dropped_and_the_rest_kept(self):
        seen = {}

        def make(**kwargs):
            if "continuous_task" in kwargs:
                raise TypeError(
                    "__init__() got an unexpected keyword argument "
                    "'continuous_task'")
            seen.update(kwargs)
            return "env"

        out = make_dropping_unknown(
            make, dict(id="CloseSubtaskTrain-v0", continuous_task=True,
                       robot_force_mult=0.001))
        self.assertEqual(out, "env")
        self.assertEqual(seen, {"id": "CloseSubtaskTrain-v0",
                                "robot_force_mult": 0.001})

    def test_an_unrelated_type_error_still_raises(self):
        def make(**kwargs):
            raise TypeError("num_envs must be an int")

        with self.assertRaises(TypeError):
            make_dropping_unknown(make, dict(id="x"))


# --------------------------------------------------------------------------- #
# The rollout
# --------------------------------------------------------------------------- #
class TestRollout(unittest.TestCase):
    def test_an_episode_is_the_horizon_long(self):
        venv, policy = _Venv([0.5] * 10), _Policy()
        trace = run_episode(venv, policy, RewardTrace(discount=1.0), 8, seed=3)
        # Terminations are ignored, so the length is the horizon, not a
        # property of the policy.
        self.assertEqual(len(trace.steps), 8)
        self.assertEqual(policy.seen, 8)
        self.assertEqual(venv.seeds, [3])
        self.assertAlmostEqual(trace.steps[-1].ret, 4.0)

    def test_truncation_ends_the_episode_early(self):
        venv = _Venv([0.5] * 10, truncate_at=4)
        trace = run_episode(venv, _Policy(), RewardTrace(discount=1.0), 8,
                            seed=0)
        self.assertEqual(len(trace.steps), 4)
        self.assertEqual(trace.first_flag("truncated"), 4)

    def test_success_is_recorded_per_step(self):
        venv = _Venv([0.1] * 6, success_from=5)
        trace = run_episode(venv, _Policy(), RewardTrace(discount=1.0), 6,
                            seed=0)
        self.assertEqual(trace.first_success_step(), 5)
        self.assertTrue(trace.summary(6)["success_within_horizon"])

    def test_each_episode_starts_from_an_empty_trace(self):
        venv, trace = _Venv([1.0] * 4), RewardTrace(discount=1.0)
        run_episode(venv, _Policy(), trace, 4, seed=0)
        run_episode(venv, _Policy(), trace, 2, seed=1)
        self.assertEqual(len(trace.steps), 2)
        self.assertEqual(venv.seeds, [0, 1])


# --------------------------------------------------------------------------- #
# What gets written
# --------------------------------------------------------------------------- #
class TestWrite(unittest.TestCase):
    def _write(self, tmp, success=True):
        ckpt = _ckpt_dir(tmp)
        args = demo.parse_args(["--ckpt-dir", str(ckpt), "--no-plot"])
        section = env_section(load_ckpt_config(ckpt), "eval_env")
        trace = run_episode(_Venv([0.25] * 4, success_from=3 if success else None),
                            _Policy(), RewardTrace(discount=1.0), 4, seed=7)
        out = Path(tmp) / "out"
        paths = write_episode(
            trace, trace.summary(200), out=out, seed=7, success=success,
            args=args, section=section,
            plan_fp=Path("/root/.maniskill/plans/fridge.json"), algo="ppo",
            horizon=200, title="Close Fridge")
        return paths, json.loads(paths["json"].read_text(encoding="utf-8"))

    def test_the_json_identifies_the_checkpoint_it_came_from(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, written = self._write(tmp)
            self.assertTrue(paths["csv"].exists())
            self.assertEqual(paths["csv"].name, "seed0007_success.csv")
        self.assertTrue(written["checkpoint"].endswith("policy.pt"))
        self.assertEqual(written["algo"], "ppo")
        self.assertEqual(written["config_section"], "eval_env")
        self.assertEqual(written["horizon"], 200)
        self.assertEqual(written["env_id"], "CloseSubtaskTrain-v0")
        self.assertEqual(written["title"], "Close Fridge")
        self.assertEqual(written["env_kwargs"]["robot_force_mult"], 0.001)
        self.assertEqual(len(written["steps"]), 4)

    def test_a_failed_episode_says_so_in_its_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, written = self._write(tmp, success=False)
            self.assertEqual(paths["csv"].name, "seed0007_failed.csv")
        self.assertFalse(written["attempt"]["success"])

    def test_the_csv_matches_the_motion_planning_demo(self):
        # Same columns, so --from-csv over there redraws what is written here.
        with tempfile.TemporaryDirectory() as tmp:
            paths, _ = self._write(tmp)
            header = paths["csv"].read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(header.split(","),
                         ["step", "reward", "return", "discounted_return",
                          "success", "terminated", "truncated"])


class TestGuards(unittest.TestCase):
    def test_a_missing_policy_file_is_refused_before_the_sim_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.yml").write_text(CONFIG, encoding="utf-8")
            args = demo.parse_args(["--ckpt-dir", tmp])
            with self.assertRaises(SystemExit) as caught:
                demo.run(args)
        self.assertIn("policy.pt", str(caught.exception))

    def test_the_default_section_is_the_evaluation_one(self):
        args = demo.parse_args(["--ckpt-dir", "."])
        self.assertEqual(args.config_section, "eval_env")
        self.assertEqual(args.max_episode_steps, 0)   # 0 means "from config"
        self.assertEqual(args.num_envs, 1)
        self.assertTrue(args.plot)
        self.assertFalse(args.allow_random)


if __name__ == "__main__":
    unittest.main()
