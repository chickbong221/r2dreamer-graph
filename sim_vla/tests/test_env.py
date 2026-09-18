"""Stage 6: the online env and replay match the demonstration contract."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from .common import REPO, require, require_torch

DEMOS = REPO / "data/sim_vla_demos"


def dataset_for(task="PickCube-v1"):
    path = DEMOS / task / "demos.h5"
    if not path.exists():
        raise unittest.SkipTest(f"no collected dataset at {path}")
    return path


class TestReplay(unittest.TestCase):
    def episode(self, steps=5, obs_rows=None):
        from sim_vla.data.replay import OnlineEpisode

        episode = OnlineEpisode()
        rows = steps + 1 if obs_rows is None else obs_rows
        for _ in range(rows):
            episode.add_observation({"proprio": np.zeros(4, np.float32)})
        for index in range(steps):
            episode.add_transition(np.zeros(8, np.float32), float(index),
                                   False, index == steps - 1, False)
        return episode

    def test_final_observation_must_be_captured(self):
        from sim_vla.data.replay import OnlineReplay

        replay = OnlineReplay()
        replay.add(self.episode(steps=5))
        self.assertEqual(len(replay), 1)
        # An episode whose observations equal its actions lost the state the
        # last action led to, which is the reset-before-capture bug.
        with self.assertRaises(ValueError):
            replay.add(self.episode(steps=5, obs_rows=5))

    def test_windows_use_the_canonical_layout(self):
        """Every array has one row per observation, the same count for all.

        The replay used to emit its own shape -- transitions for actions,
        transitions+1 for observations -- which is why a mixed batch failed on
        73 rows against 65. Both sources now go through layout.assemble.
        """
        from sim_vla.data import layout
        from sim_vla.data.replay import OnlineReplay

        replay = OnlineReplay(seed=0)
        for _ in range(4):
            replay.add(self.episode(steps=7))
        batch = replay.sample(batch=3, length=4, burn_in=1)
        expected = layout.rows(4, 1)
        for key, value in batch.items():
            self.assertEqual(value.shape[0], 3, key)
            self.assertEqual(value.shape[1], expected, key)
        # The posterior input and the actor target are separate arrays.
        for key in ("action", "action_target", "reward", "loss_mask",
                    "valid", "action_valid", "reward_valid"):
            self.assertIn(key, batch)

    def test_windows_stay_inside_one_episode(self):
        """A window never reaches past the episode it was drawn from.

        Rewards were written as the step index, so a row that crossed a
        boundary would carry a reward from the wrong episode.
        """
        from sim_vla.data.replay import OnlineReplay

        replay = OnlineReplay(seed=0)
        for _ in range(4):
            replay.add(self.episode(steps=7))
        batch = replay.sample(batch=3, length=4, burn_in=1)
        rewards, valid = batch["reward"], batch["valid"]
        for row in range(rewards.shape[0]):
            real = rewards[row][valid[row]]
            # Contiguous and non-decreasing: one episode, in order.
            self.assertTrue(np.all(np.diff(real) >= 0), real)
            self.assertLessEqual(float(real.max()), 6.0)

    def test_burn_in_is_excluded_only_where_it_exists(self):
        """A window at a reset has no history to burn in, so row 0 is scored."""
        from sim_vla.data.replay import OnlineReplay

        replay = OnlineReplay(seed=0)
        for _ in range(4):
            replay.add(self.episode(steps=7))
        batch = replay.sample(batch=6, length=4, burn_in=1)
        for row in range(batch["loss_mask"].shape[0]):
            if batch["is_first"][row, 0]:
                self.assertTrue(batch["loss_mask"][row, 0],
                                "a reset window has no burn-in to exclude")
            else:
                self.assertFalse(batch["loss_mask"][row, 0],
                                 "a mid-episode window burns in its first row")

    def test_mixture_waits_for_a_meaningful_replay(self):
        from sim_vla.data.replay import OnlineReplay, mixed_batch

        class Demo:
            def batch(self, n):
                return {"action": np.zeros((n, 4, 8), np.float32)}

        replay = OnlineReplay(seed=0)
        out = mixed_batch(Demo(), replay, batch=8, length=4, burn_in=0)
        self.assertEqual(out["action"].shape[0], 8)          # demos only


class TestOnlineEnv(unittest.TestCase):
    def build(self, graph_enabled):
        require("mani_skill")
        from sim_vla.envs.maniskill import SimVlaEnv, load_metadata

        metadata = load_metadata(dataset_for())
        return SimVlaEnv(metadata, graph_enabled=graph_enabled, max_steps=8)

    def test_observation_matches_the_dataset_contract(self):
        env = self.build(False)
        obs = env.reset(seed=0)
        try:
            for key in env.camera_keys.values():
                self.assertIn(key, obs)
                self.assertEqual(obs[key].shape[:2], tuple(env.image_size))
            self.assertEqual(obs["proprio"].shape[0], len(env.proprio_names))
        finally:
            env.close()

    def test_baseline_observation_has_no_graph_keys(self):
        from scenegraph.adapters.graph_pack import GRAPH_KEYS

        env = self.build(False)
        obs = env.reset(seed=0)
        try:
            for key in GRAPH_KEYS:
                self.assertNotIn(key, obs)
        finally:
            env.close()

    def test_graph_arm_observation_carries_the_packed_graph(self):
        from scenegraph.adapters.graph_pack import GRAPH_KEYS

        env = self.build(True)
        obs = env.reset(seed=0)
        try:
            for key in GRAPH_KEYS:
                self.assertIn(key, obs)
        finally:
            env.close()

    def test_nothing_terminates_and_the_recorded_flag_is_kept(self):
        env = self.build(False)
        obs = env.reset(seed=0)
        try:
            out = env.step(np.zeros(env.action_dim, np.float32))
            self.assertFalse(out["is_terminal"])
            self.assertIn("terminated_recorded", out)
        finally:
            env.close()

    def test_collect_episode_keeps_the_final_observation(self):
        # The env skip has to fire before the import: collect_episode lives in
        # a torch module, and an unguarded import errors where it should skip.
        env = self.build(False)
        require_torch()
        from sim_vla.training.online import collect_episode

        try:
            episode = collect_episode(
                env, lambda obs: np.zeros(env.action_dim, np.float32),
                max_steps=4, seed=0)
            arrays = episode.arrays()
            # Storage naming: the replay stores "actions" like the dataset.
            self.assertEqual(arrays["proprio"].shape[0],
                             arrays["actions"].shape[0] + 1)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
