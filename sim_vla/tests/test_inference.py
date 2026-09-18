"""Stage 9: recurrent inference, and one real turn of the online loop.

Acting is where the coordinate systems and the recurrence meet, and both fail
quietly. A policy that forgets to reset still produces rollouts; a policy that
feeds back the action it *asked for* rather than the one that was *sent* still
produces rollouts. These check the values, not the shapes.

The online test is small on purpose -- a handful of environment steps and a
couple of updates against a fake env. It is an integration test of the loop,
not a training run.
"""

from __future__ import annotations

import unittest

import numpy as np

from .common import (DummyExpert, fake_batch, obs_shapes, require_torch,
                     small_model_config)

IMAGE = 16
PROPRIO = 9
ACTION = 8


def build_model(graph_enabled=False):
    require_torch()
    from sim_vla.models.world_model import build_world_model

    _cfg, model_cfg = small_model_config(graph_enabled)
    batch = fake_batch(graph_enabled=graph_enabled)
    model = build_world_model(model_cfg, obs_shapes(batch), ACTION,
                              graph_enabled=graph_enabled)
    return model, model_cfg


def tiny_actor(feature_dim, chunk=4):
    torch = require_torch()
    from sim_vla.models.latent_adapter import LatentAdapter

    expert = DummyExpert(32, ACTION)

    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = LatentAdapter(feature_dim, 32, hidden=32, layers=1)
            self.expert = expert
            self.expert_linear = expert.linear
            self.chunk_size = chunk
            self.action_dim = ACTION
            self.flow_steps = 2

        def condition(self, features, instruction=None):
            return {"state_token": self.adapter(features),
                    "instruction": instruction}

        def velocity_fn(self):
            return self.expert

    return Actor()


def coordinates(low=-1.0, high=1.0, normalizer=None):
    from sim_vla.models.action_space import ActionBounds, ActionCoordinates

    return ActionCoordinates(
        normalizer,
        ActionBounds(low=np.full(ACTION, low, np.float32),
                     high=np.full(ACTION, high, np.float32)),
        action_dim=ACTION)


class FakeEnv:
    """Deterministic, and it records exactly what it was commanded."""

    def __init__(self, steps=6):
        self.steps = int(steps)
        self.commands: list[np.ndarray] = []
        self.resets: list[int] = []
        self._t = 0

    def _obs(self):
        rng = np.random.default_rng(self._t)
        return {"image_base": rng.integers(
                    0, 255, (IMAGE, IMAGE, 3), dtype=np.uint8),
                "proprio": rng.standard_normal(PROPRIO).astype(np.float32)}

    def reset(self, seed=None):
        self.resets.append(int(seed or 0))
        self._t = 0
        return self._obs()

    def step(self, action):
        self.commands.append(np.asarray(action, dtype=np.float32).copy())
        self._t += 1
        return {"obs": self._obs(), "reward": float(self._t),
                "is_terminal": False, "is_last": self._t >= self.steps,
                "success": False}

    def close(self):
        pass


class TestRecurrentInference(unittest.TestCase):
    def policy(self, execute=1, coords=None):
        from sim_vla.training.online import LatentPolicy

        torch = require_torch()
        torch.manual_seed(0)
        model, _cfg = build_model()
        actor = tiny_actor(model.feature_dim)
        return LatentPolicy(model, actor, device="cpu",
                            coords=coords if coords is not None
                            else coordinates(), execute=execute), model, actor

    def test_reset_clears_the_recurrent_state(self):
        torch = require_torch()
        policy, _model, _actor = self.policy()
        env = FakeEnv()

        torch.manual_seed(1)
        policy.reset()
        first = policy(env.reset(0))

        # Drive it well into an episode, then reset and repeat the first step.
        obs = env.reset(0)
        for _ in range(4):
            obs = env.step(policy(obs))["obs"]

        torch.manual_seed(1)
        policy.reset()
        again = policy(env.reset(0))
        np.testing.assert_allclose(first, again, atol=1e-5,
                                   err_msg="the policy carried state across "
                                           "reset()")

    def test_state_actually_advances_within_an_episode(self):
        """The counterpart: if nothing is carried, reset() proves nothing."""
        torch = require_torch()
        policy, _model, _actor = self.policy()
        policy.reset()
        env = FakeEnv()
        obs = env.reset(0)
        self.assertIsNone(policy._state)
        policy(obs)
        first_state = tuple(t.clone() for t in policy._state)
        obs = env.step(policy(obs))["obs"]
        policy(obs)
        moved = any(not torch.allclose(a, b)
                    for a, b in zip(first_state, policy._state))
        self.assertTrue(moved, "the recurrent state never advanced")

    def test_the_action_fed_back_is_the_action_sent(self):
        """Clipping downstream desynchronised these; clipping is in the
        policy now."""
        require_torch()
        policy, _model, _actor = self.policy(coords=coordinates(-0.01, 0.01))
        policy.reset()
        env = FakeEnv()
        obs = env.reset(0)
        for _ in range(4):
            action = policy(obs)
            np.testing.assert_allclose(policy._prev_action, action, atol=0)
            obs = env.step(action)["obs"]
        np.testing.assert_allclose(np.stack(env.commands),
                                   np.stack(env.commands).clip(-0.01, 0.01),
                                   atol=1e-6)

    def test_commands_respect_the_recorded_bounds(self):
        """Asymmetric bounds, so a symmetric clip would not pass by luck."""
        require_torch()
        policy, _model, _actor = self.policy(coords=coordinates(-0.25, 0.5))
        policy.reset()
        env = FakeEnv()
        obs = env.reset(0)
        for _ in range(5):
            action = policy(obs)
            self.assertGreaterEqual(float(action.min()), -0.25 - 1e-6)
            self.assertLessEqual(float(action.max()), 0.5 + 1e-6)
            obs = env.step(action)["obs"]

    def test_clipping_is_counted(self):
        """Bounds tight enough that clipping is certain, so the counter is
        being tested rather than the sampler's luck."""
        require_torch()
        policy, _model, _actor = self.policy(coords=coordinates(-1e-6, 1e-6))
        policy.reset()
        env = FakeEnv()
        obs = env.reset(0)
        for _ in range(3):
            obs = env.step(policy(obs))["obs"]
        self.assertGreater(policy.clipped, 0,
                           "the clip counter was never incremented")

    def test_execute_beyond_one_is_refused_for_online_training(self):
        require_torch()
        from sim_vla.training.online import LatentPolicy

        model, _cfg = build_model()
        actor = tiny_actor(model.feature_dim, chunk=4)
        with self.assertRaises(NotImplementedError):
            LatentPolicy(model, actor, device="cpu", coords=coordinates(),
                         execute=2)

    def test_execute_outside_the_chunk_is_refused(self):
        require_torch()
        from sim_vla.training.online import LatentPolicy

        model, _cfg = build_model()
        actor = tiny_actor(model.feature_dim, chunk=4)
        with self.assertRaises(ValueError):
            LatentPolicy(model, actor, device="cpu", coords=coordinates(),
                         execute=9)
        with self.assertRaises(ValueError):
            LatentPolicy(model, actor, device="cpu", coords=coordinates(),
                         execute=0)

    def test_evaluation_resets_between_episodes(self):
        require_torch()
        from sim_vla.evaluation.policy import evaluate_policy

        policy, _model, _actor = self.policy()
        env = FakeEnv(steps=3)
        report = evaluate_policy(env, policy, episodes=2, max_steps=3)
        self.assertEqual(report["episodes"], 2)
        # reset() is called per episode; the policy is stateless across them.
        self.assertEqual(len(env.resets), 2)

    def test_evaluation_does_not_reclip_a_bounded_policy(self):
        """Exactly what the policy returned is exactly what the env received.

        The evaluation loop's own ``[-1, 1]`` default is wrong for a policy
        that carries the recorded bounds, and clipping on top of it desyncs the
        command from the ``a_(t-1)`` the policy fed its own posterior.
        """
        require_torch()
        from sim_vla.evaluation.policy import evaluate_policy

        policy, _model, _actor = self.policy(coords=coordinates(-3.0, 3.0))
        env = FakeEnv(steps=3)
        returned: list[np.ndarray] = []

        class Recording:
            bounded = policy.bounded

            def reset(self):
                policy.reset()

            def __call__(self, obs):
                action = policy(obs)
                returned.append(np.asarray(action).copy())
                return action

        evaluate_policy(env, Recording(), episodes=1, max_steps=3)
        self.assertEqual(len(returned), len(env.commands))
        for index, (asked, sent) in enumerate(zip(returned, env.commands)):
            np.testing.assert_array_equal(
                asked, sent,
                err_msg=f"step {index}: the evaluation altered the command")


class DemoStub:
    """A demonstration sampler that emits exactly the online contract.

    Built from the same ``layout.assemble`` path the replay uses, so the test
    is about the loop rather than about two loaders agreeing.
    """

    def __init__(self, replay, length, burn_in):
        self.replay = replay
        self.length = length
        self.burn_in = burn_in
        self.lookahead = 0

    def batch(self, size):
        return self.replay.sample(size, self.length, self.burn_in,
                                  self.lookahead)


def fill_replay(replay, episodes=6, steps=8):
    from sim_vla.data.replay import OnlineEpisode

    rng = np.random.default_rng(0)
    for _ in range(episodes):
        episode = OnlineEpisode()
        episode.add_observation(
            {"image_base": rng.integers(0, 255, (IMAGE, IMAGE, 3), dtype=np.uint8),
             "proprio": rng.standard_normal(PROPRIO).astype(np.float32)})
        for step in range(steps):
            episode.add_transition(
                rng.uniform(-1, 1, ACTION).astype(np.float32),
                float(step), False, step == steps - 1, False)
            episode.add_observation(
                {"image_base": rng.integers(
                     0, 255, (IMAGE, IMAGE, 3), dtype=np.uint8),
                 "proprio": rng.standard_normal(PROPRIO).astype(np.float32)})
        replay.add(episode)
    return replay


class TestOnlineLoop(unittest.TestCase):
    """Collection, replay mixing, a world-model step, and an actor step."""

    def test_one_turn_of_the_loop(self):
        torch = require_torch()
        from sim_vla.data.replay import OnlineReplay
        from sim_vla.models.critics import ValueCritic
        from sim_vla.training.actor_critic import ActorCriticConfig
        from sim_vla.training.online import OnlineConfig, run_online

        torch.manual_seed(0)
        model, model_cfg = build_model()
        actor = tiny_actor(model.feature_dim)
        critic = ValueCritic(model_cfg, model.feature_dim)
        demo = DemoStub(fill_replay(OnlineReplay(seed=0)), 4, 1)

        cfg = {"task": {"instruction": "do the thing"}, "actor": {"execute": 1},
               "eval": {"seeds_start": 900000}}
        config = OnlineConfig(total_steps=8, episodes_per_collect=2,
                              updates_per_collect=2, batch_size=4,
                              sequence_length=4, burn_in=1,
                              imagination_batch=8, max_episode_steps=4,
                              demo_fraction=0.5, seed=7)
        ac = ActorCriticConfig(horizon=2, flow_steps=2, critic_warmup=0)

        before = [p.detach().clone() for p in model.parameters()]
        seen: list[dict] = []
        trainer = run_online(cfg, model, actor, critic, demo, FakeEnv(steps=4),
                             config=config, ac_config=ac, device="cpu",
                             coords=coordinates(),
                             on_metrics=seen.append)

        self.assertGreaterEqual(trainer.env_steps, 8)
        self.assertGreaterEqual(trainer.updates, 2)
        self.assertGreater(len(trainer.replay), 0, "nothing was collected")
        self.assertTrue(seen, "no metrics were reported")

        last = seen[-1]
        self.assertIn("world_loss", last)
        self.assertIn("actor_grad_norm", last)
        self.assertGreater(last["actor_grad_norm"], 0.0,
                           "the actor received no gradient after warm-up")
        self.assertGreater(last["imagination_starts"], 0)

        moved = any(not torch.equal(old, new.detach())
                    for old, new in zip(before, model.parameters()))
        self.assertTrue(moved, "the world model was never updated")

    def test_replay_and_demonstrations_mix_without_losing_keys(self):
        require_torch()
        from sim_vla.data.replay import OnlineReplay, mixed_batch

        demo_replay = fill_replay(OnlineReplay(seed=1))
        online = fill_replay(OnlineReplay(seed=2))
        batch = mixed_batch(DemoStub(demo_replay, 4, 1), online, 8, 4, 1, 0.5)
        from sim_vla.data.batch import WINDOW_REQUIRED

        for key in WINDOW_REQUIRED:
            self.assertIn(key, batch)
        self.assertEqual(batch["action"].shape[0], 8)
        self.assertIn("image_base", batch)
        self.assertIn("proprio", batch)


if __name__ == "__main__":
    unittest.main()
