"""Stage 9: progress shaping is optional, graph-dependent, and kept apart."""

from __future__ import annotations

import unittest
from pathlib import Path

from .common import require_torch, small_model_config


class TestProgressConfig(unittest.TestCase):
    def test_progress_requires_the_graph(self):
        from sim_vla.config import load_config

        with self.assertRaises(SystemExit):
            load_config("pickcube", "dreamer",
                        {"model": {"progress": {"enabled": True}}})
        cfg = load_config("pickcube", "graph_progress")
        self.assertTrue(cfg["model"]["progress"]["enabled"])

    def test_head_is_not_built_for_a_baseline(self):
        require_torch()
        from sim_vla.training.progress import build_progress

        _cfg, model_cfg = small_model_config(False)
        self.assertIsNone(build_progress(model_cfg, 32, graph_enabled=False,
                                         progress_enabled=False))
        with self.assertRaises(SystemExit):
            build_progress(model_cfg, 32, graph_enabled=False,
                           progress_enabled=True)

    def test_head_is_built_for_the_graph_arm(self):
        require_torch()
        from sim_vla.training.progress import build_progress

        _cfg, model_cfg = small_model_config(True)
        head = build_progress(model_cfg, 32, graph_enabled=True,
                              progress_enabled=True)
        self.assertIsNotNone(head)


class TestShaping(unittest.TestCase):
    def test_beta_warms_up_and_is_zero_when_disabled(self):
        # beta_at is torch-free, but it lives in a module that defines an
        # nn.Module, so the import needs the guard.
        require_torch()
        from sim_vla.training.progress import ProgressConfig, beta_at

        off = ProgressConfig(enabled=False, beta=0.5)
        self.assertEqual(beta_at(off, 10_000_000), 0.0)
        on = ProgressConfig(enabled=True, beta=0.5, warmup_start=100,
                            warmup_end=200)
        self.assertEqual(beta_at(on, 50), 0.0)
        self.assertAlmostEqual(beta_at(on, 150), 0.25)
        self.assertAlmostEqual(beta_at(on, 500), 0.5)

    def test_shaping_is_potential_based(self):
        torch = require_torch()
        from sim_vla.training.progress import build_progress, shaping_reward

        _cfg, model_cfg = small_model_config(True)
        head = build_progress(model_cfg, 8, graph_enabled=True,
                              progress_enabled=True)
        feat = torch.randn(4, 3, 8)
        out = shaping_reward(head, feat, discount=0.99)
        # gamma * phi(s') - phi(s): one value per transition, not per state.
        self.assertEqual(out.shape[0], feat.shape[0] - 1)
        phi = head.potential(feat)
        self.assertTrue(torch.allclose(out, 0.99 * phi[1:] - phi[:-1], atol=1e-5))

    def test_reward_streams_stay_separate(self):
        """The environment reward and the shaping term are added, not merged.

        Behavioural: the same rollout with and without a shaping stream must
        differ by exactly ``beta`` times that stream, and the reported
        environment reward must be the unshaped one.
        """
        torch = require_torch()
        from types import SimpleNamespace

        import sim_vla.training.actor_critic as module
        from sim_vla.training.actor_critic import ActorCriticConfig

        feat = torch.zeros(3, 1, 2)
        heads = {"reward": torch.tensor([[0.0], [1.0], [1.0]]),
                 "cont": torch.ones(3, 1)}
        original = (module.imagine, module.imagined_rewards)
        module.imagine = lambda *a, **k: {"feat": feat,
                                          "action": torch.zeros(2, 1, 2),
                                          "action_steps": []}
        module.imagined_rewards = lambda _m, _f: heads
        self.addCleanup(lambda: setattr(module, "imagine", original[0]))
        self.addCleanup(lambda: setattr(module, "imagined_rewards",
                                        original[1]))

        class Critic:
            def parameters(self):
                return iter(())

            def value(self, f):
                return torch.zeros(f.shape[0], f.shape[1])

            def target_value(self, f, *, detach=False):
                return torch.zeros(1, f.shape[1])

        world = SimpleNamespace(parameters=lambda: iter(()))
        plain = module.actor_loss(world, None, Critic(), None,
                                  ActorCriticConfig(horizon=2, discount=1.0,
                                                    lam=1.0))
        shaped = module.actor_loss(
            world, None, Critic(), None,
            ActorCriticConfig(horizon=2, discount=1.0, lam=1.0,
                              progress_beta=0.5),
            progress_reward=torch.tensor([[2.0], [4.0]]))
        # The environment reward reported is the unshaped one either way.
        self.assertEqual([float(v) for v in plain["reward"].reshape(-1)],
                         [1.0, 1.0])
        # And the return differs by beta * the shaping stream.
        self.assertAlmostEqual(float(shaped["returns"][1, 0])
                               - float(plain["returns"][1, 0]),
                               0.5 * 4.0, places=5)


class TestProgressArmIsRefused(unittest.TestCase):
    """graph_progress has no supervision contract in this package.

    Constructing the head and never training it would produce a third arm
    identical to the second, reported as a different method. That reads as a
    null result rather than a missing feature, so it is refused up front.
    """

    def cfg(self, enabled=True, graph=True):
        return {"model": {"graph": {"enabled": graph},
                          "progress": {"enabled": enabled, "beta": 0.1}}}

    def test_the_arm_is_refused_before_any_training(self):
        require_torch()
        from sim_vla.training.progress import preflight

        with self.assertRaises(SystemExit) as caught:
            preflight(self.cfg(), {})
        message = str(caught.exception)
        self.assertIn("graph_progress", message)
        # It says what is missing, not just that something is.
        self.assertIn("schedule", message)
        self.assertIn("progress", message)

    def test_the_other_arms_pass_through(self):
        require_torch()
        from sim_vla.training.progress import preflight

        preflight(self.cfg(enabled=False))
        preflight(self.cfg(enabled=False, graph=False))

    def test_the_pipeline_refuses_it_before_stage_1a(self):
        require_torch()
        from sim_vla.training import pipeline, pretrain_world_model

        called = []
        original = pretrain_world_model.run
        pretrain_world_model.run = lambda *a, **k: called.append(1)
        self.addCleanup(
            lambda: setattr(pretrain_world_model, "run", original))
        with self.assertRaises(SystemExit):
            pipeline.run(self.cfg() | {"task": {"dataset": "missing.h5"}},
                         world_steps=1, imitation_steps=0, online_steps=0,
                         device="cpu", root=Path("."))
        self.assertFalse(called, "Stage 1A ran before the arm was refused")


if __name__ == "__main__":
    unittest.main()
