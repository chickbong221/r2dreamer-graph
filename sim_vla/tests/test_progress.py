"""Stage 9: progress shaping is optional, graph-dependent, and kept apart."""

from __future__ import annotations

import unittest

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
        """The environment reward and the shaping term are added, not merged."""
        require_torch()
        import inspect

        from sim_vla.training import actor_critic

        source = inspect.getsource(actor_critic.actor_loss)
        self.assertIn("progress_beta", source)
        self.assertIn("progress_reward", source)


if __name__ == "__main__":
    unittest.main()
