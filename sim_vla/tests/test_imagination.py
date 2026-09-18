"""Stage 7: the flow sampler keeps gradients, imagination uses the prior."""

from __future__ import annotations

import unittest

from .common import (DummyExpert, fake_batch, obs_shapes, require_torch,
                     small_model_config)


class TestFlowSampler(unittest.TestCase):
    def test_sampled_action_carries_gradient_to_the_conditioning(self):
        torch = require_torch()
        from sim_vla.models.flow_sampler import (assert_gradient_reaches,
                                                 sample_actions)
        from sim_vla.models.latent_adapter import LatentAdapter

        adapter = LatentAdapter(feature_dim=16, token_dim=16, hidden=32)
        expert = DummyExpert(token_dim=16, action_dim=4)
        cond = {"state_token": adapter(torch.randn(3, 16)), "instruction": None}
        action = sample_actions(expert, cond, batch=3, chunk=4, dim=4, steps=5,
                                differentiable=True)
        assert_gradient_reaches(action, *adapter.parameters())

    def test_no_grad_sampling_is_detected_not_silent(self):
        torch = require_torch()
        from sim_vla.models.flow_sampler import (assert_gradient_reaches,
                                                 sample_actions)
        from sim_vla.models.latent_adapter import LatentAdapter

        adapter = LatentAdapter(feature_dim=16, token_dim=16, hidden=32)
        expert = DummyExpert(token_dim=16, action_dim=4)
        cond = {"state_token": adapter(torch.randn(3, 16)), "instruction": None}
        action = sample_actions(expert, cond, batch=3, chunk=4, dim=4, steps=5,
                                differentiable=False)
        # A detached actor trains forever without improving; this is where it
        # is supposed to be caught.
        with self.assertRaises(RuntimeError):
            assert_gradient_reaches(action, *adapter.parameters())


class TestImagination(unittest.TestCase):
    def rollout(self, graph_enabled, horizon=3):
        torch = require_torch()
        from sim_vla.models.world_model import build_world_model
        from sim_vla.training.imagination import flatten_start, imagine

        _cfg, model_cfg = small_model_config(graph_enabled)
        batch = fake_batch(graph_enabled=graph_enabled)
        model = build_world_model(model_cfg, obs_shapes(batch), 8,
                                  graph_enabled=graph_enabled)
        out = model.observe(batch)
        start = flatten_start(out["post"], graph_enabled)
        action_fn = lambda feat: torch.zeros(feat.shape[0], 8)
        return model, imagine(model, None, start, horizon, action_fn=action_fn)

    def test_both_arms_roll_forward(self):
        require_torch()
        for graph_enabled in (False, True):
            with self.subTest(graph=graph_enabled):
                model, out = self.rollout(graph_enabled)
                self.assertEqual(out["feat"].shape[-1], model.feature_dim)
                self.assertEqual(out["feat"].shape[0], 4)   # horizon + 1
                self.assertEqual(out["action"].shape[0], 3)

    def test_graph_arm_uses_the_semantic_prior_not_an_extractor(self):
        require_torch()
        import inspect

        from sim_vla.training import imagination

        source = inspect.getsource(imagination.imagine)
        self.assertIn("semantic_prior", source)
        # There is no scene inside imagination to extract a graph from.
        for forbidden in ("GraphEncoder", "FigureGraphSource", "pack_graph"):
            self.assertNotIn(forbidden, source)


class TestLambdaReturn(unittest.TestCase):
    def test_bootstraps_when_nothing_terminates(self):
        torch = require_torch()
        from sim_vla.training.imagination import lambda_return

        reward = torch.zeros(3, 2)
        value = torch.ones(4, 2) * 5.0
        cont = torch.ones(3, 2)
        out = lambda_return(reward, value, cont, discount=1.0, lam=1.0)
        # With no reward and a constant value the return is the bootstrap.
        self.assertTrue(torch.allclose(out, torch.full((3, 2), 5.0)))


if __name__ == "__main__":
    unittest.main()
