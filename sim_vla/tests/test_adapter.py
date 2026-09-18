"""Stage 4: the single-token adapter, and gradient flow into it."""

from __future__ import annotations

import unittest

from .common import DummyExpert, require_torch


class TestLatentAdapter(unittest.TestCase):
    def test_emits_exactly_one_token(self):
        torch = require_torch()
        from sim_vla.models.latent_adapter import LatentAdapter

        adapter = LatentAdapter(feature_dim=48, token_dim=32, hidden=64)
        out = adapter(torch.zeros(5, 48))
        # One state token. No context-token block is projected.
        self.assertEqual(tuple(out.shape), (5, 1, 32))

    def test_arms_differ_only_in_input_width(self):
        require_torch()
        from sim_vla.models.latent_adapter import LatentAdapter

        base = LatentAdapter(feature_dim=48, token_dim=32, hidden=64)
        graph = LatentAdapter(feature_dim=64, token_dim=32, hidden=64)
        self.assertEqual(base.token_dim, graph.token_dim)
        self.assertEqual(base.hidden, graph.hidden)
        gap = (graph.parameter_report()["total"]
               - base.parameter_report()["total"])
        # The graph arm's extra capacity is the wider first layer and nothing
        # else, and it is reported rather than left to be discovered.
        self.assertEqual(gap, (64 - 48) * 64)

    def test_wrong_feature_width_is_refused(self):
        torch = require_torch()
        from sim_vla.models.latent_adapter import LatentAdapter

        adapter = LatentAdapter(feature_dim=48, token_dim=32, hidden=64)
        with self.assertRaises(ValueError):
            adapter(torch.zeros(2, 64))

    def test_gradient_reaches_the_adapter_through_the_expert(self):
        torch = require_torch()
        from sim_vla.models.flow_sampler import flow_matching_loss
        from sim_vla.models.latent_adapter import LatentAdapter

        adapter = LatentAdapter(feature_dim=48, token_dim=32, hidden=64)
        expert = DummyExpert(token_dim=32, action_dim=8)
        cond = {"state_token": adapter(torch.randn(4, 48)), "instruction": None}
        loss, _ = flow_matching_loss(expert, torch.randn(4, 6, 8), cond)
        loss.backward()
        grads = [p.grad for p in adapter.parameters() if p.grad is not None]
        self.assertTrue(grads, "no gradient reached the adapter")
        self.assertTrue(any(g.abs().sum() > 0 for g in grads))

    def test_gradient_passes_through_a_frozen_module(self):
        """Frozen means requires_grad=False, not a cut graph."""
        torch = require_torch()
        from sim_vla.models.latent_adapter import LatentAdapter
        from sim_vla.models.smolvla_actor import freeze

        adapter = LatentAdapter(feature_dim=48, token_dim=32, hidden=64)
        expert = DummyExpert(token_dim=32, action_dim=8)
        frozen = freeze(expert.linear)
        self.assertGreater(frozen, 0)
        cond = {"state_token": adapter(torch.randn(3, 48)), "instruction": None}
        out = expert(torch.randn(3, 4, 8), torch.rand(3), cond)
        out.sum().backward()
        self.assertTrue(any(p.grad is not None for p in adapter.parameters()))
        self.assertTrue(all(p.grad is None for p in expert.linear.parameters()))


if __name__ == "__main__":
    unittest.main()
