"""Stage 3: the world model constructs and trains under both graph settings."""

from __future__ import annotations

import unittest

from .common import fake_batch, obs_shapes, require_torch, small_model_config


class TestWorldModelArms(unittest.TestCase):
    def build(self, graph_enabled):
        from sim_vla.models.world_model import build_world_model

        _cfg, model_cfg = small_model_config(graph_enabled)
        batch = fake_batch(graph_enabled=graph_enabled)
        model = build_world_model(model_cfg, obs_shapes(batch), 8,
                                  graph_enabled=graph_enabled)
        return model, batch

    def test_baseline_builds_no_graph_components(self):
        require_torch()
        model, _ = self.build(False)
        self.assertIsNone(model.graph_encoder)
        self.assertIsNone(model.graph_decoder)
        self.assertFalse(model.rssm.semantic)
        self.assertEqual(model.graph_dim, 0)

    def test_graph_arm_builds_them(self):
        require_torch()
        model, _ = self.build(True)
        self.assertIsNotNone(model.graph_encoder)
        self.assertIsNotNone(model.graph_decoder)
        self.assertTrue(model.rssm.semantic)
        self.assertGreater(model.graph_dim, 0)

    def test_feature_widths_differ_by_the_semantic_dim(self):
        require_torch()
        base, _ = self.build(False)
        graph, _ = self.build(True)
        self.assertEqual(graph.feature_dim, base.feature_dim + graph.graph_dim)

    def test_both_arms_take_a_gradient(self):
        torch = require_torch()
        for graph_enabled in (False, True):
            with self.subTest(graph=graph_enabled):
                model, batch = self.build(graph_enabled)
                total, losses, _ = model.loss(batch)
                self.assertTrue(torch.isfinite(total), losses)
                total.backward()
                grads = [p.grad for p in model.parameters()
                         if p.requires_grad and p.grad is not None]
                self.assertTrue(grads, "no parameter received a gradient")

    def test_graph_losses_only_in_the_graph_arm(self):
        require_torch()
        base, base_batch = self.build(False)
        _total, base_losses, _ = base.loss(base_batch)
        self.assertFalse([k for k in base_losses if k.startswith("graph")])
        graph, graph_batch = self.build(True)
        _total, graph_losses, _ = graph.loss(graph_batch)
        self.assertTrue([k for k in graph_losses if k.startswith("graph")])

    def test_baseline_refuses_a_graph_token(self):
        require_torch()
        model, batch = self.build(False)
        self.assertIsNone(model.graph_token(batch))

    def test_graph_arm_refuses_a_dataset_without_graphs(self):
        require_torch()
        from sim_vla.models.world_model import build_world_model

        _cfg, model_cfg = small_model_config(True)
        batch = fake_batch(graph_enabled=False)
        with self.assertRaises(ValueError):
            build_world_model(model_cfg, obs_shapes(batch), 8,
                              graph_enabled=True)


class TestLossEquivalence(unittest.TestCase):
    """The loss terms are the simulator's, not a second implementation."""

    def test_kl_and_semantic_terms_come_from_the_rssm(self):
        require_torch()
        import inspect

        from sim_vla.models import world_model

        source = inspect.getsource(world_model.WorldModel.loss)
        # Called on the RSSM rather than reimplemented here: a second copy of
        # the KL or the semantic alignment is how the arms drift apart.
        for call in ("self.rssm.kl_loss", "self.rssm.semantic_align_loss",
                     "self.rssm.semantic_amplitude_loss",
                     "self.rssm.semantic_prior_seq", "self.graph_decoder("):
            self.assertIn(call, source)


if __name__ == "__main__":
    unittest.main()
