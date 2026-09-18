"""Stage 4: the real checkpoint, the real forward path, real gradients.

Nothing here is a mock. The checkpoint is loaded at a resolved commit, the
adapter is built at the width the *loaded model* reports, a flow-matching loss
runs forward and backward through the frozen transformer, and an action chunk
is sampled with gradients enabled. Both arms' feature widths are exercised.

Skips are narrow and honest. A missing lerobot, or weights that cannot be
reached, skips -- and ``run_stage`` marks this stage INCOMPLETE because it is
listed as required, so a skipped stage 4 can never read as a pass. Anything
else, including an interface that has moved, **fails**: that is the result
this stage exists to produce.
"""

from __future__ import annotations

import unittest

from .common import require_torch

# Widths standing in for a baseline (h, z) and a graph arm (h, z, g). The
# adapter's input width is the only thing that differs between the arms.
BASELINE_FEATURE = 4224
GRAPH_FEATURE = 4224 + 512


def load():
    """Load once per process. An unreachable checkpoint skips; a broken one fails."""
    from sim_vla.models.pretrained import PretrainedError, load_policy

    if not hasattr(load, "_cached"):
        try:
            import lerobot  # noqa: F401
        except Exception as exc:                           # noqa: BLE001
            raise unittest.SkipTest(f"lerobot not installed: {exc}")
        try:
            load._cached = load_policy(revision="main")
        except PretrainedError as exc:
            message = str(exc)
            # Only "cannot reach it" is a skip. "It loaded and the interface is
            # wrong" is the finding.
            if any(word in message.lower() for word in
                   ("could not resolve", "connection", "offline", "401", "403",
                    "not importable")):
                raise unittest.SkipTest(message)
            raise
    return load._cached


def build_actor(feature_dim: int, action_dim: int = 8):
    from sim_vla.models.latent_adapter import LatentAdapter
    from sim_vla.models.pretrained import model_facts
    from sim_vla.models.smolvla_actor import SmolVLAActor

    loaded = load()
    facts = model_facts(loaded)
    adapter = LatentAdapter(feature_dim=feature_dim,
                            token_dim=int(facts["vlm_hidden_size"]), hidden=256)
    actor = SmolVLAActor(loaded, adapter, action_dim=action_dim,
                         instruction="pick up the cube")
    return actor, facts


class TestLoading(unittest.TestCase):
    def test_revision_resolves_to_an_immutable_commit(self):
        loaded = load()
        self.assertRegex(loaded.revision, r"^[0-9a-f]{40}$")
        self.assertNotEqual(loaded.revision, loaded.requested)

    def test_model_exposes_the_flow_interface_this_targets(self):
        loaded = load()
        for name in ("embed_prefix", "embed_suffix", "denoise_step"):
            self.assertTrue(callable(getattr(loaded.model, name, None)),
                            f"VLAFlowMatching.{name} is missing")
        for name in ("state_proj", "action_in_proj", "action_out_proj"):
            self.assertIsNotNone(getattr(loaded.model, name, None), name)

    def test_facts_come_from_the_model_not_from_defaults(self):
        from sim_vla.models.pretrained import model_facts

        facts = model_facts(load())
        for key in ("vlm_hidden_size", "expert_hidden_size", "max_action_dim",
                    "chunk_size", "num_steps"):
            self.assertGreater(int(facts[key]), 0, key)
        # The widths are read off the real layers, so they must agree with the
        # projections they were read from.
        model = load().model
        self.assertEqual(facts["vlm_hidden_size"], model.state_proj.out_features)
        self.assertEqual(facts["action_in_dim"], facts["max_action_dim"])


class TestConditioning(unittest.TestCase):
    def test_adapter_width_must_match_the_loaded_model(self):
        require_torch()
        from sim_vla.models.latent_adapter import LatentAdapter
        from sim_vla.models.pretrained import PretrainedError, model_facts
        from sim_vla.models.smolvla_actor import SmolVLAActor

        loaded = load()
        wrong = LatentAdapter(feature_dim=64,
                              token_dim=int(model_facts(loaded)["vlm_hidden_size"]) + 1)
        with self.assertRaises(PretrainedError):
            SmolVLAActor(loaded, wrong, action_dim=8)

    def test_prefix_cache_is_built_for_both_arms(self):
        torch = require_torch()
        for width in (BASELINE_FEATURE, GRAPH_FEATURE):
            with self.subTest(feature_dim=width):
                actor, _facts = build_actor(width)
                cond = actor.condition(torch.randn(2, width))
                self.assertIn("past_key_values", cond)
                self.assertIn("prefix_pad_masks", cond)
                self.assertEqual(cond["prefix_pad_masks"].shape[0], 2)
                self.assertEqual(tuple(cond["state_token"].shape),
                                 (2, 1, actor.vlm_hidden))

    def test_action_padding_matches_the_checkpoint(self):
        torch = require_torch()
        actor, facts = build_actor(BASELINE_FEATURE)
        padded = actor.pad_actions(torch.randn(2, actor.chunk_size, 8))
        self.assertEqual(padded.shape[-1], int(facts["max_action_dim"]))
        self.assertTrue(torch.all(padded[..., 8:] == 0))
        mask = actor.action_dim_mask()
        self.assertEqual(int(mask.sum()), 8)
        self.assertEqual(mask.numel(), int(facts["max_action_dim"]))


class TestFlowIntegration(unittest.TestCase):
    def test_loss_forward_backward_and_gradients(self):
        """The real expert: finite loss, gradient in adapter and expert, none frozen."""
        torch = require_torch()
        from sim_vla.models.flow_sampler import flow_matching_loss

        for width in (BASELINE_FEATURE, GRAPH_FEATURE):
            with self.subTest(feature_dim=width):
                actor, _ = build_actor(width)
                cond = actor.condition(torch.randn(2, width))
                actions = torch.randn(2, actor.chunk_size, actor.action_dim)
                loss, metrics = flow_matching_loss(
                    actor.velocity_fn(), actions, cond)
                self.assertTrue(torch.isfinite(loss), metrics)
                loss.backward()

                adapter_grads = [p.grad for p in actor.adapter.parameters()
                                 if p.grad is not None]
                self.assertTrue(adapter_grads, "no gradient reached the adapter")
                self.assertTrue(any(g.abs().sum() > 0 for g in adapter_grads),
                                "adapter gradients are all zero")

                trainable = [p for p in actor._expert_parameters()
                             if p.requires_grad]
                self.assertTrue(trainable, "no expert parameter is trainable")
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for p in trainable),
                                "no gradient reached the action expert")

                frozen = [p for p in actor.model.parameters()
                          if not p.requires_grad]
                self.assertTrue(frozen, "nothing is frozen")
                self.assertTrue(all(p.grad is None for p in frozen),
                                "a frozen parameter accumulated a gradient")

    def test_sampling_produces_finite_actions_with_gradients(self):
        torch = require_torch()
        from sim_vla.models.flow_sampler import (assert_gradient_reaches,
                                                 sample_actions)

        actor, _ = build_actor(GRAPH_FEATURE)
        cond = actor.condition(torch.randn(2, GRAPH_FEATURE))
        chunk = sample_actions(
            actor.velocity_fn(), cond, batch=2, chunk=actor.chunk_size,
            dim=actor.action_dim, steps=2, differentiable=True)
        self.assertEqual(tuple(chunk.shape),
                         (2, actor.chunk_size, actor.action_dim))
        self.assertTrue(torch.isfinite(chunk).all())
        assert_gradient_reaches(chunk, *actor.adapter.parameters())

    def test_report_records_the_pin_and_what_trains(self):
        require_torch()
        actor, _ = build_actor(BASELINE_FEATURE)
        report = actor.trainable_report()
        print("[pretrained]", report)
        self.assertRegex(report["revision"], r"^[0-9a-f]{40}$")
        self.assertIn("adapter", report["trainable_prefixes"])
        self.assertGreater(report["parameters_trainable"], 0)
        self.assertLess(report["parameters_trainable"],
                        report["parameters_total"])


if __name__ == "__main__":
    unittest.main()
