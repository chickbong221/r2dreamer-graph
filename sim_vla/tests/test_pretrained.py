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

import numpy as np

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


def build_actor(feature_dim: int, action_dim: int = 8,
                state_token_mode: str = "embedding"):
    from sim_vla.models.latent_adapter import LatentAdapter
    from sim_vla.models.pretrained import model_facts
    from sim_vla.models.smolvla_actor import SmolVLAActor

    loaded = load()
    facts = model_facts(loaded)
    # The token width is decided by the mode: embed_prefix always projects, so
    # "embedding" hands it a vlm_hidden_size token past an identity and
    # "state_proj" hands it a max_state_dim one for the real projection.
    token_dim = int(facts["vlm_hidden_size"] if state_token_mode == "embedding"
                    else facts["max_state_dim"])
    adapter = LatentAdapter(feature_dim=feature_dim, token_dim=token_dim,
                            hidden=256)
    actor = SmolVLAActor(loaded, adapter, action_dim=action_dim,
                         instruction="pick up the cube",
                         state_token_mode=state_token_mode)
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
        wrong = LatentAdapter(
            feature_dim=64,
            token_dim=int(model_facts(loaded)["vlm_hidden_size"]) + 1)
        with self.assertRaises(PretrainedError):
            SmolVLAActor(loaded, wrong, action_dim=8)

    def test_prefix_cache_is_built_for_both_arms(self):
        torch = require_torch()
        for width in (BASELINE_FEATURE, GRAPH_FEATURE):
            with self.subTest(feature_dim=width):
                actor, _facts = build_actor(width)
                cond = actor.condition(torch.randn(2, width, device=actor.device))
                self.assertIn("past_key_values", cond)
                self.assertIn("prefix_pad_masks", cond)
                self.assertEqual(cond["prefix_pad_masks"].shape[0], 2)
                self.assertEqual(tuple(cond["state_token"].shape),
                                 (2, 1, actor.vlm_hidden))

    def test_action_padding_matches_the_checkpoint(self):
        torch = require_torch()
        actor, facts = build_actor(BASELINE_FEATURE)
        padded = actor.pad_actions(
            torch.randn(2, actor.chunk_size, 8, device=actor.device))
        self.assertEqual(padded.shape[-1], int(facts["max_action_dim"]))
        self.assertTrue(torch.all(padded[..., 8:] == 0))
        mask = actor.action_dim_mask(actor.device)
        self.assertEqual(int(mask.sum()), 8)
        self.assertEqual(mask.numel(), int(facts["max_action_dim"]))


class TestStateTokenModes(unittest.TestCase):
    """Both ways of filling SmolVLA's state slot, against the real model.

    embed_prefix calls state_proj unconditionally: a 960-wide token meets a
    32->960 Linear and fails with "mat1 and mat2 shapes cannot be multiplied".
    "embedding" mode steps over that projection for the call; "state_proj"
    mode feeds it a 32-wide token instead.
    """

    def test_state_proj_mode_uses_the_real_projection(self):
        torch = require_torch()
        actor, facts = build_actor(BASELINE_FEATURE,
                                   state_token_mode="state_proj")
        self.assertEqual(actor.adapter.token_dim, int(facts["max_state_dim"]))
        cond = actor.condition(
            torch.randn(2, BASELINE_FEATURE, device=actor.device))
        self.assertIn("past_key_values", cond)

    def test_embedding_mode_steps_over_the_projection(self):
        torch = require_torch()
        actor, facts = build_actor(BASELINE_FEATURE,
                                   state_token_mode="embedding")
        self.assertEqual(actor.adapter.token_dim,
                         int(facts["vlm_hidden_size"]))
        cond = actor.condition(
            torch.randn(2, BASELINE_FEATURE, device=actor.device))
        self.assertIn("past_key_values", cond)

    def test_the_projection_is_restored_after_the_call(self):
        """The swap is for the duration of embed_prefix and nothing longer."""
        torch = require_torch()
        actor, _ = build_actor(BASELINE_FEATURE, state_token_mode="embedding")
        before = actor.model.state_proj
        actor.condition(torch.randn(2, BASELINE_FEATURE, device=actor.device))
        self.assertIs(actor.model.state_proj, before)
        self.assertTrue(hasattr(actor.model.state_proj, "in_features"))


class TestDevicePlacement(unittest.TestCase):
    def test_adapter_is_moved_to_the_pretrained_weights(self):
        """A CPU adapter and a CUDA policy meet inside an embedding lookup.

        The failure is "index is on cpu, different from other tensors on
        cuda:0", raised from the token embedding -- which names neither the
        adapter nor the policy.
        """
        require_torch()
        actor, _ = build_actor(BASELINE_FEATURE)
        weights = next(actor.model.parameters()).device
        for parameter in actor.adapter.parameters():
            self.assertEqual(parameter.device.type, weights.type)

    def test_condition_accepts_features_from_elsewhere(self):
        torch = require_torch()
        actor, _ = build_actor(BASELINE_FEATURE)
        # A world model on another device should not be the caller's problem.
        cond = actor.condition(torch.randn(2, BASELINE_FEATURE, device="cpu"))
        self.assertEqual(cond["state_token"].device.type, actor.device.type)


class TestFlowIntegration(unittest.TestCase):
    def test_loss_forward_backward_and_gradients(self):
        """The real expert: finite loss, gradient in adapter and expert, none frozen."""
        torch = require_torch()
        from sim_vla.models.flow_sampler import flow_matching_loss

        for width in (BASELINE_FEATURE, GRAPH_FEATURE):
            with self.subTest(feature_dim=width):
                actor, _ = build_actor(width)
                cond = actor.condition(torch.randn(2, width, device=actor.device))
                actions = torch.randn(2, actor.chunk_size, actor.action_dim,
                                      device=actor.device)
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
        cond = actor.condition(torch.randn(2, GRAPH_FEATURE, device=actor.device))
        chunk = sample_actions(
            actor.velocity_fn(), cond, batch=2, chunk=actor.chunk_size,
            dim=actor.action_dim, steps=2, differentiable=True,
            device=actor.device)
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


class TestChunkSynchronisation(unittest.TestCase):
    """The pretrained stack reads its own ``config.chunk_size``.

    Setting only the wrapper's copy gave a policy supervised on one horizon
    and predicting another, with nothing to show for it but a loss that
    descended.
    """

    def test_the_checkpoint_value_is_taken_by_default(self):
        require_torch()
        actor, facts = build_actor(BASELINE_FEATURE)
        self.assertEqual(actor.chunk_size, facts["chunk_size"])
        self.assertEqual(int(actor.model.config.chunk_size), actor.chunk_size)

    def test_an_override_reaches_the_model_config(self):
        require_torch()
        from sim_vla.models.latent_adapter import LatentAdapter
        from sim_vla.models.pretrained import model_facts
        from sim_vla.models.smolvla_actor import SmolVLAActor

        loaded = load()
        facts = model_facts(loaded)
        wanted = max(int(facts["chunk_size"]) // 2, 1)
        adapter = LatentAdapter(feature_dim=BASELINE_FEATURE,
                                token_dim=int(facts["vlm_hidden_size"]),
                                hidden=256)
        actor = SmolVLAActor(loaded, adapter, action_dim=8,
                             chunk_size=wanted, instruction="x")
        try:
            self.assertEqual(actor.chunk_size, wanted)
            self.assertEqual(int(actor.model.config.chunk_size), wanted)
            self.assertLessEqual(int(actor.model.config.n_action_steps), wanted)
        finally:
            # Restore, because load() caches the policy for the process.
            actor._synchronise_chunk(int(facts["chunk_size"]))


class TestRealSmokeRun(unittest.TestCase):
    """A real forward and backward through the pretrained expert, and a short
    recurrent inference run driving it.

    A DummyExpert test proves the plumbing; it does not prove the pretrained
    integration works. This does, on the real weights, in a handful of steps.
    """

    def test_forward_backward_updates_only_the_trainable_set(self):
        torch = require_torch()
        from sim_vla.models.flow_sampler import flow_matching_loss

        actor, _ = build_actor(BASELINE_FEATURE)
        feat = torch.randn(2, BASELINE_FEATURE, device=actor.device)
        cond = actor.condition(feat)
        targets = torch.zeros(2, actor.chunk_size, actor.action_dim,
                              device=actor.device)
        torch.manual_seed(0)
        loss, _metrics = flow_matching_loss(actor.velocity_fn(), targets, cond)
        loss.backward()

        got = [n for n, p in actor.named_parameters()
               if p.grad is not None and float(p.grad.abs().sum()) > 0]
        self.assertTrue(any("adapter" in n for n in got),
                        "no gradient reached the adapter")
        frozen_with_grad = [n for n, p in actor.named_parameters()
                            if not p.requires_grad and p.grad is not None]
        self.assertFalse(frozen_with_grad,
                         f"frozen parameters accumulated gradients: "
                         f"{frozen_with_grad[:5]}")
        for parameter in actor.parameters():
            parameter.grad = None

    def test_short_recurrent_inference_run(self):
        torch = require_torch()
        from sim_vla.tests.common import fake_batch, obs_shapes, \
            small_model_config
        from sim_vla.models.world_model import build_world_model
        from sim_vla.training.online import LatentPolicy
        from sim_vla.tests.test_inference import FakeEnv, coordinates

        torch.manual_seed(0)
        _cfg, model_cfg = small_model_config(False)
        batch = fake_batch(graph_enabled=False)
        model = build_world_model(model_cfg, obs_shapes(batch), 8,
                                  graph_enabled=False)
        actor, _ = build_actor(model.feature_dim)
        actor.to("cpu")
        policy = LatentPolicy(model, actor, device="cpu",
                              coords=coordinates(), execute=1)
        policy.reset()
        env = FakeEnv(steps=3)
        obs = env.reset(0)
        for _ in range(3):
            action = policy(obs)
            self.assertEqual(action.shape, (8,))
            self.assertTrue(np.isfinite(action).all())
            np.testing.assert_allclose(policy._prev_action, action, atol=0)
            obs = env.step(action)["obs"]
        self.assertEqual(len(env.commands), 3)


if __name__ == "__main__":
    unittest.main()
