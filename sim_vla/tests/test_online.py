"""Stage 8: critic targets, actor gradients, and frozen parameters staying so."""

from __future__ import annotations

import unittest

import numpy as np

from .common import (DummyExpert, fake_batch, obs_shapes, require_torch,
                     small_model_config)


def build(graph_enabled):
    torch = require_torch()
    from sim_vla.models.critics import ValueCritic
    from sim_vla.models.world_model import build_world_model

    _cfg, model_cfg = small_model_config(graph_enabled)
    batch = fake_batch(graph_enabled=graph_enabled)
    model = build_world_model(model_cfg, obs_shapes(batch), 8,
                              graph_enabled=graph_enabled)
    critic = ValueCritic(model_cfg, model.feature_dim)
    return model, critic, batch


class TestCritic(unittest.TestCase):
    def test_target_starts_equal_and_then_lags(self):
        """The slow copy moves a fraction of the way, not all of it.

        The perturbation has to be random. The value head is a 255-bin
        symexp_twohot, so adding the *same* constant to every parameter shifts
        every logit equally and the softmax -- and therefore the mode -- does
        not move at all.
        """
        torch = require_torch()
        torch.manual_seed(0)
        model, critic, _ = build(False)
        feat = torch.randn(4, model.feature_dim)
        self.assertTrue(torch.allclose(critic.value(feat),
                                       critic.target_value(feat), atol=1e-5))

        for parameter in critic.net.parameters():
            parameter.data.add_(torch.randn_like(parameter))
        live, target = critic.value(feat), critic.target_value(feat)
        self.assertFalse(torch.allclose(live, target, atol=1e-3),
                         "the target moved with the live head")

        critic.update_target()
        after = critic.target_value(feat)
        # It moved toward the live head without arriving.
        self.assertFalse(torch.allclose(after, live, atol=1e-3))
        self.assertFalse(torch.allclose(after, target, atol=1e-6))

    def test_continuation_is_read_as_a_probability(self):
        """cont is a binary head: .mode is a property, .mean is the probability."""
        torch = require_torch()
        from sim_vla.training.imagination import imagined_rewards

        model, _critic, _ = build(False)
        feat = torch.randn(3, 2, model.feature_dim)
        heads = imagined_rewards(model, feat)
        self.assertEqual(heads["cont"].shape, (3, 2))
        self.assertEqual(heads["reward"].shape, (3, 2))
        # A probability, not a 0/1 mode.
        self.assertTrue(((heads["cont"] >= 0) & (heads["cont"] <= 1)).all())
        self.assertTrue(torch.isfinite(heads["reward"]).all())

    def test_targets_are_detached(self):
        torch = require_torch()
        model, critic, _ = build(False)
        feat = torch.randn(4, model.feature_dim, requires_grad=True)
        returns = torch.randn(4, requires_grad=True)
        loss = critic.loss(feat, returns)
        loss.backward()
        # The actor's gradient must not arrive through the critic's target.
        self.assertIsNone(returns.grad)


class TestActorUpdate(unittest.TestCase):
    def make_actor(self, model):
        torch = require_torch()
        from sim_vla.models.latent_adapter import LatentAdapter

        class Actor(torch.nn.Module):
            chunk_size, action_dim = 4, 8

            def __init__(self):
                super().__init__()
                self.adapter = LatentAdapter(model.feature_dim, 16, hidden=32)
                self.expert = DummyExpert(16, 8)
                self.expert_linear = self.expert.linear

            def condition(self, feat, instruction=None):
                return {"state_token": self.adapter(feat),
                        "instruction": instruction}

            def velocity_fn(self):
                return self.expert

        return Actor()

    def test_actor_loss_reaches_the_adapter(self):
        torch = require_torch()
        from sim_vla.training.actor_critic import ActorCriticConfig, actor_loss
        from sim_vla.training.imagination import flatten_start

        model, critic, batch = build(False)
        actor = self.make_actor(model)
        start = flatten_start(model.observe(batch)["post"], False)
        out = actor_loss(model, actor, critic, start,
                         ActorCriticConfig(horizon=2, flow_steps=3))
        out["loss"].backward()
        grads = [p.grad for p in actor.adapter.parameters() if p.grad is not None]
        self.assertTrue(grads, "the return's gradient never reached the actor")

    def test_world_model_and_critic_parameters_are_unchanged(self):
        torch = require_torch()
        from sim_vla.training.actor_critic import ActorCriticConfig, actor_loss
        from sim_vla.training.imagination import flatten_start

        model, critic, batch = build(False)
        actor = self.make_actor(model)
        before = [p.detach().clone() for p in model.parameters()]
        start = flatten_start(model.observe(batch)["post"], False)
        out = actor_loss(model, actor, critic, start,
                         ActorCriticConfig(horizon=2, flow_steps=3))
        out["loss"].backward()
        opt = torch.optim.SGD(
            [p for p in actor.parameters() if p.requires_grad], lr=0.1)
        opt.step()
        for old, new in zip(before, model.parameters()):
            self.assertTrue(torch.equal(old, new.detach()),
                            "the actor update changed the world model")

    def test_freeze_restores_requires_grad(self):
        torch = require_torch()
        from sim_vla.training.actor_critic import freeze_parameters

        model, critic, _ = build(False)
        with freeze_parameters(model, critic):
            self.assertFalse(any(p.requires_grad for p in model.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.parameters()))

    def test_critic_warmup_holds_the_actor_still(self):
        torch = require_torch()
        from sim_vla.training.actor_critic import (ActorCriticConfig,
                                                   ActorCriticTrainer)
        from sim_vla.training.imagination import flatten_start

        model, critic, batch = build(False)
        actor = self.make_actor(model)
        trainer = ActorCriticTrainer(
            model, actor, critic,
            ActorCriticConfig(horizon=2, flow_steps=2, critic_warmup=2))
        start = flatten_start(model.observe(batch)["post"], False)
        before = [p.detach().clone() for p in actor.adapter.parameters()]
        metrics = trainer.update(start)
        self.assertNotEqual(metrics["critic_loss"], metrics["critic_loss"] + 1)
        for old, new in zip(before, actor.adapter.parameters()):
            self.assertTrue(torch.equal(old, new.detach()),
                            "the actor moved during critic warm-up")


class TestPostWarmupUpdates(unittest.TestCase):
    """Several full updates with no warm-up left, which is where the
    in-place/version error appeared: the critic optimizer used to step before
    the actor's outstanding backward pass had run."""

    def test_repeated_updates_after_warmup(self):
        torch = require_torch()
        from sim_vla.training.actor_critic import (ActorCriticConfig,
                                                   ActorCriticTrainer)
        from sim_vla.training.imagination import flatten_start

        model, critic, batch = build(False)
        actor = TestActorUpdate().make_actor(model)
        trainer = ActorCriticTrainer(
            model, actor, critic,
            ActorCriticConfig(horizon=2, flow_steps=2, critic_warmup=0))
        # Detached: the same start seeds three updates, and a live graph would
        # be freed by the first backward.
        start = tuple(t.detach() for t in
                      flatten_start(model.observe(batch)["post"], False))
        before = [p.detach().clone() for p in model.parameters()]

        for step in range(3):
            metrics = trainer.update(start)
            self.assertFalse(np.isnan(metrics["actor_loss"]),
                             f"actor did not update at step {step}")
            self.assertTrue(np.isfinite(metrics["actor_grad_norm"]))
            # How many parameters received a gradient, not only its norm: a
            # single zero cannot distinguish an absent gradient from an
            # unmeasured one.
            self.assertEqual(metrics["actor_params_with_grad"],
                             metrics["actor_params_trainable"],
                             f"step {step}: only "
                             f"{metrics['actor_params_with_grad']} of "
                             f"{metrics['actor_params_trainable']} actor "
                             "parameters received a gradient")
            if metrics["actor_grad_norm"] == 0.0:
                # Every parameter has a .grad (asserted above) and their norm
                # is zero, while autograd.grad on the same objective is not.
                # Compute both on one objective to locate the discrepancy.
                from sim_vla.training.actor_critic import actor_loss
                from sim_vla.training.imagination import gradient_chain

                out = actor_loss(model, actor, critic, start, trainer.config)
                params = [p for p in actor.parameters() if p.requires_grad]
                direct = torch.autograd.grad(out["loss"], params,
                                             retain_graph=True,
                                             allow_unused=True)
                direct_norm = float(torch.sqrt(sum(
                    (g ** 2).sum() for g in direct if g is not None)))
                # Inspect the live graph before backward consumes it. This is
                # only a failure diagnostic, but it must not replace the real
                # zero-gradient assertion with "backward through graph a
                # second time".
                chain = gradient_chain(out["loss"], out, actor)
                for parameter in params:
                    parameter.grad = None
                out["loss"].backward()
                backward_norm = float(torch.sqrt(sum(
                    (p.grad ** 2).sum() for p in params
                    if p.grad is not None)))
                self.fail(
                    f"actor gradient was zero at step {step}. "
                    f"autograd.grad norm={direct_norm:.3e}, "
                    f".grad norm after backward={backward_norm:.3e}, "
                    f"unused={sum(1 for g in direct if g is None)}/"
                    f"{len(params)}; chain={chain}")
            self.assertTrue(np.isfinite(metrics["critic_loss"]))

        # The world model is frozen throughout the actor optimisation.
        for old, new in zip(before, model.parameters()):
            self.assertTrue(torch.equal(old, new.detach()),
                            "an actor/critic update changed the world model")

    def test_bootstrap_uses_the_slow_target(self):
        require_torch()
        import inspect

        from sim_vla.training import actor_critic

        source = inspect.getsource(actor_critic.actor_loss)
        code = chr(10).join(line.split("#", 1)[0]
                            for line in source.splitlines())
        self.assertIn("target_value(", code,
                      "the bootstrap must come from the slow target critic")


class TestImaginationStarts(unittest.TestCase):
    """Starts are re-encoded after the world-model step, not reused."""

    def test_starts_follow_the_current_parameters(self):
        torch = require_torch()
        from sim_vla.training.imagination import start_states

        model, _critic, batch = build(False)
        first = start_states(model, batch, limit=0)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.1)
        second = start_states(model, batch, limit=0)
        self.assertFalse(torch.allclose(first[0], second[0]),
                         "starts did not change after the model did")

    def test_starts_are_detached_and_limited(self):
        torch = require_torch()
        from sim_vla.training.imagination import start_states

        model, _critic, batch = build(False)
        full = start_states(model, batch, limit=0)
        self.assertFalse(full[0].requires_grad, "starts must be detached")
        limited = start_states(model, batch, limit=3)
        self.assertEqual(limited[0].shape[0], 3)

    def test_padding_and_burn_in_are_excluded(self):
        torch = require_torch()
        from sim_vla.training.imagination import start_states

        model, _critic, batch = build(False)
        batch["loss_mask"] = torch.zeros_like(batch["loss_mask"])
        batch["loss_mask"][:, 1] = True
        starts = start_states(model, batch, limit=0)
        self.assertEqual(starts[0].shape[0], batch["loss_mask"].shape[0])


if __name__ == "__main__":
    unittest.main()
