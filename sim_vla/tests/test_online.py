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
            ActorCriticConfig(horizon=2, flow_steps=2, critic_warmup=0,
                              imagination_microbatch=5))
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


class TestMicrobatchUpdates(unittest.TestCase):
    def test_graph_progress_bfloat16_keeps_actor_gradients(self):
        torch = require_torch()
        from sim_vla.training.actor_critic import ActorCriticConfig, ActorCriticTrainer
        from sim_vla.training.imagination import start_states
        from sim_vla.training.progress import build_progress

        torch.manual_seed(4)
        model, critic, batch = build(True)
        _, model_cfg = small_model_config(True)
        actor = TestActorUpdate().make_actor(model)
        head = build_progress(model_cfg, model.feature_dim,
                              graph_enabled=True, progress_enabled=True)
        trainer = ActorCriticTrainer(
            model, actor, critic,
            ActorCriticConfig(horizon=2, flow_steps=2, critic_warmup=0,
                              imagination_microbatch=2, precision="bfloat16"),
            progress_head=head)
        starts = start_states(model, batch, limit=5)
        head_before = [p.detach().clone() for p in head.parameters()]
        for _ in range(2):
            metrics = trainer.update(starts, progress_beta=0.05)
            self.assertGreater(metrics["actor_grad_norm"], 0.0)
            self.assertTrue(np.isfinite(metrics["actor_loss"]))
            self.assertTrue(np.isfinite(metrics["shaping_reward"]))
            self.assertEqual(metrics["progress_beta"], 0.05)
        for before, after in zip(head_before, head.parameters()):
            torch.testing.assert_close(before, after)
            self.assertIsNone(after.grad)

    def test_uneven_groups_match_full_batch_gradient_and_optimizer_step(self):
        torch = require_torch()
        import copy
        from unittest.mock import patch
        from sim_vla.training.actor_critic import ActorCriticConfig, ActorCriticTrainer

        class Critic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Linear(1, 1)
                self.target_updates = 0

            def loss(self, feat, returns):
                return (self.net(feat).squeeze(-1) - returns).square().mean()

            def update_target(self):
                self.target_updates += 1

        torch.manual_seed(3)
        actor = torch.nn.Linear(1, 1)
        critic = Critic()
        start = (torch.arange(1., 6.).reshape(5, 1),)
        shaping = torch.arange(5.).reshape(1, 5)
        trainers = []
        reports = []
        for size in (0, 2):
            trainer = ActorCriticTrainer(
                torch.nn.Linear(1, 1), copy.deepcopy(actor), copy.deepcopy(critic),
                ActorCriticConfig(critic_warmup=0, imagination_microbatch=size,
                                  grad_clip=1e6, progress_beta=0.2))
            pending = []
            seen = []

            def objective(wm, policy, value, seeds, config, *,
                          differentiable=True, progress_reward=None, **kwargs):
                # The next forward must not coexist with the previous graph.
                self.assertFalse(pending)
                feat = seeds[0]
                returns = feat.square().squeeze(-1).unsqueeze(0)
                loss = policy(feat).square().mean()
                if differentiable:
                    seen.append(feat.shape[0])
                    self.assertTrue(torch.is_grad_enabled())
                    loss = loss + (policy(feat).squeeze(-1)
                                   * progress_reward.squeeze(0)).mean()
                    pending.append(True)
                    loss.register_hook(lambda grad: (pending.clear(), grad)[1])
                else:
                    self.assertFalse(torch.is_grad_enabled())
                return {"loss": loss, "returns": returns,
                        "feat": torch.stack([feat, feat]), "reward": returns,
                        "shaping": progress_reward}

            with patch("sim_vla.training.actor_critic.actor_loss", objective):
                reports.append(trainer.update(start, progress_reward=shaping))
            self.assertFalse(pending)
            self.assertEqual(seen, [5] if size == 0 else [2, 2, 1])
            self.assertEqual(trainer.critic.target_updates, 1)
            self.assertEqual(trainer.step, 1)
            trainers.append(trainer)

        for key in ("actor_loss", "critic_loss", "actor_grad_norm", "shaping_reward"):
            self.assertAlmostEqual(reports[0][key], reports[1][key], places=4)
        for name in ("actor_opt", "critic_opt"):
            left = getattr(trainers[0], name).state_dict()["state"]
            right = getattr(trainers[1], name).state_dict()["state"]
            for key in left:
                # Adam's moments verify accumulated gradients, not only weights:
                # a first Adam step can hide a wrong gradient scale.
                for field in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(left[key][field], right[key][field])
                self.assertEqual(float(right[key]["step"]), 1.0)
        for name in ("actor", "critic"):
            for a, b in zip(getattr(trainers[0], name).parameters(),
                            getattr(trainers[1], name).parameters()):
                torch.testing.assert_close(a, b)


class TestUpdateRatio(unittest.TestCase):
    def test_fractional_updates_survive_collection_boundaries(self):
        require_torch()
        from sim_vla.training.online import OnlineConfig

        config = OnlineConfig(train_ratio=64, batch_size=16, sequence_length=64)
        updates = 0
        counts = []
        for env_steps in (150, 300, 450, 600):
            due = config.updates_due(env_steps, updates)
            counts.append(due)
            updates += due
        self.assertEqual(counts, [9, 9, 10, 9])
        self.assertEqual(updates, 37)
        self.assertEqual(config.updates_due(600, updates), 0)
        self.assertEqual(OnlineConfig(train_ratio=0).updates_due(600, 0), 8)

    def test_invalid_ratio_is_rejected(self):
        require_torch()
        from sim_vla.training.online import OnlineConfig

        for ratio in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                OnlineConfig(train_ratio=ratio).updates_due(10, 0)


class TestOnlineSettings(unittest.TestCase):
    def test_model_defaults_and_cli_overrides_reach_both_trainers(self):
        require_torch()
        from unittest.mock import patch
        from sim_vla.config import load_config
        from sim_vla.models.model_config import load_model_config
        from sim_vla.training import pipeline
        from sim_vla.training.wandb_logger import RunLogger

        cfg = load_config("peginsertion", "dreamer")
        model = load_model_config(cfg)
        online, ac = pipeline.online_configs(
            cfg, model, total_steps=500_000, flow_steps=7)
        self.assertEqual(online.batch_size, 16)
        self.assertEqual(online.train_ratio, 64)
        self.assertEqual(online.imagination_batch, 256)
        self.assertEqual(ac.imagination_microbatch, 16)
        self.assertEqual(ac.horizon, model.imag_horizon)
        self.assertEqual(ac.discount, 1 - 1 / model.horizon)
        self.assertEqual(ac.lam, model.lamb)
        self.assertEqual(ac.flow_steps, 7)
        self.assertEqual(online.precision, "bfloat16")
        self.assertEqual(ac.precision, online.precision)

        # Exercise main's actual CLI-to-YAML-to-trainer configuration path,
        # without loading a pretrained checkpoint or a simulator.
        with patch.object(pipeline, "run", return_value={}) as run, \
                patch.object(pipeline, "start_run", return_value=RunLogger()):
            pipeline.main([
                "--task", "peginsertion", "--experiment", "graph_progress",
                "--device", "cpu", "--batch-size", "16", "--train-ratio", "32",
                "--online-precision", "float32", "--imagination-batch", "128",
                "--imagination-microbatch", "7", "--imag-horizon", "9"])
        cfg = run.call_args.args[0]
        online, ac = pipeline.online_configs(
            cfg, model, total_steps=12, flow_steps=6)
        self.assertEqual(online.train_ratio, 32)
        self.assertEqual(online.batch_size, 16)
        self.assertEqual(online.imagination_batch, 128)
        self.assertEqual(ac.imagination_microbatch, 7)
        self.assertEqual(ac.horizon, 9)
        self.assertEqual(ac.precision, "float32")
        self.assertEqual(ac.flow_steps, 6)

    def test_invalid_online_settings_fail_before_loading_models(self):
        require_torch()
        from sim_vla.config import load_config

        for settings in ({"train_ratio": -1}, {"imagination_microbatch": -1},
                         {"imagination_batch": -1}, {"imag_horizon": 0},
                         {"precision": "float16"}):
            with self.subTest(settings=settings), self.assertRaises(SystemExit):
                load_config("peginsertion", "dreamer", {"online": settings})


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
