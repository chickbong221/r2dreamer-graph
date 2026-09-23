"""Stage 8: the executed-chunk update -- targets, gradients, frozen parameters.

One rollout per microbatch of starts feeds both losses: the actor maximises
the chunk's bootstrapped return, the critic regresses the start state onto the
same return, detached. What these check is the wiring that is invisible in a
loss curve -- which parameters move, which stay, what the critic is trained
at, and that grouping the starts changes nothing but memory.
"""

from __future__ import annotations

import unittest

import numpy as np

from .common import (DummyExpert, fake_batch, obs_shapes, require_torch,
                     small_model_config)


def build(graph_enabled):
    require_torch()
    from sim_vla.models.critics import ValueCritic
    from sim_vla.models.world_model import build_world_model

    _cfg, model_cfg = small_model_config(graph_enabled)
    batch = fake_batch(graph_enabled=graph_enabled)
    model = build_world_model(model_cfg, obs_shapes(batch), 8,
                              graph_enabled=graph_enabled)
    critic = ValueCritic(model_cfg, model.feature_dim)
    return model, critic, batch


def make_actor(model, chunk=8):
    torch = require_torch()
    from sim_vla.models.latent_adapter import LatentAdapter

    class Actor(torch.nn.Module):
        chunk_size, action_dim = chunk, 8

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


def ac_config(**overrides):
    from sim_vla.training.actor_critic import ActorCriticConfig

    settings = dict(execute=2, flow_steps=2, critic_warmup=0,
                    imagination_microbatch=0)
    settings.update(overrides)
    return ActorCriticConfig(**settings)


def give_the_heads_an_opinion(model, critic):
    """Move the reward and value heads off their flat initialization.

    Both are ``symexp_twohot`` heads whose output layer starts at zero, so
    every state gets the same prediction and the gradient with respect to the
    feature is exactly zero -- which is why ``critic_warmup`` exists, and why
    a gradient test taken on a fresh model would pass or fail for the wrong
    reason.
    """
    torch = require_torch()
    torch.manual_seed(3)
    for parameter in critic.net.parameters():
        parameter.data.add_(torch.randn_like(parameter))
    for parameter in model.reward_head.parameters():
        parameter.data.add_(torch.randn_like(parameter))
    return model, critic


def trainer_for(graph_enabled=False, opinionated=True, **overrides):
    from sim_vla.training.actor_critic import ActorCriticTrainer
    from sim_vla.training.imagination import start_states

    model, critic, batch = build(graph_enabled)
    if opinionated:
        give_the_heads_an_opinion(model, critic)
    actor = make_actor(model)
    trainer = ActorCriticTrainer(model, actor, critic, ac_config(**overrides))
    return trainer, start_states(model, batch), model, actor, critic


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


class TestActorObjective(unittest.TestCase):
    def objective(self, **overrides):
        from sim_vla.training.actor_critic import executed_chunk_objective
        from sim_vla.training.imagination import start_states

        model, critic, batch = build(False)
        give_the_heads_an_opinion(model, critic)
        actor = make_actor(model)
        start = start_states(model, batch)
        out = executed_chunk_objective(model, actor, critic, start,
                                       ac_config(**overrides))
        return out, model, actor, critic

    def test_the_return_reaches_the_adapter(self):
        require_torch()
        out, _model, actor, _critic = self.objective()
        out["loss"].backward()
        grads = [p.grad for p in actor.adapter.parameters()
                 if p.grad is not None and float(p.grad.abs().sum()) > 0]
        self.assertTrue(grads, "the return's gradient never reached the actor")

    def test_the_world_model_and_critic_keep_their_parameters(self):
        torch = require_torch()
        out, model, actor, critic = self.objective()
        before = [p.detach().clone() for p in model.parameters()]
        out["loss"].backward()
        for name, parameter in model.named_parameters():
            self.assertIsNone(parameter.grad,
                              f"world model parameter {name} accumulated a "
                              "gradient during the actor update")
        for name, parameter in critic.named_parameters():
            self.assertIsNone(parameter.grad,
                              f"critic parameter {name} accumulated a "
                              "gradient during the actor update")
        torch.optim.SGD([p for p in actor.parameters() if p.requires_grad],
                        lr=0.1).step()
        for old, new in zip(before, model.parameters()):
            self.assertTrue(torch.equal(old, new.detach()),
                            "the actor update changed the world model")

    def test_the_progress_head_is_read_but_not_trained(self):
        torch = require_torch()
        from sim_vla.training.actor_critic import executed_chunk_objective
        from sim_vla.training.imagination import start_states
        from sim_vla.training.progress import build_progress

        model, critic, batch = build(True)
        give_the_heads_an_opinion(model, critic)
        _cfg, model_cfg = small_model_config(True)
        actor = make_actor(model)
        head = build_progress(model_cfg, model.feature_dim,
                              graph_enabled=True, progress_enabled=True)
        out = executed_chunk_objective(
            model, actor, critic, start_states(model, batch),
            ac_config(progress_beta=0.25), progress_head=head)
        self.assertIsNotNone(out["shaping"])
        # The shaped stream is what the return was built from, and it differs
        # from the environment stream it is reported beside.
        self.assertFalse(torch.allclose(out["shaped_reward"], out["reward"]))
        out["loss"].backward()
        for name, parameter in head.named_parameters():
            self.assertIsNone(parameter.grad,
                              f"the actor update trained progress head {name}")

    def test_warmup_builds_no_graph_at_all(self):
        """Including the conditioning: the adapter must not be run with grad
        during the warm-up, or its activations are kept for nothing."""
        require_torch()
        from sim_vla.training.actor_critic import executed_chunk_objective
        from sim_vla.training.imagination import start_states

        model, critic, batch = build(False)
        give_the_heads_an_opinion(model, critic)
        actor = make_actor(model)
        out = executed_chunk_objective(model, actor, critic,
                                       start_states(model, batch), ac_config(),
                                       differentiable=False)
        for key in ("loss", "returns", "feat", "action"):
            self.assertIsNone(out[key].grad_fn,
                              f"{key} kept a graph while the actor was warming "
                              "up")
        self.assertFalse(out["loss"].requires_grad)

    def test_freeze_restores_requires_grad(self):
        require_torch()
        from sim_vla.training.actor_critic import freeze_parameters

        model, critic, _ = build(False)
        with freeze_parameters(model, critic):
            self.assertFalse(any(p.requires_grad for p in model.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.parameters()))


class TestCriticTarget(unittest.TestCase):
    """What the critic is trained at, and on what."""

    def test_it_regresses_the_start_state_on_a_detached_return(self):
        torch = require_torch()
        from sim_vla.training.actor_critic import ActorCriticTrainer
        from sim_vla.training.imagination import start_states

        model, critic, batch = build(False)
        give_the_heads_an_opinion(model, critic)
        actor = make_actor(model)
        trainer = ActorCriticTrainer(model, actor, critic, ac_config())
        start = start_states(model, batch)
        seen = {}
        original = critic.loss

        def record(feat, returns, mask=None):
            seen["feat"] = feat.detach().clone()
            seen["returns_requires_grad"] = bool(returns.requires_grad)
            seen["feat_requires_grad"] = bool(feat.requires_grad)
            return original(feat, returns, mask)

        critic.loss = record
        trainer.update(start)
        critic.loss = original

        self.assertFalse(seen["returns_requires_grad"],
                         "the critic was trained on a live return")
        self.assertFalse(seen["feat_requires_grad"],
                         "the critic's input carried the actor's graph")
        # The start states themselves, not a later state of the chunk.
        from sim_vla.training.imagination import imagine_chunk

        with torch.no_grad():
            expected = imagine_chunk(model, actor, start, trainer.config.execute,
                                     flow_steps=trainer.config.flow_steps,
                                     differentiable=False)["feat"][0]
        self.assertEqual(tuple(seen["feat"].shape), tuple(expected.shape))
        torch.testing.assert_close(seen["feat"], expected)


class TestUpdateMechanics(unittest.TestCase):
    def test_critic_warmup_holds_the_actor_still(self):
        torch = require_torch()
        trainer, start, _model, actor, _critic = trainer_for(critic_warmup=2)
        before = [p.detach().clone() for p in actor.adapter.parameters()]
        metrics = trainer.update(start)
        self.assertTrue(np.isnan(metrics["actor_loss"]))
        self.assertTrue(np.isfinite(metrics["critic_loss"]))
        self.assertEqual(trainer.actor_steps, 0)
        for old, new in zip(before, actor.adapter.parameters()):
            self.assertTrue(torch.equal(old, new.detach()),
                            "the actor moved during critic warm-up")

    def test_repeated_updates_after_warmup(self):
        torch = require_torch()
        trainer, start, model, _actor, _critic = trainer_for(
            imagination_microbatch=5)
        before = [p.detach().clone() for p in model.parameters()]
        for step in range(3):
            metrics = trainer.update(start)
            self.assertFalse(np.isnan(metrics["actor_loss"]),
                             f"actor did not update at step {step}")
            self.assertGreater(metrics["actor_grad_norm"], 0.0,
                               f"no actor gradient at step {step}")
            self.assertEqual(metrics["actor_params_with_grad"],
                             metrics["actor_params_trainable"],
                             f"step {step}: only "
                             f"{metrics['actor_params_with_grad']} of "
                             f"{metrics['actor_params_trainable']} actor "
                             "parameters received a gradient")
            self.assertTrue(np.isfinite(metrics["critic_loss"]))
        self.assertEqual(trainer.actor_steps, 3)
        # The world model is frozen throughout the actor optimisation.
        for old, new in zip(before, model.parameters()):
            self.assertTrue(torch.equal(old, new.detach()),
                            "an actor/critic update changed the world model")

    def test_the_bootstrap_comes_from_the_slow_target(self):
        require_torch()
        import inspect

        from sim_vla.training import actor_critic

        source = inspect.getsource(actor_critic.executed_chunk_objective)
        code = chr(10).join(line.split("#", 1)[0]
                            for line in source.splitlines())
        self.assertIn("target_value(", code,
                      "the bootstrap must come from the slow target critic")

    def test_execute_beyond_the_chunk_is_refused(self):
        require_torch()
        from sim_vla.training.actor_critic import ActorCriticTrainer

        model, critic, _batch = build(False)
        actor = make_actor(model, chunk=4)
        with self.assertRaises(ValueError):
            ActorCriticTrainer(model, actor, critic, ac_config(execute=5))


class TestMicrobatchUpdates(unittest.TestCase):
    def test_graph_progress_bfloat16_keeps_actor_gradients(self):
        torch = require_torch()
        from sim_vla.training.actor_critic import ActorCriticTrainer
        from sim_vla.training.imagination import start_states
        from sim_vla.training.progress import build_progress

        torch.manual_seed(4)
        model, critic, batch = build(True)
        give_the_heads_an_opinion(model, critic)
        _, model_cfg = small_model_config(True)
        actor = make_actor(model)
        head = build_progress(model_cfg, model.feature_dim,
                              graph_enabled=True, progress_enabled=True)
        trainer = ActorCriticTrainer(
            model, actor, critic,
            ac_config(imagination_microbatch=2, precision="bfloat16"),
            progress_head=head)
        starts = start_states(model, batch)
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

        from sim_vla.training.actor_critic import ActorCriticTrainer

        class Critic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Linear(1, 1)
                self.target_updates = 0

            def loss(self, feat, returns, mask=None):
                return (self.net(feat).squeeze(-1) - returns).square().mean()

            def update_target(self):
                self.target_updates += 1

        torch.manual_seed(3)
        actor = torch.nn.Linear(1, 1)
        critic = Critic()
        start = (torch.arange(1., 6.).reshape(5, 1),)
        trainers = []
        reports = []
        for size in (0, 2):
            trainer = ActorCriticTrainer(
                torch.nn.Linear(1, 1), copy.deepcopy(actor),
                copy.deepcopy(critic),
                ac_config(imagination_microbatch=size, grad_clip=1e6))
            pending = []
            seen = []

            def objective(wm, policy, value, seeds, config, *,
                          differentiable=True, **kwargs):
                # The next rollout must not coexist with the previous graph.
                self.assertFalse(pending)
                self.assertTrue(differentiable)
                feat = seeds[0]
                seen.append(feat.shape[0])
                returns = policy(feat).squeeze(-1) * feat.squeeze(-1)
                loss = -returns.mean()
                pending.append(True)
                loss.register_hook(lambda grad: (pending.clear(), grad)[1])
                one = torch.ones_like(returns)
                return {"loss": loss, "returns": returns,
                        "feat": torch.stack([feat, feat]),
                        "reward": one, "cont": one, "bootstrap": one,
                        "shaping": None}

            with patch("sim_vla.training.actor_critic."
                       "executed_chunk_objective", objective):
                reports.append(trainer.update(start))
            self.assertFalse(pending)
            self.assertEqual(seen, [5] if size == 0 else [2, 2, 1])
            self.assertEqual(trainer.critic.target_updates, 1)
            self.assertEqual(trainer.step, 1)
            trainers.append(trainer)

        for key in ("actor_loss", "critic_loss", "actor_grad_norm", "return"):
            self.assertAlmostEqual(reports[0][key], reports[1][key], places=4)
        for name in ("actor_opt", "critic_opt"):
            left = getattr(trainers[0], name).state_dict()["state"]
            right = getattr(trainers[1], name).state_dict()["state"]
            for key in left:
                # Adam's moments verify accumulated gradients, not only
                # weights: a first Adam step can hide a wrong gradient scale.
                for field in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(left[key][field],
                                               right[key][field])
                self.assertEqual(float(right[key]["step"]), 1.0)
        for name in ("actor", "critic"):
            for a, b in zip(getattr(trainers[0], name).parameters(),
                            getattr(trainers[1], name).parameters()):
                torch.testing.assert_close(a, b)


class TestReturnNormalization(unittest.TestCase):
    """The RL term is divided by the running return spread; nothing else is."""

    def run_update(self, *, return_norm, spread=None, microbatch=0,
                   warmup=0):
        torch = require_torch()
        from unittest.mock import patch

        from sim_vla.training.actor_critic import ActorCriticTrainer

        class Critic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Linear(1, 1)

            def loss(self, feat, returns, mask=None):
                return (self.net(feat).squeeze(-1) - returns).square().mean()

            def update_target(self):
                pass

        def objective(wm, policy, value, seeds, config, *,
                      differentiable=True, **kwargs):
            feat = seeds[0]
            returns = policy(feat).squeeze(-1) * feat.squeeze(-1)
            one = torch.ones_like(returns)
            return {"loss": -returns.mean(), "returns": returns,
                    "feat": torch.stack([feat, feat]), "reward": one,
                    "cont": one, "bootstrap": one, "shaping": None}

        torch.manual_seed(3)
        trainer = ActorCriticTrainer(
            torch.nn.Linear(1, 1), torch.nn.Linear(1, 1), Critic(),
            ac_config(imagination_microbatch=microbatch, critic_warmup=warmup,
                      grad_clip=1e6, return_norm=return_norm))
        if spread is not None:
            trainer.return_ema.ema_vals.copy_(torch.tensor([0.0, spread]))
        start = (torch.arange(1., 6.).reshape(5, 1),)
        with patch("sim_vla.training.actor_critic.executed_chunk_objective",
                   objective):
            return trainer, trainer.update(start)

    def test_the_actor_gradient_is_divided_by_the_spread_and_the_critic_is_not(self):
        require_torch()
        _plain, plain = self.run_update(return_norm=False, spread=10.0)
        _normed, normed = self.run_update(return_norm=True, spread=10.0)
        self.assertEqual(normed["return_scale"], 10.0)
        self.assertAlmostEqual(normed["actor_grad_norm"],
                               plain["actor_grad_norm"] / 10.0, places=5)
        self.assertAlmostEqual(normed["critic_loss"], plain["critic_loss"],
                               places=6)
        self.assertAlmostEqual(normed["actor_loss"], plain["actor_loss"],
                               places=6)

    def test_a_small_spread_is_floored_at_one(self):
        require_torch()
        _plain, plain = self.run_update(return_norm=False, spread=0.2)
        _normed, normed = self.run_update(return_norm=True, spread=0.2)
        self.assertEqual(normed["return_scale"], 1.0)
        self.assertAlmostEqual(normed["actor_grad_norm"],
                               plain["actor_grad_norm"], places=6)

    def test_grouping_still_changes_nothing(self):
        require_torch()
        _whole, whole = self.run_update(return_norm=True, spread=10.0)
        _groups, groups = self.run_update(return_norm=True, spread=10.0,
                                          microbatch=2)
        self.assertAlmostEqual(whole["actor_grad_norm"],
                               groups["actor_grad_norm"], places=5)

    def test_the_scale_is_tracked_through_the_critic_warmup(self):
        torch = require_torch()
        trainer, metrics = self.run_update(return_norm=True, warmup=5)
        self.assertTrue(np.isnan(metrics["actor_loss"]))
        self.assertNotEqual(float(trainer.return_ema.ema_vals[1]), 0.0)
        self.assertEqual(metrics["return_p95"],
                         float(trainer.return_ema.ema_vals[1]))


class TestImaginationStarts(unittest.TestCase):
    """Every eligible replay position, re-encoded after the world-model step."""

    def test_starts_follow_the_current_parameters(self):
        torch = require_torch()
        from sim_vla.training.imagination import start_states

        model, _critic, batch = build(False)
        first = start_states(model, batch)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.1)
        second = start_states(model, batch)
        self.assertFalse(torch.allclose(first[0], second[0]),
                         "starts did not change after the model did")

    def test_every_eligible_position_is_a_start(self):
        require_torch()
        from sim_vla.training.imagination import start_states

        model, _critic, batch = build(False)
        starts = start_states(model, batch)
        self.assertFalse(starts[0].requires_grad, "starts must be detached")
        self.assertEqual(int(starts[0].shape[0]),
                         int(batch["loss_mask"].sum()),
                         "the starts were capped or padded rows slipped in")

    def test_padding_and_burn_in_are_excluded(self):
        torch = require_torch()
        from sim_vla.training.imagination import start_states

        model, _critic, batch = build(False)
        batch["loss_mask"] = torch.zeros_like(batch["loss_mask"])
        batch["loss_mask"][:, 1] = True
        starts = start_states(model, batch)
        self.assertEqual(starts[0].shape[0], batch["loss_mask"].shape[0])

    def test_terminal_states_are_never_starts(self):
        """Nothing continues from a terminal state, so no chunk starts there."""
        torch = require_torch()
        from sim_vla.training.imagination import start_states

        model, _critic, batch = build(False)
        eligible = int(batch["loss_mask"].sum())
        batch["is_terminal"] = torch.zeros_like(batch["loss_mask"])
        batch["is_terminal"][0, 2] = True
        starts = start_states(model, batch)
        self.assertEqual(int(starts[0].shape[0]), eligible - 1)

    def test_a_supplied_posterior_is_reused_rather_than_re_encoded(self):
        require_torch()
        from sim_vla.training.imagination import start_states

        model, _critic, batch = build(False)
        post = model.observe(batch)["post"]
        calls = []
        original = model.observe
        model.observe = lambda *a, **k: (calls.append(1), original(*a, **k))[1]
        start_states(model, batch, post=post)
        model.observe = original
        self.assertEqual(calls, [], "the batch was encoded a second time")


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
        self.assertEqual(ac.imagination_microbatch, 16)
        # One setting decides the executed chunk, and the discount still comes
        # from the root model's horizon.
        self.assertEqual(ac.execute, int(cfg["actor"]["execute"]))
        self.assertEqual(ac.execute, 5)
        self.assertEqual(ac.discount, 1 - 1 / model.horizon)
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
                "--online-precision", "float32",
                "--imagination-microbatch", "7", "--critic-warmup", "3",
                "--num-envs", "128", "--reconfiguration-freq", "1",
                "--online-world-lr", "6e-5", "--critic-lr", "6e-5",
                "--progress-lr", "6e-5"])
        cfg = run.call_args.args[0]
        self.assertEqual(cfg["online"]["num_envs"], 128)
        self.assertEqual(cfg["online"]["reconfiguration_freq"], 1)
        defaults = load_config("peginsertion", "dreamer")["online"]
        self.assertEqual(defaults["num_envs"], 16)
        self.assertIsNone(defaults["reconfiguration_freq"])
        online, ac = pipeline.online_configs(
            cfg, model, total_steps=12, flow_steps=6)
        self.assertEqual(online.train_ratio, 32)
        self.assertEqual(ac.imagination_microbatch, 7)
        self.assertEqual(ac.critic_warmup, 3)
        self.assertEqual(ac.precision, "float32")
        self.assertEqual(ac.flow_steps, 6)
        # Stage 2's world-model rate, separate from Stage 1A's --world-lr.
        self.assertEqual(online.world_lr, 6e-5)
        self.assertEqual(ac.critic_lr, 6e-5)
        self.assertEqual(online.progress_lr, 6e-5)
        self.assertIsNone(cfg["pretrain"].get("world_lr"))
        default, default_ac = pipeline.online_configs(
            load_config("peginsertion", "dreamer"), model, total_steps=12,
            flow_steps=6)
        self.assertEqual(default.world_lr, 1e-4)
        self.assertEqual(default_ac.critic_lr, 3e-4)
        self.assertEqual(default.progress_lr, 3e-4)

    def test_stage_one_learning_rates_reach_both_stages(self):
        require_torch()
        from unittest.mock import patch

        from sim_vla.config import load_config
        from sim_vla.models.model_config import load_model_config
        from sim_vla.training import pipeline
        from sim_vla.training.pretrain_world_model import world_lr
        from sim_vla.training.train_imitation import imitation_lr
        from sim_vla.training.wandb_logger import RunLogger

        cfg = load_config("peginsertion", "graph_progress")
        model = load_model_config(cfg)
        self.assertEqual(world_lr(cfg, model), float(model.lr))
        self.assertEqual(float(model.lr), 4e-5)
        self.assertEqual(imitation_lr(cfg), 1e-4)

        with patch.object(pipeline, "run", return_value={}) as run, \
                patch.object(pipeline, "start_run", return_value=RunLogger()):
            pipeline.main([
                "--task", "peginsertion", "--experiment", "graph_progress",
                "--device", "cpu", "--world-lr", "8e-5",
                "--imitation-lr", "1.5e-4", "--world-warmup-steps", "1000",
                "--world-final-lr", "1e-5", "--imitation-warmup-steps", "500",
                "--imitation-final-lr", "2.5e-6"])
        cfg = run.call_args.args[0]
        self.assertEqual(world_lr(cfg, model), 8e-5)
        self.assertEqual(imitation_lr(cfg), 1.5e-4)
        from sim_vla.training.pretrain_world_model import world_schedule
        from sim_vla.training.train_imitation import imitation_decay

        schedule = world_schedule(cfg, model, 30_000)
        self.assertEqual((schedule.peak, schedule.warmup, schedule.final,
                          schedule.total), (8e-5, 1000, 1e-5, 30_000))
        self.assertEqual(imitation_decay(cfg),
                         {"warmup_steps": 500, "final_lr": 2.5e-6})
        # Unset, both stages keep the constant rate they always had.
        default = load_config("peginsertion", "dreamer")
        self.assertIsNone(world_schedule(default, model, 10).final)
        self.assertEqual(world_schedule(default, model, 10).warmup, 0)
        self.assertEqual(imitation_decay(default),
                         {"warmup_steps": 0, "final_lr": None})

        for settings in ({"world_lr": 0}, {"imitation_lr": -1e-4},
                         {"world_final_lr": 0}, {"imitation_warmup_steps": -1},
                         {"imitation_lr": 1e-4, "imitation_final_lr": 2e-4},
                         {"world_lr": 1e-4, "world_final_lr": 1e-3}):
            with self.subTest(settings=settings), \
                    self.assertRaises(SystemExit):
                load_config("peginsertion", "dreamer", {"pretrain": settings})

    def test_invalid_online_settings_fail_before_loading_models(self):
        require_torch()
        from sim_vla.config import load_config

        for settings in ({"train_ratio": -1}, {"imagination_microbatch": -1},
                         {"precision": "float16"}, {"num_envs": 0},
                         {"reconfiguration_freq": -1}, {"critic_warmup": -1},
                         {"actor_lr": 0}, {"world_lr": 0},
                         {"world_lr": float("nan")}, {"critic_lr": 0},
                         {"progress_lr": -1e-4}, {"demo_anchor": -0.5},
                         {"anchor_rows": 0}, {"anchor_windows": 0},
                         {"progress_warmup_start": 30000},
                         {"progress_warmup_start": 100,
                          "progress_warmup_end": 100},
                         {"num_envs": 4, "train_ratio": 0}):
            with self.subTest(settings=settings), self.assertRaises(SystemExit):
                load_config("peginsertion", "dreamer", {"online": settings})

    def test_an_execute_longer_than_the_chunk_is_refused(self):
        require_torch()
        from sim_vla.config import load_config

        with self.assertRaises(SystemExit):
            load_config("peginsertion", "dreamer",
                        {"actor": {"execute": 0}})
        with self.assertRaises(SystemExit):
            load_config("peginsertion", "dreamer",
                        {"actor": {"chunk_size": 4, "execute": 5}})

    def test_an_online_chunk_shorter_than_execute_or_longer_than_imitated_is_refused(self):
        require_torch()
        from sim_vla.config import load_config

        with self.assertRaises(SystemExit):
            load_config("peginsertion", "dreamer",
                        {"online": {"chunk_size": 4}, "actor": {"execute": 5}})
        with self.assertRaises(SystemExit):
            load_config("peginsertion", "dreamer",
                        {"online": {"chunk_size": 20},
                         "actor": {"chunk_size": 10, "execute": 5}})
        # null is "keep what Stage 1B imitated at".
        load_config("peginsertion", "dreamer", {"online": {"chunk_size": None}})


class TestAnchorAndShapingSettings(unittest.TestCase):
    def test_flags_reach_the_trainer_config_and_the_shaping_schedule(self):
        require_torch()
        from unittest.mock import patch

        from sim_vla.config import load_config
        from sim_vla.models.model_config import load_model_config
        from sim_vla.training import pipeline
        from sim_vla.training.wandb_logger import RunLogger

        with patch.object(pipeline, "run", return_value={}) as run, \
                patch.object(pipeline, "start_run", return_value=RunLogger()):
            pipeline.main([
                "--task", "placesphere", "--experiment", "graph_progress",
                "--device", "cpu", "--demo-anchor", "0.5",
                "--anchor-rows", "32", "--progress-warmup-start", "30000",
                "--progress-warmup-end", "100000", "--return-norm"])
        cfg = run.call_args.args[0]
        model = load_model_config(cfg)
        _online, ac = pipeline.online_configs(cfg, model, total_steps=500_000,
                                              flow_steps=5)
        self.assertTrue(ac.return_norm)
        self.assertEqual(ac.demo_anchor, 0.5)
        self.assertEqual(ac.anchor_rows, 32)
        self.assertEqual(pipeline.shaping_warmup(cfg, 500_000),
                         (30_000, 100_000))

        default = load_config("placesphere", "graph_progress")
        _online, ac = pipeline.online_configs(default, model,
                                              total_steps=500_000,
                                              flow_steps=5)
        self.assertEqual(ac.demo_anchor, 0.0)
        self.assertFalse(ac.return_norm)
        self.assertEqual(pipeline.shaping_warmup(default, 500_000),
                         (100_000, 300_000))


class TestOnlineChunk(unittest.TestCase):
    """Stage 2 generates online.chunk_size actions; Stage 1B keeps its own."""

    class Actor:
        def __init__(self, chunk_size):
            self.chunk_size = chunk_size
            self.calls = []

        def shrink_chunk(self, chunk):
            self.calls.append(chunk)
            self.chunk_size = chunk
            return 1e-7

    def test_the_default_config_runs_stage_two_at_five_actions_and_five_steps(self):
        require_torch()
        from sim_vla.config import load_config

        cfg = load_config("peginsertion", "dreamer")
        self.assertEqual(cfg["online"]["chunk_size"], 5)
        self.assertEqual(cfg["actor"]["flow_steps"], 5)
        self.assertGreaterEqual(cfg["online"]["chunk_size"],
                                cfg["actor"]["execute"])

    def test_a_set_chunk_shrinks_the_actor_and_is_reported(self):
        require_torch()
        from sim_vla.training.pipeline import shrink_online_chunk

        actor = self.Actor(50)
        out = shrink_online_chunk({"online": {"chunk_size": 5}}, actor)
        self.assertEqual(actor.calls, [5])
        self.assertEqual(out, {"imitation_chunk_size": 50, "chunk_size": 5,
                               "shrink_difference": 1e-7})

    def test_null_leaves_the_actor_alone(self):
        require_torch()
        from sim_vla.training.pipeline import shrink_online_chunk

        for cfg in ({"online": {"chunk_size": None}}, {"online": {}}, {}):
            with self.subTest(cfg=cfg):
                actor = self.Actor(50)
                out = shrink_online_chunk(cfg, actor)
                self.assertEqual(actor.calls, [])
                self.assertEqual(out["chunk_size"], 50)
                self.assertEqual(out["imitation_chunk_size"], 50)


class TestRemovedSettingsCannotRunSilently(unittest.TestCase):
    """A stale config or launch script must fail, not run a different
    experiment under the old names."""

    def test_removed_online_settings_are_refused_by_name(self):
        require_torch()
        from sim_vla.config import REMOVED_ONLINE, load_config

        for key in REMOVED_ONLINE:
            with self.subTest(setting=key), self.assertRaises(SystemExit) as caught:
                load_config("peginsertion", "dreamer",
                            {"online": {key: 1}})
            self.assertIn(key, str(caught.exception))

    def test_removed_flags_say_what_replaced_them(self):
        require_torch()
        from sim_vla.training.pipeline import REMOVED_FLAGS, parse_args

        for flag in REMOVED_FLAGS:
            with self.subTest(flag=flag):
                for argv in ([flag, "1"], [f"{flag}=1"]):
                    with self.assertRaises(SystemExit) as caught:
                        parse_args(["--task", "peginsertion"] + argv)
                    message = str(caught.exception)
                    self.assertIn(flag, message)
                    self.assertIn("removed", message)

    def test_the_surviving_flags_still_parse(self):
        require_torch()
        from sim_vla.training.pipeline import parse_args

        args = parse_args(["--task", "peginsertion", "--imagination-microbatch",
                           "8", "--critic-warmup", "5", "--actor-lr", "1e-5",
                           "--profile-online", "--num-envs", "4",
                           "--train-ratio", "64"])
        self.assertEqual(args.imagination_microbatch, 8)
        self.assertEqual(args.critic_warmup, 5)
        self.assertTrue(args.profile_online)


class TestSeeding(unittest.TestCase):
    """One seed, applied before anything is constructed, on both paths."""

    def test_seed_flag_reaches_the_resolved_config(self):
        require_torch()
        from sim_vla.training.pipeline import parse_args

        args = parse_args(["--task", "peginsertion", "--seed", "11"])
        self.assertEqual(args.seed, 11)
        self.assertIsNone(parse_args(["--task", "peginsertion"]).seed)

    def test_seed_override_lands_in_data_seed(self):
        from sim_vla.config import load_config, validate

        cfg = load_config("peginsertion", "dreamer",
                          overrides={"data": {"seed": 11}})
        validate(cfg)
        self.assertEqual(int(cfg["data"]["seed"]), 11)

    def test_seed_everything_fixes_module_initialisation(self):
        """The property the resume path needs: construction is seeded too."""
        torch = require_torch()
        from sim_vla.training.pretrain_world_model import seed_everything

        def adapter_weights(seed):
            from sim_vla.models.latent_adapter import LatentAdapter

            seed_everything(seed)
            return [p.detach().clone()
                    for p in LatentAdapter(16, 16, hidden=32).parameters()]

        first, again = adapter_weights(0), adapter_weights(0)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, again)),
                        "seed 0 did not reproduce module initialisation")
        other = adapter_weights(7)
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(first, other)),
                         "two different seeds initialised identically")

    def test_seed_everything_covers_python_numpy_and_torch(self):
        torch = require_torch()
        import random

        from sim_vla.training.pretrain_world_model import seed_everything

        def draws(seed):
            seed_everything(seed)
            return (random.random(), float(np.random.rand()),
                    float(torch.rand(1)))

        self.assertEqual(draws(0), draws(0),
                         "seed 0 was not applied to all three generators")
        self.assertNotEqual(draws(0), draws(3))


class TestProfiling(unittest.TestCase):
    def test_profiling_off_adds_no_metrics(self):
        require_torch()
        trainer, start, _m, _a, _c = trainer_for(profile=False)
        metrics = trainer.update(start)
        self.assertFalse([k for k in metrics if k.startswith("profile_")])

    def test_profiling_on_reports_every_phase(self):
        require_torch()
        trainer, start, _m, _a, _c = trainer_for(profile=True)
        metrics = trainer.update(start)
        for phase in ("imagine", "actor_backward", "critic_backward", "step"):
            self.assertIn(f"profile_{phase}_s", metrics, phase)
            self.assertGreaterEqual(metrics[f"profile_{phase}_s"], 0.0)

    def test_the_actor_phase_is_absent_during_warmup(self):
        require_torch()
        trainer, start, _m, _a, _c = trainer_for(profile=True, critic_warmup=5)
        metrics = trainer.update(start)
        self.assertIn("profile_imagine_s", metrics)
        self.assertNotIn("profile_actor_backward_s", metrics)

    def test_table_renders_without_cuda(self):
        require_torch()
        from sim_vla.training.profiling import Phases

        phases = Phases(device="cpu", enabled=True)
        with phases("imagine"):
            pass
        text = phases.table("smoke")
        self.assertIn("imagine", text)
        self.assertIn("CUDA not in use", text)


if __name__ == "__main__":
    unittest.main()
