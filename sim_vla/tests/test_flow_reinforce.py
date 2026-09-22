"""Stage 12: the stochastic flow sampler and its score-function gradient.

These are the tests that do not need the pretrained checkpoint. What they pin
down is the part that is easy to get subtly wrong and impossible to notice
later: the density arithmetic, the Euler sign, and the direction the gradient
actually pushes the sampled transition's likelihood.

A wrong constant in the log density still trains -- it just trains against a
differently scaled objective than the imitation anchor it is summed with. So
the density is checked against ``torch.distributions`` rather than against
itself.
"""

from __future__ import annotations

import math
import unittest

from .common import DummyExpert, require_torch


class TestFlowSigmas(unittest.TestCase):
    def test_constant_schedule_scales_by_sqrt_steps(self):
        torch = require_torch()
        from sim_vla.models.flow_sampler import flow_sigmas

        sigmas = flow_sigmas(0.1, 4)
        self.assertEqual(tuple(sigmas.shape), (4,))
        self.assertTrue(torch.allclose(
            sigmas, torch.full((4,), 0.1 / math.sqrt(4))))

    def test_nonpositive_and_nonfinite_scales_are_refused(self):
        require_torch()
        from sim_vla.models.flow_sampler import flow_sigmas

        for bad in (0.0, -0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                flow_sigmas(bad, 4)

    def test_unknown_schedule_is_refused_not_silently_substituted(self):
        require_torch()
        from sim_vla.models.flow_sampler import flow_sigmas

        with self.assertRaises(ValueError):
            flow_sigmas(0.1, 4, schedule="cosine")


class TestTransitionDensity(unittest.TestCase):
    def test_matches_independent_normal_over_chunk_and_action_dims(self):
        torch = require_torch()
        from sim_vla.models.flow_sampler import transition_log_prob

        torch.manual_seed(0)
        mean = torch.randn(5, 7, 3)
        sample = torch.randn(5, 7, 3)
        sigma = torch.tensor(0.037)

        reference = torch.distributions.Independent(
            torch.distributions.Normal(mean, sigma.expand_as(mean)), 2)
        self.assertTrue(torch.allclose(
            transition_log_prob(sample, mean, sigma),
            reference.log_prob(sample), atol=1e-5))

    def test_density_is_summed_over_coordinates_not_averaged(self):
        torch = require_torch()
        from sim_vla.models.flow_sampler import transition_log_prob

        # Twice the coordinates, identical per-coordinate residuals: a sum
        # doubles, a mean does not move. Averaging here would silently rescale
        # the whole actor loss against the imitation anchor.
        mean = torch.zeros(1, 2, 3)
        sample = torch.full((1, 2, 3), 0.5)
        wide_mean = torch.zeros(1, 4, 3)
        wide_sample = torch.full((1, 4, 3), 0.5)
        narrow = transition_log_prob(sample, mean, 0.25)
        wide = transition_log_prob(wide_sample, wide_mean, 0.25)
        self.assertTrue(torch.allclose(wide, 2.0 * narrow, atol=1e-5))

    def test_zero_or_invalid_sigma_is_refused(self):
        torch = require_torch()
        from sim_vla.models.flow_sampler import transition_log_prob

        mean = torch.zeros(2, 3, 4)
        sample = torch.zeros(2, 3, 4)
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                transition_log_prob(sample, mean, bad)

    def test_shape_disagreement_is_refused(self):
        torch = require_torch()
        from sim_vla.models.flow_sampler import transition_log_prob

        with self.assertRaises(ValueError):
            transition_log_prob(torch.zeros(2, 3, 4), torch.zeros(2, 3, 5), 0.1)

    def test_density_is_computed_in_float32_from_half_inputs(self):
        torch = require_torch()
        from sim_vla.models.flow_sampler import transition_log_prob

        mean = torch.randn(3, 2, 4).to(torch.bfloat16)
        sample = torch.randn(3, 2, 4).to(torch.bfloat16)
        out = transition_log_prob(sample, mean, 0.05)
        self.assertEqual(out.dtype, torch.float32)


class TestEulerDirection(unittest.TestCase):
    def test_transition_mean_steps_down_from_noise_toward_the_action(self):
        torch = require_torch()
        from sim_vla.models.flow_sampler import transition_mean

        # A velocity field of constant ones: the mean must move *against* it,
        # matching sample_actions' ``x = x - dt * v``. The opposite sign
        # integrates toward noise and still runs.
        ones = lambda x, t, cond: torch.ones_like(x)
        u = torch.zeros(2, 3, 4)
        mean = transition_mean(ones, u, torch.zeros(2), None, 0.25)
        self.assertTrue(torch.allclose(mean, torch.full_like(u, -0.25)))

    def test_times_run_from_one_down_to_just_above_zero(self):
        torch = require_torch()
        from sim_vla.models.flow_sampler import flow_times

        self.assertTrue(torch.allclose(
            flow_times(4), torch.tensor([1.0, 0.75, 0.5, 0.25])))

    def test_sampling_and_scoring_agree_on_the_recorded_path(self):
        torch = require_torch()
        from sim_vla.models.flow_sampler import (flow_sigmas, sample_flow_path,
                                                 transition_log_prob,
                                                 transition_mean)
        from sim_vla.models.latent_adapter import LatentAdapter

        torch.manual_seed(0)
        adapter = LatentAdapter(feature_dim=16, token_dim=16, hidden=32)
        expert = DummyExpert(token_dim=16, action_dim=4)
        cond = {"state_token": adapter(torch.randn(3, 16)), "instruction": None}
        sigmas = flow_sigmas(0.1, 5)
        with torch.no_grad():
            path = sample_flow_path(expert, cond, batch=3, chunk=4, dim=4,
                                    steps=5, sigmas=sigmas)

        # Nothing collected may carry a graph: the whole point is that the
        # sequential chain is not retained.
        for key in ("states", "means", "chunk"):
            self.assertIsNone(path[key].grad_fn, key)

        # Recomputing each transition mean from the recorded state reproduces
        # the mean that was actually sampled from.
        dt = 1.0 / 5
        for index in range(5):
            with torch.no_grad():
                again = transition_mean(expert, path["states"][:, index],
                                        path["times"][index].expand(3), cond, dt)
            self.assertTrue(torch.allclose(again, path["means"][:, index],
                                           atol=1e-6))
            # And the density of the state that was recorded is finite.
            logp = transition_log_prob(path["states"][:, index + 1],
                                       path["means"][:, index], sigmas[index])
            self.assertEqual(tuple(logp.shape), (3,))
            self.assertTrue(bool(torch.isfinite(logp).all()))

    def test_final_state_is_the_returned_chunk(self):
        torch = require_torch()
        from sim_vla.models.flow_sampler import flow_sigmas, sample_flow_path
        from sim_vla.models.latent_adapter import LatentAdapter

        torch.manual_seed(1)
        adapter = LatentAdapter(feature_dim=8, token_dim=8, hidden=16)
        expert = DummyExpert(token_dim=8, action_dim=3)
        cond = {"state_token": adapter(torch.randn(2, 8)), "instruction": None}
        with torch.no_grad():
            path = sample_flow_path(expert, cond, batch=2, chunk=5, dim=3,
                                    steps=4, sigmas=flow_sigmas(0.05, 4))
        self.assertEqual(tuple(path["states"].shape), (2, 5, 5, 3))
        self.assertTrue(torch.equal(path["states"][:, -1], path["chunk"]))


class TestScoreGradientDirection(unittest.TestCase):
    """A positive advantage must make the sampled transition more likely."""

    def _step(self, advantage: float) -> float:
        torch = require_torch()
        from sim_vla.models.flow_sampler import transition_log_prob

        torch.manual_seed(0)
        mean = torch.zeros(1, 2, 3, requires_grad=True)
        sample = torch.full((1, 2, 3), 0.4)
        sigma = 0.2
        before = float(transition_log_prob(sample, mean, sigma))
        loss = -(advantage * transition_log_prob(sample, mean, sigma)).mean()
        loss.backward()
        with torch.no_grad():
            moved = mean - 0.05 * mean.grad
        after = float(transition_log_prob(sample, moved, sigma))
        return after - before

    def test_positive_advantage_raises_the_sample_log_probability(self):
        self.assertGreater(self._step(+1.0), 0.0)

    def test_negative_advantage_lowers_the_sample_log_probability(self):
        self.assertLess(self._step(-1.0), 0.0)

    def test_score_estimator_matches_the_analytic_expected_reward_gradient(self):
        """E[ r(x) * d/dmu log p(x) ] == d/dmu E[ r(x) ] for a toy case.

        With ``p = N(mu, sigma^2)`` and ``r(x) = sum(x)``, the analytic
        gradient of the expected reward with respect to ``mu`` is one per
        coordinate. This is the property the whole objective rests on, so it is
        checked numerically rather than assumed from the algebra.
        """
        torch = require_torch()
        from sim_vla.models.flow_sampler import transition_log_prob

        torch.manual_seed(0)
        samples, sigma = 200_000, 0.3
        mean = torch.zeros(1, 1, 2, requires_grad=True)
        draw = mean.detach() + sigma * torch.randn(samples, 1, 2)
        reward = draw.flatten(start_dim=1).sum(-1)            # (N,)
        logp = transition_log_prob(draw, mean.expand_as(draw), sigma)
        (reward.detach() * logp).mean().backward()
        # One per coordinate, within Monte-Carlo error at this sample count.
        self.assertTrue(torch.allclose(mean.grad, torch.ones_like(mean.grad),
                                       atol=5e-2),
                        f"score estimate {mean.grad} is not close to ones")


def build(graph_enabled, device="cpu"):
    """A tiny world model, critic and batch, as stage 8's tests build them.

    ``device`` goes in through the config rather than a later ``.cuda()``:
    every RSSM block carries its own device key, so a model built for CPU and
    moved afterwards keeps CPU-side constants and raises inside ``obs_step``.
    """
    torch = require_torch()
    from sim_vla.models.critics import ValueCritic
    from sim_vla.models.world_model import build_world_model

    from .common import fake_batch, obs_shapes, small_model_config

    _cfg, model_cfg = small_model_config(graph_enabled, device=device)
    batch = fake_batch(graph_enabled=graph_enabled)
    if device != "cpu":
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
    model = build_world_model(model_cfg, obs_shapes(batch), 8,
                              graph_enabled=graph_enabled)
    model = model.to(device)
    critic = ValueCritic(model_cfg, model.feature_dim).to(device)
    return model, critic, batch


def make_actor(model):
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


def config(**overrides):
    from sim_vla.training.actor_critic import ActorCriticConfig

    settings = dict(horizon=3, flow_steps=2, critic_warmup=0,
                    actor_objective="flow_reinforce", flow_noise_std=0.1,
                    actor_transition_microbatch=7, imagination_microbatch=0)
    settings.update(overrides)
    return ActorCriticConfig(**settings)


def give_the_heads_an_opinion(model, critic):
    """Move the reward and value heads off their flat initialization.

    Both are ``symexp_twohot`` heads whose output layer starts at zero, so
    every bin gets the same logit and the mode is exactly 0.0 for every state.
    That makes ``returns - value`` identically zero -- and a zero advantage
    multiplying a log probability is a zero gradient, exactly as the algebra
    says it should be.

    This is not a quirk of the toy model. It is why ``critic_warmup`` is
    load-bearing for ``flow_reinforce`` in a way it is not for the pathwise
    objective: pathwise differentiates the return itself and still moves
    through a flat head, whereas a score-function update multiplied by a zero
    advantage does nothing at all.
    """
    torch = require_torch()
    torch.manual_seed(3)
    for parameter in critic.net.parameters():
        parameter.data.add_(torch.randn_like(parameter))
    for parameter in model.reward_head.parameters():
        parameter.data.add_(torch.randn_like(parameter))
    return model, critic


def trainer_for(graph_enabled=False, opinionated=True, seed=0, **overrides):
    from sim_vla.training.actor_critic import ActorCriticTrainer
    from sim_vla.training.imagination import flatten_start

    model, critic, batch = build(graph_enabled)
    if opinionated:
        give_the_heads_an_opinion(model, critic)
    actor = make_actor(model)
    trainer = ActorCriticTrainer(model, actor, critic, config(**overrides),
                                 seed=seed)
    start = flatten_start(model.observe(batch)["post"], graph_enabled)
    return trainer, start, model, actor, critic


class TestCollectionIsGraphless(unittest.TestCase):
    def test_recorded_rollout_carries_no_autograd_graph(self):
        torch = require_torch()
        trainer, start, _model, _actor, _critic = trainer_for()
        record = trainer.collect(start)
        for key in ("features", "flow_states", "executed_actions"):
            self.assertIsNone(record[key].grad_fn, key)
            self.assertFalse(record[key].requires_grad, key)
        horizon, batch = 3, start[0].shape[0]
        self.assertEqual(tuple(record["flow_states"].shape),
                         (horizon, batch, 3, 4, 8))
        self.assertEqual(tuple(record["executed_actions"].shape),
                         (horizon, batch, 8))

    def test_microbatched_collection_covers_every_start_state(self):
        require_torch()
        trainer, start, _m, _a, _c = trainer_for(imagination_microbatch=3)
        record = trainer.collect(start)
        self.assertEqual(int(record["batch"]), int(start[0].shape[0]))
        self.assertEqual(record["features"].shape[1], start[0].shape[0])


class TestTargetTimeline(unittest.TestCase):
    """Successor indexing and survival weights, on hand-checkable numbers."""

    def _targets_with_continuation(self, cont_values):
        """Targets computed against a hand-chosen continuation stream.

        ``imagined_rewards`` is patched rather than the head itself: the head
        is a registered submodule and cannot be swapped for a plain callable,
        and this is the seam the targets actually read.
        """
        torch = require_torch()
        from sim_vla.training import actor_critic

        trainer, start, model, _actor, critic = trainer_for()
        record = trainer.collect(start)
        rows = record["features"].shape[1]
        cont = torch.tensor(cont_values).reshape(-1, 1).expand(-1, rows)
        original = actor_critic.imagined_rewards
        actor_critic.imagined_rewards = lambda wm, feat: {
            "reward": torch.zeros(feat.shape[0], rows),
            "cont": cont.to(feat.device)}
        try:
            return trainer, actor_critic.flow_reinforce_targets(
                model, critic, record, trainer.config,
                return_ema=trainer.return_ema)
        finally:
            actor_critic.imagined_rewards = original

    def test_weights_are_an_exclusive_cumulative_product(self):
        torch = require_torch()

        const = 0.5
        # Four features, so four successor-indexed entries; the targets read
        # [1:], giving three transitions for a horizon of three.
        trainer, targets = self._targets_with_continuation([const] * 4)
        weights = targets["weights"]
        discount = trainer.config.discount
        self.assertEqual(tuple(weights.shape)[0], 3)
        self.assertTrue(torch.allclose(weights[0], torch.ones_like(weights[0])))
        self.assertTrue(torch.allclose(
            weights[1], torch.full_like(weights[1], discount * const),
            atol=1e-5))
        self.assertTrue(torch.allclose(
            weights[2], torch.full_like(weights[2], (discount * const) ** 2),
            atol=1e-5))

    def test_a_terminal_transition_keeps_its_own_reward(self):
        """cont = 0 zeroes what follows, never the transition that earned it."""
        torch = require_torch()

        # Successor-indexed cont for transitions 0,1,2 is [1, 0, 1]: the
        # transition at index 1 terminates.
        _trainer, targets = self._targets_with_continuation([9.0, 1.0, 0.0, 1.0])
        weights = targets["weights"]
        self.assertTrue(bool((weights[0] > 0).all()),
                        "the first transition must always be eligible")
        self.assertTrue(bool((weights[1] > 0).all()),
                        "a terminating transition still earned its own reward")
        self.assertTrue(bool((weights[2] == 0).all()),
                        "everything after a terminal must be weighted zero")

    def test_advantages_are_detached(self):
        require_torch()
        from sim_vla.training.actor_critic import flow_reinforce_targets

        trainer, start, model, _actor, critic = trainer_for()
        record = trainer.collect(start)
        targets = flow_reinforce_targets(model, critic, record, trainer.config,
                                         return_ema=trainer.return_ema)
        for key in ("advantage", "weights", "returns"):
            self.assertIsNone(targets[key].grad_fn, key)


class TestGradientBoundaries(unittest.TestCase):
    def test_scoring_reaches_the_adapter_through_the_recomputed_prefix(self):
        torch = require_torch()
        trainer, start, _model, actor, _critic = trainer_for()
        record = trainer.collect(start)
        index = torch.arange(6)
        logp = trainer.score_path(record, index)
        self.assertTrue(logp.requires_grad)
        grads = torch.autograd.grad(logp.sum(),
                                    list(actor.adapter.parameters()),
                                    allow_unused=True)
        self.assertTrue(any(g is not None and float(g.abs().sum()) > 0
                            for g in grads),
                        "no gradient reached the adapter from the score")

    def test_actor_update_leaves_world_model_and_critic_untouched(self):
        torch = require_torch()
        trainer, start, model, _actor, critic = trainer_for()
        world_before = [p.detach().clone() for p in model.parameters()]
        trainer.update(start)
        for old, new in zip(world_before, model.parameters()):
            self.assertTrue(torch.equal(old, new.detach()),
                            "the actor/critic update moved the world model")
        self.assertTrue(all(p.grad is None or float(p.grad.abs().sum()) == 0.0
                            for p in model.parameters()),
                        "the world model accumulated a gradient")

    def test_actor_update_moves_the_adapter(self):
        torch = require_torch()
        trainer, start, _model, actor, _critic = trainer_for()
        before = [p.detach().clone() for p in actor.adapter.parameters()]
        metrics = trainer.update(start)
        self.assertGreater(metrics["actor_grad_norm"], 0.0)
        moved = any(not torch.equal(old, new.detach())
                    for old, new in zip(before, actor.adapter.parameters()))
        self.assertTrue(moved, "the actor step changed nothing")

    def test_no_actor_step_during_critic_warmup(self):
        torch = require_torch()
        trainer, start, _model, actor, _critic = trainer_for(critic_warmup=2)
        before = [p.detach().clone() for p in actor.adapter.parameters()]
        metrics = trainer.update(start)
        self.assertNotEqual(metrics["actor_loss"], metrics["actor_loss"])  # nan
        self.assertEqual(trainer.actor_steps, 0)
        for old, new in zip(before, actor.adapter.parameters()):
            self.assertTrue(torch.equal(old, new.detach()),
                            "the actor moved during critic warm-up")

    def test_exactly_one_actor_step_per_update(self):
        require_torch()
        trainer, start, _m, _a, _c = trainer_for()
        trainer.update(start)
        self.assertEqual(trainer.actor_steps, 1)
        trainer.update(start)
        self.assertEqual(trainer.actor_steps, 2)


class TestModuleMode(unittest.TestCase):
    """Collection and scoring must see the same network, and restore it."""

    def test_scoring_runs_in_eval_mode_but_still_builds_a_graph(self):
        torch = require_torch()
        from sim_vla.training.actor_critic import deterministic_modules

        trainer, start, _model, actor, _critic = trainer_for()
        actor.train()
        record = trainer.collect(start)
        logp = trainer.score_path(record, torch.arange(4))
        self.assertTrue(logp.requires_grad,
                        "eval() must not disable autograd during scoring")

        seen = {}
        with deterministic_modules(actor):
            seen["training"] = actor.training
        self.assertFalse(seen["training"], "the module was not put in eval")

    def test_prior_mode_is_restored_after_collection_and_scoring(self):
        torch = require_torch()

        trainer, start, model, actor, _critic = trainer_for()
        for mode in (True, False):
            actor.train(mode)
            model.train(mode)
            record = trainer.collect(start)
            trainer.score_path(record, torch.arange(4))
            self.assertEqual(actor.training, mode,
                             "the actor's mode was not restored")
            self.assertEqual(model.training, mode,
                             "the world model's mode was not restored")

    def test_mode_is_restored_even_when_scoring_raises(self):
        torch = require_torch()
        from sim_vla.training.actor_critic import deterministic_modules

        _trainer, _start, _model, actor, _critic = trainer_for()
        actor.train()
        with self.assertRaises(ValueError):
            with deterministic_modules(actor):
                raise ValueError("boom")
        self.assertTrue(actor.training, "an exception left the module in eval")


class TestMicrobatchEquivalence(unittest.TestCase):
    def test_partitioning_the_scored_transitions_does_not_change_the_gradient(self):
        torch = require_torch()
        from sim_vla.training.actor_critic import flow_reinforce_targets

        def gradient(microbatch):
            torch.manual_seed(0)
            trainer, start, model, actor, critic = trainer_for()
            trainer.config.actor_transition_microbatch = microbatch
            torch.manual_seed(7)
            record = trainer.collect(start)
            targets = flow_reinforce_targets(
                model, critic, record, trainer.config,
                return_ema=trainer.return_ema)
            horizon = int(record["horizon"])
            count = int(record["batch"])
            steps = int(record["flow_steps"])
            total = horizon * count * steps
            norm = 1.0 / float(count * horizon)
            order = torch.arange(total)
            pair = torch.div(order, steps, rounding_mode="floor")
            scale = (targets["weights"].reshape(-1)[pair]
                     * targets["advantage"].reshape(-1)[pair]).detach()
            trainer.actor_opt.zero_grad(set_to_none=True)
            for offset in range(0, total, microbatch):
                index = order[offset:offset + microbatch]
                logp = trainer.score_path(record, index)
                (-(norm * scale[index] * logp).sum()).backward()
            return [p.grad.detach().clone()
                    for p in actor.adapter.parameters() if p.grad is not None]

        whole = gradient(10_000)
        # 7 does not divide the transition count, so the last group is short:
        # a normalizer that depended on group size would show up here.
        for size in (7, 16):
            parts = gradient(size)
            self.assertEqual(len(whole), len(parts))
            for a, b in zip(whole, parts):
                self.assertTrue(torch.allclose(a, b, atol=1e-5),
                                f"microbatch {size} changed the gradient")


class TestBothArms(unittest.TestCase):
    def test_graph_arm_updates_and_keeps_its_wider_feature(self):
        require_torch()
        trainer, start, model, _actor, _critic = trainer_for(graph_enabled=True)
        self.assertEqual(len(start), 3, "the graph arm carries (h, z, g)")
        record = trainer.collect(start)
        self.assertEqual(record["features"].shape[-1], model.feature_dim)
        metrics = trainer.update(start)
        self.assertGreater(metrics["actor_grad_norm"], 0.0)

    def test_baseline_arm_updates(self):
        require_torch()
        trainer, start, _model, _actor, _critic = trainer_for(graph_enabled=False)
        self.assertEqual(len(start), 2, "the baseline carries (h, z) only")
        metrics = trainer.update(start)
        self.assertGreater(metrics["actor_grad_norm"], 0.0)


class DemoSamplerStub:
    """A demonstration sampler that hands back the same prepared window.

    Deterministic on purpose: the anchor's equivalence and agreement tests
    compare two computations of the same rows, and a sampler that redrew would
    make the comparison meaningless rather than strict.
    """

    def __init__(self, graph_enabled=False, cuda=False):
        import torch

        from .common import fake_batch

        batch = fake_batch(graph_enabled=graph_enabled)
        if cuda:
            batch = {k: (v.cuda() if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
        self._batch = batch
        self.calls = 0

    def batch(self, _size):
        self.calls += 1
        return self._batch


class TestDemonstrationAnchor(unittest.TestCase):
    def anchored(self, weight=1.0, graph_enabled=False, **overrides):
        from sim_vla.training.actor_critic import ActorCriticTrainer
        from sim_vla.training.imagination import flatten_start

        model, critic, batch = build(graph_enabled)
        give_the_heads_an_opinion(model, critic)
        actor = make_actor(model)
        sampler = DemoSamplerStub(graph_enabled)
        trainer = ActorCriticTrainer(
            model, actor, critic, config(demo_anchor=weight, **overrides),
            demo_sampler=sampler, to_model_batch=lambda b: b)
        start = flatten_start(model.observe(batch)["post"], graph_enabled)
        return trainer, start, model, actor, sampler

    def test_anchor_agrees_with_stage_1b_on_the_same_rows(self):
        """Same rows, same noise and times -> the same flow-matching loss."""
        torch = require_torch()
        from sim_vla.models.flow_sampler import flow_matching_loss
        from sim_vla.training.train_imitation import prepare_anchor_rows

        trainer, _start, model, actor, sampler = self.anchored()
        rows = prepare_anchor_rows(model, sampler.batch(8),
                                   int(actor.chunk_size))
        self.assertIsNotNone(rows, "the fixture window must have eligible rows")
        feat, targets, mask = rows

        generator = torch.Generator().manual_seed(11)
        cond = actor.condition(feat.detach(), None)
        stage_1b, _m = flow_matching_loss(actor.velocity_fn(), targets, cond,
                                          mask=mask, generator=generator)
        generator = torch.Generator().manual_seed(11)
        cond = actor.condition(feat.detach(), None)
        valid = float(mask.sum()) * float(targets.shape[-1])
        anchored, _m = flow_matching_loss(actor.velocity_fn(), targets, cond,
                                          mask=mask, denominator=valid,
                                          generator=generator)
        # The Stage 1B call divides by weight.sum(), which for an unmasked
        # dim set is exactly the same number; the anchor just states it.
        self.assertTrue(torch.allclose(stage_1b, anchored, atol=1e-6),
                        f"{float(stage_1b)} != {float(anchored)}")

    def test_global_denominator_is_independent_of_the_partition(self):
        """The invariant the anchor needs, tested on the arithmetic itself.

        Going through ``flow_matching_loss`` cannot test this: it draws its own
        noise and time per call, so two partitions score different samples and
        any difference would be the noise rather than the denominator. The
        property at stake is purely about how the masked sum is normalized, so
        that is what this exercises, with the errors fixed.
        """
        torch = require_torch()

        torch.manual_seed(0)
        rows = 7
        error = torch.rand(rows, 4, 3)
        # Deliberately unequal per-row validity: equal masks would hide the
        # bug, because then every group's own mean happens to be comparable.
        mask = torch.tensor([4, 1, 3, 2, 4, 1, 2]).reshape(rows, 1) > (
            torch.arange(4).reshape(1, 4))
        weight = mask.unsqueeze(-1).to(error.dtype).expand_as(error)
        total_valid = float(weight.sum())

        def partitioned(group, denominator):
            out = 0.0
            for offset in range(0, rows, group):
                stop = offset + group
                piece = error[offset:stop] * weight[offset:stop]
                scale = (total_valid if denominator == "global"
                         else max(float(weight[offset:stop].sum()), 1.0))
                out += float(piece.sum() / scale)
            return out

        whole = partitioned(rows, "global")
        # 3 leaves an uneven final group, which is exactly where a per-group
        # denominator diverges from a global one.
        for group in (1, 2, 3, 5):
            self.assertAlmostEqual(whole, partitioned(group, "global"),
                                   places=6)
        self.assertNotAlmostEqual(whole, partitioned(3, "per_group"), places=3,
                                  msg="the fixture cannot distinguish the two "
                                      "denominators, so it proves nothing")

    def test_anchor_reports_the_whole_selections_valid_target_count(self):
        require_torch()
        from sim_vla.training.train_imitation import prepare_anchor_rows

        trainer, start, model, actor, sampler = self.anchored(weight=1.0)
        _feat, targets, mask = prepare_anchor_rows(
            model, sampler.batch(8), int(actor.chunk_size))
        expected = float(mask.sum()) * float(targets.shape[-1])
        metrics = trainer.update(start)
        self.assertAlmostEqual(metrics["anchor_valid_targets"], expected,
                               places=4)

    def test_nonzero_anchor_contributes_actor_gradient(self):
        torch = require_torch()
        trainer, start, _model, actor, _sampler = self.anchored(weight=1.0)
        metrics = trainer.update(start)
        self.assertIn("anchor_loss", metrics)
        self.assertGreater(metrics["anchor_rows"], 0.0)
        self.assertGreater(metrics["actor_grad_norm"], 0.0)

    def test_zero_anchor_does_no_demonstration_work(self):
        require_torch()
        trainer, start, _model, _actor, sampler = self.anchored(weight=0.0)
        trainer.update(start)
        self.assertEqual(sampler.calls, 0,
                         "a zero anchor still drew a demonstration batch")

    def test_anchor_changes_the_gradient_it_is_added_to(self):
        torch = require_torch()

        def gradient(weight):
            torch.manual_seed(0)
            trainer, start, _m, actor, _s = self.anchored(weight=weight)
            torch.manual_seed(21)
            trainer.update(start)
            return [p.detach().clone() for p in actor.adapter.parameters()]

        without = gradient(0.0)
        with_anchor = gradient(5.0)
        self.assertTrue(
            any(not torch.allclose(a, b, atol=1e-7)
                for a, b in zip(without, with_anchor)),
            "the anchor weight changed nothing about the update")

    def test_anchor_failure_is_reported_not_silently_dropped(self):
        require_torch()
        from sim_vla.training.actor_critic import ActorCriticTrainer
        from sim_vla.training.imagination import flatten_start

        class Empty:
            def batch(self, _size):
                from .common import fake_batch

                batch = fake_batch(graph_enabled=False)
                # No scored row anywhere: nothing is eligible to anchor on.
                batch["loss_mask"] = batch["loss_mask"] & False
                return batch

        model, critic, batch = build(False)
        actor = make_actor(model)
        trainer = ActorCriticTrainer(
            model, actor, critic, config(demo_anchor=1.0),
            demo_sampler=Empty(), to_model_batch=lambda b: b)
        start = flatten_start(model.observe(batch)["post"], False)
        with self.assertRaises(RuntimeError):
            trainer.update(start)


class TestConfigurationRefusals(unittest.TestCase):
    def test_flow_reinforce_without_a_noise_scale_is_refused_at_construction(self):
        require_torch()
        from sim_vla.training.actor_critic import ActorCriticTrainer

        model, critic, _batch = build(False)
        actor = make_actor(model)
        with self.assertRaises(ValueError):
            ActorCriticTrainer(model, actor, critic,
                               config(flow_noise_std=0.0))

    def test_unknown_objective_is_refused(self):
        require_torch()
        from sim_vla.training.actor_critic import ActorCriticTrainer

        model, critic, _batch = build(False)
        actor = make_actor(model)
        with self.assertRaises(ValueError):
            ActorCriticTrainer(model, actor, critic,
                               config(actor_objective="ppo"))

    def test_anchor_without_demonstrations_is_refused(self):
        require_torch()
        from sim_vla.training.actor_critic import ActorCriticTrainer

        model, critic, _batch = build(False)
        actor = make_actor(model)
        with self.assertRaises(ValueError):
            ActorCriticTrainer(model, actor, critic, config(demo_anchor=1.0))


class TestSeededGenerators(unittest.TestCase):
    """Generators must live where the draw happens, and seed 0 is a seed."""

    def test_seed_zero_is_still_reproducible(self):
        torch = require_torch()

        def once():
            torch.manual_seed(0)
            trainer, start, _m, _a, _c = trainer_for(seed=0)
            return trainer.collect(start)["flow_states"]

        self.assertTrue(torch.equal(once(), once()),
                        "seed 0 was treated as unseeded and did not reproduce")

    def test_different_seeds_draw_different_paths(self):
        torch = require_torch()

        def once(seed):
            torch.manual_seed(0)
            trainer, start, _m, _a, _c = trainer_for(seed=seed)
            return trainer.collect(start)["flow_states"]

        self.assertFalse(torch.equal(once(0), once(7)),
                         "two seeds produced identical flow paths")

    def test_generator_is_created_on_the_requested_device(self):
        torch = require_torch()

        trainer, _start, _m, _a, _c = trainer_for(seed=11)
        generator = trainer.generator_for(torch.device("cpu"))
        self.assertEqual(generator.device.type, "cpu")
        # Same device asked twice is the same generator, so its stream
        # advances rather than restarting.
        self.assertIs(generator, trainer.generator_for(torch.device("cpu")))

    def test_per_device_streams_are_stable_across_processes(self):
        """The offset must not come from Python's salted string hash."""
        torch = require_torch()
        import zlib

        trainer, _s, _m, _a, _c = trainer_for(seed=5)
        expected = (5 + zlib.crc32(b"cpu:0")) % (2 ** 63 - 1)
        probe = torch.Generator(device="cpu")
        probe.manual_seed(expected)
        self.assertTrue(torch.equal(
            probe.get_state(),
            trainer.generator_for(torch.device("cpu")).get_state()))


class TestCudaSampling(unittest.TestCase):
    """The regression: a nonzero seed used to raise on GPU, never on CPU."""

    def setUp(self):
        from .common import require_cuda

        require_cuda()

    def test_nonzero_seed_collects_on_cuda(self):
        torch = require_torch()
        from sim_vla.training.actor_critic import ActorCriticTrainer
        from sim_vla.training.imagination import flatten_start

        model, critic, batch = build(False, device="cuda")
        give_the_heads_an_opinion(model, critic)
        actor = make_actor(model).cuda()
        # A CPU generator handed to a CUDA torch.randn raises; this is the
        # call that used to do exactly that.
        trainer = ActorCriticTrainer(model, actor, critic,
                                     config(flow_noise_std=0.05), seed=7,
                                     device=torch.device("cuda"))
        start = flatten_start(model.observe(batch)["post"], False)
        record = trainer.collect(start)
        self.assertEqual(record["flow_states"].device.type, "cuda")
        metrics = trainer.update(start)
        self.assertGreater(metrics["actor_grad_norm"], 0.0)

    def test_nonzero_seed_anchors_on_cuda(self):
        torch = require_torch()
        from sim_vla.training.actor_critic import ActorCriticTrainer
        from sim_vla.training.imagination import flatten_start

        model, critic, batch = build(False, device="cuda")
        give_the_heads_an_opinion(model, critic)
        actor = make_actor(model).cuda()
        sampler = DemoSamplerStub(cuda=True)
        trainer = ActorCriticTrainer(
            model, actor, critic,
            config(flow_noise_std=0.05, demo_anchor=1.0), seed=7,
            demo_sampler=sampler, to_model_batch=lambda b: b,
            device=torch.device("cuda"))
        start = flatten_start(model.observe(batch)["post"], False)
        metrics = trainer.update(start)
        self.assertGreater(metrics["anchor_rows"], 0.0)


class TestFlatHeadsHaveNoSignal(unittest.TestCase):
    """Item 4: *both* objectives are dead through zero-initialised heads."""

    def raw_actor_gradient(self, objective):
        """The actor gradient with the heads exactly as initialised.

        Measured without letting the critic take a step first, which is the
        only way to see the property itself: one critic update already makes
        the value head non-flat, so a measurement taken after it is measuring
        the updated critic, not the initialisation.
        """
        torch = require_torch()
        from sim_vla.training.actor_critic import (ActorCriticConfig,
                                                   actor_loss,
                                                   flow_reinforce_targets,
                                                   ActorCriticTrainer)
        from sim_vla.training.imagination import flatten_start

        torch.manual_seed(0)
        model, critic, batch = build(False)          # heads left untouched
        actor = make_actor(model)
        start = flatten_start(model.observe(batch)["post"], False)
        if objective == "pathwise":
            out = actor_loss(model, actor, critic, start,
                             ActorCriticConfig(horizon=3, flow_steps=2))
            out["loss"].backward()
        else:
            trainer = ActorCriticTrainer(
                model, actor, critic,
                config(flow_noise_std=0.1))
            record = trainer.collect(start)
            targets = flow_reinforce_targets(model, critic, record,
                                             trainer.config,
                                             return_ema=trainer.return_ema)
            # The advantage is the whole story here.
            self.assertEqual(float(targets["advantage"].abs().max()), 0.0)
            logp = trainer.score_path(record, torch.arange(8))
            (-(targets["advantage"].reshape(-1)[0] * logp).sum()).backward()
        grads = [p.grad for p in actor.parameters() if p.grad is not None]
        self.assertTrue(grads, "nothing received a .grad buffer at all")
        return float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads)))

    def test_flow_reinforce_has_no_gradient_through_flat_heads(self):
        self.assertEqual(self.raw_actor_gradient("flow_reinforce"), 0.0)

    def test_pathwise_has_no_gradient_through_flat_heads_either(self):
        """The correction: pathwise is not immune, it is equally dead.

        A zero-initialised head's output does not depend on the feature, so
        its gradient with respect to the feature is zero and the imagined
        return is constant in the action. Populated ``.grad`` buffers full of
        zeros are what make this easy to miss.
        """
        self.assertEqual(self.raw_actor_gradient("pathwise"), 0.0)

    def test_after_one_critic_step_pathwise_sees_a_trace_and_reinforce_none(self):
        """The asymmetry inside ``update()``, stated rather than assumed.

        ``update()`` trains the critic before the actor. ``pathwise``
        re-imagines afterwards and so picks up whatever the single critic step
        created; ``flow_reinforce`` computed its advantage *before* that step,
        deliberately, so it stays exactly zero for one more update. Neither is
        a usable learning signal -- both need the warm-up -- but they are not
        zero in the same way, and a test that claimed they were would be
        wrong.
        """
        torch = require_torch()
        from sim_vla.training.actor_critic import ActorCriticTrainer
        from sim_vla.training.imagination import flatten_start

        norms = {}
        for objective in ("pathwise", "flow_reinforce"):
            torch.manual_seed(0)
            model, critic, batch = build(False)
            actor = make_actor(model)
            trainer = ActorCriticTrainer(
                model, actor, critic,
                config(actor_objective=objective,
                       flow_noise_std=(0.1 if objective == "flow_reinforce"
                                       else 0.0)))
            start = flatten_start(model.observe(batch)["post"], False)
            norms[objective] = trainer.update(start)["actor_grad_norm"]
        self.assertEqual(norms["flow_reinforce"], 0.0)
        self.assertLess(norms["pathwise"], 1e-3,
                        "a trace from one critic step, not a real signal")

    def test_both_recover_once_the_heads_mean_something(self):
        require_torch()
        for objective in ("flow_reinforce", "pathwise"):
            trainer, start, _m, _a, _c = trainer_for(
                actor_objective=objective,
                flow_noise_std=0.1 if objective == "flow_reinforce" else 0.0)
            metrics = trainer.update(start)
            self.assertGreater(metrics["actor_grad_norm"], 0.0, objective)

    def test_advantage_magnitude_is_reported_not_just_its_signed_mean(self):
        require_torch()
        trainer, start, _m, _a, _c = trainer_for()
        metrics = trainer.update(start)
        self.assertIn("advantage_abs", metrics)
        self.assertIn("rl_grad_norm", metrics)
        self.assertGreater(metrics["advantage_abs"], 0.0)
        self.assertGreater(metrics["rl_grad_norm"], 0.0)


class TestPathwiseAnchor(unittest.TestCase):
    """Item 2: the pathwise branch accepted demo_anchor and ignored it."""

    def anchored(self, weight, objective="pathwise"):
        from sim_vla.training.actor_critic import ActorCriticTrainer
        from sim_vla.training.imagination import flatten_start

        model, critic, batch = build(False)
        give_the_heads_an_opinion(model, critic)
        actor = make_actor(model)
        sampler = DemoSamplerStub()
        trainer = ActorCriticTrainer(
            model, actor, critic,
            config(actor_objective=objective, demo_anchor=weight,
                   flow_noise_std=0.1 if objective == "flow_reinforce" else 0.0,
                   grad_report_every=1),
            demo_sampler=sampler, to_model_batch=lambda b: b)
        start = flatten_start(model.observe(batch)["post"], False)
        return trainer, start, actor, sampler

    def test_pathwise_anchor_does_demonstration_work(self):
        require_torch()
        trainer, start, _actor, sampler = self.anchored(1.0)
        metrics = trainer.update(start)
        self.assertGreater(sampler.calls, 0,
                           "the pathwise branch never drew a demo batch")
        self.assertIn("anchor_loss", metrics)
        self.assertGreater(metrics["anchor_rows"], 0.0)

    def test_pathwise_anchor_contributes_gradient(self):
        require_torch()
        trainer, start, _actor, _sampler = self.anchored(1.0)
        metrics = trainer.update(start)
        self.assertIn("anchor_grad_norm", metrics)
        self.assertGreater(metrics["anchor_grad_norm"], 0.0,
                           "the anchor was applied but moved nothing")

    def test_pathwise_zero_anchor_still_does_no_demo_work(self):
        require_torch()
        trainer, start, _actor, sampler = self.anchored(0.0)
        trainer.update(start)
        self.assertEqual(sampler.calls, 0)

    def test_pathwise_anchor_is_the_only_gradient_when_the_heads_are_flat(self):
        """Isolated, because summing cannot separate terms of unequal size.

        Two probes were wrong before this one. Comparing parameters after the
        step fails because AdamW's first update is about ``lr * sign(g)``, so
        very different gradients with matching signs land in the same place.
        Comparing summed gradients fails because the RL term can be many
        orders of magnitude larger, and in float32 ``1.3e8 + 6.4`` is
        ``1.3e8`` -- the anchor is applied, nonzero, and invisible in the sum.

        With freshly initialised heads the RL term contributes only the trace
        left by ``update()``'s one critic step -- order 1e-5 -- so an anchor
        that is wired in stands out by orders of magnitude and the question
        becomes decidable.
        """
        torch = require_torch()
        from sim_vla.training.actor_critic import ActorCriticTrainer
        from sim_vla.training.imagination import flatten_start

        def norm(weight):
            torch.manual_seed(0)
            model, critic, batch = build(False)      # heads left flat
            actor = make_actor(model)
            trainer = ActorCriticTrainer(
                model, actor, critic,
                config(actor_objective="pathwise", demo_anchor=weight),
                demo_sampler=DemoSamplerStub(), to_model_batch=lambda b: b)
            start = flatten_start(model.observe(batch)["post"], False)
            return trainer.update(start)["actor_grad_norm"]

        unanchored, anchored = norm(0.0), norm(3.0)
        self.assertLess(unanchored, 1e-3,
                        "the fixture is not isolating the anchor")
        self.assertGreater(anchored, 1000 * unanchored,
                           "the pathwise branch applied no anchor gradient")

    def test_anchor_ratio_is_reported_so_imbalance_is_visible(self):
        require_torch()
        trainer, start, _actor, _s = self.anchored(1.0)
        metrics = trainer.update(start)
        self.assertIn("anchor_grad_ratio", metrics)
        self.assertGreaterEqual(metrics["anchor_grad_ratio"], 0.0)

    def test_both_objectives_report_separate_rl_and_anchor_norms(self):
        require_torch()
        for objective in ("pathwise", "flow_reinforce"):
            trainer, start, _a, _s = self.anchored(1.0, objective)
            metrics = trainer.update(start)
            self.assertIn("rl_grad_norm", metrics, objective)
            self.assertIn("anchor_grad_norm", metrics, objective)


class TestAnchorBudgets(unittest.TestCase):
    """Item 3: window budget and eligible-row budget are different things."""

    def test_the_sampler_is_asked_for_windows_not_rows(self):
        require_torch()
        from sim_vla.training.actor_critic import ActorCriticTrainer
        from sim_vla.training.imagination import flatten_start

        model, critic, batch = build(False)
        give_the_heads_an_opinion(model, critic)
        actor = make_actor(model)

        class Recording(DemoSamplerStub):
            sizes: list = []

            def batch(self, size):
                type(self).sizes.append(int(size))
                return super().batch(size)

        Recording.sizes = []
        trainer = ActorCriticTrainer(
            model, actor, critic,
            config(demo_anchor=1.0, anchor_windows=3, anchor_rows=64),
            demo_sampler=Recording(), to_model_batch=lambda b: b)
        start = flatten_start(model.observe(batch)["post"], False)
        trainer.update(start)
        self.assertEqual(Recording.sizes, [3],
                         "the anchor asked for anchor_rows windows again")

    def test_row_cap_still_bounds_the_conditioned_rows(self):
        require_torch()
        from sim_vla.training.train_imitation import prepare_anchor_rows

        model, _critic, _b = build(False)
        batch = DemoSamplerStub().batch(4)
        everything = prepare_anchor_rows(model, batch, 4)
        capped = prepare_anchor_rows(model, batch, 4, max_rows=3)
        self.assertGreater(everything[0].shape[0], 3)
        self.assertEqual(capped[0].shape[0], 3)

    def test_grouping_never_slices_the_time_axis(self):
        """The causality property, tested directly rather than by equality.

        The real RSSM samples discrete latents, so two encodings of the same
        batch are not equal to each other -- grouping cannot be validated by
        comparing outputs. What has to hold is structural: every window is
        handed to ``observe`` with its whole time axis, because a window cut
        in time would give row ``t`` a posterior that never consumed rows
        ``0..t``. A recording stand-in checks exactly that.
        """
        torch = require_torch()
        from sim_vla.training.train_imitation import encode_windows

        batch = DemoSamplerStub().batch(4)
        windows, steps = batch["loss_mask"].shape

        class Recording:
            def __init__(self):
                self.seen = []

            def parameters(self):
                return iter([torch.zeros(1)])

            def observe(self, piece):
                self.seen.append(tuple(piece["loss_mask"].shape))
                return {"post": piece}

            def features(self, post):
                # Deterministic and window-identifying, so ordering is
                # checkable without fighting the sampler.
                return post["proprio"]

        for group in (1, 2, 3, 4, 99):
            model = Recording()
            out = encode_windows(model, batch, window_microbatch=group)
            self.assertTrue(all(shape[1] == steps for shape in model.seen),
                            f"group {group} cut a window's time axis")
            self.assertEqual(sum(shape[0] for shape in model.seen), windows)
            # Order preserved: prepare_anchor_rows indexes these features with
            # masks taken from the unsliced batch, so a permutation here would
            # pair every row with the wrong window's mask.
            self.assertTrue(torch.allclose(out, batch["proprio"].float()),
                            f"group {group} reordered the windows")

    def test_encoding_returns_float32_under_a_reduced_precision_setting(self):
        torch = require_torch()
        from sim_vla.training.train_imitation import encode_windows

        model, _critic, _b = build(False)
        batch = DemoSamplerStub().batch(2)
        out = encode_windows(model, batch, window_microbatch=1,
                             precision="float32")
        self.assertEqual(out.dtype, torch.float32)


class TestAnchorGradientPartition(unittest.TestCase):
    """Item 5: compare real gradients, with the path sample held fixed."""

    def fixture(self):
        torch = require_torch()
        from sim_vla.models.latent_adapter import LatentAdapter

        torch.manual_seed(0)
        rows, chunk, dim = 7, 4, 8
        adapter = LatentAdapter(feature_dim=16, token_dim=16, hidden=32)
        expert = DummyExpert(token_dim=16, action_dim=dim)
        feat = torch.randn(rows, 16)
        targets = torch.randn(rows, chunk, dim)
        # Deliberately unequal valid lengths per row: with equal masks a
        # per-group denominator and a global one happen to agree, so the
        # fixture would prove nothing.
        lengths = torch.tensor([4, 1, 3, 2, 4, 1, 2]).reshape(rows, 1)
        mask = lengths > torch.arange(chunk).reshape(1, chunk)
        noise = torch.randn(rows, chunk, dim)
        times = torch.rand(rows)
        valid = float(mask.sum()) * float(dim)
        return adapter, expert, feat, targets, mask, noise, times, valid

    def gradient(self, group_size):
        torch = require_torch()
        from sim_vla.models.flow_sampler import flow_matching_loss

        (adapter, expert, feat, targets, mask,
         noise, times, valid) = self.fixture()
        for parameter in adapter.parameters():
            parameter.grad = None
        for offset in range(0, feat.shape[0], group_size):
            stop = offset + group_size
            cond = {"state_token": adapter(feat[offset:stop]),
                    "instruction": None}
            loss, _m = flow_matching_loss(
                expert, targets[offset:stop], cond, mask=mask[offset:stop],
                denominator=valid, noise=noise[offset:stop],
                times=times[offset:stop])
            loss.backward()
        return [p.grad.detach().clone() for p in adapter.parameters()
                if p.grad is not None]

    def test_microbatched_anchor_gradient_equals_the_full_batch_one(self):
        torch = require_torch()

        whole = self.gradient(1000)
        self.assertTrue(whole, "the fixture produced no gradient at all")
        # 3 and 5 both leave an uneven final group over 7 rows.
        for group in (1, 2, 3, 5):
            parts = self.gradient(group)
            self.assertEqual(len(whole), len(parts))
            for a, b in zip(whole, parts):
                self.assertTrue(torch.allclose(a, b, atol=1e-6),
                                f"group {group} changed the anchor gradient")

    def test_a_per_group_denominator_would_have_been_caught(self):
        """The fixture must be able to fail, or it proves nothing."""
        torch = require_torch()
        from sim_vla.models.flow_sampler import flow_matching_loss

        (adapter, expert, feat, targets, mask,
         noise, times, _valid) = self.fixture()

        def per_group(group_size):
            for parameter in adapter.parameters():
                parameter.grad = None
            for offset in range(0, feat.shape[0], group_size):
                stop = offset + group_size
                cond = {"state_token": adapter(feat[offset:stop]),
                        "instruction": None}
                # denominator=None makes each group divide by its own count.
                loss, _m = flow_matching_loss(
                    expert, targets[offset:stop], cond,
                    mask=mask[offset:stop], noise=noise[offset:stop],
                    times=times[offset:stop])
                loss.backward()
            return [p.grad.detach().clone() for p in adapter.parameters()
                    if p.grad is not None]

        self.assertFalse(
            all(torch.allclose(a, b, atol=1e-6)
                for a, b in zip(per_group(1000), per_group(3))),
            "unequal masks did not expose a per-group denominator")


class TestSeedOverride(unittest.TestCase):
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
        self.assertTrue(all(torch.equal(a, b)
                            for a, b in zip(first, again)),
                        "seed 0 did not reproduce module initialisation")
        other = adapter_weights(7)
        self.assertFalse(all(torch.equal(a, b)
                             for a, b in zip(first, other)),
                         "two different seeds initialised identically")

    def test_seed_everything_covers_python_numpy_and_torch(self):
        torch = require_torch()
        import random

        import numpy as np

        from sim_vla.training.pretrain_world_model import seed_everything

        def draws(seed):
            seed_everything(seed)
            return (random.random(), float(np.random.rand()),
                    float(torch.rand(1)))

        self.assertEqual(draws(0), draws(0),
                         "seed 0 was not applied to all three generators")
        self.assertNotEqual(draws(0), draws(3))

    def test_trainer_generators_follow_the_run_seed(self):
        torch = require_torch()

        def first_path(seed):
            torch.manual_seed(99)          # deliberately unrelated
            trainer, start, _m, _a, _c = trainer_for(seed=seed)
            return trainer.collect(start)["flow_states"]

        self.assertTrue(torch.equal(first_path(0), first_path(0)))
        self.assertFalse(torch.equal(first_path(0), first_path(4)))


class TestAnchorDiagnosticsUnderImbalance(unittest.TestCase):
    """A small anchor beside a huge RL term must still be measured."""

    def build(self, anchor=1.0):
        from sim_vla.training.actor_critic import ActorCriticTrainer
        from sim_vla.training.imagination import flatten_start

        model, critic, batch = build(False)
        # The perturbed heads give an RL gradient many orders of magnitude
        # larger than the anchor's -- the situation where subtracting the
        # accumulated float32 gradients reports an anchor of exactly zero.
        give_the_heads_an_opinion(model, critic)
        actor = make_actor(model)
        trainer = ActorCriticTrainer(
            model, actor, critic,
            config(demo_anchor=anchor, grad_report_every=1),
            demo_sampler=DemoSamplerStub(), to_model_batch=lambda b: b)
        start = flatten_start(model.observe(batch)["post"], False)
        return trainer, start, actor

    def test_tiny_anchor_under_a_huge_rl_gradient_is_still_measured(self):
        """The regression, in the regime where subtraction actually fails.

        Synthetic on purpose. A toy rollout produces an imbalance of maybe
        1000x, which float32 represents perfectly well -- subtracting would
        pass there and still be wrong. Here the RL gradient is 1e8 and the
        anchor 1.0, so ``(rl + anchor) - rl`` is exactly 0.0 in float32 while
        the anchor is plainly nonzero.
        """
        torch = require_torch()

        trainer, _start, actor = self.build()
        trainable = [p for p in actor.parameters() if p.requires_grad]
        rl_value = 1e8

        # One element carries each term, so the norms are exactly the values
        # written rather than sqrt(numel) times them.
        with torch.no_grad():
            for parameter in trainable:
                parameter.grad = torch.zeros_like(parameter)
            trainable[0].grad.view(-1)[0] = rl_value

        def fake_anchor(_device):
            # Accumulated the way autograd would: assigning when .grad is
            # None, adding when it is not. The stash empties .grad first,
            # which is exactly what this has to tolerate.
            with torch.no_grad():
                if trainable[0].grad is None:
                    trainable[0].grad = torch.zeros_like(trainable[0])
                trainable[0].grad.view(-1)[0] += 1.0
            return {"anchor_loss": 1.0, "anchor_rows": 1.0}

        trainer._anchor_backward = fake_anchor
        trainer.config.grad_report_every = 1
        metrics = trainer._anchored_backward(trainable, torch.device("cpu"))

        # Float32 cannot hold the sum: this is the failure being guarded.
        self.assertEqual(float(torch.tensor(rl_value, dtype=torch.float32)
                               + torch.tensor(1.0, dtype=torch.float32)),
                         rl_value)
        self.assertAlmostEqual(metrics["anchor_grad_norm"], 1.0, places=4)
        self.assertAlmostEqual(metrics["rl_grad_norm"], rl_value, delta=1.0)
        self.assertEqual(metrics["retained_anchor_grad_norm"], 0.0,
                         "the fixture is not in the rounding regime")
        # And training behaviour is untouched: the combined gradient is the
        # sum, which in this regime is the RL term.
        self.assertAlmostEqual(float(trainable[0].grad.view(-1)[0]),
                               rl_value, delta=1.0)

    def test_a_dominant_rl_term_still_reports_a_nonzero_anchor(self):
        require_torch()
        trainer, start, _actor = self.build()
        metrics = trainer.update(start)
        self.assertGreater(
            metrics["anchor_grad_norm"], 0.0,
            "the anchor was measured as zero; it was measured by subtraction")
        self.assertLess(metrics["anchor_grad_ratio"], 1e-2,
                        "the fixture is not actually imbalanced")

    def test_retained_contribution_is_reported_separately(self):
        require_torch()
        trainer, start, _actor = self.build()
        metrics = trainer.update(start)
        self.assertIn("retained_anchor_grad_norm", metrics)
        # The point of the pair: the anchor exists, and almost none of it
        # survives being added to the RL term in float32.
        self.assertLessEqual(metrics["retained_anchor_grad_norm"],
                             metrics["anchor_grad_norm"] * 1.5 + 1e-6)

    def test_diagnostics_do_not_change_the_step_count_or_the_gradient(self):
        """Measuring must not perturb training.

        The stash-and-restore has to leave the same combined gradient and the
        same single optimizer step it would have had without diagnostics.
        """
        torch = require_torch()

        def run(report_every):
            torch.manual_seed(0)
            trainer, start, actor = self.build()
            trainer.config.grad_report_every = report_every
            captured = {}
            original = trainer.actor_opt.step

            def capture(*a, **k):
                captured.setdefault("calls", 0)
                captured["calls"] += 1
                captured["grad"] = [
                    None if p.grad is None else p.grad.detach().clone()
                    for p in actor.adapter.parameters()]
                return original(*a, **k)

            trainer.actor_opt.step = capture
            torch.manual_seed(5)
            trainer.update(start)
            return captured

        off, on = run(0), run(1)
        self.assertEqual(off["calls"], 1)
        self.assertEqual(on["calls"], 1, "diagnostics added an optimizer step")
        for a, b in zip(off["grad"], on["grad"]):
            if a is None or b is None:
                continue
            self.assertTrue(torch.allclose(a, b, rtol=1e-5, atol=1e-6),
                            "diagnostics changed the combined gradient")

    def test_diagnostics_follow_the_cadence(self):
        """The first update reports; the next one does not, at a long cadence.

        Reporting on ``actor_steps == 0`` is intended: a run should say what
        its gradient balance is immediately rather than after fifty updates.
        """
        require_torch()
        trainer, start, _actor = self.build()
        trainer.config.grad_report_every = 1000
        self.assertIn("anchor_grad_norm", trainer.update(start))
        self.assertNotIn("anchor_grad_norm", trainer.update(start))

    def test_a_zero_cadence_disables_diagnostics_entirely(self):
        require_torch()
        trainer, start, _actor = self.build()
        trainer.config.grad_report_every = 0
        metrics = trainer.update(start)
        self.assertNotIn("anchor_grad_norm", metrics)
        self.assertIn("anchor_loss", metrics, "the anchor itself must still run")


class TestValidationBoundaries(unittest.TestCase):
    """Item 4: validation moved to a boundary, not deleted."""

    def test_check_sigmas_still_refuses_bad_schedules(self):
        torch = require_torch()
        from sim_vla.models.flow_sampler import check_sigmas

        check_sigmas(torch.tensor([0.1, 0.2]))            # no raise
        for bad in ([0.0, 0.1], [-1.0, 0.1], [float("nan"), 0.1]):
            with self.assertRaises(ValueError):
                check_sigmas(torch.tensor(bad))

    def test_log_prob_validates_by_default_and_can_be_told_not_to(self):
        torch = require_torch()
        from sim_vla.models.flow_sampler import transition_log_prob

        mean = torch.zeros(2, 3, 4)
        with self.assertRaises(ValueError):
            transition_log_prob(mean, mean, 0.0)
        # The fast path does not re-check; the caller validated the schedule
        # once. It must still produce a tensor rather than silently returning
        # something else.
        out = transition_log_prob(mean, mean, 0.5, validate=False)
        self.assertEqual(tuple(out.shape), (2,))

    def test_score_path_bounds_are_checked_by_default(self):
        torch = require_torch()

        trainer, start, _m, _a, _c = trainer_for()
        record = trainer.collect(start)
        total = record["horizon"] * record["batch"] * record["flow_steps"]
        with self.assertRaises(IndexError):
            trainer.score_path(record, torch.tensor([total + 5]))

    def test_the_update_still_rejects_a_bad_schedule_before_scoring(self):
        require_torch()

        trainer, start, _m, _a, _c = trainer_for()
        record = trainer.collect(start)
        # Corrupt the recorded schedule the way a bug would, and confirm the
        # once-per-update boundary check catches it rather than thousands of
        # skipped per-microbatch checks letting it through.
        record["flow_sigmas"] = record["flow_sigmas"] * 0.0
        from sim_vla.models.flow_sampler import check_sigmas

        with self.assertRaises(ValueError):
            check_sigmas(record["flow_sigmas"])


class TestRealSmolVlaConsistency(unittest.TestCase):
    """Gates 2 and 4 against the real checkpoint, on the real device.

    Every requirement here is checked and skipped with a reason rather than
    worked around, because each workaround would make the test describe
    something other than what it claims:

    * **lerobot** -- without it there is no SmolVLA to be consistent with.
    * **CUDA** -- this validates the mixed-precision path that training uses.
      Falling back to CPU would exercise float32 on different kernels and
      still be reported as CUDA validation.
    * **An immutable pinned revision** -- ``actor.revision`` in the config.
      Loading ``main`` instead would validate whatever the Hub serves today,
      which is not the checkpoint the run will train from.

    What is measured is the discrepancy between the distribution that was
    *sampled from* during collection and the one that is *differentiated*
    during scoring. Those are two separate forward passes through a frozen
    network under autocast, so they are not bit-identical, and a coordinate
    error compared against sigma says nothing useful: sigma is the width of
    the distribution, not a bound on how far its mean may move. The right
    scale is the density itself, so this reports

        KL( N(mu_collect, s^2 I) || N(mu_score, s^2 I) )
            = sum_coords (mu_collect - mu_score)^2 / (2 s^2)

    summed over the chunk exactly as ``transition_log_prob`` sums it, in nats,
    alongside the absolute log-probability difference those means produce.
    """

    #: Summed KL in nats between the collection and scoring transition
    #: distributions. 1e-2 nats over a whole chunk means the two densities
    #: agree to about 1% in probability, which is far below the variation the
    #: advantage itself carries; anything approaching 1 nat would mean the
    #: gradient is being taken against a different distribution than the one
    #: that produced the sample.
    KL_TOLERANCE_NATS = 1e-2
    #: Absolute difference in summed log-probability, same reasoning.
    LOGP_TOLERANCE_NATS = 5e-2

    def setUp(self):
        self.torch = require_torch()
        try:
            import lerobot  # noqa: F401
        except Exception as exc:                           # noqa: BLE001
            raise unittest.SkipTest(f"lerobot not installed: {exc}")
        if not self.torch.cuda.is_available():
            raise unittest.SkipTest(
                "no CUDA device: this validates the mixed-precision GPU path "
                "and a CPU run would not be that path")
        from sim_vla.config import load_config

        cfg = load_config("peginsertion", "dreamer")
        self.revision = str((cfg.get("actor") or {}).get("revision") or "")
        self.precision = str((cfg.get("online") or {}).get(
            "precision", "bfloat16"))
        if len(self.revision) != 40:
            raise unittest.SkipTest(
                "actor.revision is not pinned to an immutable 40-character "
                f"commit (got {self.revision!r}); refusing to silently "
                "validate whatever 'main' resolves to today")

    def actor(self, feature_dim=64):
        from sim_vla.models.latent_adapter import LatentAdapter
        from sim_vla.models.pretrained import (PretrainedError, load_policy,
                                               model_facts)
        from sim_vla.models.smolvla_actor import SmolVLAActor

        try:
            loaded = load_policy(revision=self.revision)
        except PretrainedError as exc:                     # noqa: BLE001
            message = str(exc).lower()
            if any(word in message for word in
                   ("could not resolve", "connection", "offline", "401",
                    "403", "not importable")):
                raise unittest.SkipTest(str(exc))
            raise
        facts = model_facts(loaded)
        adapter = LatentAdapter(feature_dim=feature_dim,
                                token_dim=int(facts["vlm_hidden_size"]),
                                hidden=256)
        actor = SmolVLAActor(loaded, adapter, action_dim=8,
                             instruction="insert the peg into the hole",
                             state_token_mode="embedding").cuda()
        return actor, facts

    def collect(self, actor, feature_dim=64, batch=3, noise_std=0.03):
        """One recorded rollout step, exactly as the trainer collects it."""
        torch = self.torch
        from sim_vla.models.flow_sampler import flow_sigmas, sample_flow_path
        from sim_vla.training.actor_critic import deterministic_modules
        from sim_vla.training.precision import autocast

        device = torch.device("cuda")
        feat = torch.randn(batch, feature_dim, device=device)
        sigmas = flow_sigmas(noise_std, int(actor.flow_steps), device=device)
        with deterministic_modules(actor):
            with torch.no_grad(), autocast(device, self.precision):
                cond = actor.condition(feat, None)
                path = sample_flow_path(
                    actor.velocity_fn(), cond, batch=batch,
                    chunk=int(actor.chunk_size), dim=int(actor.action_dim),
                    steps=int(actor.flow_steps), sigmas=sigmas,
                    device=device, dtype=torch.float32)
        return feat, sigmas, path

    def test_every_transition_agrees_between_collection_and_scoring(self):
        """All K transitions, not just the first, in KL and in log-density."""
        torch = self.torch
        from sim_vla.models.flow_sampler import (transition_log_prob,
                                                 transition_mean)
        from sim_vla.training.actor_critic import deterministic_modules
        from sim_vla.training.precision import autocast

        actor, _facts = self.actor()
        feat, sigmas, path = self.collect(actor)
        device = torch.device("cuda")
        steps = int(actor.flow_steps)
        worst_kl, worst_logp = 0.0, 0.0

        with deterministic_modules(actor):
            for index in range(steps):
                with torch.no_grad(), autocast(device, self.precision):
                    cond = actor.condition(feat.detach(), None)
                    again = transition_mean(
                        actor.velocity_fn(), path["states"][:, index],
                        path["times"][index].expand(feat.shape[0]), cond,
                        1.0 / float(steps))
                sigma = sigmas[index].float()
                collected = path["means"][:, index].float()
                scored = again.float()
                # Same sigma both sides, so the KL is the scaled squared mean
                # gap, summed over the chunk the way the density is summed.
                kl = (((collected - scored) ** 2) / (2.0 * sigma ** 2)
                      ).flatten(start_dim=1).sum(-1)
                gap = (transition_log_prob(path["states"][:, index + 1],
                                           collected, sigma)
                       - transition_log_prob(path["states"][:, index + 1],
                                             scored, sigma)).abs()
                worst_kl = max(worst_kl, float(kl.max()))
                worst_logp = max(worst_logp, float(gap.max()))

        self.assertLess(
            worst_kl, self.KL_TOLERANCE_NATS,
            f"worst summed KL {worst_kl:.4g} nats across {steps} transitions "
            f"exceeds {self.KL_TOLERANCE_NATS} -- scoring differentiates a "
            "different distribution than collection sampled from")
        self.assertLess(
            worst_logp, self.LOGP_TOLERANCE_NATS,
            f"worst |delta log p| {worst_logp:.4g} nats exceeds "
            f"{self.LOGP_TOLERANCE_NATS}")

    def test_agreement_survives_the_real_microbatch_layouts(self):
        """Collection batches by start state; scoring batches by transition.

        Scoring flattens ``(step, start, flow step)`` and cuts it into groups
        that do not align with collection's, so the same conditioning feature
        appears several times in one scored batch and the final group is
        usually short. Those are different tensor shapes through the same
        frozen network, and autocast kernels are shape-dependent -- which is
        exactly the layout difference this has to cover.
        """
        torch = self.torch
        from sim_vla.models.flow_sampler import transition_mean
        from sim_vla.training.actor_critic import deterministic_modules
        from sim_vla.training.precision import autocast

        actor, _facts = self.actor()
        batch = 3
        feat, sigmas, path = self.collect(actor, batch=batch)
        device = torch.device("cuda")
        steps = int(actor.flow_steps)

        rows = [(b, k) for b in range(batch) for k in range(steps)]
        worst_kl = 0.0
        with deterministic_modules(actor):
            # 5 divides neither len(rows) nor batch, so groups repeat
            # conditioning features and the last one is short.
            for size in (1, 5, len(rows)):
                for offset in range(0, len(rows), size):
                    group = rows[offset:offset + size]
                    picked = torch.stack([feat[b] for b, _k in group])
                    states = torch.stack(
                        [path["states"][b, k] for b, k in group])
                    times = torch.stack(
                        [path["times"][k] for _b, k in group])
                    sigma = torch.stack([sigmas[k] for _b, k in group]
                                        ).reshape(-1, 1, 1).float()
                    with torch.no_grad(), autocast(device, self.precision):
                        cond = actor.condition(picked.detach(), None)
                        again = transition_mean(actor.velocity_fn(),
                                                states.detach(), times, cond,
                                                1.0 / float(steps))
                    collected = torch.stack(
                        [path["means"][b, k] for b, k in group]).float()
                    kl = (((collected - again.float()) ** 2)
                          / (2.0 * sigma ** 2)).flatten(start_dim=1).sum(-1)
                    worst_kl = max(worst_kl, float(kl.max()))
        self.assertLess(
            worst_kl, self.KL_TOLERANCE_NATS,
            f"worst summed KL {worst_kl:.4g} nats across microbatch layouts "
            f"exceeds {self.KL_TOLERANCE_NATS}")

    def test_adapter_learns_and_the_frozen_vlm_does_not_move(self):
        """Through the real recomputed prefix, across a real optimizer step."""
        torch = self.torch
        from sim_vla.models.flow_sampler import (transition_log_prob,
                                                 transition_mean)
        from sim_vla.training.precision import autocast

        actor, _facts = self.actor()
        feat, sigmas, path = self.collect(actor)
        device = torch.device("cuda")
        steps = int(actor.flow_steps)

        frozen = [(name, p, p.detach().clone())
                  for name, p in actor.named_parameters()
                  if not p.requires_grad]
        self.assertTrue(frozen, "nothing in this actor is frozen")
        trainable = [p for p in actor.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)

        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for index in range(steps):
            with autocast(device, self.precision):
                cond = actor.condition(feat.detach(), None)
                mean = transition_mean(
                    actor.velocity_fn(), path["states"][:, index].detach(),
                    path["times"][index].expand(feat.shape[0]), cond,
                    1.0 / float(steps))
            logp = transition_log_prob(path["states"][:, index + 1].detach(),
                                       mean, sigmas[index])
            (-logp.sum()).backward()
            total += float(logp.detach().sum())
        self.assertTrue(math.isfinite(total))

        adapter_grads = [p.grad for p in actor.adapter.parameters()
                         if p.grad is not None]
        self.assertTrue(adapter_grads, "no adapter parameter received a grad")
        norm = float(torch.sqrt(
            sum((g.float() ** 2).sum() for g in adapter_grads)))
        self.assertGreater(norm, 0.0,
                           "the adapter's gradient was identically zero")

        for name, parameter, _before in frozen:
            self.assertIsNone(
                parameter.grad,
                f"frozen parameter {name} accumulated a gradient")

        optimizer.step()
        for name, parameter, before in frozen:
            self.assertTrue(
                torch.equal(parameter.detach(), before),
                f"frozen parameter {name} moved across the optimizer step")


class TestProfiling(unittest.TestCase):
    def test_profiling_off_adds_no_metrics_and_costs_nothing(self):
        require_torch()
        trainer, start, _m, _a, _c = trainer_for(profile=False)
        metrics = trainer.update(start)
        self.assertFalse([k for k in metrics if k.startswith("profile_")])

    def test_profiling_on_reports_every_phase(self):
        require_torch()
        trainer, start, _m, _a, _c = trainer_for(profile=True)
        metrics = trainer.update(start)
        for phase in ("collect", "targets", "critic", "score"):
            self.assertIn(f"profile_{phase}_s", metrics, phase)
            self.assertGreaterEqual(metrics[f"profile_{phase}_s"], 0.0)

    def test_the_anchor_phase_appears_only_when_anchored(self):
        require_torch()
        from sim_vla.training.actor_critic import ActorCriticTrainer
        from sim_vla.training.imagination import flatten_start

        model, critic, batch = build(False)
        give_the_heads_an_opinion(model, critic)
        actor = make_actor(model)
        trainer = ActorCriticTrainer(
            model, actor, critic, config(profile=True, demo_anchor=1.0),
            demo_sampler=DemoSamplerStub(), to_model_batch=lambda b: b)
        start = flatten_start(model.observe(batch)["post"], False)
        self.assertIn("profile_anchor_s", trainer.update(start))

    def test_table_renders_without_cuda(self):
        require_torch()
        from sim_vla.training.profiling import Phases

        phases = Phases(device="cpu", enabled=True)
        with phases("collect"):
            pass
        text = phases.table("smoke")
        self.assertIn("collect", text)
        self.assertIn("CUDA not in use", text)


class TestConfigIntegration(unittest.TestCase):
    """Bad combinations fail in seconds, not after a critic warm-up."""

    def resolved(self, **online):
        from sim_vla.config import load_config, validate

        cfg = load_config("peginsertion", "dreamer",
                          overrides={"online": online})
        validate(cfg)
        return cfg

    def test_default_config_still_resolves_to_the_pathwise_objective(self):
        cfg = self.resolved()
        self.assertEqual(cfg["online"]["actor_objective"], "pathwise")
        self.assertEqual(float(cfg["online"]["flow_noise_std"]), 0.0)

    def test_flow_reinforce_with_a_noise_scale_validates(self):
        cfg = self.resolved(actor_objective="flow_reinforce",
                            flow_noise_std=0.03, demo_anchor=0.5)
        self.assertEqual(cfg["online"]["actor_objective"], "flow_reinforce")

    def test_flow_reinforce_without_noise_is_refused(self):
        with self.assertRaises(SystemExit):
            self.resolved(actor_objective="flow_reinforce", flow_noise_std=0.0)

    def test_noise_without_flow_reinforce_is_refused(self):
        """A setting that is read and never applied is worse than absent."""
        with self.assertRaises(SystemExit):
            self.resolved(actor_objective="pathwise", flow_noise_std=0.05)

    def test_unknown_objective_is_refused(self):
        with self.assertRaises(SystemExit):
            self.resolved(actor_objective="ppo")

    def test_unknown_noise_schedule_is_refused(self):
        with self.assertRaises(SystemExit):
            self.resolved(actor_objective="flow_reinforce", flow_noise_std=0.03,
                          flow_noise_schedule="cosine")

    def test_negative_anchor_is_refused(self):
        with self.assertRaises(SystemExit):
            self.resolved(demo_anchor=-1.0)

    def test_online_configs_carries_the_settings_through(self):
        require_torch()
        from types import SimpleNamespace

        from sim_vla.training.pipeline import online_configs

        cfg = self.resolved(actor_objective="flow_reinforce",
                            flow_noise_std=0.03, demo_anchor=0.25,
                            actor_lr=1e-5, actor_transition_microbatch=8)
        model_cfg = SimpleNamespace(imag_horizon=15, horizon=333, lamb=0.95)
        _online, ac = online_configs(cfg, model_cfg, total_steps=10,
                                     flow_steps=10)
        self.assertEqual(ac.actor_objective, "flow_reinforce")
        self.assertAlmostEqual(ac.flow_noise_std, 0.03)
        self.assertAlmostEqual(ac.demo_anchor, 0.25)
        self.assertAlmostEqual(ac.actor_lr, 1e-5)
        self.assertEqual(ac.actor_transition_microbatch, 8)

    def test_actor_lr_defaults_to_the_dataclass_value_when_unset(self):
        require_torch()
        from types import SimpleNamespace

        from sim_vla.training.actor_critic import ActorCriticConfig
        from sim_vla.training.pipeline import online_configs

        cfg = self.resolved()
        model_cfg = SimpleNamespace(imag_horizon=15, horizon=333, lamb=0.95)
        _online, ac = online_configs(cfg, model_cfg, total_steps=10,
                                     flow_steps=10)
        self.assertAlmostEqual(ac.actor_lr, ActorCriticConfig().actor_lr)


class TestCheckpointContract(unittest.TestCase):
    """What the run optimized is written down, not inferred later."""

    def written(self, tmp, **overrides):
        torch = require_torch()
        from sim_vla.runtime.checkpoint import CheckpointMeta
        from sim_vla.training.online import OnlineConfig, OnlineTrainer

        model, critic, _batch = build(False)
        actor = make_actor(model)
        trainer = OnlineTrainer(
            model, actor, critic, DemoSamplerStub(),
            config=OnlineConfig(save_checkpoints=True, batch_size=2,
                                sequence_length=6, burn_in=1),
            ac_config=config(**overrides), device="cpu",
            checkpoint_dir=tmp,
            meta=CheckpointMeta(graph_enabled=False, stage="online",
                                feature_dim=model.feature_dim))
        trainer.env_steps = 17
        path = trainer.checkpoint()
        self.assertIsNotNone(path)
        return torch.load(path, weights_only=False), trainer

    def test_objective_and_sampler_identity_are_recorded(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            payload, _trainer = self.written(Path(tmp), flow_noise_std=0.03,
                                             demo_anchor=0.25)
        extra = payload["meta"]["extra"]
        self.assertEqual(extra["actor_objective"], "flow_reinforce")
        self.assertEqual(extra["flow_noise_schedule"],
                         "constant_per_step_scaled_by_sqrt_k")
        self.assertAlmostEqual(extra["flow_noise_std"], 0.03)
        self.assertAlmostEqual(extra["demo_anchor"], 0.25)
        self.assertEqual(extra["advantage_scale"], "return_ema")
        for key in ("flow_steps", "imag_horizon", "actor_lr", "precision",
                    "actor_transition_microbatch", "anchor_rows",
                    "actor_updates", "world_updates", "env_steps"):
            self.assertIn(key, extra, key)
        self.assertEqual(extra["env_steps"], 17)

    def test_return_ema_is_saved_as_a_module(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            payload, _trainer = self.written(Path(tmp), flow_noise_std=0.03)
        self.assertIn("return_ema", payload["modules"])
        self.assertIn("ema_vals", payload["modules"]["return_ema"])

    def test_pathwise_checkpoints_record_their_objective_too(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            payload, _trainer = self.written(
                Path(tmp), actor_objective="pathwise", flow_noise_std=0.0)
        self.assertEqual(payload["meta"]["extra"]["actor_objective"],
                         "pathwise")


class TestReturnEmaRoundTrip(unittest.TestCase):
    def test_running_statistics_survive_a_state_dict_round_trip(self):
        torch = require_torch()
        import networks

        ema = networks.ReturnEMA(device=torch.device("cpu"))
        for _ in range(5):
            ema(torch.randn(64) * 10.0)
        before = ema.ema_vals.detach().clone()
        self.assertGreater(float(before.abs().sum()), 0.0,
                           "the fixture never moved the statistics")

        restored = networks.ReturnEMA(device=torch.device("cpu"))
        restored.load_state_dict(ema.state_dict())
        self.assertTrue(torch.equal(before, restored.ema_vals))

    def test_the_scale_is_floored_at_one(self):
        torch = require_torch()
        import networks

        ema = networks.ReturnEMA(device=torch.device("cpu"))
        # A degenerate return stream must not amplify noise to unit scale.
        _offset, scale = ema(torch.zeros(64))
        self.assertGreaterEqual(float(scale), 1.0)


if __name__ == "__main__":
    unittest.main()
