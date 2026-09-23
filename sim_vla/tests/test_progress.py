"""Stage 9: progress shaping is optional, graph-dependent, and kept apart."""

from __future__ import annotations

import unittest
from pathlib import Path

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
        from sim_vla.training.progress import (build_progress, predict,
                                               shaping_reward)

        _cfg, model_cfg = small_model_config(True)
        head = build_progress(model_cfg, 8, graph_enabled=True,
                              progress_enabled=True)
        feat = torch.randn(4, 3, 8)
        out = shaping_reward(head, feat, discount=0.99)
        # gamma * phi(s') - phi(s): one value per transition, not per state.
        self.assertEqual(out.shape[0], feat.shape[0] - 1)
        phi = predict(head, feat)
        self.assertTrue(torch.allclose(out, 0.99 * phi[1:] - phi[:-1], atol=1e-5))

    def test_a_terminal_transition_earns_no_successor_potential(self):
        """``cont`` belongs in the shaping, not only in the return.

        Without it a transition that ends the episode would still be credited
        with the potential of the state after it -- a state the agent never
        occupies -- and the shaping would stop being potential-based.
        """
        torch = require_torch()
        from sim_vla.training.progress import build_progress, shaping_reward

        _cfg, model_cfg = small_model_config(True)
        head = build_progress(model_cfg, 8, graph_enabled=True,
                              progress_enabled=True)
        feat = torch.randn(3, 2, 8)
        cont = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        out = shaping_reward(head, feat, 0.9, cont=cont)
        phi = head(feat).squeeze(-1)
        expected = 0.9 * cont * phi[1:] - phi[:-1]
        torch.testing.assert_close(out, expected)
        # The terminating transition is credited with -phi(s) alone.
        self.assertAlmostEqual(float(out[0, 1]), -float(phi[0, 1]), places=5)


REPO = Path(__file__).resolve().parents[2]


class TestProgressAvailability(unittest.TestCase):
    """What the arm needs, checked before anything expensive.

    An earlier version of this refused the arm unconditionally, on the belief
    that no task schedule existed and that the scorer needed decoder
    probabilities. Both were wrong -- the repository ships a schedule per task
    and the scorer reads observed labels -- so what is checked now is whether
    *this* run has the pieces.
    """

    def cfg(self, env_id="PickCube-v1", enabled=True, graph=True):
        return {"task": {"env_id": env_id},
                "model": {"graph": {"enabled": graph},
                          "progress": {"enabled": enabled, "beta": 0.05}}}

    def metadata(self, env_id="PickCube-v1"):
        return {"graph": {
            "whitelist_dir": str(
                REPO / "scenegraph" / "configs" / "subtask_whitelists" / env_id),
            "absolute_tokens": {"pad": 0, "near": 1, "above": 2}}}

    def test_the_shipped_tasks_have_everything_the_arm_needs(self):
        from sim_vla.training.progress import availability

        for env_id in ("PickCube-v1", "PlaceSphere-v1", "PegInsertionSide-v1",
                       "StackCube-v1"):
            with self.subTest(env_id=env_id):
                missing = availability(self.cfg(env_id),
                                       self.metadata(env_id))
                self.assertEqual(missing, [], f"{env_id}: {missing}")

    def test_a_task_with_no_schedule_is_refused(self):
        from sim_vla.training.progress import preflight

        with self.assertRaises(SystemExit) as caught:
            preflight(self.cfg("NoSuchTask-v9"), {})
        self.assertIn("schedule", str(caught.exception))

    def test_the_baseline_cannot_run_the_progress_arm(self):
        from sim_vla.training.progress import preflight

        with self.assertRaises(SystemExit) as caught:
            preflight(self.cfg(graph=False), self.metadata())
        self.assertIn("graph", str(caught.exception))

    def test_the_other_arms_pass_through(self):
        from sim_vla.training.progress import preflight

        preflight(self.cfg(enabled=False))
        preflight(self.cfg(enabled=False, graph=False))


class TestWarmupScaling(unittest.TestCase):
    """The warm-up has to fall inside the run, or the arm is not the arm."""

    def test_the_warmup_lands_inside_the_budget(self):
        from sim_vla.training.progress import ProgressConfig, beta_at, \
            warmup_for

        total = 200_000
        start, end = warmup_for(total)
        self.assertLess(start, end)
        self.assertLess(end, total)
        config = ProgressConfig(enabled=True, beta=0.05, warmup_start=start,
                                warmup_end=end)
        self.assertEqual(beta_at(config, 0), 0.0)
        self.assertEqual(beta_at(config, start), 0.0)
        self.assertAlmostEqual(beta_at(config, end), 0.05)
        self.assertAlmostEqual(beta_at(config, total), 0.05)
        # And it is genuinely on for most of the run.
        self.assertGreater(beta_at(config, total // 2), 0.0)

    def test_the_repository_defaults_would_never_fire_in_a_short_run(self):
        """Why the scaling exists, stated as a test rather than a comment."""
        from sim_vla.training.progress import ProgressConfig, beta_at

        absolute = ProgressConfig(enabled=True, beta=0.05,
                                  warmup_start=400_000, warmup_end=700_000)
        self.assertEqual(beta_at(absolute, 200_000), 0.0)

    def test_a_tiny_budget_still_produces_an_ordered_window(self):
        from sim_vla.training.progress import warmup_for

        for total in (0, 1, 10, 100):
            start, end = warmup_for(total)
            self.assertLess(start, end, f"total={total}")


class TestShapingReachesTheActor(unittest.TestCase):
    """The head is read by the actor objective, and its stream stays separate."""

    def rollout(self, beta, rewards=None, conts=None, execute=2):
        """The objective over a stubbed rollout with a head that says
        something. Only the rollout is stubbed; the shaping, the return and
        the reporting are the real ones."""
        torch = require_torch()
        from types import SimpleNamespace

        import sim_vla.training.actor_critic as module
        from sim_vla.training.actor_critic import ActorCriticConfig
        from sim_vla.training.progress import build_progress

        _cfg, model_cfg = small_model_config(True)
        head = build_progress(model_cfg, 2, graph_enabled=True,
                              progress_enabled=True)
        # A head that says something: its output layer is zero-initialised.
        with torch.no_grad():
            for child in head.modules():
                if isinstance(child, torch.nn.Linear) and not float(
                        child.weight.abs().sum()):
                    child.weight.copy_(
                        torch.randn(child.weight.shape) * 0.1)

        feat = torch.randn(execute + 1, 1, 2, requires_grad=True)
        heads = {
            "reward": (torch.zeros(execute + 1, 1) if rewards is None
                       else torch.as_tensor(rewards).reshape(execute + 1, 1)),
            "cont": (torch.ones(execute + 1, 1) if conts is None
                     else torch.as_tensor(conts).reshape(execute + 1, 1))}
        original = (module.imagine_chunk, module.imagined_rewards)
        module.imagine_chunk = lambda *a, **k: {
            "feat": feat, "action": torch.zeros(execute, 1, 2),
            "action_steps": [], "chunk": torch.zeros(1, execute, 2)}
        module.imagined_rewards = lambda _m, _f: heads
        self.addCleanup(lambda: setattr(module, "imagine_chunk", original[0]))
        self.addCleanup(lambda: setattr(module, "imagined_rewards",
                                        original[1]))

        class Critic:
            def parameters(self):
                return iter(())

            def value(self, f):
                return torch.zeros(f.shape[0], f.shape[1])

            def target_value(self, f, *, detach=False):
                return torch.zeros(f.shape[0])

        out = module.executed_chunk_objective(
            SimpleNamespace(parameters=lambda: iter(())), None, Critic(), None,
            ActorCriticConfig(execute=execute, discount=1.0,
                              progress_beta=beta),
            progress_head=head)
        return out, head, feat

    def test_a_progress_head_changes_the_return_and_is_logged_apart(self):
        torch = require_torch()
        out, _head, _feat = self.rollout(0.05)
        self.assertIsNotNone(out["shaping"])
        # The environment stream is reported unshaped.
        self.assertTrue(torch.allclose(out["reward"], torch.zeros(2, 1)))
        self.assertFalse(torch.allclose(out["shaped_reward"],
                                        torch.zeros(2, 1)),
                         "the shaping term never reached the reward")

    def test_actor_and_critic_share_one_shaped_return(self):
        """The return the actor maximises is the return the critic is fitted
        to, so the two cannot be optimising different rewards."""
        torch = require_torch()
        from sim_vla.training.imagination import chunk_return

        out, _head, _feat = self.rollout(0.25, rewards=[0.0, 1.0, 2.0])
        expected = chunk_return(out["shaped_reward"], out["cont"],
                                out["bootstrap"], 1.0)
        torch.testing.assert_close(out["returns"].detach(), expected)
        self.assertAlmostEqual(float(out["loss"]),
                               -float(out["returns"].mean()), places=6)

    def test_beta_zero_leaves_the_reward_untouched(self):
        torch = require_torch()
        out, _head, _feat = self.rollout(0.0, rewards=[1.0, 1.0, 1.0])
        # During warm-up beta is zero and the arm is exactly `graph`.
        self.assertIsNone(out["shaping"])
        self.assertTrue(torch.allclose(out["shaped_reward"], out["reward"]))

    def test_the_progress_head_is_frozen_during_the_actor_update(self):
        require_torch()
        out, head, _feat = self.rollout(0.05)
        out["loss"].backward()
        for name, parameter in head.named_parameters():
            self.assertIsNone(
                parameter.grad,
                f"progress head parameter {name} was trained by the actor "
                "update; it is fitted on observed targets, not on the return")

    def test_the_shaping_still_reaches_the_actor_through_the_feature(self):
        """Frozen parameters, live outputs: the shaping term has to remain a
        function of the imagined states."""
        torch = require_torch()
        out, _head, feat = self.rollout(0.05)
        grad = torch.autograd.grad(out["loss"], feat, allow_unused=True)[0]
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)


class FakePotential:
    """Stands in for SchedulePotential: a fixed ramp, valid where asked.

    ``poison`` puts NaN targets on every row the loader excludes -- burn-in
    and padding -- while still *claiming* them valid, so only the batch masks
    stand between those rows and the loss.
    """

    def __init__(self, valid=True, poison=False):
        self.valid = valid
        self.poison = poison

    def targets(self, batch):
        torch = require_torch()
        mask = batch["loss_mask"]
        steps = mask.shape[1]
        phi = torch.linspace(0.0, 1.0, steps).expand(mask.shape[0], steps)
        phi = phi.clone()
        if self.poison:
            excluded = ~mask.bool() | ~batch["valid"].bool()
            phi[excluded] = float("nan")
        return phi, torch.full(mask.shape, bool(self.valid))

    def describe(self):
        return {"env_id": "Fake-v0", "phases": 1}


def graph_progress_arm():
    """A small graph_progress world model, its configs and one batch."""
    from .common import fake_batch, obs_shapes
    from sim_vla.models.world_model import build_world_model

    cfg, model_cfg = small_model_config(True)
    cfg["model"]["progress"]["enabled"] = True
    batch = fake_batch(graph_enabled=True)
    model = build_world_model(model_cfg, obs_shapes(batch), 8,
                              graph_enabled=True)
    return cfg, model_cfg, model, batch


class TestProgressHead(unittest.TestCase):
    """The regular trainer's head and objective, not a second copy of them."""

    def head(self, feature_dim=8):
        from sim_vla.training.progress import build_progress

        _cfg, model_cfg = small_model_config(True)
        return model_cfg, build_progress(model_cfg, feature_dim,
                                         graph_enabled=True,
                                         progress_enabled=True)

    def constant(self, head, value):
        """Make ``head`` output ``value`` everywhere."""
        torch = require_torch()
        import math

        with torch.no_grad():
            head.last.weight.zero_()
            head.last.bias.fill_(math.log(value / (1.0 - value)))

    def test_the_head_is_networks_progress_head_from_its_config_block(self):
        torch = require_torch()
        import networks

        model_cfg, head = self.head(8)
        self.assertIs(type(head), networks.ProgressHead)
        block = model_cfg.progress.head
        linears = [m for m in head.mlp.layers
                   if isinstance(m, torch.nn.Linear)]
        self.assertEqual(len(linears), int(block.layers))
        self.assertEqual(linears[0].in_features, 8)
        self.assertEqual(linears[0].out_features, int(block.units))
        self.assertEqual(head.last.out_features, 1)
        # Key for key the regular trainer's, so its weights mean the same.
        self.assertEqual(sorted(head.state_dict()),
                         sorted(networks.ProgressHead(block, 8).state_dict()))

    def test_predictions_are_bounded_to_the_unit_interval(self):
        torch = require_torch()
        from sim_vla.training.progress import predict, shaping_reward

        _cfg, head = self.head(8)
        feat = torch.cat([torch.randn(5, 3, 8) * 100.0,
                          torch.full((1, 3, 8), 1e4),
                          torch.full((1, 3, 8), -1e4)])
        for logit in (60.0, -60.0, 0.0):
            with torch.no_grad():
                head.last.bias.fill_(logit)
            phi = predict(head, feat)
            self.assertEqual(tuple(phi.shape), (7, 3))
            self.assertEqual(phi.dtype, torch.float32)
            self.assertTrue(bool(torch.isfinite(phi).all()))
            self.assertTrue(bool(((phi >= 0.0) & (phi <= 1.0)).all()),
                            f"phi left [0, 1] at bias {logit}")
            # And so the shaping term is bounded too: |gamma phi' - phi| <= 1.
            self.assertLessEqual(
                float(shaping_reward(head, feat, 0.99).abs().max()), 1.0)

    def test_the_objective_is_dreamers_masked_huber(self):
        torch = require_torch()
        import torch.nn.functional as F

        from sim_vla.training.progress import (PROGRESS_HUBER_DELTA, predict,
                                               progress_loss)

        _cfg, head = self.head(8)
        torch.manual_seed(0)
        feat = torch.randn(2, 5, 8)
        target = torch.rand(2, 5)
        valid = torch.rand(2, 5) > 0.3
        valid[0, 0] = True
        loss, _metrics = progress_loss(head, feat, target, valid)
        # dreamer.py:_progress_model_loss, written out.
        phi = predict(head, feat)
        weight = valid.float()
        error = F.huber_loss(phi, target * weight, reduction="none",
                             delta=0.1)
        expected = (error * weight).sum() / weight.sum().clamp_min(1)
        self.assertEqual(PROGRESS_HUBER_DELTA, 0.1)
        torch.testing.assert_close(loss, expected)

    def test_a_large_error_costs_linearly_not_quadratically(self):
        """delta=0.1: an error of 0.6 costs 0.1 * (0.6 - 0.05) = 0.055."""
        torch = require_torch()
        from sim_vla.training.progress import progress_loss

        _cfg, head = self.head(8)
        self.constant(head, 0.2)
        loss, metrics = progress_loss(head, torch.randn(1, 4, 8),
                                      torch.full((1, 4), 0.8),
                                      torch.ones(1, 4, dtype=torch.bool))
        self.assertAlmostEqual(float(loss), 0.055, places=5)
        self.assertAlmostEqual(metrics["progress_head_mae"], 0.6, places=5)

    def test_invalid_rows_carry_no_loss_and_no_gradient(self):
        torch = require_torch()
        from sim_vla.training.progress import progress_loss

        _cfg, head = self.head(8)
        torch.manual_seed(1)
        feat = torch.randn(2, 6, 8, requires_grad=True)
        target = torch.rand(2, 6)
        mask = torch.ones(2, 6, dtype=torch.bool)
        mask[:, :2] = False
        mask[1, -1] = False
        poisoned = target.clone()
        poisoned[~mask] = float("nan")

        loss, metrics = progress_loss(head, feat, poisoned, mask)
        self.assertTrue(bool(torch.isfinite(loss)),
                        "a NaN target on an excluded row reached the loss")
        clean, _ = progress_loss(head, feat, target, mask)
        torch.testing.assert_close(loss, clean)
        self.assertAlmostEqual(metrics["progress_valid"],
                               float(mask.float().mean()))
        loss.backward()
        self.assertTrue(bool((feat.grad[~mask] == 0).all()),
                        "an excluded row sent gradient")
        self.assertGreater(float(feat.grad[mask].abs().sum()), 0.0)

    def test_a_mismatched_target_shape_is_refused(self):
        torch = require_torch()
        from sim_vla.training.progress import progress_loss

        _cfg, head = self.head(8)
        with self.assertRaises(ValueError):
            progress_loss(head, torch.randn(2, 5, 8), torch.rand(2, 5, 1),
                          torch.ones(2, 5, dtype=torch.bool))

    def test_burn_in_padding_and_unscorable_rows_are_masked(self):
        torch = require_torch()
        from sim_vla.training.progress import progress_mask

        # Row 1 is padded for its last two steps; both rows burn in for two.
        valid = torch.ones(2, 6, dtype=torch.bool)
        valid[1, 4:] = False
        loss_mask = valid.clone()
        loss_mask[:, :2] = False
        phi_valid = torch.ones(2, 6, dtype=torch.bool)
        phi_valid[0, 3] = False
        mask = progress_mask({"loss_mask": loss_mask, "valid": valid},
                             phi_valid)
        expected = torch.tensor([[0, 0, 1, 0, 1, 1],
                                 [0, 0, 1, 1, 0, 0]], dtype=torch.bool)
        self.assertTrue(torch.equal(mask, expected))
        # Padding stays out even if a loader ever left loss_mask set on it.
        loose = loss_mask | ~valid
        self.assertTrue(torch.equal(
            progress_mask({"loss_mask": loose, "valid": valid}, phi_valid),
            expected))

    def test_the_weight_is_loss_scales_progress_model_not_beta(self):
        from sim_vla.training.progress import progress_weight

        cfg, model_cfg = small_model_config(True)
        # The configured value, today.
        self.assertEqual(progress_weight(model_cfg), 1.0)
        cfg["model"]["progress"]["beta"] = 0.37
        model_cfg.progress.beta = 0.37
        model_cfg.loss_scales.progress_model = 2.5
        self.assertEqual(progress_weight(model_cfg), 2.5)
        model_cfg.loss_scales.progress_model = 0.0
        with self.assertRaises(SystemExit):
            progress_weight(model_cfg)


class TestJointProgressPretraining(unittest.TestCase):
    """Stage 1A trains the head jointly with the world model, as dreamer.py
    does. This replaces the earlier guarantee that the head left the world
    model untouched: now it is meant to reach it."""

    def arm(self):
        return graph_progress_arm()

    def test_progress_loss_alone_trains_the_head_and_what_feeds_it(self):
        torch = require_torch()
        from sim_vla.training.pretrain_world_model import build_progress_head
        from sim_vla.training.progress import joint_progress_loss

        cfg, model_cfg, model, batch = self.arm()
        head = build_progress_head(cfg, model_cfg, model, device="cpu")
        torch.manual_seed(3)
        _total, _losses, aux = model.loss(batch)
        term, _ = joint_progress_loss(head, FakePotential(), aux["feat"],
                                      batch)
        named = ([(f"head.{n}", p) for n, p in head.named_parameters()]
                 + [(f"world.{n}", p) for n, p in model.named_parameters()])
        grads = torch.autograd.grad(term, [p for _, p in named],
                                    allow_unused=True)
        reached = {name for (name, _), grad in zip(named, grads)
                   if grad is not None and float(grad.abs().sum()) > 0.0}

        self.assertTrue(any(n.startswith("head.last.") for n in reached))
        self.assertTrue(any(n.startswith("head.mlp.") for n in reached))
        # Through the attached feature, into everything that produces it.
        for prefix in ("world.encoder.", "world.rssm.",
                       "world.graph_encoder."):
            self.assertTrue(any(n.startswith(prefix) for n in reached),
                            f"no progress gradient reached {prefix}")
        # And nothing the feature does not pass through.
        for prefix in ("world.decoder.", "world.reward_head.",
                       "world.cont_head.", "world.graph_decoder."):
            self.assertFalse(any(n.startswith(prefix) for n in reached),
                             f"progress gradient reached {prefix}")

    def test_one_step_adds_progress_model_times_the_loss(self):
        import copy

        torch = require_torch()
        from sim_vla.training.pretrain_world_model import (
            build_progress_head, stage_parameters, train_step)
        from sim_vla.training.progress import joint_progress_loss

        cfg, model_cfg, model, batch = self.arm()
        head = build_progress_head(cfg, model_cfg, model, device="cpu")
        reference, reference_head = copy.deepcopy(model), copy.deepcopy(head)
        weight = 2.5

        torch.manual_seed(5)
        world_total, _losses, aux = reference.loss(batch)
        term, _ = joint_progress_loss(reference_head, FakePotential(),
                                      aux["feat"], batch)

        optimizer = torch.optim.AdamW(stage_parameters(model, head), lr=1e-3)
        head_before = [p.detach().clone() for p in head.parameters()]
        torch.manual_seed(5)
        total, last = train_step(model, optimizer, batch,
                                 progress=(head, FakePotential(), weight))
        torch.testing.assert_close(total.detach(),
                                   (world_total + weight * term).detach())
        self.assertAlmostEqual(last["progress_model"], float(term), places=5)
        for key in ("progress_valid", "progress_target_mean",
                    "progress_target_std", "progress_head_mae"):
            self.assertIn(key, last)
        self.assertTrue(any(not torch.equal(old, new.detach())
                            for old, new in zip(head_before,
                                                head.parameters())),
                        "the one optimizer step did not move the head")

    def test_the_progress_term_changes_the_world_model_update(self):
        """Same seed, same batch, with and without the term. Plain SGD, so an
        update is proportional to its gradient and the difference is exactly
        what the progress loss sent into the world model."""
        import copy

        torch = require_torch()
        from sim_vla.training.pretrain_world_model import (
            build_progress_head, stage_parameters, train_step)

        cfg, model_cfg, model, batch = self.arm()
        plain = copy.deepcopy(model)
        head = build_progress_head(cfg, model_cfg, model, device="cpu")
        torch.manual_seed(5)
        train_step(model, torch.optim.SGD(stage_parameters(model, head),
                                          lr=1e-2),
                   batch, progress=(head, FakePotential(), 1.0))
        torch.manual_seed(5)
        _total, plain_last = train_step(
            plain, torch.optim.SGD(plain.parameters(), lr=1e-2), batch)
        self.assertNotIn("progress_model", plain_last)
        moved = {name.split(".")[0]
                 for (name, joint), (_, alone) in zip(model.named_parameters(),
                                                      plain.named_parameters())
                 if not torch.equal(joint, alone)}
        for component in ("encoder", "rssm", "graph_encoder"):
            self.assertIn(component, moved,
                          f"the progress term never reached {component}")

    def test_excluded_rows_do_not_reach_the_step(self):
        """Burn-in and padding rows whose targets are NaN, claimed valid by
        the potential: the batch masks alone keep them out."""
        torch = require_torch()
        from sim_vla.training.pretrain_world_model import (
            build_progress_head, stage_parameters, train_step)

        cfg, model_cfg, model, batch = self.arm()
        batch["loss_mask"][:, :2] = False          # burn-in
        batch["valid"][1, -2:] = False             # padding
        batch["loss_mask"][1, -2:] = False
        head = build_progress_head(cfg, model_cfg, model, device="cpu")
        total, last = train_step(
            model, torch.optim.AdamW(stage_parameters(model, head), lr=1e-3),
            batch, progress=(head, FakePotential(poison=True), 1.0))
        self.assertTrue(bool(torch.isfinite(total)))
        self.assertAlmostEqual(last["progress_valid"],
                               float(batch["loss_mask"].float().mean()))
        for name, parameter in list(model.named_parameters()) + list(
                head.named_parameters()):
            self.assertTrue(bool(torch.isfinite(parameter).all()),
                            f"{name} went non-finite")

    def test_an_unscorable_batch_leaves_the_update_unchanged(self):
        import copy

        torch = require_torch()
        from sim_vla.training.pretrain_world_model import (
            build_progress_head, stage_parameters, train_step)

        cfg, model_cfg, model, batch = self.arm()
        plain = copy.deepcopy(model)
        head = build_progress_head(cfg, model_cfg, model, device="cpu")
        head_before = [p.detach().clone() for p in head.parameters()]
        torch.manual_seed(5)
        _total, last = train_step(
            model, torch.optim.SGD(stage_parameters(model, head), lr=1e-2),
            batch, progress=(head, FakePotential(valid=False), 1.0))
        torch.manual_seed(5)
        train_step(plain, torch.optim.SGD(plain.parameters(), lr=1e-2), batch)
        self.assertEqual(last["progress_valid"], 0.0)
        self.assertEqual(last["progress_model"], 0.0)
        for (name, joint), (_, alone) in zip(model.named_parameters(),
                                             plain.named_parameters()):
            self.assertTrue(torch.equal(joint, alone), name)
        for old, new in zip(head_before, head.parameters()):
            self.assertTrue(torch.equal(old, new.detach()))

    def test_building_the_head_leaves_the_cpu_random_stream_alone(self):
        torch = require_torch()
        from sim_vla.training.pretrain_world_model import build_progress_head

        cfg, model_cfg, model, _batch = self.arm()
        torch.manual_seed(11)
        expected = torch.rand(4)
        torch.manual_seed(11)
        first = build_progress_head(cfg, model_cfg, model, device="cpu")
        self.assertTrue(torch.equal(torch.rand(4), expected))
        # And the head's own weights depend on the run's seed alone.
        second = build_progress_head(cfg, model_cfg, model, device="cpu")
        for a, b in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_building_the_head_leaves_the_cuda_random_stream_alone(self):
        """torch.manual_seed reseeds CUDA too; forking only the CPU stream
        left CUDA at the run's seed after the head was built."""
        from .common import require_cuda

        torch = require_cuda()
        from sim_vla.training.pretrain_world_model import build_progress_head

        cfg, model_cfg, model, _batch = self.arm()
        torch.manual_seed(11)
        expected = torch.rand(4, device="cuda")
        torch.manual_seed(11)
        build_progress_head(cfg, model_cfg, model, device="cpu")
        self.assertTrue(torch.equal(torch.rand(4, device="cuda"), expected))

    def test_other_arms_build_no_head_and_keep_their_optimizer(self):
        require_torch()
        from sim_vla.training.pretrain_world_model import (
            build_progress_head, stage_parameters)

        cfg, model_cfg, model, _batch = self.arm()
        cfg["model"]["progress"]["enabled"] = False
        self.assertIsNone(
            build_progress_head(cfg, model_cfg, model, device="cpu"))
        self.assertEqual([id(p) for p in stage_parameters(model, None)],
                         [id(p) for p in model.parameters()])


class TestStage2FitStaysDetached(unittest.TestCase):
    """Stage 2 keeps its own policy: the same objective, head only."""

    def test_fit_progress_moves_the_head_and_not_the_world_model(self):
        torch = require_torch()
        from sim_vla.training.pretrain_world_model import build_progress_head
        from sim_vla.training.progress import PROGRESS_LR, fit_progress

        cfg, model_cfg, model, batch = graph_progress_arm()
        head = build_progress_head(cfg, model_cfg, model, device="cpu")
        _total, _losses, aux = model.loss(batch)
        head_before = [p.detach().clone() for p in head.parameters()]
        metrics = fit_progress(head,
                               torch.optim.AdamW(head.parameters(),
                                                 lr=PROGRESS_LR),
                               FakePotential(), aux["feat"], batch)
        self.assertIn("progress_loss", metrics)
        self.assertIn("progress_head_mae", metrics)
        self.assertTrue(any(not torch.equal(old, new.detach())
                            for old, new in zip(head_before,
                                                head.parameters())))
        for name, parameter in model.named_parameters():
            self.assertIsNone(parameter.grad,
                              f"Stage 2's head fit sent gradient into {name}")

    def test_an_unscorable_batch_takes_no_head_step(self):
        torch = require_torch()
        from sim_vla.training.pretrain_world_model import build_progress_head
        from sim_vla.training.progress import fit_progress

        cfg, model_cfg, model, batch = graph_progress_arm()
        head = build_progress_head(cfg, model_cfg, model, device="cpu")
        _total, _losses, aux = model.loss(batch)
        before = [p.detach().clone() for p in head.parameters()]
        metrics = fit_progress(head,
                               torch.optim.AdamW(head.parameters(), lr=1.0),
                               FakePotential(valid=False), aux["feat"], batch)
        self.assertEqual(metrics["progress_valid"], 0.0)
        for old, new in zip(before, head.parameters()):
            self.assertTrue(torch.equal(old, new.detach()))


if __name__ == "__main__":
    unittest.main()
