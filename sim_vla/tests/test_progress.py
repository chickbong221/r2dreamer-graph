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
        from sim_vla.training.progress import build_progress, shaping_reward

        _cfg, model_cfg = small_model_config(True)
        head = build_progress(model_cfg, 8, graph_enabled=True,
                              progress_enabled=True)
        feat = torch.randn(4, 3, 8)
        out = shaping_reward(head, feat, discount=0.99)
        # gamma * phi(s') - phi(s): one value per transition, not per state.
        self.assertEqual(out.shape[0], feat.shape[0] - 1)
        phi = head.potential(feat)
        self.assertTrue(torch.allclose(out, 0.99 * phi[1:] - phi[:-1], atol=1e-5))

    def test_reward_streams_stay_separate(self):
        """The environment reward and the shaping term are added, not merged.

        Behavioural: the same rollout with and without a shaping stream must
        differ by exactly ``beta`` times that stream, and the reported
        environment reward must be the unshaped one.
        """
        torch = require_torch()
        from types import SimpleNamespace

        import sim_vla.training.actor_critic as module
        from sim_vla.training.actor_critic import ActorCriticConfig

        feat = torch.zeros(3, 1, 2)
        heads = {"reward": torch.tensor([[0.0], [1.0], [1.0]]),
                 "cont": torch.ones(3, 1)}
        original = (module.imagine, module.imagined_rewards)
        module.imagine = lambda *a, **k: {"feat": feat,
                                          "action": torch.zeros(2, 1, 2),
                                          "action_steps": []}
        module.imagined_rewards = lambda _m, _f: heads
        self.addCleanup(lambda: setattr(module, "imagine", original[0]))
        self.addCleanup(lambda: setattr(module, "imagined_rewards",
                                        original[1]))

        class Critic:
            def parameters(self):
                return iter(())

            def value(self, f):
                return torch.zeros(f.shape[0], f.shape[1])

            def target_value(self, f, *, detach=False):
                return torch.zeros(1, f.shape[1])

        world = SimpleNamespace(parameters=lambda: iter(()))
        plain = module.actor_loss(world, None, Critic(), None,
                                  ActorCriticConfig(horizon=2, discount=1.0,
                                                    lam=1.0))
        shaped = module.actor_loss(
            world, None, Critic(), None,
            ActorCriticConfig(horizon=2, discount=1.0, lam=1.0,
                              progress_beta=0.5),
            progress_reward=torch.tensor([[2.0], [4.0]]))
        # The environment reward reported is the unshaped one either way.
        self.assertEqual([float(v) for v in plain["reward"].reshape(-1)],
                         [1.0, 1.0])
        # And the return differs by beta * the shaping stream.
        self.assertAlmostEqual(float(shaped["returns"][1, 0])
                               - float(plain["returns"][1, 0]),
                               0.5 * 4.0, places=5)


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

        for env_id in ("PickCube-v1", "PlaceSphere-v1", "PegInsertionSide-v1"):
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

    def test_a_progress_head_changes_the_return_and_is_logged_apart(self):
        torch = require_torch()
        from types import SimpleNamespace

        import sim_vla.training.actor_critic as module
        from sim_vla.training.actor_critic import ActorCriticConfig
        from sim_vla.training.progress import build_progress

        _cfg, model_cfg = small_model_config(True)
        head = build_progress(model_cfg, 2, graph_enabled=True,
                              progress_enabled=True)
        # A head that says something: the value head is zero-initialised.
        with torch.no_grad():
            for child in head.modules():
                if isinstance(child, torch.nn.Linear) and not float(
                        child.weight.abs().sum()):
                    child.weight.copy_(
                        torch.randn(child.weight.shape) * 0.1)

        feat = torch.randn(3, 1, 2)
        heads = {"reward": torch.zeros(3, 1), "cont": torch.ones(3, 1)}
        original = (module.imagine, module.imagined_rewards)
        module.imagine = lambda *a, **k: {"feat": feat,
                                          "action": torch.zeros(2, 1, 2),
                                          "action_steps": []}
        module.imagined_rewards = lambda _m, _f: heads
        self.addCleanup(lambda: setattr(module, "imagine", original[0]))
        self.addCleanup(lambda: setattr(module, "imagined_rewards",
                                        original[1]))

        class Critic:
            def parameters(self):
                return iter(())

            def value(self, f):
                return torch.zeros(f.shape[0], f.shape[1])

            def target_value(self, f, *, detach=False):
                return torch.zeros(1, f.shape[1])

        world = SimpleNamespace(parameters=lambda: iter(()))
        config = ActorCriticConfig(horizon=2, discount=1.0, lam=1.0,
                                   progress_beta=0.05)
        out = module.actor_loss(world, None, Critic(), None, config,
                                progress_head=head)
        self.assertIsNotNone(out["shaping"])
        # The environment stream is reported unshaped.
        self.assertTrue(torch.allclose(out["reward"], torch.zeros(2, 1)))
        self.assertFalse(torch.allclose(out["shaped_reward"],
                                        torch.zeros(2, 1)),
                         "the shaping term never reached the reward")

    def test_beta_zero_leaves_the_reward_untouched(self):
        torch = require_torch()
        from types import SimpleNamespace

        import sim_vla.training.actor_critic as module
        from sim_vla.training.actor_critic import ActorCriticConfig
        from sim_vla.training.progress import build_progress

        _cfg, model_cfg = small_model_config(True)
        head = build_progress(model_cfg, 2, graph_enabled=True,
                              progress_enabled=True)
        feat = torch.randn(3, 1, 2)
        heads = {"reward": torch.ones(3, 1), "cont": torch.ones(3, 1)}
        original = (module.imagine, module.imagined_rewards)
        module.imagine = lambda *a, **k: {"feat": feat,
                                          "action": torch.zeros(2, 1, 2),
                                          "action_steps": []}
        module.imagined_rewards = lambda _m, _f: heads
        self.addCleanup(lambda: setattr(module, "imagine", original[0]))
        self.addCleanup(lambda: setattr(module, "imagined_rewards",
                                        original[1]))

        class Critic:
            def parameters(self):
                return iter(())

            def value(self, f):
                return torch.zeros(f.shape[0], f.shape[1])

            def target_value(self, f, *, detach=False):
                return torch.zeros(1, f.shape[1])

        out = module.actor_loss(
            SimpleNamespace(parameters=lambda: iter(())), None, Critic(), None,
            ActorCriticConfig(horizon=2, discount=1.0, lam=1.0,
                              progress_beta=0.0),
            progress_head=head)
        # During warm-up beta is zero and the arm is exactly `graph`.
        self.assertTrue(torch.allclose(out["shaped_reward"], out["reward"]))

    def test_the_progress_head_is_frozen_during_the_actor_update(self):
        torch = require_torch()
        from types import SimpleNamespace

        import sim_vla.training.actor_critic as module
        from sim_vla.training.actor_critic import ActorCriticConfig
        from sim_vla.training.progress import build_progress

        _cfg, model_cfg = small_model_config(True)
        head = build_progress(model_cfg, 2, graph_enabled=True,
                              progress_enabled=True)
        feat = torch.randn(3, 1, 2, requires_grad=True)
        heads = {"reward": torch.zeros(3, 1), "cont": torch.ones(3, 1)}
        original = (module.imagine, module.imagined_rewards)
        module.imagine = lambda *a, **k: {"feat": feat,
                                          "action": torch.zeros(2, 1, 2),
                                          "action_steps": []}
        module.imagined_rewards = lambda _m, _f: heads
        self.addCleanup(lambda: setattr(module, "imagine", original[0]))
        self.addCleanup(lambda: setattr(module, "imagined_rewards",
                                        original[1]))

        class Critic:
            def parameters(self):
                return iter(())

            def value(self, f):
                return torch.zeros(f.shape[0], f.shape[1])

            def target_value(self, f, *, detach=False):
                return torch.zeros(1, f.shape[1])

        out = module.actor_loss(
            SimpleNamespace(parameters=lambda: iter(())), None, Critic(), None,
            ActorCriticConfig(horizon=2, discount=1.0, lam=1.0,
                              progress_beta=0.05),
            progress_head=head)
        out["loss"].backward()
        for name, parameter in head.named_parameters():
            self.assertIsNone(
                parameter.grad,
                f"progress head parameter {name} was trained by the actor "
                "update; it is fitted on observed targets, not on the return")


class FakePotential:
    """Stands in for SchedulePotential: a fixed ramp, valid where asked."""

    def __init__(self, valid=True):
        self.valid = valid

    def targets(self, batch):
        torch = require_torch()
        mask = batch["loss_mask"]
        steps = mask.shape[1]
        phi = torch.linspace(0.0, 1.0, steps).expand(mask.shape[0], steps)
        return phi.clone(), torch.full(mask.shape, bool(self.valid))


class TestProgressPretraining(unittest.TestCase):
    """Stage 1A fits the head on demonstrations, and only the head."""

    def arm(self):
        from .common import fake_batch, obs_shapes
        from sim_vla.models.world_model import build_world_model

        cfg, model_cfg = small_model_config(True)
        cfg["model"]["progress"]["enabled"] = True
        batch = fake_batch(graph_enabled=True)
        model = build_world_model(model_cfg, obs_shapes(batch), 8,
                                  graph_enabled=True)
        return cfg, model_cfg, model, batch

    def test_the_head_learns_and_the_world_model_does_not_notice(self):
        """Same seed, same batch, with and without the head: the world model
        comes out identical, so the graph_progress arm's world model is the
        graph arm's."""
        import copy

        torch = require_torch()
        from sim_vla.training.pretrain_world_model import (
            build_progress_head, train_step)
        from sim_vla.training.progress import PROGRESS_LR

        cfg, model_cfg, model, batch = self.arm()
        plain = copy.deepcopy(model)
        head = build_progress_head(cfg, model_cfg, model, device="cpu")
        before = [p.detach().clone() for p in head.parameters()]
        head_opt = torch.optim.AdamW(head.parameters(), lr=PROGRESS_LR)

        torch.manual_seed(5)
        _total, last = train_step(
            model, torch.optim.AdamW(model.parameters(), lr=1e-3), batch,
            progress=(head, head_opt, FakePotential()))
        torch.manual_seed(5)
        _total, plain_last = train_step(
            plain, torch.optim.AdamW(plain.parameters(), lr=1e-3), batch)

        self.assertIn("progress_loss", last)
        self.assertNotIn("progress_loss", plain_last)
        self.assertTrue(any(not torch.equal(old, new.detach())
                            for old, new in zip(before, head.parameters())),
                        "the progress head did not train")
        for (name, trained), (_, alone) in zip(model.named_parameters(),
                                               plain.named_parameters()):
            self.assertTrue(torch.equal(trained, alone),
                            f"fitting the head moved the world model: {name}")

    def test_an_unscorable_batch_takes_no_head_step(self):
        torch = require_torch()
        from sim_vla.training.pretrain_world_model import (
            build_progress_head, train_step)

        cfg, model_cfg, model, batch = self.arm()
        head = build_progress_head(cfg, model_cfg, model, device="cpu")
        before = [p.detach().clone() for p in head.parameters()]
        _total, last = train_step(
            model, torch.optim.AdamW(model.parameters(), lr=1e-3), batch,
            progress=(head, torch.optim.AdamW(head.parameters(), lr=1.0),
                      FakePotential(valid=False)))
        self.assertEqual(last["progress_valid"], 0.0)
        for old, new in zip(before, head.parameters()):
            self.assertTrue(torch.equal(old, new.detach()))

    def test_building_the_head_leaves_the_random_stream_alone(self):
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

    def test_other_arms_fit_no_head(self):
        require_torch()
        from sim_vla.training.pretrain_world_model import build_progress_head

        cfg, model_cfg, model, _batch = self.arm()
        cfg["model"]["progress"]["enabled"] = False
        self.assertIsNone(
            build_progress_head(cfg, model_cfg, model, device="cpu"))


if __name__ == "__main__":
    unittest.main()
