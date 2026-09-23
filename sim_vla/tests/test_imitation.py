"""Stage 5: chunking, masking, and a real imitation update.

The chunk tests use a nonconstant action sequence -- ``a_i = i`` -- so that an
off-by-one in the gather is a wrong *number*, not a wrong shape. A fixture of
zeros passes every alignment bug this stage exists to catch.
"""

from __future__ import annotations

import unittest

import numpy as np

from .common import (DummyExpert, fake_batch, obs_shapes, require_torch,
                     small_model_config)


def masks(steps, *, available=None, eligible=None, batch=1):
    torch = require_torch()
    ones = torch.ones(batch, steps, dtype=torch.bool)
    return (ones.clone() if available is None else available,
            ones.clone() if eligible is None else eligible)


class TestChunking(unittest.TestCase):
    def test_a_chunk_is_the_actions_at_and_after_its_row(self):
        torch = require_torch()
        from sim_vla.training.train_imitation import chunk_targets

        targets = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1)
        available, eligible = masks(5)
        gathered, mask, _ = chunk_targets(targets, available, eligible, 3)
        self.assertEqual(tuple(gathered.shape), (1, 5, 3, 1))
        # Row 1 is supervised on a_1, a_2, a_3 -- not on a_0.
        self.assertEqual(gathered[0, 1].reshape(-1).tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(gathered[0, 0].reshape(-1).tolist(), [0.0, 1.0, 2.0])

    def test_chunks_are_masked_past_the_end_of_the_window(self):
        torch = require_torch()
        from sim_vla.training.train_imitation import chunk_targets

        targets = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1)
        available, eligible = masks(5)
        _gathered, mask, _ = chunk_targets(targets, available, eligible, 3)
        self.assertEqual(mask[0, 3].tolist(), [True, True, False])
        self.assertEqual(mask[0, 4].tolist(), [True, False, False])

    def test_availability_stops_at_the_first_missing_action(self):
        """A row marked available after a gap must not resurrect the chunk."""
        torch = require_torch()
        from sim_vla.training.train_imitation import chunk_targets

        targets = torch.zeros(1, 6, 2)
        available = torch.tensor([[True, True, False, True, True, True]])
        _gathered, mask, _ = chunk_targets(targets, available,
                                           torch.ones(1, 6, dtype=torch.bool), 4)
        # From row 0: a_0, a_1 exist, a_2 does not -- and a_3 cannot count.
        self.assertEqual(mask[0, 0].tolist(), [True, True, False, False])

    def test_eligibility_excludes_burn_in_rows(self):
        """A burn-in row's chunk reaches scored rows; it still must not
        contribute, because the state conditioning it was never established."""
        torch = require_torch()
        from sim_vla.training.train_imitation import chunk_targets

        targets = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)
        available = torch.ones(1, 6, dtype=torch.bool)
        scored = torch.tensor([[False, False, True, True, True, True]])
        _gathered, _mask, eligible = chunk_targets(
            targets, available, scored & available, 3)
        self.assertEqual(eligible[0].tolist(),
                         [False, False, True, True, True, True])

    def test_eligibility_excludes_rows_with_no_target_of_their_own(self):
        torch = require_torch()
        from sim_vla.training.train_imitation import chunk_targets

        targets = torch.zeros(1, 4, 1)
        available = torch.tensor([[True, True, True, False]])
        scored = torch.ones(1, 4, dtype=torch.bool)
        _g, _m, eligible = chunk_targets(targets, available,
                                         scored & available, 2)
        self.assertFalse(bool(eligible[0, 3]))


def tiny_actor(feature_dim, action_dim=8, chunk=2, token=32):
    torch = require_torch()
    from sim_vla.models.latent_adapter import LatentAdapter

    expert = DummyExpert(token, action_dim)

    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = LatentAdapter(feature_dim, token, hidden=32,
                                         layers=1)
            self.expert = expert
            self.expert_linear = expert.linear
            self.chunk_size = chunk
            self.action_dim = action_dim
            self.flow_steps = 3

        def condition(self, features, instruction=None):
            return {"state_token": self.adapter(features),
                    "instruction": instruction}

        def velocity_fn(self):
            return self.expert

    return Actor()


class TestImitationUpdate(unittest.TestCase):
    """A real update: the policy learns, the world model does not move."""

    def build(self):
        torch = require_torch()
        from sim_vla.models.world_model import build_world_model
        from sim_vla.training.train_imitation import (ImitationConfig,
                                                      ImitationTrainer)

        torch.manual_seed(0)
        _cfg, model_cfg = small_model_config(False)
        batch = fake_batch(graph_enabled=False)
        model = build_world_model(model_cfg, obs_shapes(batch), 8,
                                  graph_enabled=False)
        actor = tiny_actor(model.feature_dim)
        config = ImitationConfig(chunk_size=actor.chunk_size, flow_steps=3,
                                 batch_size=2)
        trainer = ImitationTrainer(model, actor, None, config, device="cpu")
        return trainer, model, actor, batch

    def test_the_update_moves_the_adapter_and_the_expert(self):
        torch = require_torch()
        trainer, _model, actor, batch = self.build()
        before = {name: p.detach().clone()
                  for name, p in actor.named_parameters() if p.requires_grad}
        loss, metrics = trainer.loss(batch)
        loss.backward()
        moved = [name for name, p in actor.named_parameters()
                 if p.requires_grad and p.grad is not None
                 and float(p.grad.abs().sum()) > 0]
        self.assertTrue(any("adapter" in n for n in moved),
                        f"no adapter gradient; got {moved}")
        self.assertTrue(any("expert" in n for n in moved),
                        f"no action-expert gradient; got {moved}")
        self.assertGreater(metrics["eligible_rows"], 0)

    def test_the_world_model_does_not_move(self):
        torch = require_torch()
        trainer, model, _actor, batch = self.build()
        before = [p.detach().clone() for p in model.parameters()]
        loss, _metrics = trainer.loss(batch)
        loss.backward()
        trainer.optimizer.step()
        for name, parameter in model.named_parameters():
            self.assertIsNone(parameter.grad,
                              f"{name} accumulated a gradient while frozen")
        for old, new in zip(before, model.parameters()):
            self.assertTrue(torch.equal(old, new.detach()),
                            "the imitation update changed the world model")

    def test_the_loss_reads_the_target_not_the_previous_action(self):
        """Swapping only ``action_target`` must change the loss."""
        torch = require_torch()
        trainer, _model, _actor, batch = self.build()
        torch.manual_seed(1)
        first, _ = trainer.loss(batch)
        altered = dict(batch)
        altered["action_target"] = batch["action_target"] + 5.0
        torch.manual_seed(1)
        second, _ = trainer.loss(altered)
        self.assertNotAlmostEqual(float(first), float(second), places=4,
                                  msg="action_target does not reach the loss")

    def test_the_loss_ignores_the_previous_action_column(self):
        torch = require_torch()
        trainer, _model, _actor, batch = self.build()
        torch.manual_seed(1)
        first, _ = trainer.loss(batch)
        altered = dict(batch)
        # "action" feeds the posterior, which is under no_grad and is recomputed
        # -- it changes the features, so the loss may move. What must not happen
        # is the target being taken from here.
        altered["action_target"] = batch["action"].clone()
        torch.manual_seed(1)
        second, _ = trainer.loss(altered)
        self.assertNotAlmostEqual(float(first), float(second), places=4)

    def test_a_batch_with_no_eligible_rows_is_reported_not_crashed(self):
        torch = require_torch()
        trainer, _model, _actor, batch = self.build()
        empty = dict(batch)
        empty["loss_mask"] = torch.zeros_like(batch["loss_mask"])
        loss, metrics = trainer.loss(empty)
        self.assertEqual(metrics["skipped"], 1.0)
        self.assertEqual(float(loss), 0.0)
        self.assertFalse(loss.requires_grad)

    def test_a_mismatched_chunk_size_is_refused(self):
        require_torch()
        from sim_vla.models.world_model import build_world_model
        from sim_vla.training.train_imitation import (ImitationConfig,
                                                      ImitationTrainer)

        _cfg, model_cfg = small_model_config(False)
        batch = fake_batch(graph_enabled=False)
        model = build_world_model(model_cfg, obs_shapes(batch), 8,
                                  graph_enabled=False)
        actor = tiny_actor(model.feature_dim, chunk=2)
        with self.assertRaises(ValueError) as caught:
            ImitationTrainer(model, actor, None,
                             ImitationConfig(chunk_size=7), device="cpu")
        self.assertIn("chunk", str(caught.exception))

    def updates(self, trainer, batch, count):
        # update() takes a sampler's numpy batch; the fixture is already
        # tensors, so the conversion is the one thing stepped over.
        trainer.to_torch = lambda given: given
        return [trainer.update(batch) for _ in range(count)]

    def test_each_update_runs_at_its_scheduled_rate(self):
        from sim_vla.training.lr_schedule import LRSchedule

        trainer, _model, _actor, batch = self.build()
        trainer.schedule = LRSchedule(peak=1e-3, total=6, warmup=2,
                                      final=1e-5)
        rates = [row["lr"] for row in self.updates(trainer, batch, 6)]
        for step, rate in enumerate(rates):
            self.assertAlmostEqual(rate, trainer.schedule.at(step))
        self.assertAlmostEqual(rates[0], 5e-4)
        self.assertAlmostEqual(rates[-1], 1e-5)
        self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], 1e-5)

    def test_the_config_builds_the_schedule_and_its_default_is_constant(self):
        require_torch()
        from sim_vla.models.world_model import build_world_model
        from sim_vla.training.train_imitation import (ImitationConfig,
                                                      ImitationTrainer)

        _cfg, model_cfg = small_model_config(False)
        batch = fake_batch(graph_enabled=False)
        model = build_world_model(model_cfg, obs_shapes(batch), 8,
                                  graph_enabled=False)
        actor = tiny_actor(model.feature_dim)
        scheduled = ImitationTrainer(
            model, actor, None,
            ImitationConfig(chunk_size=actor.chunk_size, lr=1e-4, steps=30,
                            warmup_steps=5, final_lr=2.5e-6), device="cpu")
        self.assertEqual((scheduled.schedule.total, scheduled.schedule.warmup,
                          scheduled.schedule.final), (30, 5, 2.5e-6))
        trainer, _model, _actor, batch = self.build()
        rates = [row["lr"] for row in self.updates(trainer, batch, 3)]
        self.assertEqual(rates, [1e-4] * 3)


class TestLRSchedule(unittest.TestCase):
    """Linear warmup, then cosine decay to the final rate at the last step."""

    def schedule(self, **kwargs):
        from sim_vla.training.lr_schedule import LRSchedule

        return LRSchedule(**({"peak": 1e-4, "total": 30_000} | kwargs))

    def test_no_warmup_and_no_final_rate_is_the_old_constant_rate(self):
        constant = self.schedule()
        self.assertEqual({constant.at(step) for step in (0, 1, 15_000, 29_999)},
                         {1e-4})

    def test_warmup_ramps_linearly_to_the_peak(self):
        warm = self.schedule(warmup=1000, final=2.5e-6)
        self.assertAlmostEqual(warm.at(0), 1e-7)
        self.assertAlmostEqual(warm.at(499), 5e-5)
        self.assertAlmostEqual(warm.at(999), 1e-4)
        self.assertAlmostEqual(warm.at(1000), 1e-4)

    def test_cosine_decay_ends_on_the_final_rate_at_the_last_step(self):
        decay = self.schedule(warmup=1000, final=2.5e-6)
        self.assertAlmostEqual(decay.at(29_999), 2.5e-6)
        # Halfway through the decay, halfway between peak and final.
        middle = 1000 + (30_000 - 1000 - 1) // 2
        self.assertAlmostEqual(decay.at(middle), (1e-4 + 2.5e-6) / 2,
                               delta=1e-8)
        rates = [decay.at(step) for step in range(1000, 30_000, 97)]
        self.assertTrue(all(a >= b for a, b in zip(rates, rates[1:])),
                        "the rate rose during the decay")
        # Past the last step it holds rather than coming back up.
        self.assertAlmostEqual(decay.at(40_000), 2.5e-6)

    def test_apply_sets_every_parameter_group(self):
        from types import SimpleNamespace

        optimizer = SimpleNamespace(param_groups=[{"lr": 1.0}, {"lr": 1.0}])
        rate = self.schedule(warmup=10).apply(optimizer, 4)
        self.assertAlmostEqual(rate, 5e-5)
        self.assertEqual([g["lr"] for g in optimizer.param_groups],
                         [rate, rate])

    def test_nonsense_is_refused(self):
        for kwargs in ({"final": 2e-4}, {"final": 0.0}, {"warmup": -1},
                       {"peak": 0.0}, {"final": float("nan")}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                self.schedule(**kwargs)


class TestActionWidth(unittest.TestCase):
    """Stage 1B's entry point, against the canonical window keys."""

    def test_width_comes_from_action_target(self):
        require_torch()
        from sim_vla.training.train_imitation import action_width

        class Sampler:
            def batch(self, size):
                return {"action": np.zeros((size, 4, 7), np.float32),
                        "action_target": np.zeros((size, 4, 7), np.float32)}

        self.assertEqual(action_width({"task": {}}, Sampler()), 7)

    def test_a_sampler_without_the_key_is_reported_clearly(self):
        require_torch()
        from sim_vla.training.train_imitation import action_width

        class Sampler:
            def batch(self, size):
                return {"action": np.zeros((size, 4, 7), np.float32)}

        with self.assertRaises(KeyError):
            action_width({"task": {}}, Sampler())

    def test_a_configured_width_that_disagrees_is_refused(self):
        require_torch()
        from sim_vla.training.train_imitation import action_width

        class Sampler:
            def batch(self, size):
                return {"action_target": np.zeros((size, 4, 7), np.float32)}

        with self.assertRaises(ValueError):
            action_width({"task": {"action_dim": 8}}, Sampler())


class TestFlowLoss(unittest.TestCase):
    def test_masked_steps_do_not_contribute(self):
        torch = require_torch()
        from sim_vla.models.flow_sampler import flow_matching_loss

        expert = DummyExpert(token_dim=8, action_dim=4)
        cond = {"state_token": torch.zeros(2, 1, 8), "instruction": None}
        actions = torch.randn(2, 5, 4)
        mask = torch.ones(2, 5, dtype=torch.bool)
        mask[:, 3:] = False
        torch.manual_seed(0)
        a, _ = flow_matching_loss(expert, actions, cond, mask=mask)
        wild = actions.clone()
        wild[:, 3:] = 1e4
        torch.manual_seed(0)
        b, _ = flow_matching_loss(expert, wild, cond, mask=mask)
        self.assertAlmostEqual(float(a), float(b), places=3)

    def test_overfits_a_single_constant_chunk(self):
        """A small, real optimisation: the loss must actually go down."""
        torch = require_torch()
        from sim_vla.models.flow_sampler import flow_matching_loss
        from sim_vla.models.latent_adapter import LatentAdapter

        torch.manual_seed(0)
        adapter = LatentAdapter(feature_dim=16, token_dim=16, hidden=64)
        expert = DummyExpert(token_dim=16, action_dim=4)
        params = list(adapter.parameters()) + list(expert.parameters())
        opt = torch.optim.Adam(params, lr=3e-3)
        feat = torch.randn(8, 16)
        target = torch.ones(8, 4, 4) * 0.5

        first = None
        for _ in range(300):
            cond = {"state_token": adapter(feat), "instruction": None}
            loss, _ = flow_matching_loss(expert, target, cond)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            first = float(loss) if first is None else first
        self.assertLess(float(loss), first * 0.7,
                        f"flow loss did not fall: {first} -> {float(loss)}")

    def test_the_chunk_dimension_survives_into_the_loss(self):
        """Collapsing H would turn chunk imitation into single-action
        imitation, and the loss would still descend."""
        torch = require_torch()
        from sim_vla.models.flow_sampler import flow_matching_loss

        expert = DummyExpert(token_dim=8, action_dim=4)
        cond = {"state_token": torch.zeros(2, 1, 8), "instruction": None}
        actions = torch.randn(2, 5, 4)
        torch.manual_seed(0)
        whole, _ = flow_matching_loss(expert, actions, cond)
        only_first = actions.clone()
        only_first[:, 1:] = actions[:, :1]
        torch.manual_seed(0)
        collapsed, _ = flow_matching_loss(expert, only_first, cond)
        self.assertNotAlmostEqual(float(whole), float(collapsed), places=4)


if __name__ == "__main__":
    unittest.main()
