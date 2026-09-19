"""SOLD with SmolVLA: the slot adapter, imitation, and the actor hook.

The action expert is a stub; the adapter, the slot encoder, the dynamics, the
reward head and the critic are upstream's. Three properties carry most of the
weight here:

* the conditioning is causal and its **context is the same length** in
  imitation, in imagination and at inference
* the imagined return's gradient reaches the adapter and the expert
* the entropy term is gone and nothing replaced it; REINFORCE is refused
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from . import common as C
from .test_sold_native import build_parts, sold_source, tiny_world


def build(*, context=2, chunk=3, bounded=True, action_dim=7, image=16,
          episode=24):
    from ..action_space import converter_for
    from ..sold.policy import LatentSlotPolicy

    source = sold_source(bounded_actions=bounded, action=action_dim,
                         image=image, steps=episode)
    parts = build_parts(image=image, action_dim=action_dim,
                        max_episode_steps=episode)
    actor = C.stub_slot_actor(num_slots=int(parts["num_slots"]),
                              slot_dim=int(parts["slot_dim"]),
                              action_dim=action_dim, context=context,
                              chunk_size=chunk)
    converter = converter_for(source.metadata, action_dim=action_dim,
                              mode="identity", device="cpu")
    policy = LatentSlotPolicy(actor, converter, lr=1e-3, max_batch=32,
                              min_num_context=context)
    return source, parts, actor, converter, policy


class Adapter(unittest.TestCase):
    def test_it_reads_the_backends_own_slots(self):
        torch = C.require_torch()
        _source, parts, actor, _conv, _policy = build()
        self.assertEqual(actor.adapter.num_slots, int(parts["num_slots"]))
        self.assertEqual(actor.adapter.slot_dim, int(parts["slot_dim"]))
        slots = torch.randn(2, 5, int(parts["num_slots"]), int(parts["slot_dim"]))
        token = actor.adapter(slots)
        self.assertEqual(tuple(token.shape)[:2], (2, 1))

    def test_the_token_ignores_frames_before_the_context(self):
        """The bounded window is real: older frames cannot change the token."""
        torch = C.require_torch()
        _source, parts, actor, _conv, _policy = build(context=2)
        slots = torch.randn(2, 6, int(parts["num_slots"]), int(parts["slot_dim"]))
        first = actor.adapter(slots)
        older = slots.clone()
        older[:, :-2] = torch.randn_like(older[:, :-2])
        self.assertTrue(torch.allclose(first, actor.adapter(older), atol=1e-6))

    def test_the_token_ignores_nothing_inside_the_context(self):
        torch = C.require_torch()
        _source, parts, actor, _conv, _policy = build(context=2)
        slots = torch.randn(2, 6, int(parts["num_slots"]), int(parts["slot_dim"]))
        first = actor.adapter(slots)
        inside = slots.clone()
        inside[:, -2] = torch.randn_like(inside[:, -2])
        self.assertFalse(torch.allclose(first, actor.adapter(inside)))

    def test_batched_and_incremental_conditioning_agree(self):
        """A training row and the same moment online produce one token.

        This is the property the bounded context exists for. SOLD's ALiBi mask
        biases a key by its index *within the window*, so a growing history and
        a fixed training window would put different weight on the same slots.
        """
        torch = C.require_torch()
        _source, parts, actor, _conv, policy = build(context=2)
        slots = torch.randn(1, 6, int(parts["num_slots"]), int(parts["slot_dim"]))

        for step in range(1, 6):
            # Training: row `step` of a long window, via select_windows.
            index = torch.tensor([step])
            window = actor.adapter.select_windows(slots, index)
            batched = actor.adapter(window)
            # Inference: the history as `select_action` would have grown it.
            incremental = actor.adapter(slots[:, :step + 1])
            self.assertTrue(torch.allclose(batched, incremental, atol=1e-6),
                            f"row {step} conditions differently in training "
                            "and at inference")

    def test_a_row_without_a_full_history_is_refused(self):
        torch = C.require_torch()
        _source, parts, actor, _conv, _policy = build(context=3)
        slots = torch.randn(2, 6, int(parts["num_slots"]), int(parts["slot_dim"]))
        with self.assertRaises(ValueError) as raised:
            actor.adapter.select_windows(slots, torch.tensor([0]))
        self.assertIn("burn_in", str(raised.exception))

    def test_a_context_longer_than_the_imagination_start_is_refused(self):
        """The first imagined step would otherwise see a shorter window."""
        C.require_torch()

        from ..sold.policy import LatentSlotPolicy

        _source, _parts, actor, converter, _policy = build(context=3)
        with self.assertRaises(ValueError) as raised:
            LatentSlotPolicy(actor, converter, min_num_context=2)
        self.assertIn("num_context", str(raised.exception))


class Imitation(unittest.TestCase):
    def trainer(self, *, steps=5, context=2, chunk=3, sequence_length=8):
        from ..sold import stages

        source, parts, actor, converter, _policy = build(context=context,
                                                         chunk=chunk)
        config = stages.ImitationConfig(steps=steps, batch_size=4, lr=3e-3,
                                        grad_clip=1.0, log_every=0,
                                        sequence_length=sequence_length)
        return source, parts, actor, stages.ImitationTrainer(
            parts, actor, source, converter, config=config, device="cpu")

    def test_the_world_model_is_frozen(self):
        torch = C.require_torch()
        _source, parts, _actor, trainer = self.trainer()
        self.assertGreater(trainer.frozen, 0)
        watched = ("autoencoder", "dynamics", "reward", "critic", "actor")
        before = {key: {n: p.detach().clone()
                        for n, p in parts[key].named_parameters()}
                  for key in watched}
        for _ in range(3):
            trainer.update()
        for key in watched:
            for name, parameter in parts[key].named_parameters():
                self.assertTrue(torch.allclose(before[key][name], parameter),
                                f"{key}.{name} moved during imitation")
                self.assertIsNone(parameter.grad,
                                  f"{key}.{name} accumulated a gradient")

    def test_burn_in_covers_the_context(self):
        C.require_torch()
        _source, _parts, _actor, trainer = self.trainer(context=2)
        self.assertEqual(trainer.burn_in, 1)
        _s, _p, _a, trainer3 = self.trainer(context=3)
        self.assertEqual(trainer3.burn_in, 2)

    def test_the_adapter_and_the_expert_get_gradients(self):
        torch = C.require_torch()
        _source, _parts, actor, trainer = self.trainer()
        raw = trainer.windows.raw_batch(4)
        loss, metrics = trainer.loss(raw)
        self.assertFalse(metrics["skipped"])
        self.assertGreater(metrics["eligible_rows"], 0)
        loss.backward()
        adapter = [p.grad for p in actor.adapter.parameters()
                   if p.grad is not None]
        self.assertTrue(adapter, "no gradient reached the slot adapter")
        self.assertGreater(float(sum(g.abs().sum() for g in adapter)), 0.0)
        expert = actor.actor.expert.linear.weight.grad
        self.assertIsNotNone(expert)
        self.assertGreater(float(expert.abs().sum()), 0.0)

    def test_a_short_run_overfits(self):
        torch = C.require_torch()
        _source, _parts, _actor, trainer = self.trainer(steps=150,
                                                        sequence_length=6)
        torch.manual_seed(0)
        first = [trainer.update()["loss"] for _ in range(20)]
        for _ in range(150):
            trainer.update()
        last = [trainer.update()["loss"] for _ in range(20)]
        self.assertLess(float(np.mean(last)), 0.75 * float(np.mean(first)),
                        f"flow loss did not descend: {np.mean(first):.4f} -> "
                        f"{np.mean(last):.4f}")


class ImaginedObjective(unittest.TestCase):
    """The gradient path the actor update depends on, without Lightning.

    ``imagine_ahead`` is a method on a ``LightningModule``, so the test that
    runs upstream's own loop is in ``test_sold_online`` and skips where
    Lightning is absent. This one rebuilds the same *path* -- sample, step the
    real dynamics, score with the real reward and critic heads, take a
    lambda-style return, differentiate -- from the real modules, so the
    property is checked on any machine.
    """

    def rollout(self, horizon=3):
        torch = C.require_torch()
        source, parts, actor, converter, policy = build(context=2)
        slots = torch.randn(2, 2, int(parts["num_slots"]),
                            int(parts["slot_dim"]))
        actions = torch.zeros(2, 1, int(source.action_dim))
        context = slots
        for _ in range(horizon):
            action = policy.sample(context, start=context.shape[1] - 1,
                                   grad=True)
            actions = torch.cat([actions, action.unsqueeze(1)], dim=1)
            predicted = parts["dynamics"].predict_slots(
                context, actions, steps=1, num_context=context.shape[1])
            context = torch.cat([context, predicted], dim=1)
        rewards = parts["reward"](context, start=2).mean.squeeze(-1)
        values = parts["critic"](context, start=2).mean.squeeze(-1)
        return parts, actor, policy, rewards, values

    def test_the_return_reaches_the_adapter_and_the_expert(self):
        torch = C.require_torch()
        parts, actor, policy, rewards, values = self.rollout()
        objective = -(rewards + 0.96 * values).mean()
        policy.zero_grad()
        objective.backward()
        for name, parameter in actor.adapter.named_parameters():
            self.assertIsNotNone(parameter.grad,
                                 f"no gradient reached the adapter at {name}")
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0,
                               f"the gradient into {name} is exactly zero")
        expert = actor.actor.expert.linear.weight
        self.assertIsNotNone(expert.grad)
        self.assertGreater(float(expert.grad.abs().sum()), 0.0)

    def test_the_world_model_does_not_move_during_an_actor_update(self):
        torch = C.require_torch()
        parts, actor, policy, rewards, values = self.rollout()
        watched = ("dynamics", "reward", "critic")
        before = {key: {n: p.detach().clone()
                        for n, p in parts[key].named_parameters()}
                  for key in watched}
        policy.zero_grad()
        (-(rewards + 0.96 * values).mean()).backward()
        policy.step(10.0)
        for key in watched:
            for name, parameter in parts[key].named_parameters():
                self.assertTrue(torch.allclose(before[key][name], parameter),
                                f"{key}.{name} moved during an actor update")

    def test_the_action_carries_gradient_at_every_imagined_step(self):
        torch = C.require_torch()

        from ..latent_actor import assert_gradient_reaches

        _source, parts, actor, _conv, policy = build(context=2)
        slots = torch.randn(2, 2, int(parts["num_slots"]),
                            int(parts["slot_dim"]))
        action = policy.sample(slots, grad=True)
        assert_gradient_reaches(action, actor.trainable_parameters())
        detached = policy.sample(slots, grad=False)
        self.assertFalse(detached.requires_grad)

    def test_the_action_is_bounded_and_in_native_units(self):
        torch = C.require_torch()
        _source, parts, _actor, converter, policy = build(context=2)
        slots = torch.randn(4, 2, int(parts["num_slots"]),
                            int(parts["slot_dim"])) * 5.0
        action = policy.sample(slots)
        self.assertLessEqual(float(action.abs().max()), 1.0 + 1e-6)
        self.assertEqual(tuple(action.shape), (4, int(converter.action_dim)))

    def test_only_the_last_row_may_be_conditioned_on(self):
        torch = C.require_torch()
        _source, parts, _actor, _conv, policy = build(context=2)
        slots = torch.randn(2, 4, int(parts["num_slots"]),
                            int(parts["slot_dim"]))
        with self.assertRaises(ValueError):
            policy.sample(slots, start=1)


class ActorObjective(unittest.TestCase):
    """The intentional differences, stated as tests."""

    def test_no_log_prob_and_no_entropy(self):
        C.require_torch()

        from ..latent_actor import ActorCapabilityError

        _source, _parts, actor, _conv, _policy = build()
        for call in (actor.log_prob, actor.entropy, actor.rsample):
            with self.assertRaises(ActorCapabilityError):
                call()

    def test_compute_actor_loss_drops_the_entropy_term(self):
        torch = C.require_torch()

        from ..sold import stages
        from ..vendor import SOLD

        try:
            SOLDModule = SOLD.get("train_sold", "SOLDModule")
        except Exception as exc:                           # noqa: BLE001
            raise unittest.SkipTest(
                f"train_sold needs Lightning, gym and Hydra: {exc}")

        horizon = 4
        holder = stages.Holder(
            actor_gradients="dynamics", actor_entropy_loss_weight=0.0003,
            return_moments=SOLD.get("modeling.distributions", "Moments")(),
            discounts=torch.ones(1, horizon))
        returns = torch.randn(2, horizon)
        values = torch.randn(2, horizon)
        entropies = torch.randn(2, horizon)
        log_probs = torch.randn(2, horizon)

        flow = SOLDModule.compute_actor_loss(holder, returns, values, None, None)
        gaussian = SOLDModule.compute_actor_loss(holder, returns, values,
                                                 log_probs, entropies)
        self.assertAlmostEqual(float(flow["actor_loss"]),
                               float(flow["actor_return_loss"]), places=6)
        self.assertEqual(float(flow["actor_entropy_loss"]), 0.0)
        # The return term is the same one; only the entropy differs.
        self.assertNotAlmostEqual(float(gaussian["actor_loss"]),
                                  float(flow["actor_loss"]), places=6)

    def test_reinforce_is_refused(self):
        torch = C.require_torch()

        from ..vendor import SOLD

        try:
            SOLDModule = SOLD.get("train_sold", "SOLDModule")
        except Exception as exc:                           # noqa: BLE001
            raise unittest.SkipTest(
                f"train_sold needs Lightning, gym and Hydra: {exc}")

        _source, _parts, actor, converter, policy = build()

        class Stub:
            actor_gradients = "reinforce"
            attach_policy = SOLDModule.attach_policy

        with self.assertRaises(ValueError) as raised:
            Stub().attach_policy(policy)
        self.assertIn("log-probability", str(raised.exception))


class Checkpoints(unittest.TestCase):
    def meta(self, cfg_like, parts, source, converter, actor, *,
             stage="imitation", smolvla=True):
        from ..sold import stages

        return stages.stage_meta(cfg_like, stage=stage, parts=parts,
                                 source=source, smolvla=smolvla,
                                 converter=converter, actor=actor,
                                 max_episode_steps=24)

    def cfg_like(self):
        return {"world_model": tiny_world(), "smolvla": {"execute": 1}}

    def test_round_trip(self):
        torch = C.require_torch()

        from ..checkpoint import load
        from ..sold import stages

        source, parts, actor, converter, _policy = build()
        cfg = self.cfg_like()
        with tempfile.TemporaryDirectory() as tmp:
            path = stages.save_stage(
                Path(tmp) / "stage2.pt",
                self.meta(cfg, parts, source, converter, actor),
                parts, actor=actor)
            self.assertTrue(path.with_suffix(".json").exists())

            s2, p2, a2, c2, _ = build()
            load(path, self.meta(cfg, p2, s2, c2, a2),
                 {"autoencoder": p2["autoencoder"], "dynamics": p2["dynamics"],
                  "reward": p2["reward"], "adapter": a2.adapter,
                  "smolvla_actor": a2}, strict=False)
            for (name, left), (_n, right) in zip(
                    actor.adapter.named_parameters(),
                    a2.adapter.named_parameters()):
                self.assertTrue(torch.allclose(left, right), name)

    def test_a_different_adapter_context_is_refused(self):
        torch = C.require_torch()

        from ..checkpoint import load
        from ..sold import stages

        source, parts, actor, converter, _policy = build(context=2)
        cfg = self.cfg_like()
        with tempfile.TemporaryDirectory() as tmp:
            path = stages.save_stage(
                Path(tmp) / "stage2.pt",
                self.meta(cfg, parts, source, converter, actor),
                parts, actor=actor)
            s2, p2, a2, c2, _ = build(context=1)
            with self.assertRaises(SystemExit) as raised:
                load(path, self.meta(cfg, p2, s2, c2, a2),
                     {"autoencoder": p2["autoencoder"]}, strict=False)
            self.assertIn("architecture", str(raised.exception))

    def test_a_native_checkpoint_is_refused_for_a_smolvla_run(self):
        torch = C.require_torch()

        from ..checkpoint import load
        from ..sold import stages

        source, parts, actor, converter, _policy = build()
        cfg = self.cfg_like()
        with tempfile.TemporaryDirectory() as tmp:
            path = stages.save_stage(
                Path(tmp) / "native.pt",
                self.meta(cfg, parts, source, converter, None,
                          stage="world_model", smolvla=False),
                parts)
            with self.assertRaises(SystemExit):
                load(path, self.meta(cfg, parts, source, converter, None,
                                     stage="world_model", smolvla=True),
                     {"autoencoder": parts["autoencoder"]}, strict=False)


if __name__ == "__main__":
    unittest.main()
