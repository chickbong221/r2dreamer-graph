"""Upstream's ``SOLDModule``, with the flow actor attached.

Everything here builds the real ``SOLDModule`` and drives the real
``imagine_ahead`` and ``select_action``. It therefore needs Lightning, gym and
Hydra -- the same dependencies upstream's own entry point needs -- and skips
with that named when they are absent. The properties it checks are the ones
that cannot be checked from the pieces alone: that the imagined rollout runs
with a flow actor in it, that the actor loss that comes out has no entropy
term, that the actor step moves the adapter and nothing else, and that the
episode's slot history resets.
"""

from __future__ import annotations

import unittest

import numpy as np

from . import common as C
from .test_sold_native import tiny_world


def require_sold_module():
    from ..vendor import SOLD

    try:
        return SOLD.get("train_sold", "SOLDModule")
    except Exception as exc:                               # noqa: BLE001
        raise unittest.SkipTest(
            f"train_sold needs Lightning, gym and Hydra: {exc}")


class FakeEnv:
    """The four things SOLD reads off an environment, and a steppable body."""

    def __init__(self, *, action_dim=7, image=16, max_episode_steps=8):
        from ..sold.env import BoxSpace

        self.action_space = BoxSpace(low=np.full(action_dim, -1.0, np.float32),
                                     high=np.full(action_dim, 1.0, np.float32))
        self.max_episode_steps = int(max_episode_steps)
        self.image = int(image)
        self._steps = 0
        self._rng = np.random.default_rng(0)

    def _obs(self):
        import torch

        return torch.as_tensor(
            self._rng.integers(0, 256, (3, self.image, self.image)).astype(
                np.uint8))

    def reset(self):
        self._steps = 0
        return self._obs()

    def step(self, action):
        self._steps += 1
        done = self._steps >= self.max_episode_steps
        return self._obs(), 0.5, done, {"success": False}


def build_module(*, context=2, chunk=3, action_dim=7, image=16,
                 max_episode_steps=8):
    """The real SOLDModule at a size that runs on CPU, plus a flow policy."""
    torch = C.require_torch()
    SOLDModule = require_sold_module()

    from functools import partial

    from ..action_space import converter_for
    from ..sold import model as build
    from ..sold.policy import LatentSlotPolicy
    from ..vendor import SOLD

    world = tiny_world()
    env = FakeEnv(action_dim=action_dim, image=image,
                  max_episode_steps=max_episode_steps)
    autoencoder = build.build_autoencoder(world["autoencoder_spec"],
                                          image_size=(image, image),
                                          action_dim=action_dim)

    with SOLD.active():
        from modeling.sold import prediction
        from modeling.sold.dynamics import make_ocvp_seq_dynamics_model

        def strip(block):
            return {k: v for k, v in dict(block).items()
                    if not k.startswith("_") and v != "???"}

        def head(block):
            name = str(block["_target_"]).rsplit(".", 1)[-1]
            return partial(getattr(prediction, name), **strip(block))

        module = SOLDModule(
            autoencoder=autoencoder,
            dynamics_predictor=partial(make_ocvp_seq_dynamics_model,
                                       **strip(world["dynamics_predictor"])),
            actor=head(world["actor"]), critic=head(world["critic"]),
            reward_predictor=head(world["reward_predictor"]),
            env=env, max_steps=100, num_seed=0, update_freq=1, num_updates=1,
            eval_freq=1000, num_eval_episodes=1, batch_size=2,
            buffer_capacity=1000, save_replay_buffer=False,
            dynamics_learning_rate=1e-4, dynamics_grad_clip=3.0,
            actor_learning_rate=3e-5, actor_grad_clip=10.0,
            critic_learning_rate=3e-5, critic_grad_clip=10.0,
            reward_learning_rate=1e-4, reward_grad_clip=10.0,
            finetune_autoencoder=False, autoencoder_learning_rate=1e-4,
            autoencoder_grad_clip=0.05,
            num_context=world["num_context"],
            imagination_horizon=world["imagination_horizon"],
            start_imagination_from_every=False,
            actor_entropy_loss_weight=3e-4, actor_gradients="dynamics",
            return_lambda=0.95, discount_factor=0.96, critic_ema_decay=0.98)

    actor = C.stub_slot_actor(num_slots=autoencoder.num_slots,
                              slot_dim=autoencoder.slot_dim,
                              action_dim=action_dim, context=context,
                              chunk_size=chunk)
    converter = converter_for(None, action_dim=action_dim, mode="identity")
    policy = LatentSlotPolicy(actor, converter, lr=1e-3, max_batch=32,
                              min_num_context=int(world["num_context"]))
    return module, env, actor, policy


class NativeBehaviour(unittest.TestCase):
    def test_no_policy_attached_by_default(self):
        module, _env, _actor, _policy = build_module()
        self.assertIsNone(module.latent_policy)

    def test_imagination_runs_with_the_gaussian_actor(self):
        torch = C.require_torch()
        module, _env, _actor, _policy = build_module()
        slots = torch.randn(2, module.max_num_context + module.imagination_horizon,
                            module.autoencoder.num_slots,
                            module.autoencoder.slot_dim)
        actions = torch.zeros(2, slots.shape[1], 7)
        out = module.imagine_ahead(slots, actions)
        returns, values, dist, log_probs, entropies = out
        self.assertIsNotNone(log_probs)
        self.assertIsNotNone(entropies)
        loss = module.compute_actor_loss(returns, values, log_probs, entropies)
        self.assertNotEqual(float(loss["actor_entropy_loss"]), 0.0)


class WithSmolVLA(unittest.TestCase):
    def test_imagination_runs_with_the_flow_actor(self):
        torch = C.require_torch()
        module, _env, actor, policy = build_module()
        module.attach_policy(policy)
        slots = torch.randn(2, module.max_num_context + module.imagination_horizon,
                            module.autoencoder.num_slots,
                            module.autoencoder.slot_dim)
        actions = torch.zeros(2, slots.shape[1], 7)
        returns, values, dist, log_probs, entropies = module.imagine_ahead(
            slots, actions)
        self.assertIsNone(log_probs, "a flow policy reported a log-probability")
        self.assertIsNone(entropies, "a flow policy reported an entropy")
        self.assertEqual(tuple(returns.shape)[0], 2)
        self.assertEqual(policy.usage()["calls"]["imagine"],
                         module.imagination_horizon)

    def test_the_actor_loss_has_no_entropy_term_and_moves_the_adapter(self):
        torch = C.require_torch()
        module, _env, actor, policy = build_module()
        module.attach_policy(policy)
        slots = torch.randn(2, module.max_num_context + module.imagination_horizon,
                            module.autoencoder.num_slots,
                            module.autoencoder.slot_dim)
        actions = torch.zeros(2, slots.shape[1], 7)
        returns, values, _dist, log_probs, entropies = module.imagine_ahead(
            slots, actions)
        out = module.compute_actor_loss(returns, values, log_probs, entropies)
        self.assertEqual(float(out["actor_entropy_loss"]), 0.0)
        self.assertAlmostEqual(float(out["actor_loss"]),
                               float(out["actor_return_loss"]), places=6)

        before = {n: p.detach().clone()
                  for n, p in actor.adapter.named_parameters()}
        world_before = {n: p.detach().clone()
                        for n, p in module.dynamics_predictor.named_parameters()}
        policy.zero_grad()
        out["actor_loss"].backward()
        policy.step(module.actor_grad_clip)
        moved = [n for n, p in actor.adapter.named_parameters()
                 if not torch.allclose(before[n], p)]
        self.assertTrue(moved, "the imagined return did not move the adapter")
        for name, parameter in module.dynamics_predictor.named_parameters():
            self.assertTrue(torch.allclose(world_before[name], parameter),
                            f"dynamics.{name} moved during an actor update")

    def test_the_gaussian_actor_is_not_sampled_after_handoff(self):
        torch = C.require_torch()
        module, _env, _actor, policy = build_module()
        module.attach_policy(policy)
        calls = []
        original = module.actor.forward

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        module.actor.forward = spy                         # type: ignore[assignment]
        slots = torch.randn(2, module.max_num_context + module.imagination_horizon,
                            module.autoencoder.num_slots,
                            module.autoencoder.slot_dim)
        module.imagine_ahead(slots, torch.zeros(2, slots.shape[1], 7))
        self.assertEqual(calls, [],
                         "the Gaussian actor was still sampled after handoff")

    def test_reinforce_is_refused(self):
        module, _env, _actor, policy = build_module()
        module.actor_gradients = "reinforce"
        with self.assertRaises(ValueError) as raised:
            module.attach_policy(policy)
        self.assertIn("log-probability", str(raised.exception))
        self.assertIsNone(module.latent_policy)


class Inference(unittest.TestCase):
    def test_select_action_uses_the_policy_and_resets_per_episode(self):
        torch = C.require_torch()
        module, env, _actor, policy = build_module(max_episode_steps=5)
        module.attach_policy(policy)

        obs = env.reset()
        module.last_action[:] = 0.0
        first = module.select_action(obs, is_first=True, mode="train")
        self.assertEqual(tuple(first.shape), (7,))
        self.assertEqual(module._slot_history.shape[1], 1)
        for _ in range(3):
            obs, _r, _d, _i = env.step(first)
            module.last_action[:] = first
            first = module.select_action(obs, is_first=False, mode="train")
        self.assertEqual(module._slot_history.shape[1], 4)

        # A new episode starts from nothing, not from the last one's state.
        obs = env.reset()
        module.select_action(obs, is_first=True, mode="train")
        self.assertEqual(module._slot_history.shape[1], 1)
        self.assertGreater(policy.usage()["calls"]["act"], 0)

    def test_the_action_is_bounded(self):
        torch = C.require_torch()
        module, env, _actor, policy = build_module()
        module.attach_policy(policy)
        obs = env.reset()
        module.last_action[:] = 0.0
        action = module.select_action(obs, is_first=True, mode="eval")
        self.assertLessEqual(float(action.abs().max()), 1.0 + 1e-6)


class CheckpointHooks(unittest.TestCase):
    def test_the_policy_optimizer_and_settings_are_carried(self):
        module, _env, _actor, policy = build_module()
        module.attach_policy(policy)
        payload = {}
        module.on_save_checkpoint(payload)
        self.assertIn("latent_policy_optimizer", payload)
        self.assertIn("latent_policy_meta", payload)
        self.assertEqual(payload["latent_policy_meta"]["context"],
                         policy.context)

    def test_a_mismatched_policy_is_refused_on_load(self):
        module, _env, _actor, policy = build_module(context=2)
        module.attach_policy(policy)
        payload = {}
        module.on_save_checkpoint(payload)
        payload["latent_policy_meta"] = {**payload["latent_policy_meta"],
                                         "context": 3}
        payload.setdefault("num_steps", 0)
        payload.setdefault("num_episodes", 1)
        with self.assertRaises(SystemExit):
            module.on_load_checkpoint(payload)


if __name__ == "__main__":
    unittest.main()
