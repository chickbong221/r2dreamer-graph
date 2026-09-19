"""The real SmolVLA checkpoint, the real dataset, the real simulator.

Nothing here is mocked. The checkpoint is loaded at a resolved commit, each
backend's adapter is built at the width the *loaded model* reports, a
flow-matching loss runs forward and backward through the frozen transformer,
and an action chunk is sampled with gradients enabled.

Skips are narrow and they name what was missing. A checkpoint that cannot be
reached skips; a checkpoint that loads and whose interface has moved **fails**,
because that is the result this module exists to produce. ``run_stage`` lists
this module as required, so a skipped run reports INCOMPLETE rather than green.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from . import common as C

REPO = C.REPO
DEMOS = REPO / "data/sim_vla_demos"


def dataset_for(task: str = "PickCube-v1") -> Path:
    path = DEMOS / task / "demos.h5"
    if not path.exists():
        raise unittest.SkipTest(f"no collected dataset at {path}")
    return path


def pretrained():
    """The loaded checkpoint, shared across this module."""
    from ...tests.test_pretrained import load

    return load()


class TdmpcActor(unittest.TestCase):
    """TD-MPC2's latent, through the real expert."""

    def actor(self, latent_dim=576, action_dim=8):
        torch = C.require_torch()
        loaded = pretrained()

        from ...models.latent_adapter import LatentAdapter
        from ...models.pretrained import model_facts
        from ...models.smolvla_actor import SmolVLAActor
        from ..latent_actor import LatentActor

        facts = model_facts(loaded)
        adapter = LatentAdapter(latent_dim, int(facts["vlm_hidden_size"]),
                                hidden=256)
        inner = SmolVLAActor(loaded, adapter, action_dim=action_dim,
                             instruction="pick up the cube",
                             state_token_mode="embedding")
        return LatentActor(inner, instruction="pick up the cube"), facts

    def test_a_chunk_samples_at_the_checkpoints_own_width(self):
        torch = C.require_torch()
        actor, facts = self.actor()
        feature = torch.zeros(2, 576, device=actor.device)
        chunk = actor.sample_chunk(feature, differentiable=False, steps=2)
        self.assertEqual(tuple(chunk.shape),
                         (2, int(facts["chunk_size"]), 8))

    def test_the_gradient_reaches_the_adapter_through_the_frozen_stack(self):
        torch = C.require_torch()

        from ..latent_actor import assert_gradient_reaches

        actor, _facts = self.actor()
        feature = torch.zeros(2, 576, device=actor.device)
        chunk = actor.sample_chunk(feature, differentiable=True, steps=2)
        assert_gradient_reaches(chunk, list(actor.adapter.parameters())[:2],
                                what="the real expert's sampled chunk")

    def test_the_flow_loss_descends_on_one_chunk(self):
        torch = C.require_torch()
        actor, facts = self.actor()
        chunk = int(facts["chunk_size"])
        feature = torch.zeros(1, 576, device=actor.device)
        target = torch.zeros(1, chunk, 8, device=actor.device)
        optimizer = torch.optim.AdamW(actor.trainable_parameters(), lr=1e-4)
        torch.manual_seed(0)
        first, last = None, None
        for step in range(12):
            loss, _metrics = actor.flow_loss(feature, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step == 0:
                first = float(loss.detach())
            last = float(loss.detach())
        self.assertLess(last, first,
                        f"the real flow loss did not descend: {first} -> {last}")

    def test_the_policy_returns_native_units(self):
        torch = C.require_torch()

        from ..action_space import ActionConverter
        from ..tdmpc2.policy import LatentPolicy

        actor, _facts = self.actor()
        converter = ActionConverter(action_dim=8, device=actor.device)
        policy = LatentPolicy(actor, converter, lr=1e-5, max_batch=4)
        z = torch.zeros(3, 576, device=actor.device)
        action = policy.sample(z, None, site="plan_proposals")
        self.assertEqual(tuple(action.shape), (3, 8))
        self.assertLessEqual(float(action.abs().max()), 1.0 + 1e-6)


class SoldActor(unittest.TestCase):
    """SOLD's slot history, through the real expert."""

    def actor(self, *, num_slots=7, slot_dim=128, action_dim=8, context=3):
        torch = C.require_torch()
        loaded = pretrained()

        from ...models.pretrained import model_facts
        from ...models.smolvla_actor import SmolVLAActor
        from ..latent_actor import LatentActor
        from ..sold.adapter import SlotHistoryAdapter

        facts = model_facts(loaded)
        adapter = SlotHistoryAdapter(
            num_slots=num_slots, slot_dim=slot_dim,
            token_dim=int(facts["vlm_hidden_size"]), context=context,
            max_episode_steps=context, head_token_dim=128, hidden_dim=256,
            num_heads=4, num_layers=2)
        inner = SmolVLAActor(loaded, adapter, action_dim=action_dim,
                             instruction="pick up the cube",
                             state_token_mode="embedding")
        return LatentActor(inner, instruction="pick up the cube"), facts

    def test_a_slot_history_conditions_the_real_expert(self):
        torch = C.require_torch()
        actor, facts = self.actor()
        slots = torch.zeros(2, 3, 7, 128, device=actor.device)
        chunk = actor.sample_chunk(slots, differentiable=False, steps=2)
        self.assertEqual(tuple(chunk.shape), (2, int(facts["chunk_size"]), 8))

    def test_the_gradient_reaches_the_slot_adapter(self):
        torch = C.require_torch()

        from ..latent_actor import assert_gradient_reaches

        actor, _facts = self.actor()
        slots = torch.zeros(2, 3, 7, 128, device=actor.device)
        chunk = actor.sample_chunk(slots, differentiable=True, steps=2)
        assert_gradient_reaches(chunk, list(actor.adapter.parameters())[:2],
                                what="the real expert's sampled chunk")

    def test_the_bounded_context_holds_with_the_real_expert(self):
        torch = C.require_torch()
        actor, _facts = self.actor(context=2)
        history = torch.randn(1, 6, 7, 128, device=actor.device)
        torch.manual_seed(0)
        long_history = actor.actor.condition(history)["state_token"]
        torch.manual_seed(0)
        short = actor.actor.condition(history[:, -2:])["state_token"]
        self.assertTrue(torch.allclose(long_history, short, atol=1e-5))


class Simulator(unittest.TestCase):
    """The collected dataset and the live environment, for both backends."""

    def test_tdmpc2_reads_the_collected_demonstrations(self):
        torch = C.require_torch()
        C.require("h5py")
        path = dataset_for()

        from ..tdmpc2 import data as demo_data

        source = demo_data.open_demos(path, render_size=64, include_state=False)
        buffer = demo_data.DemoBuffer(source, horizon=3, batch_size=2,
                                      device="cpu")
        obs, action, reward, _task = buffer.sample()
        self.assertEqual(obs.dtype, torch.uint8)
        self.assertEqual(tuple(obs.shape)[0], 4)
        self.assertEqual(tuple(action.shape), (3, 2, source.action_dim))
        self.assertEqual(tuple(reward.shape), (3, 2, 1))

    def test_sold_reads_the_collected_demonstrations(self):
        torch = C.require_torch()
        C.require("h5py")
        path = dataset_for()

        from ..sold import data as sold_data

        source = sold_data.open_demos(path, image_size=(64, 64))
        loader = sold_data.DemoLoader(source, sequence_length=6, batch_size=2,
                                      device="cpu")
        batch = loader.sample()
        self.assertEqual(batch["obs"].dtype, torch.uint8)
        self.assertEqual(tuple(batch["obs"].shape), (2, 6, 3, 64, 64))
        self.assertEqual(tuple(batch["action"].shape),
                         (2, 6, source.action_dim))

    def test_the_live_environment_matches_the_dataset_for_sold(self):
        torch = C.require_torch()
        C.require("h5py")
        C.require("mani_skill")
        path = dataset_for()

        from ...envs.maniskill import load_metadata
        from ..sold import data as sold_data
        from ..sold.env import SoldManiSkillEnv, check_contract

        source = sold_data.open_demos(path, image_size=(64, 64))
        env = SoldManiSkillEnv(load_metadata(path), image_size=(64, 64),
                               camera=source.images.cameras[0],
                               max_episode_steps=5)
        try:
            check_contract(env, source, log=None)
            obs = env.reset()
            self.assertEqual(tuple(obs.shape), (3, 64, 64))
            self.assertEqual(obs.dtype, torch.uint8)
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            self.assertEqual(tuple(obs.shape), (3, 64, 64))
            self.assertIsInstance(float(reward), float)
            self.assertIn("success", info)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
