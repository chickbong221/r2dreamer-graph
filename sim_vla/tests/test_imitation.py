"""Stage 5: chunking, masking, and a small overfit of the flow loss."""

from __future__ import annotations

import unittest

from .common import DummyExpert, require_torch


class TestChunking(unittest.TestCase):
    def test_chunks_are_masked_past_the_end_of_the_episode(self):
        torch = require_torch()
        from sim_vla.training.train_imitation import chunk_targets

        actions = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1)
        valid = torch.ones(1, 5, dtype=torch.bool)
        targets, mask = chunk_targets(actions, valid, chunk=3)
        self.assertEqual(tuple(targets.shape), (1, 5, 3, 1))
        # The chunk at t=3 holds actions 3 and 4 and then nothing real.
        self.assertTrue(mask[0, 3].tolist() == [True, True, False])
        self.assertTrue(mask[0, 4].tolist() == [True, False, False])

    def test_a_chunk_never_crosses_into_invalid_steps(self):
        torch = require_torch()
        from sim_vla.training.train_imitation import chunk_targets

        actions = torch.zeros(1, 6, 2)
        valid = torch.tensor([[True, True, True, False, False, False]])
        _targets, mask = chunk_targets(actions, valid, chunk=4)
        self.assertFalse(mask[0, 0, 3])
        self.assertFalse(mask[0, 2, 1:].any())


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


if __name__ == "__main__":
    unittest.main()
