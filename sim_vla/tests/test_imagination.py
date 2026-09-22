"""Stage 7: the flow sampler keeps gradients, imagination executes one chunk.

An imagined rollout is one generated chunk and the first ``execute`` of its
actions stepped through the dynamics -- the same thing the online policy does
between two replans. What is easy to get wrong and impossible to notice later
is how many transitions one chunk produces, which of its actions are used and
in what order, and the arithmetic of the return computed from them.
"""

from __future__ import annotations

import unittest

from .common import (DummyExpert, fake_batch, obs_shapes, require_torch,
                     small_model_config)


class TestFlowSampler(unittest.TestCase):
    def test_sampled_action_carries_gradient_to_the_conditioning(self):
        torch = require_torch()
        from sim_vla.models.flow_sampler import (assert_gradient_reaches,
                                                 sample_actions)
        from sim_vla.models.latent_adapter import LatentAdapter

        adapter = LatentAdapter(feature_dim=16, token_dim=16, hidden=32)
        expert = DummyExpert(token_dim=16, action_dim=4)
        cond = {"state_token": adapter(torch.randn(3, 16)), "instruction": None}
        action = sample_actions(expert, cond, batch=3, chunk=4, dim=4, steps=5,
                                differentiable=True)
        assert_gradient_reaches(action, *adapter.parameters())

    def test_no_grad_sampling_is_detected_not_silent(self):
        torch = require_torch()
        from sim_vla.models.flow_sampler import (assert_gradient_reaches,
                                                 sample_actions)
        from sim_vla.models.latent_adapter import LatentAdapter

        adapter = LatentAdapter(feature_dim=16, token_dim=16, hidden=32)
        expert = DummyExpert(token_dim=16, action_dim=4)
        cond = {"state_token": adapter(torch.randn(3, 16)), "instruction": None}
        action = sample_actions(expert, cond, batch=3, chunk=4, dim=4, steps=5,
                                differentiable=False)
        # A detached actor trains forever without improving; this is where it
        # is supposed to be caught.
        with self.assertRaises(RuntimeError):
            assert_gradient_reaches(action, *adapter.parameters())


def build(graph_enabled):
    require_torch()
    from sim_vla.models.world_model import build_world_model

    _cfg, model_cfg = small_model_config(graph_enabled)
    batch = fake_batch(graph_enabled=graph_enabled)
    model = build_world_model(model_cfg, obs_shapes(batch), 8,
                              graph_enabled=graph_enabled)
    return model, batch


class CountingChunks:
    """A policy that hands back a known chunk and counts how often it is asked.

    One call per rollout is the property under test: the actor is conditioned
    and sampled once, and the chunk's actions are then executed in order.
    """

    def __init__(self, batch, chunk, dim):
        torch = require_torch()
        self.calls = 0
        # Each (start, chunk position) pair gets a distinct constant, so an
        # action executed out of order is visible in the value.
        self.chunk = (torch.arange(batch * chunk, dtype=torch.float32)
                      .reshape(batch, chunk, 1).expand(batch, chunk, dim)
                      .contiguous())

    def __call__(self, feat):
        self.calls += 1
        return self.chunk


class TestChunkRollout(unittest.TestCase):
    def rollout(self, graph_enabled, execute=5, chunk=8):
        torch = require_torch()
        from sim_vla.training.imagination import (flatten_start, imagine_chunk,
                                                  start_states)

        model, batch = build(graph_enabled)
        start = start_states(model, batch)
        actions = CountingChunks(int(start[0].shape[0]), chunk, 8)
        out = imagine_chunk(model, None, start, execute, action_fn=actions)
        return model, start, actions, out

    def test_one_chunk_becomes_exactly_execute_transitions(self):
        require_torch()
        for graph_enabled in (False, True):
            with self.subTest(graph=graph_enabled):
                model, start, actions, out = self.rollout(graph_enabled)
                starts = int(start[0].shape[0])
                # One generation, five transitions, six states.
                self.assertEqual(actions.calls, 1)
                self.assertEqual(tuple(out["feat"].shape),
                                 (6, starts, model.feature_dim))
                self.assertEqual(tuple(out["action"].shape), (5, starts, 8))
                self.assertEqual(len(out["action_steps"]), 5)

    def test_the_executed_actions_are_the_chunks_first_five_in_order(self):
        torch = require_torch()
        _model, _start, actions, out = self.rollout(False)
        for step in range(5):
            self.assertTrue(
                torch.equal(out["action"][step], actions.chunk[:, step]),
                f"step {step} did not execute chunk position {step}")

    def test_a_chunk_that_cannot_supply_the_executed_actions_is_refused(self):
        require_torch()
        with self.assertRaises(ValueError):
            self.rollout(False, execute=5, chunk=4)

    def test_the_states_actually_advance(self):
        torch = require_torch()
        _model, _start, _actions, out = self.rollout(False)
        feat = out["feat"]
        for step in range(1, feat.shape[0]):
            self.assertFalse(torch.allclose(feat[step], feat[step - 1]),
                             f"state {step} equals its predecessor")

    def test_graph_arm_uses_the_semantic_prior_not_an_extractor(self):
        require_torch()
        import inspect

        from sim_vla.training import imagination

        source = inspect.getsource(imagination.imagine_chunk)
        # Comments mention these names, so match the call rather than the word.
        code = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
        # There is no scene inside imagination to extract a graph from.
        for forbidden in ("GraphEncoder(", "FigureGraphSource(", "pack_graph("):
            self.assertNotIn(forbidden, code)
        # g is advanced by img_step, which returns it; nothing here calls the
        # prior a second time.
        self.assertNotIn("semantic_prior(", code)
        self.assertIn("img_step(", code)

    def test_img_step_returns_the_semantic_state_it_advanced(self):
        """img_step owns the semantic step and returns four values.

        The prior is a function of ``deter`` alone -- ``semantic_prior(deter,
        prev_sem)`` ignores ``prev_sem`` -- so calling it again after img_step
        is redundant rather than wrong. What was actually broken was the
        arity: unpacking two values from a four-value return.
        """
        torch = require_torch()
        from sim_vla.training.imagination import flatten_start

        model, batch = build(True)
        stoch, deter, sem = flatten_start(model.observe(batch)["post"], True)
        action = torch.zeros(stoch.shape[0], 8)

        result = model.rssm.img_step(stoch, deter, action, sem)
        self.assertEqual(len(result), 4, "graph img_step returns 4 values")
        _s, next_deter, next_sem, _logit = result
        # The sem it returned is the prior of the deter it returned, which is
        # why a second call would change nothing.
        expected, _ = model.rssm.semantic_prior(next_deter, next_sem)
        self.assertTrue(torch.allclose(next_sem, expected))

    def test_baseline_img_step_returns_two_values(self):
        torch = require_torch()
        from sim_vla.training.imagination import flatten_start

        model, batch = build(False)
        stoch, deter = flatten_start(model.observe(batch)["post"], False)
        result = model.rssm.img_step(stoch, deter,
                                     torch.zeros(stoch.shape[0], 8))
        self.assertEqual(len(result), 2)


class TestChunkReturn(unittest.TestCase):
    """Hand-computed, because a recurrence that is wrong by one step still
    produces plausible numbers."""

    def test_one_transition_is_reward_plus_discounted_bootstrap(self):
        torch = require_torch()
        from sim_vla.training.imagination import chunk_return

        out = chunk_return(torch.tensor([[2.0]]), torch.tensor([[1.0]]),
                           torch.tensor([5.0]), 0.9)
        self.assertEqual(tuple(out.shape), (1,))
        self.assertAlmostEqual(float(out[0]), 2.0 + 0.9 * 5.0, places=6)

    def test_five_transitions_accumulate_backwards(self):
        torch = require_torch()
        from sim_vla.training.imagination import chunk_return

        # gamma = 0.5, cont = 1, bootstrap 8, computed inside out:
        #   5 + .5*8   = 9        4 + .5*9    = 8.5      3 + .5*8.5 = 7.25
        #   2 + .5*7.25 = 5.625   1 + .5*5.625 = 3.8125
        reward = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        cont = torch.ones(5, 1)
        out = chunk_return(reward, cont, torch.tensor([8.0]), 0.5)
        self.assertAlmostEqual(float(out[0]), 3.8125, places=6)

    def test_a_terminal_transition_keeps_its_reward_and_drops_the_rest(self):
        torch = require_torch()
        from sim_vla.training.imagination import chunk_return

        # Continuation zero at transition 1: r_0 and r_1 count, r_2 and the
        # bootstrap do not.
        reward = torch.tensor([[1.0], [2.0], [100.0]])
        cont = torch.tensor([[1.0], [0.0], [1.0]])
        out = chunk_return(reward, cont, torch.tensor([1000.0]), 0.9)
        self.assertAlmostEqual(float(out[0]), 1.0 + 0.9 * 2.0, places=6)

    def test_with_no_reward_it_is_the_discounted_bootstrap(self):
        torch = require_torch()
        from sim_vla.training.imagination import chunk_return

        out = chunk_return(torch.zeros(3, 2), torch.ones(3, 2),
                           torch.full((2,), 5.0), 1.0)
        self.assertTrue(torch.allclose(out, torch.full((2,), 5.0)))

    def test_mismatched_shapes_are_refused(self):
        torch = require_torch()
        from sim_vla.training.imagination import chunk_return

        with self.assertRaises(ValueError):
            chunk_return(torch.zeros(3, 2), torch.ones(2, 2),
                         torch.zeros(2), 0.9)
        with self.assertRaises(ValueError):
            chunk_return(torch.zeros(3, 2), torch.ones(3, 2),
                         torch.zeros(3), 0.9)


if __name__ == "__main__":
    unittest.main()
