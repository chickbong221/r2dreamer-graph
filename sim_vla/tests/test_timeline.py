"""Stage 9: the imagined timeline, and the gradients that depend on it.

Every assertion here is about *which index* a quantity is read at. That is not
a stylistic question: the reward head predicts the reward that arrived at a
state, so the reward belonging to a transition is the successor's. Reading it
one step early takes a reward no imagined action can influence and drops the
reward earned by the last action -- and the run trains, the return descends,
and the final action of every rollout learns from nothing.

The lambda-return cases are hand-computed and written out, horizon one
included, because a recurrence that is wrong by one step still produces
plausible numbers.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from .common import (DummyExpert, fake_batch, obs_shapes, require_torch,
                     small_model_config)


def excite(module, scale: float = 0.1, seed: int = 0):
    """Move a zero-initialised output layer off zero.

    ``configs/model/_base_.yaml`` sets ``outscale: 0.0`` for the value head and
    the reward head, so ``MLPHead`` multiplies its last layer's weights by zero
    at construction. A fresh head therefore predicts a constant and its
    gradient with respect to its *input* is exactly zero.

    That is deliberate in training -- a critic should start at zero -- but it
    makes a gradient-path test vacuous: an all-zero gradient cannot distinguish
    "the path is severed" from "the head has not learned anything yet". These
    tests are about the path, so the fixture gives the head something to say.
    """
    torch = require_torch()
    generator = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, torch.nn.Linear) and not float(
                    child.weight.abs().sum()):
                child.weight.copy_(torch.randn(
                    child.weight.shape, generator=generator) * scale)
    return module


def start_from(model, batch):
    """Detached imagination starts, the way the online loop makes them.

    ``flatten_start`` alone keeps the graph back through the encoder, so a
    backward from the actor objective reaches world-model parameters that were
    still trainable when ``observe`` ran -- ``freeze_parameters`` sets
    ``requires_grad`` afterwards and cannot retract a graph already built.
    ``start_states`` detaches, which is what production does.
    """
    from sim_vla.training.imagination import start_states

    return start_states(model, batch)


def build(graph_enabled=False, lively=True):
    require_torch()
    from sim_vla.models.critics import ValueCritic
    from sim_vla.models.world_model import build_world_model

    _cfg, model_cfg = small_model_config(graph_enabled)
    batch = fake_batch(graph_enabled=graph_enabled)
    model = build_world_model(model_cfg, obs_shapes(batch), 8,
                              graph_enabled=graph_enabled)
    critic = ValueCritic(model_cfg, model.feature_dim)
    if lively:
        excite(model.reward_head, seed=1)
        excite(critic.net, seed=2)
        # The slow copy was deepcopied from a zero head, so it has to be
        # re-synced or the bootstrap stays constant.
        critic.target.load_state_dict(critic.net.state_dict())
    return model, critic, batch


class TestLambdaReturn(unittest.TestCase):
    """Hand-computed, including horizon one."""

    def test_horizon_one(self):
        torch = require_torch()
        from sim_vla.training.imagination import lambda_return

        # carry starts at value[-1]; with one step the lam mixture collapses,
        # because value[step + 1] and carry are the same tensor:
        #     G_0 = r_0 + gamma * c_0 * v_1
        reward = torch.tensor([[2.0]])
        value = torch.tensor([[10.0], [5.0]])
        cont = torch.tensor([[1.0]])
        out = lambda_return(reward, value, cont, 0.9, 0.95)
        self.assertEqual(tuple(out.shape), (1, 1))
        self.assertAlmostEqual(float(out[0, 0]), 2.0 + 0.9 * 5.0, places=5)

    def test_horizon_one_with_a_terminal_successor(self):
        """cont = 0 removes the bootstrap entirely."""
        torch = require_torch()
        from sim_vla.training.imagination import lambda_return

        out = lambda_return(torch.tensor([[2.0]]),
                            torch.tensor([[10.0], [5.0]]),
                            torch.tensor([[0.0]]), 0.9, 0.95)
        self.assertAlmostEqual(float(out[0, 0]), 2.0, places=6)

    def test_horizon_three(self):
        torch = require_torch()
        from sim_vla.training.imagination import lambda_return

        # gamma = 0.9, lam = 0.5, cont = 1 throughout.
        #   carry  = v3                                    = 40
        #   G_2    = 3 + .9 * (.5 * v3 + .5 * 40)          = 39.0
        #   G_1    = 2 + .9 * (.5 * v2 + .5 * 39)          = 33.05
        #   G_0    = 1 + .9 * (.5 * v1 + .5 * 33.05)       = 24.8725
        reward = torch.tensor([[1.0], [2.0], [3.0]])
        value = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
        cont = torch.ones(3, 1)
        out = lambda_return(reward, value, cont, 0.9, 0.5)
        for index, expected in enumerate((24.8725, 33.05, 39.0)):
            self.assertAlmostEqual(float(out[index, 0]), expected, places=4)


class StubCritic:
    """A value head with a known, linear answer and no parameters."""

    def __init__(self, values):
        self.values = values
        self.calls = []

    def parameters(self):
        return iter(())

    def value(self, feat):
        self.calls.append("value")
        return self.values

    def target_value(self, feat, *, detach=False):
        self.calls.append("target")
        return self.values[-1:]


class TestImaginedTimeline(unittest.TestCase):
    """The reward and continuation belonging to a transition are the
    successor's."""

    def stub(self, rewards, conts, horizon=3):
        torch = require_torch()
        import sim_vla.training.actor_critic as module

        feat = torch.arange((horizon + 1) * 2, dtype=torch.float32
                            ).reshape(horizon + 1, 1, 2)
        heads = {"reward": torch.as_tensor(rewards).reshape(horizon + 1, 1),
                 "cont": torch.as_tensor(conts).reshape(horizon + 1, 1)}
        original = (module.imagine, module.imagined_rewards)

        module.imagine = lambda *a, **k: {
            "feat": feat, "action": torch.zeros(horizon, 1, 2),
            "action_steps": []}
        module.imagined_rewards = lambda _model, _feat: heads
        self.addCleanup(
            lambda: setattr(module, "imagine", original[0]))
        self.addCleanup(
            lambda: setattr(module, "imagined_rewards", original[1]))
        return module, feat

    def test_reward_is_read_at_the_successor(self):
        torch = require_torch()
        from sim_vla.training.actor_critic import ActorCriticConfig

        # 100.0 is the reward that arrived at the rollout's *start*: no
        # imagined action produced it and it must not appear in the return.
        module, feat = self.stub([100.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0])
        critic = StubCritic(torch.tensor([[10.0], [20.0], [30.0], [40.0]]))
        out = module.actor_loss(
            SimpleNamespace(parameters=lambda: iter(())), None, critic, None,
            ActorCriticConfig(horizon=3, discount=0.9, lam=0.5))
        self.assertEqual([float(v) for v in out["reward"].reshape(-1)],
                         [1.0, 2.0, 3.0])
        # The same numbers as the hand-computed lambda-return case.
        self.assertAlmostEqual(float(out["returns"][0, 0]), 24.8725, places=4)

    def test_continuation_is_read_at_the_successor(self):
        torch = require_torch()
        from sim_vla.training.actor_critic import ActorCriticConfig

        # Terminal on arrival at the final imagined state: the last
        # transition's bootstrap is cut, the earlier ones are not.
        module, _feat = self.stub([0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0])
        critic = StubCritic(torch.tensor([[10.0], [20.0], [30.0], [40.0]]))
        out = module.actor_loss(
            SimpleNamespace(parameters=lambda: iter(())), None, critic, None,
            ActorCriticConfig(horizon=3, discount=0.9, lam=0.5))
        self.assertEqual([float(v) for v in out["cont"].reshape(-1)],
                         [1.0, 1.0, 0.0])
        # With cont_2 = 0 the last return is its reward alone.
        self.assertAlmostEqual(float(out["returns"][2, 0]), 1.0, places=6)

    def test_the_last_action_contributes_a_reward(self):
        """Reading ``[:-1]`` dropped it, so the horizon's action was free."""
        torch = require_torch()
        from sim_vla.training.actor_critic import ActorCriticConfig

        critic = StubCritic(torch.tensor([[0.0], [0.0], [0.0], [0.0]]))
        module, _ = self.stub([0.0, 0.0, 0.0, 7.0], [1.0] * 4)
        out = module.actor_loss(
            SimpleNamespace(parameters=lambda: iter(())), None, critic, None,
            ActorCriticConfig(horizon=3, discount=1.0, lam=1.0))
        self.assertAlmostEqual(float(out["returns"][0, 0]), 7.0, places=6)


class TestBootstrapGradient(unittest.TestCase):
    """The slow target's parameters are frozen; its output is not a constant."""

    def test_target_value_is_differentiable_in_its_input(self):
        torch = require_torch()
        model, critic, _ = build()
        feat = torch.randn(3, model.feature_dim, requires_grad=True)
        critic.target_value(feat).sum().backward()
        self.assertIsNotNone(feat.grad)
        self.assertGreater(float(feat.grad.abs().sum()), 0.0,
                           "no gradient reached the bootstrap's input")

    def test_target_value_can_still_be_detached_on_request(self):
        torch = require_torch()
        model, critic, _ = build()
        feat = torch.randn(3, model.feature_dim, requires_grad=True)
        value = critic.target_value(feat, detach=True)
        self.assertFalse(value.requires_grad)

    def test_target_parameters_never_receive_a_gradient(self):
        torch = require_torch()
        model, critic, _ = build()
        feat = torch.randn(3, model.feature_dim, requires_grad=True)
        critic.target_value(feat).sum().backward()
        for parameter in critic.target.parameters():
            self.assertIsNone(parameter.grad,
                              "the frozen target head accumulated a gradient")

    def make_actor(self, model, action_dim=8):
        torch = require_torch()
        from sim_vla.models.latent_adapter import LatentAdapter

        expert = DummyExpert(32, action_dim)

        class Actor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.adapter = LatentAdapter(model.feature_dim, 32, hidden=32,
                                             layers=1)
                self.expert = expert
                self.expert_linear = expert.linear
                self.chunk_size = 2
                self.action_dim = action_dim
                self.flow_steps = 3

            def condition(self, features, instruction=None):
                return {"state_token": self.adapter(features),
                        "instruction": instruction}

            def velocity_fn(self):
                return self.expert

        return Actor()

    def test_gradient_from_the_bootstrap_alone(self):
        """Constant immediate reward: the only path left is the terminal
        value."""
        torch = require_torch()
        import sim_vla.training.actor_critic as module
        from sim_vla.training.actor_critic import ActorCriticConfig

        torch.manual_seed(0)
        model, critic, batch = build()
        actor = self.make_actor(model)
        original = module.imagined_rewards
        # A reward that does not depend on the imagined state at all.
        module.imagined_rewards = lambda _m, feat: {
            "reward": torch.ones(feat.shape[0], feat.shape[1]),
            "cont": torch.ones(feat.shape[0], feat.shape[1])}
        self.addCleanup(
            lambda: setattr(module, "imagined_rewards", original))

        start = start_from(model, batch)
        out = module.actor_loss(model, actor, critic, start,
                                ActorCriticConfig(horizon=2, flow_steps=3))
        out["loss"].backward()
        grads = [p.grad for p in actor.adapter.parameters()
                 if p.grad is not None and float(p.grad.abs().sum()) > 0]
        self.assertTrue(
            grads,
            "with a constant reward the only gradient path is the bootstrap "
            "value, and none arrived")

    def test_gradient_from_the_immediate_successor_reward(self):
        """Constant critic: the only path left is the reward head."""
        torch = require_torch()
        from sim_vla.training.actor_critic import ActorCriticConfig, actor_loss

        torch.manual_seed(0)
        model, _critic, batch = build()
        actor = self.make_actor(model)
        flat = StubCritic(None)

        class ConstantCritic(StubCritic):
            def value(self, feat):
                return torch.zeros(feat.shape[0], feat.shape[1])

            def target_value(self, feat, *, detach=False):
                return torch.zeros(feat.shape[0], feat.shape[1])

        start = start_from(model, batch)
        out = actor_loss(model, actor, ConstantCritic(None), start,
                         ActorCriticConfig(horizon=2, flow_steps=3))
        out["loss"].backward()
        grads = [p.grad for p in actor.adapter.parameters()
                 if p.grad is not None and float(p.grad.abs().sum()) > 0]
        self.assertTrue(grads,
                        "with a constant critic the only gradient path is the "
                        "reward head, and none arrived")

    def test_frozen_modules_accumulate_no_gradients(self):
        torch = require_torch()
        from sim_vla.training.actor_critic import ActorCriticConfig, actor_loss

        torch.manual_seed(0)
        model, critic, batch = build()
        actor = self.make_actor(model)
        start = start_from(model, batch)
        out = actor_loss(model, actor, critic, start,
                         ActorCriticConfig(horizon=2, flow_steps=3))
        out["loss"].backward()
        for name, parameter in model.named_parameters():
            self.assertIsNone(parameter.grad,
                              f"world model parameter {name} accumulated a "
                              "gradient during the actor update")
        for name, parameter in critic.named_parameters():
            self.assertIsNone(parameter.grad,
                              f"critic parameter {name} accumulated a "
                              "gradient during the actor update")


class TestGradientPolicy(unittest.TestCase):
    """Critic targets must build no actor graph; actor updates must keep it."""

    def make_actor(self, model):
        return TestBootstrapGradient.make_actor(self, model)

    def test_non_differentiable_rollout_detaches_the_actions(self):
        torch = require_torch()
        from sim_vla.training.imagination import imagine

        torch.manual_seed(0)
        model, _critic, batch = build()
        actor = self.make_actor(model)
        start = start_from(model, batch)
        rollout = imagine(model, actor, start, 2, flow_steps=3,
                          differentiable=False)
        for step, action in enumerate(rollout["action_steps"]):
            self.assertFalse(action.requires_grad,
                             f"imagined action {step} kept a graph while the "
                             "rollout was only supplying critic targets")

    def test_differentiable_rollout_keeps_the_actions(self):
        torch = require_torch()
        from sim_vla.training.imagination import imagine

        torch.manual_seed(0)
        model, _critic, batch = build()
        actor = self.make_actor(model)
        start = start_from(model, batch)
        rollout = imagine(model, actor, start, 2, flow_steps=3,
                          differentiable=True)
        self.assertTrue(all(a.requires_grad
                            for a in rollout["action_steps"]))

    def test_no_grad_alone_does_not_stop_the_sampler(self):
        """Why the flag exists: sample_actions re-enables grad internally."""
        torch = require_torch()
        from sim_vla.models.flow_sampler import sample_actions

        linear = torch.nn.Linear(3, 3)

        def velocity(x_t, t, cond):
            return linear(x_t)

        with torch.no_grad():
            kept = sample_actions(velocity, None, batch=2, chunk=1, dim=3,
                                  steps=2, differentiable=True)
            dropped = sample_actions(velocity, None, batch=2, chunk=1, dim=3,
                                     steps=2, differentiable=False)
        self.assertTrue(kept.requires_grad)
        self.assertFalse(dropped.requires_grad)


if __name__ == "__main__":
    unittest.main()
