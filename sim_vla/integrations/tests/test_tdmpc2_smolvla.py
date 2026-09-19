"""TD-MPC2 with SmolVLA: conditioning, imitation, and the five policy sites.

The action expert is a stub here -- a small linear velocity field with the same
signature -- so these run on CPU without the 450M checkpoint. What they test is
the integration: which latent the policy is conditioned on, that the world
model is frozen when it should be, that the gradient of TD-MPC2's own Q
objective reaches the adapter and the expert, and that MPC is still what picks
the action. ``test_tdmpc2_real.py`` repeats the load-bearing ones against the
real checkpoint.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from . import common as C


def build(*, chunk=4, bounded=False, sites=None, live_critic=False, **cfg_over):
    from ..action_space import converter_for
    from ..tdmpc2 import agent as build_agent
    from ..tdmpc2.policy import SITES, LatentPolicy

    source = C.fake_source(bounded_actions=bounded)
    cfg = C.tiny_tdmpc2_cfg(action_dim=source.action_dim, **cfg_over)
    agent = build_agent.build_agent(cfg)
    actor = C.stub_latent_actor(int(cfg.true_latent_dim), source.action_dim,
                                chunk_size=chunk)
    converter = converter_for(source.metadata, action_dim=source.action_dim,
                              mode="identity", device="cpu")
    policy = LatentPolicy(actor, converter,
                          sites=SITES if sites is None else sites,
                          lr=1e-3, max_batch=64)
    if live_critic:
        # Upstream zero-initialises the Q heads' last layer, so dQ/da is
        # exactly zero on a fresh agent. See common.unzero_critic.
        C.unzero_critic(agent)
    return source, cfg, agent, actor, converter, policy


class CausalConditioning(unittest.TestCase):
    """What the policy is conditioned on, and what it cannot see."""

    def test_the_latent_at_row_t_ignores_later_observations(self):
        torch = C.require_torch()

        from ..tdmpc2 import data as demo_data

        source, cfg, agent, _actor, _conv, _policy = build()
        windows = demo_data.DemoWindows(source, horizon=6, seed=0, stride=1)
        batch = windows.raw_batch(2)
        obs = demo_data.observations(batch, source, device="cpu")

        torch.manual_seed(11)
        first = agent.model.encode(obs, None)

        scrambled = obs.clone()
        scrambled[3:] = torch.flip(scrambled[3:], dims=(0,)) // 2 + 7
        torch.manual_seed(11)
        second = agent.model.encode(scrambled, None)

        self.assertTrue(torch.allclose(first[:3], second[:3]),
                        "rows 0..2 changed when only rows 3.. were altered")
        self.assertFalse(torch.allclose(first[3:], second[3:]),
                         "the scramble did not change anything, so the test "
                         "proves nothing")

    def test_incremental_extraction_matches_the_sequence(self):
        """One row at a time, and a whole window, have to agree.

        TD-MPC2's encoder applies a random shift on every call -- upstream's
        behaviour, in training and inside ``act`` alike -- so the comparison
        fixes the seed before each call. Each call consumes exactly one draw,
        so the two paths see the same augmentation and any difference is the
        code path, which is what is under test.
        """
        torch = C.require_torch()

        from ..tdmpc2 import data as demo_data

        source, cfg, agent, _actor, _conv, _policy = build()
        windows = demo_data.DemoWindows(source, horizon=5, seed=0, stride=1)
        obs = demo_data.observations(windows.raw_batch(2), source, device="cpu")

        for step in range(obs.shape[0]):
            torch.manual_seed(100 + step)
            sequence = agent.model.encode(obs[step:step + 1], None)[0]
            torch.manual_seed(100 + step)
            single = agent.model.encode(obs[step], None)
            self.assertTrue(torch.allclose(sequence, single, atol=1e-6),
                            f"row {step} differs between the batched and the "
                            "incremental path")

    def test_the_adapter_reads_the_backends_own_latent(self):
        """No second encoder, no extra branch: ``z`` in, one token out."""
        torch = C.require_torch()

        _source, cfg, _agent, actor, _conv, _policy = build()
        self.assertEqual(actor.adapter.feature_dim, int(cfg.true_latent_dim))
        token = actor.adapter(torch.zeros(2, int(cfg.true_latent_dim)))
        self.assertEqual(token.shape[-2], 1)


class Imitation(unittest.TestCase):
    def trainer(self, *, bounded=True, chunk=4, steps=5, sequence_length=8,
                burn_in=2):
        from ..tdmpc2 import stages

        source, cfg, agent, actor, converter, _policy = build(
            chunk=chunk, bounded=bounded)
        config = stages.ImitationConfig(
            steps=steps, batch_size=4, lr=3e-3, log_every=0,
            sequence_length=sequence_length, burn_in=burn_in)
        return source, agent, actor, stages.ImitationTrainer(
            agent, actor, source, converter, config=config, device="cpu")

    def test_the_world_model_is_frozen(self):
        torch = C.require_torch()
        _source, agent, _actor, trainer = self.trainer()
        self.assertGreater(trainer.frozen, 0)
        self.assertFalse(any(p.requires_grad for p in agent.model.parameters()))

        before = {n: p.detach().clone()
                  for n, p in agent.model.named_parameters()}
        for _ in range(3):
            trainer.update()
        for name, parameter in agent.model.named_parameters():
            self.assertTrue(torch.allclose(before[name], parameter),
                            f"{name} moved during imitation")
            self.assertIsNone(parameter.grad, f"{name} accumulated a gradient")

    def test_the_adapter_and_the_expert_get_gradients(self):
        torch = C.require_torch()
        _source, _agent, actor, trainer = self.trainer()
        batch = trainer.windows.raw_batch(4)
        loss, metrics = trainer.loss(batch)
        self.assertFalse(metrics["skipped"])
        loss.backward()
        adapter = [p.grad for p in actor.adapter.parameters() if p.grad is not None]
        self.assertTrue(adapter, "no gradient reached the adapter")
        self.assertGreater(float(sum(g.abs().sum() for g in adapter)), 0.0)
        expert = actor.actor.expert.linear.weight.grad
        self.assertIsNotNone(expert)
        self.assertGreater(float(expert.abs().sum()), 0.0)

    def test_burn_in_rows_are_excluded_before_the_actor_runs(self):
        torch = C.require_torch()

        from .. import chunking

        _source, _agent, actor, trainer = self.trainer(burn_in=3,
                                                       sequence_length=6)
        batch = trainer.windows.raw_batch(4)
        tensors = {k: torch.as_tensor(np.asarray(v)) for k, v in batch.items()}
        selection = chunking.select(tensors, int(actor.chunk_size),
                                    converter=trainer.converter)
        rows = int(tensors["loss_mask"].shape[1])
        self.assertLess(int(selection.rows.numel()),
                        int(tensors["loss_mask"].shape[0]) * rows)
        eligible = selection.eligible
        self.assertFalse(eligible[:, :3].any(),
                         "burn-in rows reached the actor")

    def test_a_short_run_overfits(self):
        """The loss has to actually descend on a handful of episodes."""
        torch = C.require_torch()
        _source, _agent, _actor, trainer = self.trainer(
            bounded=True, steps=200, sequence_length=6, burn_in=0)
        torch.manual_seed(0)
        first = [trainer.update()["loss"] for _ in range(20)]
        for _ in range(180):
            trainer.update()
        last = [trainer.update()["loss"] for _ in range(20)]
        self.assertLess(float(np.mean(last)), 0.75 * float(np.mean(first)),
                        f"flow loss did not descend: {np.mean(first):.4f} -> "
                        f"{np.mean(last):.4f}")


class PolicySites(unittest.TestCase):
    def test_all_five_sites_are_served_by_default(self):
        C.require_torch()
        from ..tdmpc2.policy import SITES

        _source, _cfg, _agent, _actor, _conv, policy = build()
        self.assertEqual(set(policy.sites), set(SITES))
        self.assertTrue(policy.dormant_gaussian)
        self.assertIn("dormant", policy.descriptor()["gaussian_after_handoff"])

    def test_a_fallback_site_keeps_the_gaussian_training(self):
        torch = C.require_torch()

        from ..tdmpc2 import data as demo_data

        source, cfg, agent, _actor, _conv, policy = build(
            sites=("plan_proposals", "td_target", "update_pi"))
        agent.attach_policy(policy)
        self.assertTrue(policy.needs_gaussian)
        self.assertIn("trained", policy.descriptor()["gaussian_after_handoff"])

        buffer = demo_data.DemoBuffer(source, horizon=int(cfg.horizon),
                                      batch_size=int(cfg.batch_size),
                                      device="cpu")
        before = agent.model._pi[0].weight.detach().clone()
        agent.update(buffer)
        self.assertFalse(torch.allclose(before, agent.model._pi[0].weight),
                         "a site still uses the Gaussian prior but it was "
                         "left frozen at the end of pretraining")

    def test_the_gaussian_is_dormant_when_smolvla_serves_everything(self):
        torch = C.require_torch()

        from ..tdmpc2 import data as demo_data

        source, cfg, agent, _actor, _conv, policy = build()
        agent.attach_policy(policy)
        buffer = demo_data.DemoBuffer(source, horizon=int(cfg.horizon),
                                      batch_size=int(cfg.batch_size),
                                      device="cpu")
        before = {n: p.detach().clone()
                  for n, p in agent.model._pi.named_parameters()}
        agent.update(buffer)
        for name, parameter in agent.model._pi.named_parameters():
            self.assertTrue(torch.allclose(before[name], parameter),
                            f"the dormant Gaussian prior moved at {name}")

    def test_mpc_selects_the_action_the_policy_only_proposes(self):
        """MPC stays the final selector; SmolVLA seeds and scores it."""
        torch = C.require_torch()

        from ..tdmpc2 import data as demo_data

        source, cfg, agent, actor, _conv, policy = build()
        agent.attach_policy(policy)
        buffer = demo_data.DemoBuffer(source, horizon=int(cfg.horizon),
                                      batch_size=int(cfg.num_envs), device="cpu")
        obs = buffer.sample()[0][0][: int(cfg.num_envs)]
        action = agent.act(obs, t0=True, eval_mode=False)

        rows = policy.usage()["rows"]
        self.assertEqual(rows["plan_proposals"],
                         int(cfg.num_envs) * int(cfg.num_pi_trajs) * int(cfg.horizon))
        self.assertEqual(rows["estimate_value"],
                         int(cfg.num_envs) * int(cfg.num_samples) * int(cfg.iterations))
        self.assertEqual(rows["act"], 0, "MPC was bypassed")
        self.assertLessEqual(float(action.abs().max()), 1.0 + 1e-6)

    def test_a_chunk_is_not_a_planning_horizon(self):
        """The proposal at a latent is the chunk's first action, and no more."""
        torch = C.require_torch()

        _source, cfg, _agent, actor, _conv, policy = build(chunk=6)
        self.assertNotEqual(int(actor.chunk_size), int(cfg.horizon))
        z = torch.randn(4, int(cfg.true_latent_dim))
        torch.manual_seed(5)
        one = policy.sample(z, None)
        torch.manual_seed(5)
        chunk = actor.sample_chunk(z.to(actor.device), differentiable=False)
        self.assertTrue(torch.allclose(policy.converter.to_env(chunk[:, 0]), one))

    def test_proposals_come_from_the_advanced_latent(self):
        """The planner rolls the dynamics between proposals; record that it did."""
        torch = C.require_torch()

        seen = []

        _source, cfg, agent, _actor, _conv, policy = build()
        agent.attach_policy(policy)
        original = policy.sample

        def spy(z, task=None, **kwargs):
            if kwargs.get("site") == "plan_proposals":
                seen.append(z.detach().clone())
            return original(z, task, **kwargs)

        policy.sample = spy                                # type: ignore[assignment]
        z = torch.randn(int(cfg.num_envs), int(cfg.true_latent_dim))
        agent.plan(z, t0=True, eval_mode=False, task=None)
        self.assertEqual(len(seen), int(cfg.horizon))
        for earlier, later in zip(seen, seen[1:]):
            self.assertFalse(torch.allclose(earlier, later),
                             "the proposal latent did not advance between steps")


class OnlineObjective(unittest.TestCase):
    def test_the_q_objective_reaches_the_adapter_and_the_expert(self):
        """With a Q that actually depends on its action.

        On a freshly built agent it does not: upstream zeroes the last layer
        of every Q head, so ``dQ/da`` is identically zero and this test would
        pass or fail for reasons that have nothing to do with the integration.
        """
        torch = C.require_torch()

        source, cfg, agent, actor, _conv, policy = build(live_critic=True)
        agent.attach_policy(policy)
        zs = torch.randn(int(cfg.horizon) + 1, int(cfg.batch_size),
                         int(cfg.true_latent_dim))
        policy.zero_grad()
        agent.model.track_q_grad(False)
        pis = policy.sample(zs, None, grad=True, site="update_pi")
        (-agent.model.Q(zs, pis, None, return_type="avg")).mean().backward()
        agent.model.track_q_grad(True)
        for name, parameter in actor.adapter.named_parameters():
            self.assertIsNotNone(parameter.grad,
                                 f"no gradient reached the adapter at {name}")
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0,
                               f"the gradient into {name} is exactly zero")
        expert = actor.actor.expert.linear.weight
        self.assertIsNotNone(expert.grad)
        self.assertGreater(float(expert.grad.abs().sum()), 0.0)

        # ...and the whole update runs and moves them.
        adapter_before = {n: p.detach().clone()
                          for n, p in actor.adapter.named_parameters()}
        loss = agent.update_pi_latent(policy, zs, None)
        self.assertTrue(np.isfinite(loss))
        moved = [n for n, p in actor.adapter.named_parameters()
                 if not torch.allclose(adapter_before[n], p)]
        self.assertEqual(len(moved), len(adapter_before),
                         f"the Q objective left {set(adapter_before) - set(moved)} "
                         "untouched")

    def test_the_world_model_does_not_move_during_a_policy_update(self):
        """Frozen model, and the action gradient is not cut to achieve it."""
        torch = C.require_torch()

        source, cfg, agent, actor, _conv, policy = build(live_critic=True)
        agent.attach_policy(policy)
        zs = torch.randn(int(cfg.horizon) + 1, int(cfg.batch_size),
                         int(cfg.true_latent_dim))
        before = {n: p.detach().clone()
                  for n, p in agent.model.named_parameters()}
        agent.update_pi_latent(policy, zs, None)
        for name, parameter in agent.model.named_parameters():
            self.assertTrue(torch.allclose(before[name], parameter),
                            f"{name} moved during a policy-only update")
        # ...and the path the gradient took is still live.
        action = policy.sample(zs[0], None, grad=True)
        self.assertTrue(action.requires_grad)
        grads = torch.autograd.grad(
            action.sum(), list(actor.adapter.parameters())[:2],
            allow_unused=True)
        self.assertTrue(all(g is not None for g in grads))

    def test_no_entropy_term_and_no_substitute(self):
        """The objective is the scaled Q with rho weighting, and nothing else."""
        torch = C.require_torch()

        source, cfg, agent, actor, _conv, policy = build(live_critic=True)
        agent.attach_policy(policy)
        zs = torch.randn(int(cfg.horizon) + 1, int(cfg.batch_size),
                         int(cfg.true_latent_dim))

        import copy

        # The update steps the optimizer, so the recomputation has to start
        # from the weights the update started from -- and from the same draws:
        # the sampler is seeded by torch and `Q(return_type='avg')` picks its
        # two heads with numpy.
        snapshot = copy.deepcopy(actor.state_dict())
        torch.manual_seed(2)
        np.random.seed(2)
        reported = agent.update_pi_latent(policy, zs, None)

        actor.load_state_dict(snapshot)
        torch.manual_seed(2)
        np.random.seed(2)
        agent.model.track_q_grad(False)
        pis = policy.sample(zs, None, grad=True, site="update_pi")
        qs = agent.model.Q(zs, pis, None, return_type="avg")
        agent.model.track_q_grad(True)
        # `update_pi_latent` updated the running scale before scaling, so the
        # recomputation uses the value it left behind.
        scaled = qs * (1.0 / agent.scale.value)
        rho = torch.pow(cfg.rho, torch.arange(len(scaled)))
        expected = ((-scaled).mean(dim=(1, 2)) * rho).mean()
        self.assertAlmostEqual(reported, float(expected), places=5)

        # And the entropy term the Gaussian branch carries is really absent:
        # adding it back changes the number.
        with_entropy = expected + cfg.entropy_coef * 1.0
        self.assertNotAlmostEqual(reported, float(with_entropy), places=5)

    def test_log_prob_is_never_asked_for(self):
        C.require_torch()

        from ..latent_actor import ActorCapabilityError

        _source, _cfg, _agent, actor, _conv, _policy = build()
        with self.assertRaises(ActorCapabilityError):
            actor.log_prob()


class Checkpoints(unittest.TestCase):
    def meta(self, cfg, source, converter, actor, *, stage="imitation",
             smolvla=True):
        from ..checkpoint import StageMeta
        from ..tdmpc2.config import architecture

        return StageMeta(
            backend="tdmpc2", stage=stage, env_id=str(cfg.env_id),
            smolvla=smolvla, architecture=architecture(cfg),
            normalization=converter.descriptor(),
            dataset_identity=source.identity(),
            actor={"revision": "abc123", "chunk_size": int(actor.chunk_size),
                   "action_dim": int(actor.action_dim)})

    def test_round_trip(self):
        torch = C.require_torch()

        from ..checkpoint import load, save

        source, cfg, agent, actor, converter, _policy = build()
        meta = self.meta(cfg, source, converter, actor)
        with tempfile.TemporaryDirectory() as tmp:
            path = save(Path(tmp) / "stage2.pt", meta,
                        {"model": agent.model, "adapter": actor.adapter,
                         "actor": actor})
            self.assertTrue(path.with_suffix(".json").exists())

            source2, cfg2, agent2, actor2, converter2, _p2 = build()
            restored = load(path, self.meta(cfg2, source2, converter2, actor2),
                            {"model": agent2.model, "adapter": actor2.adapter,
                             "actor": actor2})
            self.assertEqual(restored.backend, "tdmpc2")
            self.assertEqual(restored.stage, "imitation")
            for (name, left), (_n, right) in zip(
                    actor.adapter.named_parameters(),
                    actor2.adapter.named_parameters()):
                self.assertTrue(torch.allclose(left, right), name)

    def test_a_different_normalization_is_refused(self):
        torch = C.require_torch()

        from ..action_space import ActionConverter
        from ..checkpoint import load, save
        from ...data.normalization import FieldStats, Normalizer

        source, cfg, agent, actor, converter, _policy = build()
        with tempfile.TemporaryDirectory() as tmp:
            path = save(Path(tmp) / "stage2.pt",
                        self.meta(cfg, source, converter, actor),
                        {"model": agent.model})
            stats = FieldStats(mean=[0.0] * source.action_dim,
                               std=[1.0] * source.action_dim,
                               low=[-1.0] * source.action_dim,
                               high=[1.0] * source.action_dim,
                               minimum=[-1.0] * source.action_dim,
                               maximum=[1.0] * source.action_dim, count=1)
            other = ActionConverter(
                action_dim=source.action_dim, mode="mean_std",
                normalizer=Normalizer(fields={"actions": stats}, identity={},
                                      mode="mean_std"))
            wanted = self.meta(cfg, source, other, actor)
            with self.assertRaises(SystemExit) as raised:
                load(path, wanted, {"model": agent.model})
            self.assertIn("normalization", str(raised.exception))

    def test_a_native_checkpoint_is_refused_for_a_smolvla_run(self):
        torch = C.require_torch()

        from ..checkpoint import load, save

        source, cfg, agent, actor, converter, _policy = build()
        with tempfile.TemporaryDirectory() as tmp:
            path = save(Path(tmp) / "native.pt",
                        self.meta(cfg, source, converter, actor,
                                  stage="world_model", smolvla=False),
                        {"model": agent.model})
            with self.assertRaises(SystemExit):
                load(path, self.meta(cfg, source, converter, actor,
                                     stage="world_model", smolvla=True),
                     {"model": agent.model})

    def test_the_agents_own_save_carries_the_policy(self):
        torch = C.require_torch()

        source, cfg, agent, actor, _conv, policy = build()
        agent.attach_policy(policy)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent.pt"
            agent.save(path)
            payload = torch.load(path, weights_only=False)
            self.assertIn("latent_policy", payload)
            self.assertIn("latent_policy_meta", payload)

            source2, cfg2, agent2, actor2, conv2, policy2 = build()
            agent2.attach_policy(policy2)
            agent2.load(payload)
            for (name, left), (_n, right) in zip(
                    actor.adapter.named_parameters(),
                    actor2.adapter.named_parameters()):
                self.assertTrue(torch.allclose(left, right), name)


class OnlineTurn(unittest.TestCase):
    """One turn of the loop: plan, step, store, update.

    Upstream's ``OnlineTrainer`` needs ManiSkill and a CUDA replay, so what is
    driven here is the agent's own half of the loop -- the part the
    integration touches -- against a fake environment. The trainer itself is
    upstream's and is exercised by the real Stage 3 run.
    """

    class FakeEnv:
        def __init__(self, *, num_envs, channels, size, action_dim, steps=6):
            self.num_envs, self.action_dim, self.steps = num_envs, action_dim, steps
            self.shape = (num_envs, channels, size, size)
            self._rng = np.random.default_rng(0)
            self._t = 0

        def _obs(self):
            import torch

            return torch.as_tensor(
                self._rng.integers(0, 256, self.shape).astype(np.uint8))

        def reset(self):
            self._t = 0
            return self._obs()

        def step(self, action):
            import torch

            self._t += 1
            reward = torch.full((self.num_envs,), 0.25)
            return self._obs(), reward, self._t >= self.steps

    def test_collect_and_update(self):
        torch = C.require_torch()

        from ..tdmpc2 import data as demo_data

        source, cfg, agent, actor, _conv, policy = build(live_critic=True)
        agent.attach_policy(policy)
        buffer = demo_data.DemoBuffer(source, horizon=int(cfg.horizon),
                                      batch_size=int(cfg.batch_size),
                                      device="cpu")
        env = self.FakeEnv(num_envs=int(cfg.num_envs),
                           channels=int(cfg.obs_shape["rgb"][0]),
                           size=int(cfg.obs_shape["rgb"][1]),
                           action_dim=int(cfg.action_dim))
        obs = env.reset()
        adapter_before = {n: p.detach().clone()
                          for n, p in actor.adapter.named_parameters()}
        done = False
        step = 0
        metrics = {}
        while not done:
            action = agent.act(obs, t0=step == 0, eval_mode=False)
            self.assertEqual(tuple(action.shape),
                             (int(cfg.num_envs), int(cfg.action_dim)))
            self.assertLessEqual(float(action.abs().max()), 1.0 + 1e-6)
            obs, _reward, done = env.step(action)
            metrics = agent.update(buffer)
            step += 1
        self.assertTrue(np.isfinite(metrics["pi_loss"]))
        moved = [n for n, p in actor.adapter.named_parameters()
                 if not torch.allclose(adapter_before[n], p)]
        self.assertTrue(moved, "the online updates did not move the adapter")
        rows = policy.usage()["rows"]
        self.assertGreater(rows["plan_proposals"], 0)
        self.assertGreater(rows["td_target"], 0)
        self.assertGreater(rows["update_pi"], 0)

    def test_the_cost_report_counts_every_site(self):
        C.require_torch()

        from ..tdmpc2.policy import planner_cost

        _source, cfg, _agent, actor, _conv, policy = build()
        cost = planner_cost(cfg, sites=policy.sites,
                            flow_steps=int(actor.flow_steps))
        self.assertEqual(cost["rows_per_env_step"]["estimate_value"],
                         int(cfg.num_envs) * int(cfg.num_samples)
                         * int(cfg.iterations))
        self.assertGreater(cost["total_denoise_passes_per_env_step"],
                           cost["total_rows_per_env_step"])
        cheaper = planner_cost(cfg, sites=("update_pi",),
                               flow_steps=int(actor.flow_steps))
        self.assertEqual(cheaper["rows_per_env_step"]["estimate_value"], 0)


if __name__ == "__main__":
    unittest.main()
