"""TD-MPC2 with the integration disabled, and the sizing measurement.

Everything here runs against the upstream agent as it is: no adapter, no
actor, no policy attached. If any of it fails, the hooks changed behaviour
they were supposed to leave alone.
"""

from __future__ import annotations

import unittest

import numpy as np

from . import common as C


def build():
    from ..tdmpc2 import agent as build_agent
    from ..tdmpc2 import data as demo_data

    source = C.fake_source()
    cfg = C.tiny_tdmpc2_cfg(action_dim=source.action_dim)
    agent = build_agent.build_agent(cfg)
    buffer = demo_data.DemoBuffer(source, horizon=int(cfg.horizon),
                                  batch_size=int(cfg.batch_size), device="cpu")
    return source, cfg, agent, buffer


class NativeBehaviour(unittest.TestCase):
    def test_no_policy_attached_by_default(self):
        C.require_torch()
        _source, _cfg, agent, _buffer = build()
        self.assertIsNone(agent.latent_policy)

    def test_pi_action_is_the_gaussian_sample(self):
        """The hook has to be the call it replaced, exactly."""
        torch = C.require_torch()
        _source, cfg, agent, _buffer = build()
        z = torch.randn(3, int(cfg.true_latent_dim))

        torch.manual_seed(7)
        hook = agent.pi_action(z, None, site="td_target")
        torch.manual_seed(7)
        native = agent.model.pi(z, None)[1]
        self.assertTrue(torch.allclose(hook, native))

        torch.manual_seed(7)
        hook_mu = agent.pi_action(z, None, site="act", deterministic=True)
        torch.manual_seed(7)
        native_mu = agent.model.pi(z, None)[0]
        self.assertTrue(torch.allclose(hook_mu, native_mu))

    def test_update_trains_the_native_components(self):
        """Stage 1 is upstream's coupled update, and this says which weights move."""
        torch = C.require_torch()
        _source, _cfg, agent, buffer = build()
        before = {name: parameter.detach().clone()
                  for name, parameter in agent.model.named_parameters()}
        metrics = agent.update(buffer)
        for key in ("consistency_loss", "reward_loss", "value_loss", "pi_loss"):
            self.assertIn(key, metrics)

        moved = {name for name, parameter in agent.model.named_parameters()
                 if not torch.allclose(before[name], parameter)}
        for prefix in ("_encoder", "_dynamics", "_reward", "_Qs", "_pi"):
            self.assertTrue(any(name.startswith(prefix) for name in moved),
                            f"{prefix} did not train in the native update")
        # The target critics move by Polyak, not by an optimizer.
        self.assertTrue(any(name.startswith("_target_Qs") for name in moved))

    def test_mpc_is_the_action_selector(self):
        torch = C.require_torch()
        _source, cfg, agent, buffer = build()
        obs = buffer.sample()[0][0][: int(cfg.num_envs)]
        action = agent.act(obs, t0=True, eval_mode=False)
        self.assertEqual(tuple(action.shape),
                         (int(cfg.num_envs), int(cfg.action_dim)))
        self.assertLessEqual(float(action.abs().max()), 1.0 + 1e-6)

    def test_save_holds_only_the_model_when_nothing_is_attached(self):
        torch = C.require_torch()
        import tempfile
        from pathlib import Path

        _source, _cfg, agent, _buffer = build()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent.pt"
            agent.save(path)
            payload = torch.load(path, weights_only=False)
            self.assertEqual(sorted(payload), ["model"])
            agent.load(payload)

    def test_a_policy_serving_no_site_is_the_native_agent(self):
        """`sites: []` has to be indistinguishable from no policy at all."""
        torch = C.require_torch()

        from ..action_space import converter_for
        from ..tdmpc2.policy import LatentPolicy

        source, cfg, agent, _buffer = build()
        actor = C.stub_latent_actor(int(cfg.true_latent_dim), source.action_dim)
        converter = converter_for(source.metadata, action_dim=source.action_dim)
        agent.attach_policy(LatentPolicy(actor, converter, sites=()))

        z = torch.randn(2, int(cfg.true_latent_dim))
        for site in agent.POLICY_SITES:
            torch.manual_seed(3)
            hook = agent.pi_action(z, None, site=site)
            torch.manual_seed(3)
            self.assertTrue(torch.allclose(hook, agent.model.pi(z, None)[1]),
                            f"site {site} was taken over by a policy that "
                            "serves nothing")
        self.assertTrue(agent.latent_policy.needs_gaussian)


class Timeline(unittest.TestCase):
    """The action/reward alignment, against countable demonstrations."""

    def test_action_t_is_taken_at_observation_t(self):
        torch = C.require_torch()

        from ..tdmpc2 import data as demo_data

        source = C.fake_source()
        windows = demo_data.DemoWindows(source, horizon=4, seed=0, stride=1)
        window = [w for w in windows.windows if w.start == 3][0]
        loaded = windows.sampler.load(window)
        batch = {k: np.asarray(v)[None] for k, v in loaded.items()}
        obs, action, reward, _task = demo_data.native_batch(
            batch, source, horizon=4, device="cpu")

        episode = window.episode_id
        # FakeDemos: actions[t] = episode * 1000 + t on every dimension, and
        # rewards[t] = -(episode * 1000 + t).
        for step in range(4):
            expected = float(episode * 1000 + window.start + step)
            self.assertAlmostEqual(float(action[step, 0, 0]), expected, places=3,
                                   msg=f"action at row {step}")
            self.assertAlmostEqual(float(reward[step, 0, 0]), -expected, places=3,
                                   msg=f"reward at row {step}")
        self.assertEqual(tuple(obs.shape)[:2], (5, 1))

    def test_the_layout_reward_is_the_one_that_arrived(self):
        """The two conventions differ by one row, and that is the whole shift."""
        C.require_torch()

        from ..tdmpc2 import data as demo_data

        source = C.fake_source()
        windows = demo_data.DemoWindows(source, horizon=4, seed=0, stride=1)
        window = [w for w in windows.windows if w.start == 5][0]
        loaded = windows.sampler.load(window)
        # layout: reward[t] = r_(t-1); TD-MPC2: reward[t] = r_t.
        layout_reward = np.asarray(loaded["reward"])
        expected_previous = -float(window.episode_id * 1000 + window.start - 1)
        self.assertAlmostEqual(float(layout_reward[0]), expected_previous, places=3)

        batch = {k: np.asarray(v)[None] for k, v in loaded.items()}
        _obs, _action, reward, _task = demo_data.native_batch(
            batch, source, horizon=4, device="cpu")
        self.assertAlmostEqual(
            float(reward[0, 0, 0]),
            -float(window.episode_id * 1000 + window.start), places=3)

    def test_windows_never_cross_an_episode(self):
        C.require_torch()

        from ..tdmpc2 import data as demo_data

        source = C.fake_source()
        windows = demo_data.DemoWindows(source, horizon=4, seed=0, stride=1)
        for window in windows.windows:
            self.assertLessEqual(window.stop, source.dataset.episodes[0].steps)
            self.assertEqual(window.pad, 0)

    def test_native_windows_are_unpadded(self):
        """`TDMPC2.update` has no mask, so a padded window would be trained on."""
        C.require_torch()

        from ..tdmpc2 import data as demo_data

        source = C.fake_source(steps=10)
        windows = demo_data.DemoWindows(source, horizon=4, seed=0, stride=1)
        self.assertTrue(all(w.pad == 0 for w in windows.windows))
        self.assertLess(len(windows.windows), len(windows.sampler.windows))


class Sizing(unittest.TestCase):
    """Measured counts, and the decision they support."""

    def test_components_are_reported_separately_and_once(self):
        torch = C.require_torch()

        from ..tdmpc2 import agent as build_agent

        source, cfg, agent, _buffer = build()
        report = build_agent.parameters(agent)
        names = [c["name"] for c in report["components"]]
        for wanted in ("encoder", "dynamics", "reward", "policy_prior(gaussian)",
                       "critic(Q ensemble)", "critic_target"):
            self.assertIn(wanted, names)
        by_name = {c["name"]: c for c in report["components"]}
        # Nothing is left over: the catch-all finds no unattributed tensors.
        self.assertEqual(by_name["world_model(rest)"]["unique"], 0)
        # The distinct total is the model's own count of distinct tensors.
        distinct = {id(p): p.numel() for p in agent.model.parameters()}
        self.assertEqual(report["total"], sum(distinct.values()))
        # Target critics are a deep copy: real memory, and not trainable.
        self.assertEqual(by_name["critic_target"]["unique"],
                         by_name["critic(Q ensemble)"]["unique"])
        self.assertEqual(by_name["critic_target"]["trainable"], 0)

    def test_preset_5_is_measured_under_the_budget(self):
        """The sizing decision, as a measurement rather than a claim.

        Upstream's preset 5 is what this project's own TD-MPC2 run script
        uses. Measured at the integration's observation contract it is far
        below the 50M guideline, so nothing is resized: picking a larger
        preset to approach the budget would be choosing an architecture by its
        parameter count.
        """
        torch = C.require_torch()

        from ..params import render
        from ..tdmpc2 import agent as build_agent
        from ..tdmpc2.config import build_cfg

        cfg = build_cfg({"model_size": 5, "obs": "rgb", "include_state": False,
                         "render_size": 64, "device": "cpu",
                         "num_cameras": 2},
                        obs_shape={"rgb": (6, 64, 64)}, action_dim=8,
                        episode_length=150)
        agent = build_agent.build_agent(cfg)
        report = build_agent.parameters(agent)
        self.assertLess(report["total"], 50_000_000,
                        render(report, title="preset 5"))
        self.assertGreater(report["total"], 1_000_000)


if __name__ == "__main__":
    unittest.main()
