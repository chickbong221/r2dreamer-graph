"""SOLD with the integration disabled, and the sizing measurement.

Everything here runs against upstream's own modules: SAVi, the OCVP dynamics,
the ALiBi predictors. If any of it fails, something that was supposed to be
left alone was not.
"""

from __future__ import annotations

import unittest

import numpy as np

from . import common as C


def world_config():
    from ..sold.config import native_defaults

    return native_defaults()


def tiny_world():
    """The same architecture family at a width that builds on CPU quickly.

    The slot count, the image size and the layer counts are what get made
    small; the classes, the masks and the losses are upstream's.
    """
    world = world_config()
    world["imagination_horizon"] = 4
    world["num_context"] = 2
    world["dynamics_predictor"] = {**world["dynamics_predictor"],
                                   "token_dim": 32, "hidden_dim": 64,
                                   "num_layers": 1, "num_heads": 2}
    for head in ("actor", "critic", "reward_predictor"):
        world[head] = {**world[head], "token_dim": 32, "hidden_dim": 64,
                       "num_layers": 1, "num_heads": 2}
    spec = {k: dict(v) if isinstance(v, dict) else v
            for k, v in world["autoencoder_spec"].items()}
    spec["corrector"] = {**spec["corrector"], "num_slots": 3, "slot_dim": 16,
                         "feature_dim": 16, "hidden_dim": 16}
    # Strides stay 1, as upstream's do: SAVi's positional embedding is built
    # for the *image* grid and added to the encoder's feature map, so a
    # downsampling stride makes the two different sizes.
    spec["encoder"] = {**spec["encoder"], "num_channels": [8, 8],
                       "kernel_sizes": [5, 5], "strides": [1, 1],
                       "feature_dim": 16}
    spec["decoder"] = {**spec["decoder"], "num_channels": [8, 8],
                       "kernel_sizes": [5, 5], "strides": [1, 1],
                       "in_channels": 16}
    spec["initializer"] = {**spec["initializer"], "num_slots": 3,
                           "slot_dim": 16}
    spec["predictor"] = {**spec["predictor"], "slot_dim": 16}
    world["autoencoder_spec"] = spec
    return world


def build_parts(*, image=16, action_dim=7, max_episode_steps=24):
    from ..sold import model as build

    return build.components(tiny_world(), image_size=(image, image),
                            action_dim=action_dim,
                            max_episode_steps=max_episode_steps)


def sold_source(**kwargs):
    from ..observations import contract_for
    from ..sold.data import SoldDemoSource

    demos = C.FakeDemos(**kwargs)
    images = contract_for(demos.metadata, backend="sold", size=(16, 16),
                          max_cameras=1)
    return SoldDemoSource(dataset=demos, images=images,
                          action_dim=demos.action_dim,
                          episode_length=demos.episodes[0].steps)


class Components(unittest.TestCase):
    def test_the_config_comes_from_solds_own_files(self):
        world = world_config()
        self.assertEqual(world["actor_gradients"], "dynamics")
        self.assertEqual(world["num_context"], 3)
        self.assertEqual(world["imagination_horizon"], 15)
        self.assertEqual(world["discount_factor"], 0.96)
        self.assertEqual(world["return_lambda"], 0.95)
        self.assertEqual(world["critic_ema_decay"], 0.98)
        self.assertFalse(world["finetune_autoencoder"])
        # ${..corrector.slot_dim} style interpolations are resolved.
        spec = world["autoencoder_spec"]
        self.assertEqual(spec["corrector"]["feature_dim"],
                         spec["encoder"]["feature_dim"])
        self.assertEqual(spec["decoder"]["in_channels"],
                         spec["corrector"]["slot_dim"])
        self.assertEqual(spec["initializer"]["num_slots"],
                         spec["corrector"]["num_slots"])

    def test_teacher_forcing_stays_off(self):
        """It is future-observation leakage in its batched form."""
        from ..sold import model as build

        world = tiny_world()
        self.assertFalse(world["dynamics_predictor"]["teacher_forcing"])
        world["dynamics_predictor"] = {**world["dynamics_predictor"],
                                       "teacher_forcing": True}
        with self.assertRaises(SystemExit) as raised:
            build.build_dynamics(world, num_slots=3, slot_dim=16, action_dim=7,
                                 imagination_horizon=4, sequence_length=6)
        self.assertIn("leakage", str(raised.exception))

    def test_everything_builds_and_runs(self):
        torch = C.require_torch()
        parts = build_parts()
        images = torch.rand(2, 6, 3, 16, 16)
        actions = torch.zeros(2, 6, 7)
        slots = parts["autoencoder"].encode(images, actions[:, 1:])
        self.assertEqual(tuple(slots.shape), (2, 6, 3, 16))
        decoded = parts["autoencoder"].decode(slots)
        self.assertEqual(tuple(decoded["reconstructions"].shape),
                         (2, 6, 3, 16, 16))
        predicted = parts["dynamics"].predict_slots(
            slots, actions[:, 1:], steps=2, num_context=2)
        self.assertEqual(tuple(predicted.shape), (2, 2, 3, 16))
        self.assertEqual(tuple(parts["reward"](slots).mean.shape), (2, 6, 1))
        self.assertEqual(tuple(parts["actor"](slots).sample().shape), (2, 6, 7))


class Losses(unittest.TestCase):
    def test_the_reconstruction_loss_matches_upstreams(self):
        """The inlined objective and ``AutoencoderModule``'s are one number."""
        torch = C.require_torch()

        from ..sold import stages

        parts = build_parts()
        images = torch.rand(2, 5, 3, 16, 16)
        actions = torch.zeros(2, 5, 7)
        torch.manual_seed(0)
        mine = stages.reconstruction_loss(parts["autoencoder"], images, actions)
        try:
            torch.manual_seed(0)
            theirs = stages.native_reconstruction_loss(
                parts["autoencoder"], images, actions)
        except Exception as exc:                           # noqa: BLE001
            raise unittest.SkipTest(
                f"upstream's AutoencoderModule needs Lightning: {exc}")
        self.assertAlmostEqual(float(mine["reconstruction_loss"]),
                               float(theirs["reconstruction_loss"]), places=6)

    def test_the_native_dynamics_and_reward_losses_run(self):
        torch = C.require_torch()

        from ..sold import stages
        from ..vendor import SOLD

        try:
            SOLDModule = SOLD.get("train_sold", "SOLDModule")
        except Exception as exc:                           # noqa: BLE001
            raise unittest.SkipTest(
                f"train_sold needs Lightning, gym and Hydra: {exc}")

        parts = build_parts()
        holder = stages.Holder(
            autoencoder=parts["autoencoder"], dynamics_predictor=parts["dynamics"],
            reward_predictor=parts["reward"], imagination_horizon=4,
            min_num_context=2, max_num_context=2)
        images = torch.rand(2, 6, 3, 16, 16)
        actions = torch.zeros(2, 6, 7)
        slots = parts["autoencoder"].encode(images, actions[:, 1:]).detach()
        out = SOLDModule.compute_dynamics_loss(holder, images, slots, actions)
        self.assertIn("dynamics_loss", out)
        rewards = torch.randn(2, 6)
        rewards[:, 0] = float("nan")
        reward = SOLDModule.compute_reward_loss(
            holder, images, images, slots, rewards)
        self.assertIn("reward_loss", reward)
        self.assertTrue(np.isfinite(float(reward["reward_loss"])))


class Causality(unittest.TestCase):
    def test_the_predictors_cannot_see_the_future(self):
        """The ALiBi mask is causal; the cls token at row t proves it."""
        torch = C.require_torch()
        parts = build_parts()
        slots = torch.randn(2, 6, 3, 16)
        first = parts["reward"](slots).mean

        scrambled = slots.clone()
        scrambled[:, 3:] = torch.randn_like(scrambled[:, 3:])
        second = parts["reward"](scrambled).mean

        self.assertTrue(torch.allclose(first[:, :3], second[:, :3], atol=1e-5),
                        "rows 0..2 changed when only rows 3.. were altered")
        self.assertFalse(torch.allclose(first[:, 3:], second[:, 3:]))

    def test_savi_encodes_causally(self):
        torch = C.require_torch()
        parts = build_parts()
        images = torch.rand(2, 6, 3, 16, 16)
        actions = torch.zeros(2, 6, 7)
        first = parts["autoencoder"].encode(images, actions[:, 1:])

        scrambled = images.clone()
        scrambled[:, 4:] = torch.rand_like(scrambled[:, 4:])
        second = parts["autoencoder"].encode(scrambled, actions[:, 1:])
        self.assertTrue(torch.allclose(first[:, :4], second[:, :4], atol=1e-5))
        self.assertFalse(torch.allclose(first[:, 4:], second[:, 4:]))


class Timeline(unittest.TestCase):
    """SOLD's row convention, against countable demonstrations."""

    def test_action_and_reward_are_the_previous_ones(self):
        torch = C.require_torch()

        from ..sold import data as sold_data

        source = sold_source()
        windows = sold_data.SoldWindows(source, length=5, seed=0, stride=1)
        window = [w for w in windows.windows if w.start == 4][0]
        loaded = windows.sampler.load(window)
        batch = {k: np.asarray(v)[None] for k, v in loaded.items()}
        out = sold_data.sold_batch(batch, source, device="cpu")

        episode, start = window.episode_id, window.start
        # action[t] is a_(t-1) and reward[t] is r_(t-1).
        for step in range(1, 5):
            expected = float(episode * 1000 + start + step - 1)
            self.assertAlmostEqual(float(out["action"][0, step, 0]), expected,
                                   places=3, msg=f"action at row {step}")
            self.assertAlmostEqual(float(out["reward"][0, step]), -expected,
                                   places=3, msg=f"reward at row {step}")
        # Mid-episode, row 0 does have a previous action and reward.
        self.assertAlmostEqual(float(out["reward"][0, 0]),
                               -float(episode * 1000 + start - 1), places=3)

    def test_an_episode_start_carries_nan_not_zero(self):
        """``compute_reward_loss`` masks with ``isnan``; a zero would train."""
        torch = C.require_torch()

        from ..sold import data as sold_data

        source = sold_source()
        windows = sold_data.SoldWindows(source, length=5, seed=0, stride=1)
        window = [w for w in windows.windows if w.start == 0][0]
        loaded = windows.sampler.load(window)
        batch = {k: np.asarray(v)[None] for k, v in loaded.items()}
        out = sold_data.sold_batch(batch, source, device="cpu")
        self.assertTrue(bool(torch.isnan(out["reward"][0, 0])))
        self.assertTrue(bool(torch.isnan(out["action"][0, 0]).all()))
        self.assertFalse(bool(torch.isnan(out["reward"][0, 1:]).any()))

    def test_images_reach_the_model_as_bytes(self):
        torch = C.require_torch()

        from ..sold import data as sold_data

        source = sold_source()
        windows = sold_data.SoldWindows(source, length=5, seed=0, stride=1)
        batch = windows.raw_batch(2)
        out = sold_data.sold_batch(batch, source, device="cpu")
        self.assertEqual(out["obs"].dtype, torch.uint8)
        self.assertEqual(tuple(out["obs"].shape), (2, 6, 3, 16, 16))


class ReplayResume(unittest.TestCase):
    """A focused regression on the vendored replay's resume path.

    ``OnlineModule.on_load_checkpoint`` called ``load_from_files``;
    ``RingBufferDataset`` defines ``load_from_disk_storage`` and nothing else.
    Resuming a run with ``save_replay_buffer: True`` therefore raised
    AttributeError from inside the load hook, after Lightning had restored the
    weights -- so the failure named neither the buffer nor the resume. Stage 3
    writes exactly those checkpoints, which is why it is fixed here.
    """

    def test_the_replay_exposes_the_method_the_resume_path_calls(self):
        import inspect

        from ..vendor import SOLD

        with SOLD.active():
            try:
                from datasets.ring_buffer import RingBufferDataset
            except Exception as exc:                       # noqa: BLE001
                raise unittest.SkipTest(f"ring_buffer needs termcolor/tqdm: {exc}")
            from utils import training as training_source

        self.assertTrue(hasattr(RingBufferDataset, "load_from_disk_storage"))
        source = inspect.getsource(training_source.OnlineModule.on_load_checkpoint)
        self.assertIn("load_from_disk_storage", source)
        self.assertNotIn("load_from_files", source)


class Sizing(unittest.TestCase):
    def test_components_are_reported_separately_and_once(self):
        torch = C.require_torch()

        from ..sold import model as build

        parts = build_parts()
        report = build.parameters(parts)
        names = [c["name"] for c in report["components"]]
        for wanted in ("autoencoder.encoder", "autoencoder.decoder",
                       "autoencoder.corrector", "dynamics", "reward",
                       "policy_prior(gaussian actor)", "critic",
                       "critic_target"):
            self.assertIn(wanted, names)
        by_name = {c["name"]: c for c in report["components"]}
        self.assertEqual(by_name["autoencoder(rest)"]["unique"], 0)
        self.assertEqual(by_name["critic_target"]["unique"],
                         by_name["critic"]["unique"])

    def test_the_shipped_config_is_measured_under_the_budget(self):
        """The sizing decision, as a measurement rather than a claim."""
        torch = C.require_torch()

        from ..params import render
        from ..sold import model as build

        parts = build.components(world_config(), image_size=(64, 64),
                                 action_dim=8, max_episode_steps=150)
        report = build.parameters(parts)
        self.assertLess(report["total"], 50_000_000,
                        render(report, title="sold/configs/train_sold.yaml"))
        self.assertGreater(report["total"], 1_000_000)


if __name__ == "__main__":
    unittest.main()
