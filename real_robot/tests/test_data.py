"""Episode selection, packed-episode windows in the RSSM convention, and the input transforms."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from scenegraph.adapters.graph_pack import GRAPH_KEYS

from ..common import parse_episodes
from ..data.episode_dataset import ActionTransform, BuiltEpisodeStore, resize_image, state_features, transform_boxes
from ..data.manifest import DatasetManifest
from ..data.selection import make_selection, selection_problems, spread_by_length
from ..data.sequence_dataset import SequenceSampler, episode_window, full_episode, last_observation
from ..graphs.pack import empty_frame
from . import synthetic as syn


def recorded_lengths(n: int = 150, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {e: int(v) for e, v in enumerate(rng.integers(203, 384, size=n))}


class Selection(unittest.TestCase):
    def setUp(self):
        self.lengths = recorded_lengths()
        self.episodes = sorted(self.lengths)

    def test_every_episode_trains_exactly_once(self):
        data = make_selection(self.episodes, self.lengths, 10, "v1")
        self.assertEqual(data["training"], self.episodes)
        self.assertEqual(len(set(data["training"])), len(data["training"]))
        self.assertEqual(selection_problems(data, self.episodes), [])

    def test_diagnostic_episodes_are_training_episodes_spread_across_lengths(self):
        data = make_selection(self.episodes, self.lengths, 10, "v1")
        diagnostic = data["diagnostic"]
        self.assertEqual(len(diagnostic), 10)
        self.assertTrue(set(diagnostic) <= set(data["training"]))
        self.assertTrue(data["diagnostic_in_training"])
        ordered = sorted(self.episodes, key=lambda e: (self.lengths[e], e))
        self.assertIn(ordered[0], diagnostic)
        self.assertIn(ordered[-1], diagnostic)
        ranks = sorted(ordered.index(e) for e in diagnostic)
        self.assertGreater(min(np.diff(ranks)), 5)          # spread out, not clustered

    def test_annotated_outcomes_are_all_represented(self):
        outcomes = {e: "success" for e in self.episodes}
        failures = [7, 58, 131]
        for e in failures:
            outcomes[e] = "failure"
        data = make_selection(self.episodes, self.lengths, 10, "v1", outcomes=outcomes,
                              outcome_source="annotations/full_episode")
        chosen = set(data["diagnostic"])
        self.assertEqual(len(chosen), 10)
        self.assertTrue(chosen & set(failures))
        self.assertTrue(chosen - set(failures))
        ordered = sorted(self.episodes, key=lambda e: (self.lengths[e], e))
        self.assertIn(ordered[0], chosen)
        self.assertIn(ordered[-1], chosen)
        self.assertTrue(data["rule"]["outcomes_used"])

    def test_deterministic(self):
        a = make_selection(self.episodes, self.lengths, 10, "v1")
        b = make_selection(self.episodes, self.lengths, 10, "v1")
        self.assertEqual(a["diagnostic"], b["diagnostic"])
        self.assertEqual(spread_by_length(self.episodes, self.lengths, 10), spread_by_length(self.episodes,
                                                                                            self.lengths, 10))

    def test_problems_are_named(self):
        data = make_selection(self.episodes, self.lengths, 10, "v1")
        broken = {**data, "diagnostic": data["diagnostic"] + [999]}
        self.assertTrue(any("not training episodes" in p for p in selection_problems(broken)))
        repeated = {**data, "training": data["training"] + [3]}
        self.assertTrue(any("more than once" in p for p in selection_problems(repeated)))
        missing = {**data, "training": data["training"][:-1],
                   "diagnostic": [e for e in data["diagnostic"] if e != data["training"][-1]]}
        self.assertTrue(any("missing from training" in p for p in selection_problems(missing, self.episodes)))

    def test_split_names_are_refused(self):
        selections = {"training": self.episodes, "diagnostic": [1, 2]}
        for name in ("train", "val", "test", "val:3"):
            with self.assertRaises(KeyError) as caught:
                parse_episodes(name, selections, self.episodes)
            self.assertIn("no train/val/test splits", str(caught.exception))
        self.assertEqual(parse_episodes("training", selections, self.episodes), self.episodes)
        self.assertEqual(parse_episodes("diagnostic:1", selections, self.episodes), [1])


def packed_episode(n: int, completion: int = -1, action_dim: int = 7, state_dim: int = 16, size: int = 8):
    spec = syn.graph_spec()
    graph = empty_frame(spec)
    arrays = {
        "state": np.arange(n, dtype=np.float32)[:, None].repeat(state_dim, 1),
        "action": (np.arange(n, dtype=np.float32)[:, None] + 1000).repeat(action_dim, 1),
        "reward": np.arange(n, dtype=np.float32) * 0.01 - 1.0,
        "done": np.zeros(n, dtype=bool),
        "transition_valid": np.zeros(n, dtype=bool),
        "task_terminal": np.zeros(n, dtype=bool),
        "obs_valid": np.ones(n, dtype=bool),
        "graph_valid": np.ones(n, dtype=bool),
        **{key: np.repeat(value[None], n, axis=0) for key, value in graph.items()},
    }
    last = completion if completion >= 0 else n - 1
    arrays["obs_valid"][last + 1:] = False
    arrays["transition_valid"][:last] = True
    if completion >= 0:
        arrays["task_terminal"][completion] = True
        arrays["done"][completion - 1] = True
    images = {"image_high": np.arange(n, dtype=np.uint8)[:, None, None, None].repeat(size, 1).repeat(size, 2).repeat(3, 3)}
    images["image_wrist_right"] = images["image_high"].copy()
    return arrays, images


class Windows(unittest.TestCase):
    def setUp(self):
        self.arrays, images = packed_episode(20)
        self.arrays.update(images)
        self.keys = ["image_high", "image_wrist_right"]

    def test_previous_action_and_arrival_reward(self):
        w = episode_window(self.arrays, self.keys, start=5, burn_in=3, length=4)
        frames = w["frame"].tolist()
        self.assertEqual(frames, [2, 3, 4, 5, 6, 7, 8])
        for j, t in enumerate(frames):
            self.assertEqual(w["prev_action"][j, 0], 1000 + t - 1)
            self.assertAlmostEqual(float(w["reward_in"][j]), (t - 1) * 0.01 - 1.0, places=5)
            self.assertEqual(w["state"][j, 0], t)
            self.assertEqual(int(w["image_high"][j, 0, 0, 0]), t)
        self.assertEqual(w["learn"].tolist(), [False] * 3 + [True] * 4)

    def test_episode_start_resets_and_pads_before_it(self):
        w = episode_window(self.arrays, self.keys, start=1, burn_in=3, length=4)
        self.assertEqual(w["frame"].tolist(), [-1, -1, 0, 1, 2, 3, 4])
        self.assertEqual(w["is_first"].tolist(), [False, False, True, False, False, False, False])
        self.assertEqual(w["obs_valid"].tolist(), [False, False] + [True] * 5)
        self.assertFalse(w["reward_in_valid"][2])
        self.assertTrue((w["prev_action"][:3] == 0).all())
        self.assertTrue((w["image_high"][:2] == 0).all())

    def test_nothing_after_a_terminal_frame(self):
        arrays, images = packed_episode(20, completion=12)
        arrays.update(images)
        self.assertEqual(last_observation(arrays), 12)
        w = episode_window(arrays, self.keys, start=10, burn_in=0, length=6)
        self.assertEqual(w["frame"].tolist(), [10, 11, 12, -1, -1, -1])
        self.assertEqual(w["is_terminal"].tolist(), [False, False, True, False, False, False])
        self.assertEqual(w["obs_valid"].tolist(), [True, True, True, False, False, False])
        self.assertTrue(w["reward_in_valid"][2])          # the terminal arrival carries its reward

    def test_final_recorded_frame_is_a_valid_observation(self):
        w = episode_window(self.arrays, self.keys, start=18, burn_in=0, length=2)
        self.assertEqual(w["obs_valid"].tolist(), [True, True])
        self.assertTrue(w["graph_valid"][1])


def write_manifest(root: str, lengths, diagnostic, built=None):
    manifest = DatasetManifest.create(
        root, identity={"x": 1},
        selection={"version": "v1", "training": sorted(lengths), "diagnostic": list(diagnostic)},
        shapes={"state": [16], "action": [7]},
        model_inputs={"cameras": ["high", "wrist_right"]}, graph={"vocab_sizes": {}}, action={})
    for episode in (sorted(lengths) if built is None else built):
        n = lengths[episode]
        arrays, images = packed_episode(n)
        directory = manifest.episode_dir(episode)
        os.makedirs(directory)
        np.savez(os.path.join(directory, "arrays.npz"), **arrays)
        for key, value in images.items():
            np.save(os.path.join(directory, f"{key}.npy"), value)
        manifest.record_episode(episode, {"n_frames": n})
    manifest.save()
    return manifest


class Store(unittest.TestCase):
    def test_sampler_over_a_packed_directory(self):
        with tempfile.TemporaryDirectory() as root:
            write_manifest(root, {0: 20, 1: 30}, diagnostic=[1])
            store = BuiltEpisodeStore(root)
            self.assertEqual(store.episodes("training"), [0, 1])
            self.assertEqual(store.episodes("diagnostic"), [1])
            sampler = SequenceSampler(store, "training", burn_in=4, length=8, batch_size=3, seed=1)
            batch = sampler.sample()
            self.assertEqual(batch["image_high"].shape, (3, 12, 8, 8, 3))
            self.assertEqual(batch["prev_action"].shape, (3, 12, 7))
            for key in GRAPH_KEYS:
                self.assertIn(key, batch)
            whole = full_episode(store, 1)
            self.assertEqual(whole["state"].shape, (1, 30, 16))
            self.assertTrue(whole["is_first"][0, 0])

    def test_windows_are_contiguous_frames_of_one_episode(self):
        with tempfile.TemporaryDirectory() as root:
            write_manifest(root, {0: 20, 1: 30, 2: 25}, diagnostic=[2])
            store = BuiltEpisodeStore(root)
            sampler = SequenceSampler(store, "training", burn_in=6, length=10, batch_size=16, seed=3,
                                      begin_fraction=0.2)
            batch = sampler.sample()
            for row in range(16):
                episode = set(batch["episode"][row].tolist())
                self.assertEqual(len(episode), 1)
                frames = batch["frame"][row]
                real = frames[frames >= 0]
                self.assertTrue(np.all(np.diff(real) == 1), frames)
                # images and state belong to those same frames, from that one episode
                self.assertTrue(np.all(batch["state"][row][frames >= 0][:, 0] == real))

    def test_split_names_are_not_selections(self):
        with tempfile.TemporaryDirectory() as root:
            write_manifest(root, {0: 20, 1: 30}, diagnostic=[1])
            store = BuiltEpisodeStore(root)
            for name in ("train", "val", "test"):
                with self.assertRaises(KeyError):
                    store.episodes(name)

    def test_coverage_reports_unpacked_training_episodes(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = write_manifest(root, {0: 20, 1: 30, 2: 25}, diagnostic=[2], built=[0, 1])
            coverage = manifest.coverage()
            self.assertFalse(coverage["complete"])
            self.assertEqual(coverage["missing"], [2])
            self.assertEqual(coverage["diagnostic_missing"], [2])
            with self.assertRaises(SystemExit):
                manifest.require_complete(False, "test")
            self.assertEqual(manifest.require_complete(True, "test")["built"], 2)

    def test_a_diagnostic_episode_outside_training_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                DatasetManifest.create(root, identity={}, selection={"version": "v1", "training": [0, 1],
                                                                      "diagnostic": [5]},
                                       shapes={}, model_inputs={}, graph={}, action={})


class Transforms(unittest.TestCase):
    def test_euler_angles_enter_as_sine_and_cosine(self):
        state = np.zeros((2, 13))
        state[0, 10] = np.pi - 1e-3
        state[1, 10] = -np.pi + 1e-3
        features = state_features(state, None, None, ["eef_rot_sincos"])
        self.assertLess(np.abs(features[0] - features[1]).max(), 1e-2)

    def test_actions_normalise_and_invert(self):
        spec = {"command_indices": [0, 6], "command_names": ["waist", "gripper"],
                "representation": "absolute_joint_position", "units": {"waist": "rad", "gripper": "rad"},
                "normalization": {"low": [-1.0, 0.6], "high": [1.0, 1.6], "margin": 0.05}}
        transform = ActionTransform.from_spec(spec)
        raw = np.zeros((3, 13))
        raw[:, 0] = [-1.0, 0.0, 1.0]
        raw[:, 6] = [0.6, 1.1, 1.6]
        normed = transform.normalize(raw)
        self.assertTrue(np.all(np.abs(normed) < 1.0))
        np.testing.assert_allclose(transform.denormalize(normed), raw[:, [0, 6]], atol=1e-6)

    def test_letterbox_boxes_follow_the_image(self):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        image[12:24, 16:32] = 255
        box = np.array([[16 / 64, 32 / 64, 12 / 48, 24 / 48]], dtype=np.float32)
        out = resize_image(image, (32, 32), "letterbox")
        moved = transform_boxes(box, (48, 64), (32, 32), "letterbox")[0]
        ys, xs = np.nonzero(out[..., 0] > 127)
        self.assertAlmostEqual(xs.min() / 32, moved[0], delta=1.5 / 32)
        self.assertAlmostEqual((ys.max() + 1) / 32, moved[3], delta=1.5 / 32)
        np.testing.assert_array_equal(transform_boxes(box, (48, 64), (32, 32), "stretch"), box)


if __name__ == "__main__":
    unittest.main()
