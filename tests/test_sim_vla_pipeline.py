"""Phase 1: the audit, the sequence loader, normalization, and the graph switch.

The acceptance test the spec names is the last class here: with everything else
fixed, corrupting or deleting the stored graphs must leave a baseline batch
identical. It is asserted by comparison of the actual arrays rather than by
inspecting which code path ran, because what matters is that no graph value
reaches the model, not which branch intended not to fetch it.

A synthetic dataset stands in for the collected one. It is written through the
real :class:`~sim_vla.data.writer.DatasetWriter`, so the layout, the metadata
and the T/T+1 contract under test are the ones the collector produces.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scenegraph.adapters.graph_pack import GRAPH_KEYS
from sim_vla.config import check_dataset_compatibility, deep_merge, load_config
from sim_vla.data.audit import audit_dataset
from sim_vla.data.dataset import DemoDataset, batch_fields
from sim_vla.data.normalization import Normalizer, fit_normalizer
from sim_vla.data.sequences import SequenceSampler, load_window, plan_windows
from sim_vla.data.writer import DatasetWriter, field_kinds

H = W = 4
PROPRIO = 25
ACTION = 8


def make_dataset(path, episodes=((30, 24), (18, 14), (50, 45)), pad=5,
                 cameras=("image_base",), flicker=None):
    """Write a dataset shaped like a collected one.

    Each entry is ``(recorded_steps, settled)``: success turns on at
    ``settled`` and stays on, so ``terminated`` does too -- which is what the
    real tasks do, and what puts the collector's pad after a terminal step.
    """
    metadata = {
        "env_id": "PickCube-v1",
        "camera_keys": {c.replace("image_", "") + "_camera": c for c in cameras},
        "image_size": [H, W],
        "proprio_names": [f"p{i}" for i in range(PROPRIO)],
        "controller": {"control_mode": "pd_joint_pos", "action_dim": ACTION,
                       "control_freq": 20},
        "graph": {"relation_tokens": {"pad": 0, "grasp": 1}, "n_max": 8,
                  "e_max": 168, "whitelist_digest": "a" * 40,
                  "thresholds_digest": "b" * 40},
        "budget": {"max_steps_to_success": 150, "pad_after_success": pad},
        "field_kinds": field_kinds(cameras, GRAPH_KEYS),
        "versions": {"repo_revision": "abc123"},
    }
    rng = np.random.default_rng(0)
    with DatasetWriter(path, metadata) as writer:
        for index, (steps, settled) in enumerate(episodes):
            success = np.zeros(steps, bool)
            success[settled:] = True
            if flicker:
                success[flicker[0]:flicker[1]] = True
            writer.add(
                images={c: rng.integers(0, 255, (steps + 1, H, W, 3), dtype=np.uint8)
                        for c in cameras},
                proprio=rng.normal(size=(steps + 1, PROPRIO)).astype(np.float32),
                graphs={k: np.full((steps + 1, 8), index + 1, np.uint8)
                        for k in GRAPH_KEYS},
                actions=rng.uniform(-1, 1, (steps, ACTION)).astype(np.float32),
                rewards=np.arange(steps, dtype=np.float32),
                terminated=success.copy(),          # ManiSkill: terminated = success
                truncated=np.zeros(steps, bool),
                success=success,
                env_states={"actors": {"cube": np.zeros((steps + 1, 13))}},
                privileged={"extra.goal_pos": np.zeros((steps + 1, 3))},
                info={"seed": 100 + index, "end_reason": "success_cut",
                      "settled_steps": settled + 1},
            )
    return metadata


class TestAllowlist(unittest.TestCase):
    def test_graph_fields_only_for_the_graph_arm(self):
        meta = {"camera_keys": {"base_camera": "image_base"}}
        baseline = batch_fields(meta, graph_enabled=False)
        graph = batch_fields(meta, graph_enabled=True)
        self.assertEqual(baseline.graph, ())
        self.assertEqual(graph.graph, tuple(GRAPH_KEYS))
        # Turning the graph off is not a reason to take a camera away.
        self.assertEqual(baseline.images, graph.images)
        self.assertEqual(baseline.observations, graph.observations)
        # Diagnostics are in neither arm's batch.
        for name in ("env_states", "privileged"):
            self.assertNotIn(name, baseline.all)
            self.assertNotIn(name, graph.all)


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "demos.h5"
        self.meta = make_dataset(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_policy_matches_the_online_env(self):
        """ignore_terminations=True, so every recorded step is usable.

        The online env is built with ignore_terminations=True, so nothing
        terminates at rollout time and the collector's post-success steps are
        ordinary steps rather than a tail after an ending.
        """
        with DemoDataset(self.path, graph_enabled=False) as data:
            self.assertEqual([ref.steps for ref in data.episodes], [30, 18, 50])
            self.assertFalse(any(ref.terminal for ref in data.episodes))
            ref = data.episodes[0]
            window = plan_windows(ref, length=ref.steps, burn_in=0)[0]
            batch = load_window(data, ref, window)
            # The recording says terminated; the trainer must not.
            self.assertTrue(batch["terminated"].any())
            self.assertFalse(batch["is_terminal"].any())
            self.assertTrue(batch["is_last"][ref.steps - 1])

    def test_honouring_terminations_cuts_at_the_first_one(self):
        with DemoDataset(self.path, graph_enabled=False,
                         ignore_terminations=False) as data:
            # settled 24 -> terminal at index 24 -> 25 usable actions.
            self.assertEqual([ref.steps for ref in data.episodes], [25, 15, 46])
            self.assertTrue(all(ref.terminal for ref in data.episodes))
            ref = data.episodes[0]
            window = plan_windows(ref, length=ref.steps, burn_in=0)[0]
            batch = load_window(data, ref, window)
            self.assertTrue(batch["is_terminal"][ref.steps - 1])

    def test_first_success_is_not_settled_success(self):
        """A flickering flag makes the two different numbers."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "flicker.h5"
            make_dataset(path, episodes=((30, 24),), flicker=(5, 9))
            with DemoDataset(path, graph_enabled=False) as data:
                ref = data.episodes[0]
                self.assertEqual(ref.first_success, 6)     # the flicker
                self.assertEqual(ref.settled_success, 25)  # where it holds
                self.assertNotEqual(ref.first_success, ref.settled_success)

    def test_observations_are_one_longer_than_transitions(self):
        with DemoDataset(self.path, graph_enabled=True) as data:
            ref = data.episodes[0]
            block = data.read(ref, 0, ref.steps)
            self.assertEqual(block["actions"].shape[0], ref.steps)
            self.assertEqual(block["rewards"].shape[0], ref.steps)
            self.assertEqual(block["image_base"].shape[0], ref.steps + 1)
            self.assertEqual(block["proprio"].shape[0], ref.steps + 1)
            self.assertEqual(block["graph_node_ent"].shape[0], ref.steps + 1)

    def test_reads_stay_inside_the_episode(self):
        with DemoDataset(self.path, graph_enabled=False) as data:
            ref = data.episodes[0]
            with self.assertRaises(IndexError):
                data.read(ref, 0, ref.steps + 1)

    def test_diagnostics_are_a_separate_call(self):
        with DemoDataset(self.path, graph_enabled=False) as data:
            ref = data.episodes[0]
            block = data.read(ref, 0, 5)
            self.assertNotIn("env_states", block)
            self.assertNotIn("privileged", block)
            diag = data.diagnostics(ref, 0, 5)
            self.assertEqual(diag["env_states"]["actors"]["cube"].shape[0], 6)


class TestSequences(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "demos.h5"
        make_dataset(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_windows_never_cross_an_episode(self):
        with DemoDataset(self.path, graph_enabled=False) as data:
            for ref in data.episodes:
                for window in plan_windows(ref, length=10, burn_in=4):
                    self.assertGreaterEqual(window.start, 0)
                    self.assertLessEqual(window.stop, ref.steps)

    def test_burn_in_is_clamped_at_the_episode_start(self):
        with DemoDataset(self.path, graph_enabled=False) as data:
            ref = data.episodes[0]
            windows = plan_windows(ref, length=10, burn_in=4)
            # Nothing precedes the first window, so it has no burn-in and is
            # the one that carries is_first.
            self.assertEqual(windows[0].burn_in, 0)
            self.assertEqual(windows[0].start, 0)
            self.assertEqual(windows[1].burn_in, 4)
            batch = load_window(data, ref, windows[0])
            self.assertTrue(batch["is_first"][0])
            later = load_window(data, ref, windows[1])
            self.assertFalse(later["is_first"].any())

    def test_masks_exclude_burn_in_and_padding(self):
        with DemoDataset(self.path, graph_enabled=False) as data:
            ref = data.episodes[1]                       # 15 usable steps
            windows = plan_windows(ref, length=10, burn_in=4)
            last = windows[-1]
            batch = load_window(data, ref, last)
            self.assertEqual(batch["actions"].shape[0], batch["valid"].shape[0])
            self.assertEqual(batch["image_base"].shape[0],
                             batch["valid"].shape[0] + 1)
            # Padding is outside valid; burn-in is valid but unscored.
            self.assertEqual(int(batch["valid"].sum()), last.stop - last.start)
            self.assertEqual(int(batch["loss_mask"].sum()),
                             last.stop - last.start - last.burn_in)
            self.assertFalse(batch["loss_mask"][: last.burn_in].any())

    def test_alignment_of_actions_rewards_and_observations(self):
        """o_t -> a_t -> r_t, o_{t+1}, with the window's own offset applied."""
        with DemoDataset(self.path, graph_enabled=False) as data:
            ref = data.episodes[2]
            window = plan_windows(ref, length=8, burn_in=2)[2]
            batch = load_window(data, ref, window)
            # rewards were written as arange, so they identify their index.
            expected = np.arange(window.start, window.stop, dtype=np.float32)
            self.assertTrue(np.array_equal(
                batch["rewards"][: window.stop - window.start], expected))

    def test_episode_end_bootstraps_under_the_online_policy(self):
        with DemoDataset(self.path, graph_enabled=False) as data:
            ref = data.episodes[0]
            window = plan_windows(ref, length=ref.steps, burn_in=0)[0]
            batch = load_window(data, ref, window)
            self.assertTrue(batch["is_last"][ref.steps - 1])
            self.assertFalse(batch["is_terminal"].any())

    def test_sampler_is_deterministic_given_a_seed(self):
        with DemoDataset(self.path, graph_enabled=False) as data:
            a = SequenceSampler(data, length=8, burn_in=2, seed=7).batch(4)
            b = SequenceSampler(data, length=8, burn_in=2, seed=7).batch(4)
            for key in a:
                self.assertTrue(np.array_equal(a[key], b[key]), key)


class TestNormalization(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "demos.h5"
        make_dataset(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fit_round_trip(self):
        with DemoDataset(self.path, graph_enabled=False) as data:
            norm = fit_normalizer(data)
            self.assertEqual(len(norm.fields["actions"].mean), ACTION)
            self.assertEqual(len(norm.fields["proprio"].mean), PROPRIO)
            sample = data.read(data.episodes[0], 0, 5)["actions"]
            back = norm.denormalize("actions", norm.normalize("actions", sample))
            np.testing.assert_allclose(back, sample, rtol=1e-4, atol=1e-4)

    def test_both_arms_get_the_same_statistics(self):
        with DemoDataset(self.path, graph_enabled=False) as base, \
                DemoDataset(self.path, graph_enabled=True) as graph:
            a, b = fit_normalizer(base), fit_normalizer(graph)
            self.assertEqual(a.fields["actions"].mean, b.fields["actions"].mean)
            self.assertEqual(a.identity, b.identity)

    def test_a_normalizer_from_another_dataset_is_refused(self):
        with DemoDataset(self.path, graph_enabled=False) as data:
            norm = fit_normalizer(data)
            target = Path(self.tmp.name) / "norm.json"
            norm.save(target)
            Normalizer.load(target, norm.identity)       # matching: fine
            with self.assertRaises(SystemExit):
                Normalizer.load(target, dict(norm.identity, env_id="Other-v1"))


class TestConfig(unittest.TestCase):
    def test_three_arms_resolve(self):
        base = load_config("pickcube", "dreamer")
        graph = load_config("pickcube", "graph")
        progress = load_config("peginsertion", "graph_progress")
        self.assertFalse(base["model"]["graph"]["enabled"])
        self.assertTrue(graph["model"]["graph"]["enabled"])
        self.assertFalse(graph["model"]["progress"]["enabled"])
        self.assertTrue(progress["model"]["progress"]["enabled"])
        # rep_loss stays the same across arms, so the comparison has one
        # difference rather than two.
        self.assertEqual(base["model"]["rep_loss"], graph["model"]["rep_loss"])
        self.assertTrue(
            progress["task"]["dataset"].endswith("PegInsertionSide-v1/demos.h5"))

    def test_progress_without_graph_is_refused(self):
        with self.assertRaises(SystemExit):
            load_config("pickcube", "dreamer",
                        {"model": {"progress": {"enabled": True}}})

    def test_deep_merge_is_per_leaf(self):
        merged = deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}})
        self.assertEqual(merged, {"a": {"x": 1, "y": 3}})

    def test_capacity_mismatch_is_refused(self):
        cfg = load_config("pickcube", "graph")
        check_dataset_compatibility(cfg, {"graph": {
            "relation_tokens": {"pad": 0}, "n_max": 8, "e_max": 168}})
        with self.assertRaises(SystemExit):
            check_dataset_compatibility(cfg, {"graph": {
                "relation_tokens": {"pad": 0}, "n_max": 8, "e_max": 256}})
        # A baseline needs nothing from the graph metadata either way.
        check_dataset_compatibility(load_config("pickcube", "dreamer"), {})


class TestAudit(unittest.TestCase):
    def test_a_good_dataset_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demos.h5"
            make_dataset(path)
            report = audit_dataset(path, graph_required=True)
            self.assertTrue(report.ok, report.render())
            self.assertEqual(report.stats["distinct_seeds"], 3)

    def test_internal_truncation_is_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demos.h5"
            make_dataset(path)
            import h5py

            with h5py.File(path, "r+") as handle:
                data = handle["traj_0"]["truncated"][()]
                data[5] = True                  # a boundary mid-episode
                handle["traj_0"]["truncated"][...] = data
            report = audit_dataset(path)
            self.assertFalse(report.ok)
            self.assertTrue(any("truncation" in f for f in report.failures))

    def test_mismatched_lengths_are_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demos.h5"
            make_dataset(path)
            import h5py

            with h5py.File(path, "r+") as handle:
                del handle["traj_1"]["env_states"]["actors"]["cube"]
                handle["traj_1"]["env_states"]["actors"].create_dataset(
                    "cube", data=np.zeros((2, 13)))
            report = audit_dataset(path)
            self.assertFalse(report.ok)
            self.assertTrue(any("env_states" in f for f in report.failures))

    def test_duplicate_seeds_are_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demos.h5"
            make_dataset(path)
            side = path.with_suffix(".json")
            payload = json.loads(side.read_text())
            payload["episodes"][1]["seed"] = payload["episodes"][0]["seed"]
            side.write_text(json.dumps(payload))
            report = audit_dataset(path)
            self.assertFalse(report.ok)
            self.assertTrue(any("seeds" in f for f in report.failures))


class TestGraphIsolation(unittest.TestCase):
    """The acceptance test: a baseline must not see the graph, at all.

    Asserted by corrupting and then deleting the stored graph arrays and
    comparing the baseline's batches byte for byte. Checking which code path
    ran would be weaker -- what matters is that no graph value reaches the
    model, not that a branch intended not to fetch one.
    """

    def batches(self, path, graph_enabled):
        with DemoDataset(path, graph_enabled=graph_enabled) as data:
            sampler = SequenceSampler(data, length=8, burn_in=2, seed=3)
            return sampler.batch(6)

    def test_corrupting_graphs_leaves_the_baseline_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demos.h5"
            make_dataset(path)
            before = self.batches(path, False)

            import h5py

            with h5py.File(path, "r+") as handle:
                for key in handle:
                    for name in GRAPH_KEYS:
                        node = handle[key]["obs"][name]
                        node[...] = np.full(node.shape, 199, node.dtype)
            after = self.batches(path, False)
            self.assertEqual(sorted(before), sorted(after))
            for name in before:
                self.assertTrue(np.array_equal(before[name], after[name]), name)
            # ... and the graph arm does see the change, or the test above
            # would pass for an arm that reads nothing at all.
            graph_batch = self.batches(path, True)
            self.assertIn("graph_node_ent", graph_batch)
            self.assertTrue((graph_batch["graph_node_ent"] == 199).all())

    def test_the_baseline_runs_with_no_graph_arrays_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demos.h5"
            make_dataset(path)
            before = self.batches(path, False)

            import h5py

            with h5py.File(path, "r+") as handle:
                for key in handle:
                    for name in GRAPH_KEYS:
                        del handle[key]["obs"][name]
            after = self.batches(path, False)
            for name in before:
                self.assertTrue(np.array_equal(before[name], after[name]), name)
            # The graph arm, on the same file, fails rather than training on
            # nothing.
            with self.assertRaises(Exception):
                self.batches(path, True)

    def test_no_graph_key_reaches_a_baseline_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demos.h5"
            make_dataset(path)
            batch = self.batches(path, False)
            for name in GRAPH_KEYS:
                self.assertNotIn(name, batch)


if __name__ == "__main__":
    unittest.main()
