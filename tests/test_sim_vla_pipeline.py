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
            batch = load_window(data, ref, window, ref.steps, 0)
            # is_last marks the final *observation*, one past the last action.
            self.assertFalse(batch["is_terminal"].any())
            self.assertTrue(batch["is_last"][ref.steps])
            self.assertFalse(batch["action_valid"][ref.steps])

    def test_honouring_terminations_cuts_at_the_first_one(self):
        with DemoDataset(self.path, graph_enabled=False,
                         ignore_terminations=False) as data:
            # settled 24 -> terminal at index 24 -> 25 usable actions.
            self.assertEqual([ref.steps for ref in data.episodes], [25, 15, 46])
            self.assertTrue(all(ref.terminal for ref in data.episodes))
            ref = data.episodes[0]
            window = plan_windows(ref, length=ref.steps, burn_in=0)[0]
            batch = load_window(data, ref, window, ref.steps, 0)
            # The terminal transition arrives at the observation after it.
            self.assertTrue(batch["is_terminal"][ref.steps])

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
            batch = load_window(data, ref, last, 10, 4)
            # Every array now has one row per observation, same count.
            rows = batch["valid"].shape[0]
            for key in ("action", "action_target", "reward", "image_base"):
                self.assertEqual(batch[key].shape[0], rows, key)
            # Padding is outside valid; burn-in is valid but unscored.
            self.assertEqual(int(batch["valid"].sum()),
                             last.stop - last.start + 1)
            self.assertFalse(batch["loss_mask"][: last.burn_in].any())

    def test_alignment_of_actions_rewards_and_observations(self):
        """o_t -> a_t -> r_t, o_{t+1}, with the window's own offset applied."""
        with DemoDataset(self.path, graph_enabled=False) as data:
            ref = data.episodes[2]
            window = plan_windows(ref, length=8, burn_in=2)[2]
            batch = load_window(data, ref, window, 8, 2)
            # rewards were written as arange, so they identify their index.
            # r_(t-1) arrives at o_t, so row i holds reward[start + i - 1].
            span = window.stop - window.start
            expected = np.arange(window.start - 1, window.stop - 1,
                                 dtype=np.float32)
            self.assertTrue(np.array_equal(batch["reward"][: span], expected),
                            f"{batch['reward'][:span]} vs {expected}")

    def test_episode_end_bootstraps_under_the_online_policy(self):
        with DemoDataset(self.path, graph_enabled=False) as data:
            ref = data.episodes[0]
            window = plan_windows(ref, length=ref.steps, burn_in=0)[0]
            batch = load_window(data, ref, window, ref.steps, 0)
            self.assertTrue(batch["is_last"][ref.steps])
            self.assertFalse(batch["is_terminal"].any())

    def test_sampler_is_deterministic_given_a_seed(self):
        with DemoDataset(self.path, graph_enabled=False) as data:
            a = SequenceSampler(data, length=8, burn_in=2, seed=7).batch(4)
            b = SequenceSampler(data, length=8, burn_in=2, seed=7).batch(4)
            for key in a:
                self.assertTrue(np.array_equal(a[key], b[key]), key)


class TestCausalAlignment(unittest.TestCase):
    """The posterior at o_t must not contain a_t, which the actor predicts.

    RSSM.obs_step takes *prev_action*. Pairing embed[i] with actions[i] put the
    action taken at o_t into the state the actor is trained to predict that
    same action from -- the target inside its own input.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "demos.h5"
        make_dataset(self.path, episodes=((20, 15),))

    def tearDown(self):
        self.tmp.cleanup()

    def window(self, length=6, burn_in=2, index=0):
        from sim_vla.data.sequences import load_window, plan_windows

        data = DemoDataset(self.path, graph_enabled=False)
        ref = data.episodes[0]
        windows = plan_windows(ref, length=length, burn_in=burn_in)
        return data, ref, windows[index], load_window(
            data, ref, windows[index], length, burn_in)

    def test_action_is_the_previous_action(self):
        """row t holds a_(t-1); the action taken at o_t is action_target."""
        data, ref, window, out = self.window(index=1)
        block = data.read(ref, window.start, window.stop)
        actions = np.asarray(block["actions"])
        before = np.asarray(
            data.read(ref, window.start - 1, window.start)["actions"])[-1]
        # a_(t-1): what preceded the window, then the window's own actions.
        np.testing.assert_allclose(out["action"][0], before)
        np.testing.assert_allclose(out["action"][1], actions[0])
        # a_t: the action taken at this observation.
        np.testing.assert_allclose(out["action_target"][0], actions[0])
        np.testing.assert_allclose(out["action_target"][1], actions[1])
        # ...which is exactly the leak: they must differ.
        self.assertFalse(np.allclose(out["action"][1], out["action_target"][1]))
        data.close()

    def test_a_t_is_absent_from_every_input_row_up_to_t(self):
        """Changing a_t must not change any posterior input at or before t."""
        data, ref, window, out = self.window(index=1)
        target = out["action_target"][3].copy()
        # a_3 appears as an input only from row 4 onward.
        for row in range(4):
            self.assertFalse(np.allclose(out["action"][row], target),
                             f"a_t leaked into the posterior input at row {row}")
        np.testing.assert_allclose(out["action"][4], target)
        data.close()

    def test_reward_belongs_to_the_arriving_transition(self):
        """r_(t-1) arrives at o_t, and there is none at a reset."""
        data, ref, window, out = self.window(index=0)
        # rewards were written as arange, so they identify their own index.
        self.assertFalse(out["reward_valid"][0], "a reset has no incoming reward")
        self.assertTrue(out["reward_valid"][1])
        self.assertAlmostEqual(float(out["reward"][1]), 0.0)   # r_0 arrives at o_1
        self.assertAlmostEqual(float(out["reward"][2]), 1.0)
        data.close()

    def test_final_observation_is_kept_and_has_no_action(self):
        data, ref, window, out = self.window(length=ref_steps_of(self.path),
                                             burn_in=0)
        last = int(out["valid"].sum()) - 1
        self.assertTrue(out["is_last"][last])
        self.assertFalse(out["action_valid"][last],
                         "no action was taken at the final observation")
        data.close()


def ref_steps_of(path):
    with DemoDataset(path, graph_enabled=False) as data:
        return data.episodes[0].steps


class TestWindowLayout(unittest.TestCase):
    """One shape for every window, from either source."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "demos.h5"
        # start, middle and a short final window
        make_dataset(self.path, episodes=((30, 24), (9, 6), (70, 60)))

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_demonstration_window_has_the_same_rows(self):
        from sim_vla.data import layout
        from sim_vla.data.sequences import SequenceSampler

        with DemoDataset(self.path, graph_enabled=False) as data:
            sampler = SequenceSampler(data, length=16, burn_in=4, seed=0)
            expected = layout.rows(16, 4)
            seen = set()
            for window in sampler.windows:
                out = sampler.load(window)
                seen.add(tuple(sorted(
                    {k: v.shape[0] for k, v in out.items()}.values())))
                layout.check(out, 16, 4)
            self.assertEqual(len(seen), 1, f"window shapes differ: {seen}")
            batch = sampler.batch(8)
            self.assertEqual(batch["proprio"].shape[:2], (8, expected))

    def test_demonstration_and_replay_windows_stack(self):
        from sim_vla.data import layout
        from sim_vla.data.replay import (OnlineEpisode, OnlineReplay,
                                         mixed_batch)
        from sim_vla.data.sequences import SequenceSampler

        with DemoDataset(self.path, graph_enabled=False) as data:
            sampler = SequenceSampler(data, length=16, burn_in=4, seed=0)
            replay = OnlineReplay(seed=0)
            for _ in range(6):
                episode = OnlineEpisode()
                for _ in range(25):
                    episode.add_observation({
                        "image_base": np.zeros((H, W, 3), np.uint8),
                        "proprio": np.zeros(PROPRIO, np.float32)})
                for index in range(24):
                    episode.add_transition(np.zeros(ACTION, np.float32),
                                           float(index), False,
                                           index == 23, False)
                replay.add(episode)
            mixed = mixed_batch(sampler, replay, batch=8, length=16, burn_in=4,
                                demo_fraction=0.5)
            for key, value in mixed.items():
                self.assertEqual(value.shape[0], 8, key)
                self.assertEqual(value.shape[1], layout.rows(16, 4), key)
            self.assertIn("action", mixed)
            self.assertIn("action_target", mixed)


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

    def test_camera_count_must_match_the_dataset(self):
        """The packed bbox is (n_max, n_cams, 4); the encoder is 5*n_cams+3.

        The model preset defaults to 2 cameras and PickCube records 1, so
        inheriting the preset would build an encoder of the wrong width against
        arrays of the wrong shape.
        """
        from sim_vla.config import check_dataset_compatibility

        recorded = {"graph": {"relation_tokens": {"pad": 0}, "n_max": 8,
                              "e_max": 168, "n_cams": 1}}
        # 0 means "take the dataset's", so it must not trip the check.
        cfg = load_config("pickcube", "graph")
        self.assertEqual(cfg["model"]["graph"]["n_cams"], 0)
        check_dataset_compatibility(cfg, recorded)
        # An explicit disagreement is refused.
        wrong = load_config("pickcube", "graph",
                            {"model": {"graph": {"n_cams": 2}}})
        with self.assertRaises(SystemExit):
            check_dataset_compatibility(wrong, recorded)
        # An explicit agreement passes.
        right = load_config("pickcube", "graph",
                            {"model": {"graph": {"n_cams": 1}}})
        check_dataset_compatibility(right, recorded)

    def test_device_reaches_every_block(self):
        """Each block carries its own device key, resolved from ${device}.

        Setting only the top level left RSSM building its initial state on the
        default while the batch sat elsewhere -- reported as "expected all
        tensors to be on the same device" from inside obs_step.
        """
        from sim_vla.models.model_config import load_model_config

        cfg = load_model_config(load_config("pickcube", "graph",
                                            {"device": "cpu"}))
        for block in ("rssm", "reward", "cont", "critic", "actor"):
            self.assertEqual(getattr(cfg, block).device, "cpu", block)
        self.assertEqual(cfg.device, "cpu")

    def test_capacity_mismatch_is_refused(self):
        cfg = load_config("pickcube", "graph")
        check_dataset_compatibility(cfg, {"graph": {
            "relation_tokens": {"pad": 0}, "n_max": 8, "e_max": 168}})
        with self.assertRaises(SystemExit):
            check_dataset_compatibility(cfg, {"graph": {
                "relation_tokens": {"pad": 0}, "n_max": 8, "e_max": 256}})
        # A baseline needs nothing from the graph metadata either way.
        check_dataset_compatibility(load_config("pickcube", "dreamer"), {})


class TestModelConfig(unittest.TestCase):
    """The simulator's config, resolved. Torch-free, so it runs anywhere.

    The bug this exists for: ``${env.encoder.cnn_keys}`` was left unresolved,
    matched no observation key, and MultiEncoder raised a bare
    NotImplementedError from the line that discovers it has no encoders.
    Nothing in that traceback mentioned an interpolation.
    """

    ARMS = (("pickcube", "dreamer"), ("pickcube", "graph"),
            ("peginsertion", "graph_progress"))

    def unresolved(self, node, path=""):
        import re as _re

        found = []
        if isinstance(node, dict):
            for key, value in node.items():
                found += self.unresolved(value, f"{path}.{key}" if path else key)
        elif isinstance(node, list):
            for value in node:
                found += self.unresolved(value, path)
        elif isinstance(node, str) and _re.search(r"\$\{", node):
            found.append(f"{path} = {node}")
        return found

    def test_no_interpolation_survives_for_any_arm(self):
        from sim_vla.models.model_config import load_model_config

        for task, arm in self.ARMS:
            with self.subTest(task=task, arm=arm):
                cfg = load_model_config(load_config(task, arm))
                left = self.unresolved(cfg.to_dict())
                self.assertEqual(left, [], f"unresolved: {left}")

    def test_encoder_regexes_match_the_keys_the_loader_produces(self):
        import re as _re

        from sim_vla.models.model_config import load_model_config

        cfg = load_model_config(load_config("pickcube", "graph"))
        fields = batch_fields({"camera_keys": {"base_camera": "image_base"}},
                              graph_enabled=True)
        # Every image key the loader emits must reach the CNN encoder, and the
        # proprioception key must reach the MLP encoder. A regex that matches
        # nothing leaves that encoder unbuilt.
        for key in fields.images:
            self.assertTrue(_re.match(cfg.encoder.cnn_keys, key), key)
            self.assertTrue(_re.match(cfg.decoder.cnn_keys, key), key)
        self.assertTrue(_re.match(cfg.encoder.mlp_keys, "proprio"))
        self.assertTrue(_re.match(cfg.decoder.mlp_keys, "proprio"))
        # ... and must not swallow the graph arrays, which go to the graph
        # encoder and would otherwise be learned twice.
        for key in GRAPH_KEYS:
            self.assertIsNone(_re.match(cfg.encoder.cnn_keys, key), key)
            self.assertIsNone(_re.match(cfg.encoder.mlp_keys, key), key)

    def test_config_writes_back_like_dictconfig(self):
        """MultiDecoder sets config.mlp.shape before building its head.

        networks.py:175 assigns into the config and then reads it one line
        later. A node that returned a fresh copy per access let that write land
        on a temporary, and MLPHead was built from shape=None -- reported as
        "'NoneType' object is not subscriptable", naming neither the config nor
        the assignment.
        """
        from sim_vla.models.model_config import load_model_config

        cfg = load_model_config(load_config("pickcube", "graph"))
        self.assertIsNone(cfg.decoder.mlp.shape)
        cfg.decoder.mlp.shape = (25,)
        self.assertEqual(cfg.decoder.mlp.shape, (25,))
        self.assertEqual(cfg.decoder.mlp.shape[0], 25)
        # Repeated access is the same object, not two views.
        self.assertIs(cfg.decoder.mlp, cfg.decoder.mlp)
        # ... and the mutation is confined to this config, not the yaml.
        fresh = load_model_config(load_config("pickcube", "graph"))
        self.assertIsNone(fresh.decoder.mlp.shape)

    def test_config_is_both_mapping_and_attributes(self):
        """networks.py:172 uses one node both ways in a single expression."""
        from sim_vla.models.model_config import load_model_config

        cfg = load_model_config(load_config("pickcube", "graph"))
        self.assertEqual(cfg.decoder.cnn_dist.name,
                         dict(**cfg.decoder.cnn_dist)["name"])
        self.assertIsInstance(dict(cfg.loss_scales), dict)
        self.assertGreater(len(dict(cfg.loss_scales)), 0)

    def test_heads_that_need_a_shape_have_one(self):
        """reward and cont are built straight from config.shape[0]."""
        from sim_vla.models.model_config import load_model_config

        cfg = load_model_config(load_config("pickcube", "graph"))
        for head in ("reward", "cont"):
            shape = getattr(cfg, head).shape
            self.assertIsNotNone(shape, f"{head}.shape is None")
            self.assertGreaterEqual(int(shape[0]), 1)

    def test_every_loss_key_has_a_scale(self):
        """Wrong scale names do not fail; they silently reweight the objective.

        The simulator keys reward as "rew" and continuation as "con", expands
        one "recon" scale across the decoder's output keys, and lets the graph
        decoder's own names through. sim_vla must use those same names.
        """
        from sim_vla.models.model_config import load_model_config

        cfg = load_model_config(load_config("pickcube", "graph"))
        scales = dict(cfg.loss_scales)
        for key in ("rew", "con", "recon", "dyn", "rep",
                    "graphdyn", "graphrep", "graphamp",
                    "node", "nodetgt", "relabs", "reltemp"):
            self.assertIn(key, scales, f"loss scale {key!r} is missing")
        # These are the names an earlier version used, and none of them exist.
        for wrong in ("reward", "cont", "image", "vector"):
            self.assertNotIn(wrong, scales)

    def test_unknown_interpolation_is_refused(self):
        from sim_vla.models.model_config import _resolve

        with self.assertRaises(KeyError):
            _resolve({"a": "${nope.deep}"}, {}, {})
        with self.assertRaises(KeyError):
            _resolve({"a": "${unknown_global}"}, {}, {}, globals_root={})


class TestBatchNaming(unittest.TestCase):
    """Every source of a batch must agree on its key names.

    Two bugs came from this. ImitationTrainer converted to tensors without
    renaming, so observe() could not find "action". And the online replay
    emitted the singular names while the demonstration sampler emitted the
    plural ones, so mixed_batch -- which concatenates the keys both sources
    share -- dropped the actions and rewards entirely.
    """

    def test_dataset_and_replay_agree(self):
        from sim_vla.data.batch import STEP_KEYS
        from sim_vla.data.dataset import SUPERVISION_FIELDS
        from sim_vla.data.replay import OnlineEpisode

        episode = OnlineEpisode()
        for _ in range(3):
            episode.add_observation({"proprio": np.zeros(4, np.float32)})
        for index in range(2):
            episode.add_transition(np.zeros(8, np.float32), float(index),
                                   False, index == 1, False)
        arrays = episode.arrays()
        for field in SUPERVISION_FIELDS:
            if field in ("terminated", "truncated"):
                continue     # the replay records is_terminal / is_last instead
            self.assertIn(field, arrays,
                          f"replay must emit {field!r} like the dataset does")
        for key in SUPERVISION_FIELDS:
            self.assertIn(key, STEP_KEYS)

    def test_mixed_batch_refuses_to_drop_actions(self):
        from sim_vla.data.replay import OnlineEpisode, OnlineReplay, mixed_batch

        class Demo:
            def batch(self, n):
                # A source that renamed early: the failure mode being guarded.
                return {"action": np.zeros((n, 4, 8), np.float32),
                        "proprio": np.zeros((n, 5, 4), np.float32)}

        replay = OnlineReplay(seed=0)
        for _ in range(6):
            episode = OnlineEpisode()
            for _ in range(5):
                episode.add_observation({"proprio": np.zeros(4, np.float32)})
            for index in range(4):
                episode.add_transition(np.zeros(8, np.float32), 0.0, False,
                                       index == 3, False)
            replay.add(episode)
        with self.assertRaises(KeyError):
            mixed_batch(Demo(), replay, batch=8, length=4, burn_in=0)

    def test_storage_names_are_renamed_once(self):
        from sim_vla.data.batch import RENAMES, storage_keys

        self.assertEqual(RENAMES, {"actions": "action", "rewards": "reward"})
        storage_keys({"actions": 1, "rewards": 2})
        with self.assertRaises(KeyError):
            storage_keys({"action": 1})


class TestEntryPoints(unittest.TestCase):
    """Every module the README and the test runner name must exist."""

    def test_documented_modules_are_importable(self):
        import importlib.util

        for name in ("sim_vla.training.pretrain_world_model",
                     "sim_vla.training.train_imitation",
                     "sim_vla.training.online",
                     "sim_vla.data.audit",
                     "sim_vla.download_pretrained",
                     "sim_vla.doctor"):
            self.assertIsNotNone(importlib.util.find_spec(name),
                                 f"{name} does not exist")


class TestPreprocessing(unittest.TestCase):
    """Images must reach the encoder as floats, and only be scaled once."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "demos.h5"
        make_dataset(self.path, episodes=((20, 15),))

    def tearDown(self):
        self.tmp.cleanup()

    def test_uint8_images_become_unit_range_floats(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")
        from sim_vla.data.batch import to_model_batch
        from sim_vla.data.sequences import SequenceSampler

        with DemoDataset(self.path, graph_enabled=False) as data:
            raw = SequenceSampler(data, length=6, burn_in=2, seed=0).batch(2)
            self.assertEqual(raw["image_base"].dtype, np.uint8)
            self.assertGreater(raw["image_base"].max(), 1)
            out = to_model_batch(raw)
            self.assertTrue(out["image_base"].is_floating_point())
            self.assertLessEqual(float(out["image_base"].max()), 1.0)
            self.assertGreaterEqual(float(out["image_base"].min()), 0.0)

    def test_preprocessing_is_not_applied_twice(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        from sim_vla.data.batch import to_model_batch
        from sim_vla.data.sequences import SequenceSampler

        with DemoDataset(self.path, graph_enabled=False) as data:
            raw = SequenceSampler(data, length=6, burn_in=2, seed=0).batch(2)
            once = to_model_batch(raw)
            twice = to_model_batch(once)
            self.assertTrue(torch.allclose(once["image_base"],
                                           twice["image_base"]),
                            "a second pass rescaled the images again")

    def test_normalizer_is_applied_when_given(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        from sim_vla.data.batch import to_model_batch
        from sim_vla.data.sequences import SequenceSampler

        with DemoDataset(self.path, graph_enabled=False) as data:
            norm = fit_normalizer(data)
            raw = SequenceSampler(data, length=6, burn_in=2, seed=0).batch(2)
            plain = to_model_batch(raw)
            scaled = to_model_batch(raw, normalizer=norm)
            self.assertFalse(torch.allclose(plain["proprio"],
                                            scaled["proprio"]),
                             "the normalizer was not applied")


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
