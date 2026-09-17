"""The sim_vla demonstration dataset: its contract, its writer, its collector.

The collector cannot run without a simulator, so the rollout tests drive
``_run_one`` against stubbed ManiSkill, scenegraph and env modules -- the same
approach ``test_maniskill_env_branch`` takes for the training env. What that
buys is the part worth testing: which episodes are accepted, where they are
cut, what lengths the arrays end up with, and that nothing is overwritten.

The one thing a stub cannot check is that ``gym.make`` really receives the step
budget, so that is asserted against the source.
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import types
import unittest
import unittest.mock
from argparse import Namespace
from pathlib import Path

import numpy as np

from scenegraph.adapters.graph_pack import GRAPH_KEYS
from sim_vla.data.config import apply_defaults, load_training_settings
from sim_vla.data.schema import (
    DEFAULT_PROPRIO, dir_digest, flatten_proprio, json_safe, merge_conflicts,
    privileged_fields, resolved_thresholds_path, unbatch,
)
from sim_vla.data.writer import DatasetWriter, field_kinds, incomplete_groups

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
class TestSchema(unittest.TestCase):
    def obs(self):
        return {
            "agent": {"qpos": np.arange(9.)[None], "qvel": np.zeros((1, 9))},
            "extra": {"tcp_pose": np.zeros((1, 7)),
                      "is_grasped": np.array([True]),
                      "goal_pos": np.zeros((1, 3))},
        }

    def test_unbatch_keeps_a_length_one_feature_axis(self):
        self.assertEqual(unbatch(np.zeros((1, 9))).shape, (9,))
        # A squeeze would make this () and silently narrow the column.
        self.assertEqual(unbatch(np.zeros((1, 1))).shape, (1,))

    def test_proprio_is_an_allowlist(self):
        vector, names = flatten_proprio(self.obs(), DEFAULT_PROPRIO)
        self.assertEqual(vector.shape, (25,))          # 9 qpos + 9 qvel + 7 tcp
        self.assertEqual(names[0], "agent.qpos[0]")
        self.assertEqual(names[-1], "extra.tcp_pose[6]")
        # The privileged fields are what the allowlist did not take, and they
        # are recorded rather than dropped.
        self.assertEqual(privileged_fields(self.obs(), DEFAULT_PROPRIO),
                         [("extra", "goal_pos"), ("extra", "is_grasped")])

    def test_missing_proprio_field_is_refused(self):
        with self.assertRaises(KeyError):
            flatten_proprio({"agent": {"qpos": np.zeros((1, 9))}},
                            DEFAULT_PROPRIO)

    def test_json_safe_keeps_nested_config_blocks(self):
        cfg = {"contact": {"eps_force": 0.05}, "grasp": {"max_angle": 30},
               "_cache": {"never": 1}, "arr": np.arange(3)}
        out = json_safe(cfg)
        # The earlier scalar-only filter dropped these entirely, which is how
        # two datasets could agree on every recorded graph setting and still
        # disagree on what an edge label means.
        self.assertEqual(out["contact"], {"eps_force": 0.05})
        self.assertEqual(out["grasp"], {"max_angle": 30})
        self.assertEqual(out["arr"], [0, 1, 2])
        self.assertNotIn("_cache", out)               # runtime state, not config

    def test_default_thresholds_are_hashed_not_labelled(self):
        path = resolved_thresholds_path("")
        self.assertTrue(Path(path).is_file(), path)
        digest = dir_digest(path)
        self.assertNotIn(digest, ("default", "missing"))
        self.assertEqual(len(digest), 40)

    def test_merge_conflicts_cover_more_than_vocabulary(self):
        base = {"env_id": "PickCube-v1", "image_size": [112, 112],
                "graph": {"n_max": 8}}
        self.assertEqual(merge_conflicts(base, dict(base)), [])
        self.assertIn("image_size",
                      merge_conflicts(base, dict(base, image_size=[64, 64])))
        self.assertIn("env_id",
                      merge_conflicts(base, dict(base, env_id="PlaceSphere-v1")))


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #
class TestWriter(unittest.TestCase):
    T = 4

    def payload(self, **overrides):
        T, H, W = self.T, 2, 2
        data = dict(
            images={"image_base": np.zeros((T + 1, H, W, 3), np.uint8)},
            proprio=np.zeros((T + 1, 25), np.float32),
            graphs={k: np.zeros((T + 1, 8), np.uint8) for k in GRAPH_KEYS},
            actions=np.zeros((T, 8), np.float32),
            rewards=np.zeros(T, np.float32),
            terminated=np.zeros(T, bool),
            truncated=np.zeros(T, bool),
            success=np.array([0] * (T - 1) + [1], bool),
            env_states={"actors": {"cube": np.zeros((T + 1, 13))}},
            privileged={"extra.goal_pos": np.zeros((T + 1, 3))},
            info={"seed": 1},
        )
        return data | overrides

    def test_round_trip_lengths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.h5"
            with DatasetWriter(path, {"env_id": "X"}) as writer:
                writer.add(**self.payload())
            import h5py

            with h5py.File(path) as handle:
                group = handle["traj_0"]
                self.assertEqual(group["actions"].shape[0], self.T)
                self.assertEqual(group["obs/image_base"].shape[0], self.T + 1)
                self.assertEqual(
                    group["env_states/actors/cube"].shape[0], self.T + 1)
                self.assertTrue(group.attrs["complete"])
            self.assertEqual(incomplete_groups(path), [])

    def test_existing_output_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.h5"
            DatasetWriter(path, {}).close()
            # Re-running a collection command must not erase eight hours of it.
            with self.assertRaises(SystemExit):
                DatasetWriter(path, {})
            DatasetWriter(path, {}, overwrite=True).close()

    def test_nested_observation_arrays_are_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = DatasetWriter(Path(tmp) / "d.h5", {})
            # One row of simulator state against five observations: every flat
            # array agrees, and only the nested tree is wrong.
            with self.assertRaises(ValueError) as caught:
                writer.add(**self.payload(
                    env_states={"actors": {"cube": np.zeros((1, 13))}}))
            self.assertIn("env_states.actors.cube", str(caught.exception))
            with self.assertRaises(ValueError):
                writer.add(**self.payload(
                    privileged={"extra.goal_pos": np.zeros((1, 3))}))
            writer.close()

    def test_sidecar_is_written_atomically_and_has_no_leftovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.h5"
            with DatasetWriter(path, {"env_id": "X"}) as writer:
                writer.add(**self.payload())
            side = json.loads(path.with_suffix(".json").read_text())
            self.assertEqual(side["count"], 1)
            self.assertEqual(side["episodes"][0]["steps"], self.T)
            self.assertFalse(list(Path(tmp).glob("*.tmp")))

    def test_field_kinds_name_both_lengths(self):
        kinds = field_kinds(["image_base"], GRAPH_KEYS)
        self.assertEqual(kinds["image_base"], "obs")
        self.assertEqual(kinds["graph_node_ent"], "obs")
        self.assertEqual(kinds["rewards"], "step")
        self.assertEqual(kinds["actions"], "step")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class TestConfig(unittest.TestCase):
    def test_reads_the_training_config(self):
        settings = load_training_settings(
            REPO / "configs/env/maniskill.yaml",
            REPO / "configs/model/size50M_graph_simple.yaml")
        # ${model.graph.n_max} has to be resolved against the model config, or
        # the collector packs at a capacity the trainer does not use.
        self.assertEqual(settings["n_max"], 8)
        self.assertEqual(settings["e_max"], 168)
        self.assertEqual(settings["shader"], "minimal")
        self.assertEqual(settings["sensor_size"], [112, 112])
        self.assertEqual(settings["visibility_policy"], "keep_tabletop")

    def test_explicit_flags_win(self):
        args = Namespace(shader="rt", n_max=None, e_max=None,
                         sensor_size=None, visibility_policy=None,
                         use_target_flag=None, object_object_spatial=None,
                         thresholds_path=None, whitelist_dir=None)
        taken = apply_defaults(args, {"shader": "minimal", "n_max": 8})
        self.assertEqual(args.shader, "rt")            # not overwritten
        self.assertEqual(args.n_max, 8)
        self.assertEqual(taken["from_config"], {"n_max": 8})

    def test_missing_config_is_not_an_error(self):
        self.assertEqual(load_training_settings("nope.yaml", "nope.yaml"), {})


# --------------------------------------------------------------------------- #
# The collector
# --------------------------------------------------------------------------- #
def flags(n, settle):
    """Per-step success flags: True from ``settle`` onward, or never."""
    return [i >= settle if settle is not None else False for i in range(n)]


def stub_modules(script, size=2):
    """The simulator, graph builder and runner the collector talks to."""
    state = {"flags": []}

    class Env:
        control_mode, control_freq, sim_freq, robot_uids = "pd_joint_pos", 20, 100, "panda"

        def __init__(self):
            self.unwrapped = self
            self.spec = types.SimpleNamespace(kwargs={"obs_mode": "rgb+segmentation"})
            self.action_space = types.SimpleNamespace(
                shape=(8,), low=-np.ones(8), high=np.ones(8))
            self.single_action_space = self.action_space
            self.agent = types.SimpleNamespace(
                controller=types.SimpleNamespace(configs={}))
            self.t = 0

        def _obs(self):
            return {"agent": {"qpos": np.arange(9.)[None],
                              "qvel": np.zeros((1, 9))},
                    "extra": {"tcp_pose": np.zeros((1, 7)),
                              "goal_pos": np.zeros((1, 3))},
                    "sensor_data": {"base_camera": {
                        "rgb": np.zeros((1, size, size, 3), np.uint8)}}}

        def reset(self, **kwargs):
            self.t = 0
            return self._obs(), {"success": np.array([False])}

        def step(self, action):
            self.t += 1
            run = state["flags"]
            ok = bool(run[self.t - 1]) if self.t - 1 < len(run) else False
            return (self._obs(), np.array([0.5]), np.array([False]),
                    np.array([False]), {"success": np.array([ok])})

        def get_state_dict(self):
            return {"actors": {"cube": np.full((1, 13), float(self.t))}}

        def close(self):
            pass

    class Graph:
        def __init__(self, frame):
            self.frame = frame

        def to_dict(self):
            return {"frame": self.frame, "nodes": ["ee"], "edges": []}

    class Source:
        cfg = {"contact": {"eps_force": 0.05}, "whitelist_dir": "wl"}
        whitelist_dir = str(REPO / "scenegraph/configs/subtask_whitelists/PickCube-v1")
        cameras = ["base_camera"]

        def __init__(self, *a, **k):
            self.frame = 0

        def on_reset(self):
            self.frame = 0

        def step(self, obs):
            self.frame += 1
            return Graph(self.frame)

    class Runner:
        def __init__(self, env, env_id, **kwargs):
            self.env, self.index = env, 0

        def attempt(self, seed):
            state["flags"] = script[self.index % len(script)]
            self.index += 1
            self.env.reset(seed=seed)
            for _ in range(len(state["flags"])):
                self.env.step(np.zeros(8, np.float32))
            return Namespace(seed=seed, success=any(state["flags"]),
                             steps=len(state["flags"]), error=None)

    class Vocab:
        sizes = {"entity": 4, "relation": 3, "absolute": 5, "temporal": 3}
        entity = relation = absolute = temporal = types.SimpleNamespace(
            token_to_id={"pad": 0, "grasp": 1})

    def module(name, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        return mod

    gym = module("gymnasium")

    class Wrapper:
        def __init__(self, env):
            self.env = env

        def __getattr__(self, name):
            return getattr(self.env, name)

    gym.Wrapper = Wrapper

    def make_env(kwargs, fallback):
        kwargs["reward_mode"] = "normalized_dense"
        made.append(dict(kwargs))
        return Env()

    made: list = []
    mods = {
        "gymnasium": gym,
        "mani_skill": module("mani_skill"),
        "mani_skill.envs": module("mani_skill.envs"),
        "mani_skill.utils": module("mani_skill.utils"),
        "mani_skill.utils.gym_utils": module(
            "mani_skill.utils.gym_utils",
            find_max_episode_steps_value=lambda env: 150),
        "envs.maniskill": module(
            "envs.maniskill",
            _make_with_supported_reward=make_env,
            camera_obs_key=lambda c: "image_" + c.replace("_camera", ""),
            rendered_cameras=lambda env: ["base_camera"]),
        "scenegraph.figures.graph_source": module(
            "scenegraph.figures.graph_source", FigureGraphSource=Source),
        "scenegraph.figures.rollout": module(
            "scenegraph.figures.rollout", MotionPlanRunner=Runner,
            capture_wrapper=None),
        "scenegraph.adapters.graph_pack": module(
            "scenegraph.adapters.graph_pack", GRAPH_KEYS=GRAPH_KEYS,
            pack_graph=lambda g, v, **k: {
                key: np.zeros(8, np.uint8) for key in GRAPH_KEYS}),
        "scenegraph.adapters.graph_vocab": module(
            "scenegraph.adapters.graph_vocab",
            build_graph_vocab=lambda d: Vocab()),
        "scenegraph.configs.loader": module(
            "scenegraph.configs.loader", default_temporal_k=lambda p=None: 5),
    }
    return mods, made


def collector_args(out_dir, **overrides):
    base = dict(
        env_id="PickCube-v1", num_traj=4, max_steps=150, pad=5, num_procs=1,
        start_seed=0, seed_stride=0, out_dir=str(out_dir), name="demos",
        instruction="", control_mode="pd_joint_pos", sim_backend="cpu",
        shader="minimal", sensor_size=[2, 2], reward_mode="normalized_dense",
        reward_fallback=["sparse"], n_max=8, e_max=168,
        visibility_policy="keep_tabletop", use_target_flag=False,
        object_object_spatial=True, thresholds_path="", whitelist_dir="",
        graph_sample=0, log_every=10_000, overwrite=False, config_source={},
    )
    return Namespace(**(base | overrides))


class TestCollector(unittest.TestCase):
    def test_split_targets_adds_up(self):
        from sim_vla.data.collect import split_targets

        # Rounding every worker up asked for 504 when 500 was wanted.
        self.assertEqual(sum(split_targets(500, 8)), 500)
        self.assertEqual(split_targets(500, 8), [63] * 4 + [62] * 4)
        self.assertEqual(sum(split_targets(7, 3)), 7)

    def test_env_is_built_with_the_step_budget(self):
        """A stub cannot see gym.make, so read the call from the source.

        Left at the task's registration, PickCube truncates at 50 and
        PegInsertionSide at 100, and a 130-step demo would carry a truncation
        flag a third of the way through.
        """
        source = (REPO / "sim_vla/data/collect.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        func = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "make_env")
        keywords = {k.arg for call in ast.walk(func)
                    if isinstance(call, ast.Call) for k in call.keywords}
        self.assertIn("max_episode_steps", keywords)

    def run_collection(self, script, tmp, **overrides):
        mods, made = stub_modules(script)
        with unittest.mock.patch.dict(sys.modules, mods):
            import sim_vla.data.collect as collect

            args = collector_args(tmp, **overrides)
            shard, stats = collect._run_one(
                args, 0, 1000, args.num_traj, overrides.pop("cap", 400))
        return shard, stats, made

    def test_accepts_within_budget_and_rejects_beyond(self):
        with tempfile.TemporaryDirectory() as tmp:
            shard, stats, _ = self.run_collection(
                [flags(140, 120), flags(200, 180), flags(90, 70)], tmp)
            self.assertEqual(stats["kept"], 4)
            self.assertEqual(stats["settled"], [121, 71, 121, 71])
            self.assertEqual(sum(stats["rejected"].values()), 2)
            # Every rejection is recorded with its seed, not only counted.
            self.assertTrue(all("seed" in row for row in stats["rejected_seeds"]))

    def test_trim_never_exceeds_the_budget(self):
        """Success at the budget is a demo of ``max_steps`` actions, not more.

        ``settled + pad`` would have written 155 for a budget of 150.
        """
        with tempfile.TemporaryDirectory() as tmp:
            shard, stats, _ = self.run_collection(
                [flags(160, 149)], tmp, num_traj=1)
            self.assertEqual(stats["settled"], [150])
            import h5py

            with h5py.File(shard) as handle:
                steps = handle["traj_0"]["actions"].shape[0]
                self.assertEqual(steps, 150)
                self.assertEqual(
                    handle["traj_0"]["obs/image_base"].shape[0], 151)

    def test_episode_arrays_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            shard, stats, made = self.run_collection(
                [flags(140, 120)], tmp, num_traj=2)
            self.assertEqual(made[0]["max_episode_steps"], 150)
            import h5py

            with h5py.File(shard) as handle:
                group = handle["traj_0"]
                steps = group["actions"].shape[0]
                self.assertEqual(steps, 126)             # settled 121 + pad 5
                for key in ("obs/proprio", "obs/graph_node_ent",
                            "env_states/actors/cube",
                            "privileged/extra.goal_pos"):
                    self.assertEqual(group[key].shape[0], steps + 1, key)
                self.assertTrue(bool(group["success"][-1]))
                self.assertFalse(bool(group["success"][0]))
            meta = json.loads(
                Path(shard).with_suffix(".json").read_text())["metadata"]
            self.assertEqual(meta["graph"]["n_max"], 8)
            self.assertEqual(meta["graph"]["e_max"], 168)
            self.assertEqual(meta["graph"]["visibility_policy"], "keep_tabletop")
            # The nested threshold blocks survive, and the default thresholds
            # file is hashed rather than labelled.
            self.assertEqual(meta["graph"]["config"]["contact"],
                             {"eps_force": 0.05})
            self.assertNotEqual(meta["graph"]["thresholds_digest"], "default")
            self.assertEqual(meta["graph"]["relation_tokens"],
                             {"pad": 0, "grasp": 1})

    def test_summary_reaches_the_sidecar_of_a_single_process_run(self):
        """The selection record is the part a reader cannot reconstruct."""
        from sim_vla.data.collect import attach_summary

        with tempfile.TemporaryDirectory() as tmp:
            shard, stats, _ = self.run_collection(
                [flags(140, 120), flags(200, 180)], tmp, num_traj=2)
            summary = {"attempts": stats["attempts"],
                       "rejected_counts": stats["rejected"],
                       "rejected_seeds": stats["rejected_seeds"]}
            attach_summary(Path(shard), summary)
            meta = json.loads(
                Path(shard).with_suffix(".json").read_text())["metadata"]
            self.assertEqual(meta["collection"]["attempts"], stats["attempts"])
            self.assertTrue(meta["collection"]["rejected_seeds"])
            self.assertFalse(list(Path(shard).parent.glob("*.tmp")))

    def test_exhausted_seed_block_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, stats, _ = self.run_collection(
                [flags(200, 180)], tmp, num_traj=5, cap=4)
            self.assertTrue(stats["exhausted"])
            self.assertEqual(stats["attempts"], 4)
            self.assertEqual(stats["kept"], 0)


class TestMerge(unittest.TestCase):
    def shard(self, tmp, name, seeds, metadata):
        import h5py

        path = Path(tmp) / f"{name}.h5"
        with h5py.File(path, "w") as handle:
            for index, _ in enumerate(seeds):
                handle.create_group(f"traj_{index}").create_dataset(
                    "actions", data=np.zeros((3, 8), np.float32))
        path.with_suffix(".json").write_text(json.dumps({
            "metadata": metadata,
            "episodes": [{"episode_id": i, "seed": s}
                         for i, s in enumerate(seeds)],
        }))
        return path

    def metadata(self, **overrides):
        return {"env_id": "PickCube-v1", "graph": {"n_max": 8},
                "camera_keys": {"base_camera": "image_base"},
                "cameras": ["base_camera"], "image_size": [112, 112],
                "proprio_names": ["a"], "proprio_fields": [["agent", "qpos"]],
                "privileged_fields": [], "controller": {"action_dim": 8},
                "reward_mode": "normalized_dense", "field_kinds": {},
                "budget": {"max_steps_to_success": 150}} | overrides

    def test_merge_renumbers_and_keeps_seeds(self):
        from sim_vla.data.collect import merge_shards

        with tempfile.TemporaryDirectory() as tmp:
            a = self.shard(tmp, "a", [1, 2], self.metadata())
            b = self.shard(tmp, "b", [7, 8], self.metadata())
            out = Path(tmp) / "merged.h5"
            merge_shards(out, [a, b])
            import h5py

            with h5py.File(out) as handle:
                self.assertEqual(sorted(handle.keys()),
                                 [f"traj_{i}" for i in range(4)])
            side = json.loads(out.with_suffix(".json").read_text())
            self.assertEqual([e["episode_id"] for e in side["episodes"]],
                             [0, 1, 2, 3])
            self.assertEqual([e["seed"] for e in side["episodes"]], [1, 2, 7, 8])

    def test_conflicting_shards_leave_the_destination_untouched(self):
        """Validation happens before a byte is written.

        The earlier version opened the destination first and raised after
        copying, which left a half-merged file beside the previous sidecar.
        """
        from sim_vla.data.collect import merge_shards

        with tempfile.TemporaryDirectory() as tmp:
            a = self.shard(tmp, "a", [1], self.metadata())
            b = self.shard(tmp, "b", [2], self.metadata(image_size=[64, 64]))
            out = Path(tmp) / "merged.h5"
            out.write_bytes(b"previous dataset")
            with self.assertRaises(SystemExit):
                merge_shards(out, [a, b], overwrite=True)
            self.assertEqual(out.read_bytes(), b"previous dataset")
            self.assertFalse(list(Path(tmp).glob("*.tmp")))

    def test_existing_destination_is_refused(self):
        from sim_vla.data.collect import merge_shards

        with tempfile.TemporaryDirectory() as tmp:
            a = self.shard(tmp, "a", [1], self.metadata())
            out = Path(tmp) / "merged.h5"
            out.write_bytes(b"previous dataset")
            with self.assertRaises(SystemExit):
                merge_shards(out, [a])
            self.assertEqual(out.read_bytes(), b"previous dataset")


if __name__ == "__main__":
    unittest.main()
