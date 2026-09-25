"""--data real: task selection, and the SO-101 converter on a synthetic snapshot.

The snapshot is LeRobot v3.0 laid out as hungho77/so101-multitask is: two
episodes in one parquet and one video per camera. The graphs directory is what
``real_robot.preprocessing.pack_graphs`` writes, with the repository's own
real-robot graph config and vocabulary.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scenegraph.adapters.graph_pack import GRAPH_KEYS
from sim_vla.config import check_dataset_compatibility, load_config

try:
    import av  # noqa: F401
    import cv2  # noqa: F401
    import h5py  # noqa: F401
    import pandas as pd
    import pyarrow  # noqa: F401
    MISSING = None
except ImportError as exc:
    MISSING = str(exc)

FPS = 30
LENGTHS = (12, 9)
REVISION = "a" * 40
NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
         "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


class TestRealConfig(unittest.TestCase):
    def test_the_real_tasks_resolve(self):
        for task, env_id in (("stackcube", "so101/blue_on_red"),
                             ("cubes_in_cup", "so101/cubes_in_cup")):
            cfg = load_config(task, "graph_progress", data="real")
            self.assertEqual(cfg["task"]["env_id"], env_id)
            self.assertTrue(cfg["task"]["dataset"].endswith(
                f"data/sim_vla_real/{task}/demos.h5"))
            self.assertEqual(cfg["eval"]["episodes"], 0)
            self.assertEqual(cfg["experiment"],
                             {"task": task, "arm": "graph_progress", "data": "real"})
            self.assertEqual(cfg["model"]["progress"]["schedule_dir"],
                             "sim_vla/configs/real/schedules")
            weights = [p["weight"] for p in cfg["task"]["progress_schedules"]]
            self.assertAlmostEqual(sum(weights), 1.0)

    def test_the_sim_configuration_is_unchanged(self):
        cfg = load_config("stackcube", "graph_progress")
        self.assertEqual(cfg["experiment"], {"task": "stackcube", "arm": "graph_progress"})
        self.assertEqual(cfg["task"]["env_id"], "StackCube-v1")
        self.assertTrue(cfg["task"]["dataset"].endswith("data/sim_vla_demos/StackCube-v1/demos.h5"))
        self.assertEqual(cfg["eval"]["episodes"], 20)
        self.assertEqual(cfg["model"]["progress"]["schedule_dir"], "scenegraph/configs/schedules")
        self.assertNotIn("progress_schedules", cfg["task"])
        self.assertEqual(load_config("stackcube", "graph_progress", data="sim"), cfg)

    def test_a_task_of_the_other_kind_names_the_flag(self):
        with self.assertRaises(SystemExit) as caught:
            load_config("cubes_in_cup", "dreamer")
        self.assertIn("--data real", str(caught.exception))
        with self.assertRaises(SystemExit) as caught:
            load_config("pickcube", "dreamer", data="real")
        self.assertIn("--data sim", str(caught.exception))
        with self.assertRaises(SystemExit):
            load_config("stackcube", "dreamer", data="both")

    def test_a_dataset_converted_for_another_task_is_refused(self):
        cfg = load_config("stackcube", "dreamer", data="real")
        check_dataset_compatibility(cfg, {"env_id": "so101/blue_on_red"})
        with self.assertRaises(SystemExit):
            check_dataset_compatibility(cfg, {"env_id": "so101/cubes_in_cup"})
        check_dataset_compatibility(load_config("stackcube", "dreamer"), {"env_id": "anything"})

    def test_wandb_keeps_real_runs_apart(self):
        from sim_vla.training.wandb_logger import WandbSettings

        real = WandbSettings.from_config(load_config("stackcube", "graph_progress", data="real"))
        sim = WandbSettings.from_config(load_config("stackcube", "graph_progress"))
        self.assertEqual((real.group, real.name), ("real-stackcube", "real-stackcube-graph_progress"))
        self.assertIn("real", real.tags)
        self.assertEqual((sim.group, sim.name), ("stackcube", "stackcube-graph_progress"))
        self.assertNotIn("real", sim.tags)


class TestPipelineRefusals(unittest.TestCase):
    def setUp(self):
        try:
            from sim_vla.training.pipeline import check_real_run
        except ImportError as exc:
            self.skipTest(f"sim_vla.training.pipeline needs torch ({exc})")
        self.check = check_real_run

    def test_recorded_data_has_no_simulator_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "stackcube" / "demos.h5"
            dataset.parent.mkdir()
            dataset.write_bytes(b"")
            unchecked = {"root": tmp}, {"source": {"graphs": str(Path(tmp) / "no_graphs")}}
            cfg = load_config("stackcube", "dreamer", {"data": unchecked[0], "task": unchecked[1]}, data="real")
            self.check(cfg, online_steps=0)
            with self.assertRaises(SystemExit):
                self.check(cfg, online_steps=10)
            with self.assertRaises(SystemExit):
                self.check(load_config("stackcube", "dreamer", {"data": unchecked[0], "task": unchecked[1],
                                                                "eval": {"episodes": 5}}, data="real"),
                           online_steps=0)
        with self.assertRaises(SystemExit) as caught:
            self.check(load_config("stackcube", "dreamer", {"data": {"root": "missing"}}, data="real"),
                       online_steps=0)
        self.assertIn("prepare_real --task stackcube", str(caught.exception))
        self.check(load_config("stackcube", "dreamer"), online_steps=10)


def gray(value: int) -> np.ndarray:
    return np.full((48, 64, 3), int(value) % 256, dtype=np.uint8)


def write_snapshot(root: Path, revision: str = REVISION) -> None:
    from real_robot.preprocessing.prepare_videos import write_video

    feature = lambda: {"dtype": "float32", "shape": [6], "names": list(NAMES)}  # noqa: E731
    video = {"dtype": "video", "shape": [48, 64, 3]}
    info = {"codebase_version": "v3.0", "fps": FPS, "robot_type": "so_follower",
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {"observation.state": feature(), "action": feature(),
                         "observation.images.top": video, "observation.images.wrist": video}}
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "source.json").write_text(json.dumps(
        {"repo_id": "hungho77/so101-multitask", "resolved_revision": revision}), encoding="utf-8")
    total = sum(LENGTHS)
    for camera, offset in (("top", 0), ("wrist", 100)):
        path = root / "videos" / f"observation.images.{camera}" / "chunk-000" / "file-000.mp4"
        write_video((gray(offset + 5 * i) for i in range(total)), str(path), FPS, crf=0)
    data, episodes, start = [], [], 0
    for episode, length in enumerate(LENGTHS):
        for frame in range(length):
            step = start + frame
            data.append({"episode_index": episode, "frame_index": frame, "index": step,
                         "timestamp": frame / FPS, "task_index": 0,
                         "observation.state": [float(step + j) for j in range(6)],
                         "action": [float(100 * step + j) for j in range(6)]})
        row = {"episode_index": episode, "tasks": ["Pick blue cube and place on red cube"],
               "length": length, "data/chunk_index": 0, "data/file_index": 0,
               "dataset_from_index": start, "dataset_to_index": start + length}
        for key in ("observation.images.top", "observation.images.wrist"):
            row |= {f"videos/{key}/chunk_index": 0, f"videos/{key}/file_index": 0,
                    f"videos/{key}/from_timestamp": start / FPS,
                    f"videos/{key}/to_timestamp": (start + length) / FPS}
        episodes.append(row)
        start += length
    (root / "data" / "chunk-000").mkdir(parents=True)
    pd.DataFrame(data).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    pd.DataFrame(episodes).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")


def graph_manifest(task: str) -> dict:
    from real_robot.graphs.schema import load_graph_config
    from real_robot.graphs.vocabulary import build_vocab, vocab_tables

    config = load_graph_config()
    return {"format": "real_robot/so101-graphs-v1", "graph": config.identity(),
            "vocab": vocab_tables(build_vocab(config)), "name": f"{task}_test",
            "source": {"repo_id": "hungho77/so101-multitask", "revision": REVISION},
            "episodes": {}}


def write_graphs(root: Path, task: str = "blue_on_red", lengths=LENGTHS) -> dict:
    manifest = graph_manifest(task)
    root.mkdir(parents=True)
    arrays, start = {}, 0
    for episode, n in enumerate(lengths):
        rng = np.random.default_rng(episode)
        packed = {
            "graph_node_ent": rng.integers(0, 9, (n, 8)).astype(np.uint8),
            "graph_node_bbox": rng.random((n, 8, 2, 4)).astype(np.float16),
            "graph_node_centroid": np.zeros((n, 8, 3), np.float32),
            "graph_node_target": rng.integers(0, 2, (n, 8)).astype(np.uint8),
            **{key: rng.integers(0, 12, (n, 168)).astype(np.uint8)
               for key in ("graph_edge_src", "graph_edge_dst", "graph_edge_rel",
                           "graph_edge_abs", "graph_edge_temp")},
            "graph_valid": np.ones(n, bool),
            "episode_index": np.full(n, episode, np.int64),
            "frame_index": np.arange(n, dtype=np.int64),
            "index": np.arange(start, start + n, dtype=np.int64),
        }
        name = f"episode_{episode:06d}.npz"
        np.savez_compressed(root / name, **packed)
        manifest["episodes"][str(episode)] = {"task": task, "n_frames": n, "valid_frames": n,
                                              "file": name, "annotation": f"a{episode}",
                                              "answer_hash": f"h{episode}"}
        arrays[episode] = packed
        start += n
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return arrays


@unittest.skipIf(MISSING, f"needs PyAV, OpenCV, h5py, pandas and pyarrow ({MISSING})")
class TestConvert(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_snapshot(self.root / "source")
        self.arrays = write_graphs(self.root / "graphs")
        self.out = self.root / "out" / "demos.h5"

    def tearDown(self):
        self.tmp.cleanup()

    def convert(self, **kwargs):
        from sim_vla.data.convert_real import convert

        settings = dict(env_id="so101/blue_on_red", lerobot=self.root / "source",
                        graphs_dir=self.root / "graphs", out=self.out, size=(16, 16),
                        log=lambda _: None)
        return convert(**(settings | kwargs))

    def test_every_frame_lines_up_with_its_graph(self):
        from sim_vla.data.dataset import DemoDataset

        self.convert()
        with DemoDataset(self.out, graph_enabled=True) as data:
            self.assertEqual([ref.steps for ref in data.episodes], [11, 8])
            start = 0
            for ref, n in zip(data.episodes, LENGTHS):
                block = data.read(ref, 0, ref.steps)
                steps = np.arange(start, start + n)
                np.testing.assert_array_equal(block["proprio"][:, 0], steps)
                np.testing.assert_array_equal(block["actions"][:, 0], 100 * steps[:-1])
                self.assertEqual(block["image_top"].shape, (n, 16, 16, 3))
                np.testing.assert_allclose(block["image_top"].mean(axis=(1, 2, 3)), 5 * steps, atol=3)
                np.testing.assert_allclose(block["image_wrist"].mean(axis=(1, 2, 3)), 100 + 5 * steps,
                                           atol=3)
                for key in GRAPH_KEYS:
                    np.testing.assert_array_equal(block[key], self.arrays[ref.episode_id][key])
                self.assertFalse(block["rewards"].any() or block["terminated"].any())
                start += n

    def test_the_metadata_is_what_training_reads(self):
        from sim_vla.data.dataset import DemoDataset
        from sim_vla.data.sequences import SequenceSampler

        self.convert()
        with DemoDataset(self.out, graph_enabled=True) as data:
            meta = data.metadata
            self.assertEqual(meta["env_id"], "so101/blue_on_red")
            self.assertEqual(meta["camera_keys"], {"top": "image_top", "wrist": "image_wrist"})
            self.assertEqual(meta["proprio_names"], NAMES)
            self.assertEqual((meta["graph"]["n_max"], meta["graph"]["e_max"], meta["graph"]["n_cams"]),
                             (8, 168, 2))
            self.assertEqual(meta["graph"]["vocab_sizes"],
                             {"entity": 9, "relation": 12, "absolute": 19, "temporal": 6})
            self.assertIn(["ee", "actor:blue_cube", "grasp"], meta["graph"]["facts"])
            self.assertIn(["actor:blue_cube", "actor:red_cube", "support"], meta["graph"]["facts"])
            self.assertEqual(meta["controller"]["action_low"][0], 0.0)
            self.assertEqual(meta["controller"]["action_high"][0], 100.0 * (sum(LENGTHS) - 2))
            self.assertEqual(meta["source"]["revision"], REVISION)
            check_dataset_compatibility(load_config("stackcube", "graph_progress", data="real"), meta)
            batch = SequenceSampler(data, length=4, burn_in=2).batch(2)
            for key in ("image_top", "image_wrist", "proprio", *GRAPH_KEYS):
                self.assertEqual(batch[key].shape[:2], (2, 7), key)

    def test_a_subset_can_be_converted(self):
        from sim_vla.data.dataset import DemoDataset

        self.convert(episodes=[1])
        with DemoDataset(self.out, graph_enabled=False) as data:
            self.assertEqual([ref.steps for ref in data.episodes], [8])

    def test_an_existing_dataset_is_kept_unless_overwritten(self):
        from sim_vla.data.convert_real import ConversionError

        self.convert()
        with self.assertRaises(ConversionError):
            self.convert()
        self.convert(overwrite=True)

    def change_graphs(self):
        packed = dict(self.arrays[1], graph_node_target=np.zeros((LENGTHS[1], 8), np.uint8))
        np.savez_compressed(self.root / "graphs" / "episode_000001.npz", **packed)

    def recorded_digest(self):
        sidecar = json.loads(self.out.with_suffix(".json").read_text(encoding="utf-8"))
        return sidecar["metadata"]["source"]["graphs_digest"]

    def test_a_dataset_is_stale_once_its_graphs_change(self):
        from sim_vla.data.convert_real import stale_reason

        graphs = self.root / "graphs"
        self.assertIn("does not exist", stale_reason(self.out, graphs))
        self.convert()
        self.assertIsNone(stale_reason(self.out, graphs, (16, 16)))
        self.assertIn("image size", stale_reason(self.out, graphs, (112, 112)))
        self.change_graphs()
        self.assertIn("changed", stale_reason(self.out, graphs))

    def test_prepare_converts_only_when_something_changed(self):
        from sim_vla.data import prepare_real

        args = ["--task", "stackcube", "--lerobot", str(self.root / "source"),
                "--graphs", str(self.root / "graphs"), "--out", str(self.out), "--image-size", "16", "16"]
        prepare_real.main(args)
        written, first = self.out.stat().st_mtime_ns, self.recorded_digest()
        prepare_real.main(args)
        self.assertEqual(self.out.stat().st_mtime_ns, written)
        self.change_graphs()
        prepare_real.main(args)
        self.assertNotEqual(self.recorded_digest(), first)
        (self.root / "source" / "source.json").write_text(json.dumps({"resolved_revision": "b" * 40}))
        with self.assertRaises(SystemExit) as caught:
            prepare_real.main(args)
        self.assertIn("holds revision", str(caught.exception))

    def test_training_refuses_a_dataset_older_than_its_graphs(self):
        try:
            from sim_vla.training.pipeline import check_real_run
        except ImportError as exc:
            self.skipTest(f"sim_vla.training.pipeline needs torch ({exc})")
        self.out = self.root / "data" / "stackcube" / "demos.h5"
        self.convert()
        cfg = load_config("stackcube", "dreamer", {"data": {"root": str(self.root / "data")},
                                                   "task": {"source": {"graphs": str(self.root / "graphs")}}},
                          data="real")
        check_real_run(cfg, online_steps=0)
        self.change_graphs()
        with self.assertRaises(SystemExit) as caught:
            check_real_run(cfg, online_steps=0)
        self.assertIn("prepare_real --task stackcube", str(caught.exception))

    def refused(self, needle, **kwargs):
        from sim_vla.data.convert_real import ConversionError

        with self.assertRaises(ConversionError) as caught:
            self.convert(**kwargs)
        self.assertIn(needle, str(caught.exception))
        self.assertFalse(self.out.exists())

    def test_an_incomplete_graph_is_refused(self):
        path = self.root / "graphs" / "episode_000001.npz"
        packed = dict(self.arrays[1], graph_valid=np.r_[True, False, np.ones(7, bool)])
        np.savez_compressed(path, **packed)
        self.refused("no complete graph")

    def test_a_length_mismatch_is_refused(self):
        manifest = json.loads((self.root / "graphs" / "manifest.json").read_text())
        manifest["episodes"]["0"]["n_frames"] = 11
        (self.root / "graphs" / "manifest.json").write_text(json.dumps(manifest))
        self.refused("recorded frames")

    def test_another_tasks_graphs_are_refused(self):
        self.refused("so101/cubes_in_cup", env_id="so101/cubes_in_cup")

    def test_graphs_annotated_on_another_revision_are_refused(self):
        (self.root / "source" / "source.json").write_text(json.dumps({"resolved_revision": "b" * 40}))
        self.refused("revision")


if __name__ == "__main__":
    unittest.main()
