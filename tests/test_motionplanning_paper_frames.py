"""Motion-planning difference frames: the wrist mount, the success gate, the frame cap.

No simulator: the hooks are driven by hand with frames whose values walk with
the step, so a difference paired with the wrong step shows up as a wrong number.
"""

import json
import pathlib
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from scenegraph.figures.rollout import Attempt
from scenegraph.tools.render_motionplanning_paper_frames import (
    DEFAULT_MAX_FRAMES, WRIST_MOUNT_CHAIN, DiffCapture, chain_pose,
    episode_metadata, parse_args, quat_rotate, rpy_quat, sensor_frames,
)

H = W = 6
HUMAN = 8


class StubEnv:
    def __init__(self):
        self.value = 0

    def render(self):
        return np.full((1, HUMAN, HUMAN, 3), self.value, dtype=np.uint8)


def _obs(value):
    frame = np.full((1, H, W, 3), value, dtype=np.uint8)
    return {"sensor_data": {"base_camera": {"rgb": frame},
                            "hand_camera": {"rgb": frame + 1}}}


def _args(root, **overrides):
    base = dict(
        out=str(root), env_id="PlaceSphere-v1", seed=0, title="",
        overwrite=False, max_frames=DEFAULT_MAX_FRAMES, sensor_size=[H, W],
        human_size=[HUMAN, HUMAN], head_camera="base_camera",
        wrist_camera="hand_camera", no_human=False, diffs_only=False,
        diff_percentile=99.5, diff_gain=0.0, diff_max_gain=16.0,
        diff_invert=False, control_mode="pd_joint_pos", sim_backend="cpu",
        sensor_shader="default", human_shader="default",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _play(capture, env, successes, head_step=3):
    capture.on_reset(_obs(10))
    for t, flag in enumerate(successes, start=1):
        env.value = t
        capture.on_transition(_obs(10 + head_step * t), np.array([0.1 * t]),
                              np.array([False]), np.array([False]),
                              {"success": np.array([bool(flag)])})


def _commit(capture, args, seed=0):
    metadata = episode_metadata(
        args, capture, Attempt(seed, True, capture.steps), gym_id="X-v1",
        reward_mode="normalized_dense", horizon=50, robot_uids="panda",
        wrist_source="mounted")
    return capture.close(commit=True, metadata=metadata)


class TempRoot(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)


class WristMountTest(unittest.TestCase):
    def test_camera_sits_where_panda_wristcam_puts_it_and_looks_down_the_fingers(self):
        p, q = chain_pose(WRIST_MOUNT_CHAIN)
        np.testing.assert_allclose(p, [0.0465, -0.02, 0.036], atol=1e-3)
        np.testing.assert_allclose(quat_rotate(q, [1.0, 0.0, 0.0]),
                                   [0.0, 0.0, 1.0], atol=1e-3)

    def test_rpy_is_the_urdf_fixed_axis_convention(self):
        q = rpy_quat(np.pi / 2, 0.0, np.pi / 2)
        np.testing.assert_allclose(quat_rotate(q, [1.0, 0, 0]), [0, 1.0, 0], atol=1e-9)
        np.testing.assert_allclose(quat_rotate(q, [0, 1.0, 0]), [0, 0, 1.0], atol=1e-9)


class SensorFramesTest(unittest.TestCase):
    def test_one_env_row_per_camera(self):
        frames = sensor_frames(_obs(7))
        self.assertEqual(sorted(frames), ["base_camera", "hand_camera"])
        self.assertEqual(frames["base_camera"].shape, (H, W, 3))
        self.assertEqual(int(frames["hand_camera"][0, 0, 0]), 8)


class CaptureTest(TempRoot):
    def test_success_inside_the_cap_is_kept_with_its_manifest(self):
        env, args = StubEnv(), _args(self.root, max_frames=5)
        capture = DiffCapture(env, args)
        capture.prepare(3)
        _play(capture, env, [0, 0, 1, 1, 1, 1, 1])
        self.assertEqual(capture.first_success, 3)
        self.assertEqual(capture.steps, 7)
        path, gains = _commit(capture, args, seed=3)

        self.assertEqual(path.name, "PlaceSphere-v1_seed0003")
        self.assertEqual(len(list((path / "frames/head").iterdir())), 5)
        self.assertEqual(len(list((path / "frames/human").iterdir())), 5)
        self.assertEqual(len(list((path / "diff_vis/wrist").iterdir())), 4)
        manifest = json.loads((path / "episode.json").read_text())
        self.assertEqual(manifest["title"], "PlaceSphere-v1")
        self.assertEqual(manifest["attempt"]["first_success_step"], 3)
        self.assertEqual(manifest["attempt"]["steps"], 4)
        self.assertEqual(manifest["attempt"]["planned_steps"], 7)
        self.assertEqual(manifest["diff"]["gains"], gains)
        self.assertEqual([s["step"] for s in manifest["steps"]], [0, 1, 2, 3, 4])
        self.assertAlmostEqual(manifest["steps"][2]["reward"], 0.2)
        self.assertEqual(manifest["steps"][1]["head_diff_mean"], 3.0)
        self.assertFalse(any(p.name.startswith(".staging")
                             for p in self.root.iterdir()))

    def test_success_after_the_cap_does_not_count(self):
        env, args = StubEnv(), _args(self.root, max_frames=3)
        capture = DiffCapture(env, args)
        _play(capture, env, [0, 0, 0, 0, 1, 1])
        self.assertIsNone(capture.first_success)
        self.assertEqual(capture.writer.count, 3)
        capture.close(commit=False)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_a_new_reset_drops_the_previous_attempt(self):
        env, args = StubEnv(), _args(self.root, max_frames=4)
        capture = DiffCapture(env, args)
        _play(capture, env, [1, 1])
        _play(capture, env, [0])
        self.assertIsNone(capture.first_success)
        self.assertEqual(capture.writer.count, 2)

    def test_a_taken_name_is_refused_before_the_attempt(self):
        (self.root / "PlaceSphere-v1_seed0000").mkdir()
        capture = DiffCapture(StubEnv(), _args(self.root))
        with self.assertRaises(SystemExit):
            capture.prepare(0)
        DiffCapture(StubEnv(), _args(self.root, overwrite=True)).prepare(0)

    def test_no_human_writes_no_human_frames(self):
        env, args = StubEnv(), _args(self.root, max_frames=3, no_human=True)
        capture = DiffCapture(env, args)
        _play(capture, env, [1, 1])
        path, _ = _commit(capture, args)
        self.assertFalse((path / "frames/human").exists())


class ArgsTest(unittest.TestCase):
    def test_defaults(self):
        args = parse_args(["--env-id", "PullCubeTool-v1"])
        self.assertEqual(args.max_frames, 250)
        self.assertEqual((args.head_camera, args.wrist_camera),
                         ("base_camera", "hand_camera"))
        self.assertEqual(args.sensor_size, [500, 500])

    def test_one_frame_is_refused(self):
        with self.assertRaises(SystemExit):
            parse_args(["--env-id", "PullCubeTool-v1", "--max-frames", "1"])


if __name__ == "__main__":
    unittest.main()
