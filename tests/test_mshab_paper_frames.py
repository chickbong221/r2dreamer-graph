"""MS-HAB difference figure: the difference maths, the layout, the replay check.

No simulator. The env is a stub that returns frames shaped the way ManiSkill
shapes them, because what can go wrong here is arithmetic (a uint8 subtraction
that wraps, a gain derived from a static scene), pairing (a difference written
beside the wrong frame), file layout (a failed replay left on disk, or a
published figure overwritten), and the one check the whole two-pass design
rests on -- that the replayed episode is the episode the policy actually rolled.
"""

import json
import pathlib
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from scenegraph.figures.diff_writer import (
    DIFF_DIR, DIFF_VIS_DIR, FRAME_DIR, HUMAN_ROLE, LEVELS, STAGING_PREFIX,
    DiffEpisodeWriter, abs_diff, amplify, episode_path, gain_from_histogram,
)
from scenegraph.tools.render_mshab_paper_frames import (
    DEFAULT_HEAD_CAMERA, DEFAULT_WRIST_CAMERA, PAPER_HUMAN_SIZE,
    PAPER_SENSOR_SIZE, POLICY_SENSOR_SIZE, RecordedEpisode, parse_args,
    preflight, replay_and_export, seeded_random_policy,
)

HEAD, WRIST = DEFAULT_HEAD_CAMERA, DEFAULT_WRIST_CAMERA
ROLES = {"head": HEAD, "wrist": WRIST}
# Small enough to write a few hundred PNGs in a test; the one case that cares
# about the printed resolution asks for the real 500px.
H = W = 8
HUMAN_H = HUMAN_W = 12


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #
class StubVenv:
    """A vector env that only does what the exporter reads from one.

    Every frame it returns is a constant image whose value walks with the step,
    so a difference between two consecutive frames has a known magnitude and a
    figure paired with the wrong step is visible as the wrong number.
    """

    def __init__(self, rewards, successes, *, step_value=3,
                 sensor_shape=(H, W), human_shape=(HUMAN_H, HUMAN_W)):
        self.rewards = list(rewards)
        self.successes = list(successes)
        self.step_value = int(step_value)
        self.sensor_shape = tuple(sensor_shape)
        self.human_shape = tuple(human_shape)
        self.t = 0
        self.resets = []
        self.actions = []
        self.closed = False

    # -- gymnasium surface -------------------------------------------------
    def reset(self, seed=None, options=None):
        self.resets.append(seed)
        self.t = 0
        return {}, {}

    def step(self, action):
        self.actions.append(np.array(action, copy=True))
        index, self.t = self.t, self.t + 1
        info = {"success": np.array([float(self.successes[index])])}
        return ({}, np.array([self.rewards[index]]), np.array([False]),
                np.array([False]), info)

    def render(self):
        return np.full((1, *self.human_shape, 3), self._value(), dtype=np.uint8)

    def close(self):
        self.closed = True

    # -- what read_unwrapped_rgbs reaches for ------------------------------
    @property
    def unwrapped(self):
        return self

    def get_obs(self):
        frame = np.full((1, *self.sensor_shape, 3), self._value(), dtype=np.uint8)
        return {"sensor_data": {HEAD: {"rgb": frame},
                                WRIST: {"rgb": frame + 1}}}

    def _value(self):
        return 10 + self.step_value * self.t


def _args(**overrides):
    base = dict(sensor_size=[H, W], human_size=[HUMAN_H, HUMAN_W],
                max_frames=100, print_every=1000)
    base.update(overrides)
    return SimpleNamespace(**base)


def _camera_frames(height=H, width=W, value=10):
    """Keyed by camera name, the way ``read_unwrapped_rgbs`` returns them."""
    return {HEAD: np.full((height, width, 3), value, dtype=np.uint8),
            WRIST: np.full((height, width, 3), value, dtype=np.uint8)}


def _role_frames(height=H, width=W, value=10):
    """Keyed by role, the way the writer takes them: the alias is the caller's."""
    return {"head": np.full((height, width, 3), value, dtype=np.uint8),
            "wrist": np.full((height, width, 3), value, dtype=np.uint8)}


class TempRoot(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# The difference itself
# --------------------------------------------------------------------------- #
class TestDifferenceMaths(unittest.TestCase):
    def test_a_darker_frame_is_a_small_difference_not_a_huge_one(self):
        """The int16 promotion is the whole point: uint8 subtraction wraps."""
        current = np.full((2, 2, 3), 10, dtype=np.uint8)
        previous = np.full((2, 2, 3), 11, dtype=np.uint8)
        diff = abs_diff(current, previous)
        self.assertEqual(diff.dtype, np.uint8)
        self.assertTrue(np.all(diff == 1))

    def test_the_difference_is_symmetric_and_bounded(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 256, size=(4, 4, 3), dtype=np.uint8)
        b = rng.integers(0, 256, size=(4, 4, 3), dtype=np.uint8)
        self.assertTrue(np.array_equal(abs_diff(a, b), abs_diff(b, a)))
        self.assertLessEqual(int(abs_diff(a, b).max()), 255)

    def test_a_still_scene_does_not_ask_for_an_infinite_gain(self):
        """Every pixel identical puts the percentile at zero magnitude."""
        histogram = np.zeros(LEVELS, dtype=np.int64)
        histogram[0] = 1000
        gain = gain_from_histogram(histogram, percentile=99.5, max_gain=16.0)
        self.assertEqual(gain, 16.0)

    def test_the_gain_maps_the_percentile_to_full_scale(self):
        histogram = np.zeros(LEVELS, dtype=np.int64)
        histogram[0] = 990          # 99% of the pixels did not move
        histogram[51] = 10
        gain = gain_from_histogram(histogram, percentile=99.5, max_gain=16.0)
        self.assertAlmostEqual(gain, 5.0, places=6)

    def test_an_empty_histogram_is_a_gain_of_one(self):
        self.assertEqual(
            gain_from_histogram(np.zeros(LEVELS, dtype=np.int64),
                                percentile=99.5, max_gain=16.0),
            1.0,
        )

    def test_amplification_clips_instead_of_wrapping(self):
        diff = np.full((2, 2, 3), 100, dtype=np.uint8)
        out = amplify(diff, 8.0)
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(np.all(out == 255))

    def test_inverting_puts_the_difference_on_a_white_ground(self):
        diff = np.zeros((2, 2, 3), dtype=np.uint8)
        self.assertTrue(np.all(amplify(diff, 4.0, invert=True) == 255))


# --------------------------------------------------------------------------- #
# The layout on disk
# --------------------------------------------------------------------------- #
class TestWriterLayout(TempRoot):
    def _write(self, writer, steps=3):
        for index in range(steps):
            writer.write_step(
                step=index, human=np.full((HUMAN_H, HUMAN_W, 3),
                                          10 + index, dtype=np.uint8),
                sensors=_role_frames(value=10 + 4 * index),
                extra={"reward": float(index)},
            )

    def test_there_is_one_fewer_difference_than_frame(self):
        writer = DiffEpisodeWriter(self.root, "ep", roles=["head", "wrist"],
                                   human_size=(HUMAN_H, HUMAN_W),
                                   sensor_size=(H, W))
        writer.open()
        self._write(writer, steps=4)
        writer.write_amplified(percentile=99.5)
        path = writer.commit({"env_id": "CloseSubtaskTrain-v0"})

        for role in ("head", "wrist"):
            frames = sorted((path / FRAME_DIR / role).glob("frame_*.png"))
            diffs = sorted((path / DIFF_DIR / role).glob("diff_*.png"))
            vis = sorted((path / DIFF_VIS_DIR / role).glob("diff_*.png"))
            self.assertEqual([p.name for p in frames],
                             [f"frame_{i:04d}.png" for i in range(4)])
            # Index 0 has no predecessor, so the differences start at 0001.
            self.assertEqual([p.name for p in diffs],
                             [f"diff_{i:04d}.png" for i in range(1, 4)])
            self.assertEqual(len(vis), len(diffs))
        human = sorted((path / FRAME_DIR / HUMAN_ROLE).glob("frame_*.png"))
        self.assertEqual(len(human), 4)

    def test_the_difference_on_disk_is_the_difference_of_its_two_neighbours(self):
        writer = DiffEpisodeWriter(self.root, "ep", roles=["head"],
                                   human_size=(HUMAN_H, HUMAN_W),
                                   sensor_size=(H, W))
        writer.open()
        self._write(writer, steps=3)
        writer.write_amplified(percentile=99.5, gain=1.0)
        path = writer.commit()

        for index in (1, 2):
            a = np.asarray(Image.open(
                path / FRAME_DIR / "head" / f"frame_{index - 1:04d}.png"))
            b = np.asarray(Image.open(
                path / FRAME_DIR / "head" / f"frame_{index:04d}.png"))
            stored = np.asarray(Image.open(
                path / DIFF_DIR / "head" / f"diff_{index:04d}.png"))
            self.assertTrue(np.array_equal(stored[..., :3], abs_diff(b, a)))
            self.assertTrue(np.all(stored[..., :3] == 4))

    def test_the_manifest_names_every_file_it_wrote(self):
        writer = DiffEpisodeWriter(self.root, "ep", roles=["head", "wrist"],
                                   human_size=(HUMAN_H, HUMAN_W),
                                   sensor_size=(H, W))
        writer.open()
        self._write(writer, steps=3)
        gains = writer.write_amplified(percentile=99.5)
        path = writer.commit({"env_id": "CloseSubtaskTrain-v0"})

        manifest = json.loads((path / "episode.json").read_text())
        self.assertEqual(manifest["exported_frames"], 3)
        self.assertEqual(manifest["exported_diffs"], 2)
        self.assertEqual(sorted(gains), ["head", "wrist"])
        first, second = manifest["steps"][0], manifest["steps"][1]
        self.assertNotIn("head_diff", first)
        self.assertIn("head_diff", second)
        for record in manifest["steps"]:
            for key, value in record.items():
                if isinstance(value, str) and value.endswith(".png"):
                    self.assertTrue((path / value).exists(), value)

    def test_dropping_the_frames_keeps_the_differences(self):
        writer = DiffEpisodeWriter(self.root, "ep", roles=["head"],
                                   human_size=(HUMAN_H, HUMAN_W),
                                   sensor_size=(H, W), save_frames=False,
                                   save_human=False)
        writer.open()
        for index in range(3):
            writer.write_step(step=index, sensors=_role_frames(value=10 + 4 * index))
        writer.write_amplified(percentile=99.5)
        path = writer.commit()
        self.assertFalse((path / FRAME_DIR).exists())
        self.assertEqual(len(list((path / DIFF_DIR / "head").glob("*.png"))), 2)

    def test_a_frame_of_the_wrong_size_is_refused(self):
        """The figure is uncropped and unresized, so a size mismatch is an
        env built with different camera configs, not something to paper over."""
        writer = DiffEpisodeWriter(self.root, "ep", roles=["head"],
                                   human_size=(HUMAN_H, HUMAN_W),
                                   sensor_size=(H, W))
        writer.open()
        with self.assertRaises(ValueError):
            writer.write_step(
                step=0, human=np.zeros((HUMAN_H, HUMAN_W, 3), dtype=np.uint8),
                sensors={"head": np.zeros((H + 1, W, 3), dtype=np.uint8)})

    def test_a_missing_role_names_the_role(self):
        writer = DiffEpisodeWriter(self.root, "ep", roles=["head", "wrist"],
                                   human_size=(HUMAN_H, HUMAN_W),
                                   sensor_size=(H, W))
        writer.open()
        with self.assertRaises(KeyError):
            writer.write_step(
                step=0, human=np.zeros((HUMAN_H, HUMAN_W, 3), dtype=np.uint8),
                sensors={"head": np.zeros((H, W, 3), dtype=np.uint8)})


class TestWriterIsolation(TempRoot):
    def test_a_published_figure_is_not_overwritten_by_accident(self):
        episode_path(self.root, "ep").mkdir(parents=True)
        writer = DiffEpisodeWriter(self.root, "ep", roles=["head"])
        with self.assertRaises(FileExistsError):
            writer.open()

    def test_overwrite_replaces_only_that_episode(self):
        keep = episode_path(self.root, "other")
        keep.mkdir(parents=True)
        (keep / "frame.png").write_bytes(b"")
        stale = episode_path(self.root, "ep")
        stale.mkdir(parents=True)
        (stale / "stale.png").write_bytes(b"")

        writer = DiffEpisodeWriter(self.root, "ep", roles=["head"],
                                   sensor_size=(H, W), save_human=False,
                                   overwrite=True)
        writer.open()
        writer.write_step(step=0, sensors=_role_frames())
        path = writer.commit()
        self.assertFalse((path / "stale.png").exists())
        self.assertTrue((keep / "frame.png").exists())

    def test_a_discarded_episode_leaves_nothing_behind(self):
        writer = DiffEpisodeWriter(self.root, "ep", roles=["head"],
                                   sensor_size=(H, W), save_human=False)
        writer.open()
        writer.write_step(step=0, sensors=_role_frames())
        writer.discard()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_a_name_with_a_separator_is_refused(self):
        """``commit`` deletes by name; a name that is a path could aim it."""
        with self.assertRaises(ValueError):
            DiffEpisodeWriter(self.root, "../ep", roles=["head"])

    def test_the_staging_prefix_is_this_writers_own(self):
        from scenegraph.figures import multicamera_writer, writer as single

        self.assertNotEqual(STAGING_PREFIX, multicamera_writer.STAGING_PREFIX)
        self.assertNotEqual(STAGING_PREFIX,
                            getattr(single, "STAGING_PREFIX", ""))


# --------------------------------------------------------------------------- #
# The camera contract
# --------------------------------------------------------------------------- #
class TestPreflight(unittest.TestCase):
    def test_a_complete_set_of_cameras_passes(self):
        preflight(_camera_frames(), np.zeros((HUMAN_H, HUMAN_W, 3), dtype=np.uint8),
                  roles=ROLES, sensor_size=(H, W),
                  human_size=(HUMAN_H, HUMAN_W))

    def test_a_missing_camera_names_what_the_env_does_render(self):
        frames = _camera_frames()
        frames.pop(WRIST)
        with self.assertRaises(SystemExit) as caught:
            preflight(frames, np.zeros((HUMAN_H, HUMAN_W, 3), dtype=np.uint8),
                      roles=ROLES, sensor_size=(H, W),
                      human_size=(HUMAN_H, HUMAN_W))
        message = str(caught.exception)
        self.assertIn(WRIST, message)
        self.assertIn(HEAD, message)

    def test_every_problem_is_reported_by_one_run(self):
        with self.assertRaises(SystemExit) as caught:
            preflight(_camera_frames(height=H + 2), None, roles=ROLES,
                      sensor_size=(H, W), human_size=(HUMAN_H, HUMAN_W))
        message = str(caught.exception)
        self.assertIn("human render", message)
        self.assertEqual(message.count("expected"), 2)

    def test_two_roles_on_one_camera_is_not_two_views(self):
        with self.assertRaises(SystemExit) as caught:
            preflight(_camera_frames(), np.zeros((HUMAN_H, HUMAN_W, 3), dtype=np.uint8),
                      roles={"head": HEAD, "wrist": HEAD}, sensor_size=(H, W),
                      human_size=(HUMAN_H, HUMAN_W))
        self.assertIn("same camera", str(caught.exception))


# --------------------------------------------------------------------------- #
# The replay, and the check that makes it trustworthy
# --------------------------------------------------------------------------- #
class TestReplay(TempRoot):
    def _writer(self, **kwargs):
        writer = DiffEpisodeWriter(
            self.root, "ep", roles=["head", "wrist"],
            human_size=(HUMAN_H, HUMAN_W), sensor_size=(H, W), **kwargs)
        writer.open()
        return writer

    def test_a_faithful_replay_reports_no_gap_and_exports_every_step(self):
        rewards = [0.25, 0.5, 0.75]
        episode = RecordedEpisode(seed=0, actions=[np.zeros(3)] * 3,
                                  rewards=list(rewards),
                                  successes=[False, False, True])
        venv = StubVenv(rewards, episode.successes)
        writer = self._writer()
        result = replay_and_export(venv, episode, writer, roles=ROLES,
                                   args=_args())
        self.assertEqual(result["max_reward_abs_diff"], 0.0)
        self.assertTrue(result["success_flags_match"])
        # Frame 0 is the reset, then one per action.
        self.assertEqual(writer.count, 4)
        self.assertEqual([r["step"] for r in writer.records], [0, 1, 2, 3])
        self.assertIsNone(writer.records[0]["reward"])
        self.assertEqual(writer.records[3]["reward"], 0.75)
        self.assertTrue(writer.records[3]["success"])

    def test_both_passes_reset_the_same_number_of_times(self):
        """Pass one resets twice before its first action -- once to shape the
        policy, once to start -- so the replay has to do the same."""
        episode = RecordedEpisode(seed=7, actions=[np.zeros(3)],
                                  rewards=[0.1], successes=[False])
        venv = StubVenv([0.1], [False])
        replay_and_export(venv, episode, self._writer(), roles=ROLES,
                          args=_args())
        self.assertEqual(venv.resets, [7, 7])

    def test_a_diverged_replay_is_reported_as_a_gap(self):
        episode = RecordedEpisode(seed=0, actions=[np.zeros(3)] * 2,
                                  rewards=[0.25, 0.5], successes=[False, False])
        venv = StubVenv([0.25, 0.9], [False, False])
        result = replay_and_export(venv, episode, self._writer(), roles=ROLES,
                                   args=_args())
        self.assertAlmostEqual(result["max_reward_abs_diff"], 0.4, places=6)

    def test_a_success_that_moved_is_reported_even_when_the_reward_matches(self):
        episode = RecordedEpisode(seed=0, actions=[np.zeros(3)],
                                  rewards=[0.25], successes=[True])
        venv = StubVenv([0.25], [False])
        result = replay_and_export(venv, episode, self._writer(), roles=ROLES,
                                   args=_args())
        self.assertEqual(result["max_reward_abs_diff"], 0.0)
        self.assertFalse(result["success_flags_match"])

    def test_max_frames_stops_the_export_before_the_actions_run_out(self):
        episode = RecordedEpisode(seed=0, actions=[np.zeros(3)] * 5,
                                  rewards=[0.1] * 5, successes=[False] * 5)
        venv = StubVenv([0.1] * 5, [False] * 5)
        writer = self._writer()
        replay_and_export(venv, episode, writer, roles=ROLES,
                          args=_args(max_frames=3))
        self.assertEqual(writer.count, 3)

    def test_the_replay_feeds_back_the_recorded_actions_in_order(self):
        actions = [np.full(3, float(i)) for i in range(3)]
        episode = RecordedEpisode(seed=0, actions=actions, rewards=[0.1] * 3,
                                  successes=[False] * 3)
        venv = StubVenv([0.1] * 3, [False] * 3)
        replay_and_export(venv, episode, self._writer(), roles=ROLES,
                          args=_args())
        self.assertEqual(len(venv.actions), 3)
        for sent, recorded in zip(venv.actions, actions):
            self.assertTrue(np.array_equal(sent, recorded))

    def test_a_camera_the_env_does_not_render_fails_before_any_step(self):
        episode = RecordedEpisode(seed=0, actions=[np.zeros(3)], rewards=[0.1],
                                  successes=[False])
        venv = StubVenv([0.1], [False])
        with self.assertRaises(SystemExit):
            replay_and_export(venv, episode, self._writer(),
                              roles={"head": HEAD, "wrist": "no_such_camera"},
                              args=_args())
        self.assertEqual(venv.actions, [])


class TestRecordedEpisode(unittest.TestCase):
    def test_the_first_success_is_a_step_index_not_a_list_index(self):
        """Step 1 is after the first action; the reward CSV counts the same way."""
        episode = RecordedEpisode(seed=0, actions=[np.zeros(3)] * 3,
                                  rewards=[0.1, 0.2, 0.3],
                                  successes=[False, True, True])
        self.assertEqual(episode.first_success_step, 2)
        self.assertEqual(episode.steps, 3)

    def test_an_episode_that_never_succeeded_says_so(self):
        episode = RecordedEpisode(seed=0, actions=[np.zeros(3)],
                                  rewards=[0.1], successes=[False])
        self.assertIsNone(episode.first_success_step)


# --------------------------------------------------------------------------- #
# CLI defaults
# --------------------------------------------------------------------------- #
class TestCli(unittest.TestCase):
    def test_the_defaults_are_the_figure_this_tool_exists_for(self):
        args = parse_args(["--ckpt-dir", "ckpt"])
        self.assertEqual(tuple(args.sensor_size), PAPER_SENSOR_SIZE)
        self.assertEqual(tuple(args.human_size), PAPER_HUMAN_SIZE)
        self.assertEqual(tuple(args.policy_sensor_size), POLICY_SENSOR_SIZE)
        self.assertEqual(args.head_camera, DEFAULT_HEAD_CAMERA)
        self.assertEqual(args.wrist_camera, DEFAULT_WRIST_CAMERA)
        self.assertEqual(args.config_section, "env")
        self.assertEqual(args.max_frames, 60)
        self.assertFalse(args.overwrite)

    def test_the_policy_and_figure_resolutions_are_separate_knobs(self):
        """Raising the figure's cameras must not raise the policy's."""
        args = parse_args(["--ckpt-dir", "ckpt", "--sensor-size", "800", "800"])
        self.assertEqual(tuple(args.sensor_size), (800, 800))
        self.assertEqual(tuple(args.policy_sensor_size), POLICY_SENSOR_SIZE)

    def test_the_checkpoint_is_the_default_policy(self):
        self.assertFalse(parse_args(["--ckpt-dir", "ckpt"]).random_policy)
        self.assertTrue(parse_args(["--ckpt-dir", "ckpt",
                                    "--random-policy"]).random_policy)


# --------------------------------------------------------------------------- #
# The random policy
# --------------------------------------------------------------------------- #
class StubActionSpace:
    """Just the two methods a gym Box is asked for, drawing from its own RNG."""

    def __init__(self, shape=(1, 13)):
        self.shape = shape
        self.rng = np.random.default_rng()

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def sample(self):
        return self.rng.uniform(-1.0, 1.0, size=self.shape).astype(np.float32)


class TestRandomPolicy(unittest.TestCase):
    def _roll(self, seed, steps=5):
        venv = SimpleNamespace(action_space=StubActionSpace())
        policy = seeded_random_policy(venv, seed)
        return policy, [policy.act({}) for _ in range(steps)]

    def test_it_says_it_is_random(self):
        """The manifest's ``checkpoint`` is nulled on this kind."""
        policy, _ = self._roll(0)
        self.assertEqual(policy.kind, "random")

    def test_the_same_seed_rolls_the_same_actions(self):
        _, first = self._roll(3)
        _, second = self._roll(3)
        for a, b in zip(first, second):
            self.assertTrue(np.array_equal(a, b))

    def test_a_different_seed_rolls_different_actions(self):
        _, first = self._roll(3)
        _, second = self._roll(4)
        self.assertFalse(np.array_equal(first[0], second[0]))


if __name__ == "__main__":
    unittest.main()
