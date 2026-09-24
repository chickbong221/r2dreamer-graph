"""A LeRobot v3.0 snapshot: the episode index, the task mapping, and one episode cut out of a shared video."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

try:
    import av  # noqa: F401
    import pandas as pd
    import pyarrow  # noqa: F401
    import PIL  # noqa: F401
    MISSING = None
except ImportError as exc:
    MISSING = str(exc)

from ..common import write_json
from . import synthetic

FPS = 30
LENGTHS = (12, 9)
TASKS = ("Pick blue cube and place on red cube", "Pick all cubes and place into cup")
KEYS = {"top": "observation.images.top", "wrist": "observation.images.wrist"}


def gray(value: int) -> np.ndarray:
    return np.full((64, 64, 3), value, dtype=np.uint8)


@unittest.skipIf(MISSING, f"needs pandas, pyarrow, PyAV and Pillow ({MISSING})")
class Snapshot(unittest.TestCase):
    def setUp(self):
        from ..preprocessing.prepare_videos import write_video

        self.tmp = tempfile.TemporaryDirectory()
        root = os.path.join(self.tmp.name, "source")
        write_json(os.path.join(root, "meta", "info.json"), {
            "codebase_version": "v3.0", "fps": FPS,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"})
        total = sum(LENGTHS)
        for key in KEYS.values():
            path = os.path.join(root, "videos", key, "chunk-000", "file-000.mp4")
            write_video((gray(10 * i) for i in range(total)), path, FPS, crf=0)
        rows, start = [], 0
        for episode, (length, task) in enumerate(zip(LENGTHS, TASKS)):
            row = {"episode_index": episode, "tasks": [task], "length": length, "data/chunk_index": 0,
                   "data/file_index": 0, "dataset_from_index": start, "dataset_to_index": start + length}
            for key in KEYS.values():
                row.update({f"videos/{key}/chunk_index": 0, f"videos/{key}/file_index": 0,
                            f"videos/{key}/from_timestamp": start / FPS,
                            f"videos/{key}/to_timestamp": (start + length) / FPS})
            rows.append(row)
            start += length
        index = os.path.join(root, "meta", "episodes", "chunk-000")
        os.makedirs(index)
        pd.DataFrame(rows).to_parquet(os.path.join(index, "file-000.parquet"))

        self.configs = synthetic.configs()
        paths = self.configs["dataset"]["paths"]
        paths["source"] = root
        paths["videos"] = os.path.join(self.tmp.name, "videos")
        from ..data.source import LeRobotSource
        self.source = LeRobotSource(self.configs)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_index_and_the_tasks(self):
        self.assertEqual(self.source.available(), [0, 1])
        self.assertEqual(self.source.lengths(), {0: 12, 1: 9})
        self.assertEqual([self.source.task_key(e) for e in (0, 1)], ["blue_on_red", "cubes_in_cup"])
        self.assertEqual(self.source.select("pilot"), [0, 1])
        self.assertEqual(self.source.select("cubes_in_cup"), [1])
        np.testing.assert_array_equal(self.source.global_index(1), np.arange(12, 21))

    def test_an_episode_is_cut_out_of_the_shared_video(self):
        frames = list(self.source.frames(1, "top"))
        self.assertEqual([index for index, _ in frames], list(range(9)))
        levels = [float(rgb.mean()) for _, rgb in frames]
        np.testing.assert_allclose(levels, [10 * (12 + i) for i in range(9)], atol=3)

    def test_the_prepared_copy_shows_every_third_frame_with_its_number(self):
        from ..preprocessing.prepare_videos import iter_frames, prepare_episode, prepared_status

        videos = dict(self.configs["annotation"]["videos"])
        record = prepare_episode(self.source, 1, videos)
        self.assertEqual((record["stride"], record["shown"], record["fps"]), (3, 4, 10.0))
        path = os.path.join(self.configs["dataset"]["paths"]["videos"], "episode_000001", "top.mp4")
        decoded = [rgb for _, _, rgb in iter_frames(path)]
        self.assertEqual(len(decoded), 4)
        self.assertLess(float(decoded[0][:4, :4].mean()), 40.0)
        np.testing.assert_allclose(float(decoded[1][40:, 40:].mean()), 10 * (12 + 3), atol=4)
        np.testing.assert_allclose(float(decoded[-1][40:, 40:].mean()), 10 * (12 + 8), atol=4)
        self.assertEqual(prepared_status(self.source, 1, videos)[1], "current")
        videos["crf"] = 23
        self.assertEqual(prepared_status(self.source, 1, videos)[1], "the video settings changed")


if __name__ == "__main__":
    unittest.main()
