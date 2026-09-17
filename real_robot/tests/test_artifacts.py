"""Artifacts are reused only for unchanged inputs, and a rebuilt episode changes the dataset's key."""

from __future__ import annotations

import os
import tempfile
import time
import unittest

import numpy as np

from ..common import write_json
from ..data.manifest import DatasetManifest
from ..preprocessing.artifacts import file_digest, reusable, stale_reason, source_video_digest
from ..preprocessing.prepare_videos import index_path, prepared_status, prepared_video_path, video_settings

VIDEO_CFG = {"overlay": True, "codec": "libx264", "crf": 18, "font_size": 18}


class Reuse(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "episode_000000.npz")
        np.savez(self.path, values=np.arange(3))
        self.inputs = {"annotation": "a1", "tracking": "t1", "videos": {"high": "v1"}}
        write_json(self.path[:-4] + ".json", {"inputs": self.inputs})

    def tearDown(self):
        self.tmp.cleanup()

    def test_unchanged_inputs_are_reused(self):
        self.assertEqual(reusable(self.path, dict(self.inputs)), (True, "current"))

    def test_a_changed_input_is_named(self):
        current, reason = reusable(self.path, {**self.inputs, "annotation": "a2"})
        self.assertFalse(current)
        self.assertIn("annotation", reason)
        self.assertIsNone(stale_reason(self.inputs, dict(self.inputs)))

    def test_missing_outputs_or_records_are_not_reused(self):
        self.assertFalse(reusable(os.path.join(self.tmp.name, "absent.npz"), self.inputs)[0])
        os.remove(self.path[:-4] + ".json")
        self.assertEqual(reusable(self.path, self.inputs), (False, "no record of its inputs"))

    def test_a_rewritten_file_has_a_new_digest(self):
        first = file_digest(self.path)
        time.sleep(0.01)
        np.savez(self.path, values=np.arange(4))
        self.assertNotEqual(file_digest(self.path), first)
        self.assertIsNone(file_digest(os.path.join(self.tmp.name, "absent")))


class VideoSource:
    """Just enough of a source for the prepared-video index: two cameras, one ten-frame episode."""

    def __init__(self, root):
        self.source_root = os.path.join(root, "source")
        self.dataset_cfg = {"paths": {"videos": os.path.join(root, "videos")},
                            "source": {"cameras": {"high": "h", "wrist_right": "w"}}}

    def lengths(self):
        return {0: 10}

    def fps(self):
        return 15.0

    def video_path(self, episode, camera):
        return os.path.join(self.source_root, f"{camera}.mp4")

    def source_record(self):
        raise FileNotFoundError("no source.json")


def write_bytes(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)


class PreparedVideos(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.source = VideoSource(self.tmp.name)
        record = {"episode_index": 0, "rows": 10, "fps": 15.0, "settings_hash": None, "cameras": {}}
        from ..common import stable_hash

        record["settings_hash"] = stable_hash(video_settings(VIDEO_CFG))
        for camera in ("high", "wrist_right"):
            write_bytes(self.source.video_path(0, camera), f"source {camera}".encode())
            copy = prepared_video_path(self.source.dataset_cfg, 0, camera)
            write_bytes(copy, f"labelled {camera}".encode())
            record["cameras"][camera] = {"source_sha256": source_video_digest(self.source, 0, camera),
                                         "sha256": file_digest(copy)}
        write_json(index_path(self.source.dataset_cfg, 0), record)

    def tearDown(self):
        self.tmp.cleanup()

    def status(self, cfg=VIDEO_CFG):
        return prepared_status(self.source, 0, cfg)[1]

    def test_copies_are_current_while_settings_sources_and_copies_match(self):
        self.assertEqual(self.status(), "current")
        self.assertIn("settings", self.status({**VIDEO_CFG, "crf": 23}))

    def test_a_changed_source_video_makes_the_copies_stale(self):
        time.sleep(0.01)
        write_bytes(self.source.video_path(0, "wrist_right"), b"re-downloaded, longer source video")
        self.assertIn("wrist_right source video changed", self.status())

    def test_a_missing_or_edited_copy_is_stale(self):
        copy = prepared_video_path(self.source.dataset_cfg, 0, "high")
        time.sleep(0.01)
        write_bytes(copy, b"edited copy, another length")
        self.assertIn("high copy changed", self.status())
        os.remove(copy)
        self.assertIn("high copy is missing", self.status())


class DatasetContent(unittest.TestCase):
    def test_rebuilding_an_episode_from_new_inputs_changes_the_dataset_key(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = DatasetManifest.create(root, identity={"x": 1},
                                              selection={"version": "v1", "training": [0, 1], "diagnostic": [1]},
                                              shapes={}, model_inputs={}, graph={}, action={})
            manifest.record_episode(0, {"inputs": {"annotation": "a"}})
            manifest.record_episode(1, {"inputs": {"annotation": "b"}})
            before = manifest.dataset_key()
            manifest.record_episode(1, {"inputs": {"annotation": "b-corrected"}})
            after = manifest.dataset_key()
            self.assertEqual(before["identity"], after["identity"])
            self.assertNotEqual(before["content"], after["content"])


if __name__ == "__main__":
    unittest.main()
