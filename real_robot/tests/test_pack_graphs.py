"""Regression checks for stale or misjoined training exports."""

import copy
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ..common import read_json, write_json
from ..graphs.validate import build_annotation
from ..preprocessing.annotate_episode import EpisodeAnnotator
from ..preprocessing.pack_graphs import current_annotation, main
from . import synthetic


class Export(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.configs = synthetic.configs()
        for key in ("annotations", "graphs"):
            self.configs["dataset"]["paths"][key] = os.path.join(self.tmp.name, key)
        graph = synthetic.graph_config()
        self.spec = graph.spec("blue_on_red")
        self.source = SimpleNamespace(
            configs=self.configs, graph=graph, fps=lambda: 30.0, spec=lambda e: self.spec,
            lengths=lambda: {7: 90}, select=lambda text: [7],
            global_index=lambda e: np.arange(200, 290),
            source_record=lambda: {"repo_id": "test", "resolved_revision": "abc"})
        self.prepared = {"rows": 90, "fps": 10.0, "shown": 31,
                         "cameras": {c: {"sha256": c} for c in self.spec.cameras}}
        self.annotator = EpisodeAnnotator(self.configs, self.source)
        annotation = build_annotation(self.spec, episode_index=7, n_frames=90, fps=30.0,
                                      answer=synthetic.complete_answer(self.spec, 90),
                                      settings=synthetic.settings())
        annotation.input_identity = self.annotator.input_identity(self.spec, self.prepared)
        self.data = annotation.to_json(self.spec)
        self.path = os.path.join(self.configs["dataset"]["paths"]["annotations"], "episode_000007.json")
        write_json(self.path, self.data)
        self.status = patch("real_robot.preprocessing.pack_graphs.prepared_status",
                            return_value=(self.prepared, "current"))
        self.status.start()
        self.addCleanup(self.status.stop)

    def test_rejects_wrong_episode_and_changed_labels_or_videos(self):
        current_annotation(self.annotator, 7, self.data)
        wrong = dict(self.data, episode_index=8)
        with self.assertRaisesRegex(ValueError, "different episode"):
            current_annotation(self.annotator, 7, wrong)
        self.annotator.labels_cfg["version"] = "changed"
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            current_annotation(self.annotator, 7, self.data)
        self.annotator.labels_cfg["version"] = synthetic.configs()["labels"]["version"]
        self.prepared["cameras"]["top"]["sha256"] = "changed video"
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            current_annotation(self.annotator, 7, self.data)

    def run_export(self, *extra):
        with patch("real_robot.preprocessing.pack_graphs.load_configs", return_value=self.configs), \
                patch("real_robot.data.source.LeRobotSource", return_value=self.source):
            main(["--name", "test", *extra])
        return read_json(os.path.join(self.configs["dataset"]["paths"]["graphs"], "test", "manifest.json"))

    def test_skipped_episode_is_removed_from_existing_manifest(self):
        first = self.run_export()
        self.assertEqual(list(first["episodes"]), ["7"])
        root = os.path.join(self.configs["dataset"]["paths"]["graphs"], "test")
        with np.load(os.path.join(root, first["episodes"]["7"]["file"])) as arrays:
            np.testing.assert_array_equal(arrays["index"], np.arange(200, 290))
        broken = copy.deepcopy(self.data)
        broken["answer"]["facts"] = []
        broken["status"] = "invalid"
        write_json(self.path, broken)
        second = self.run_export()
        self.assertEqual(second["episodes"], {})
        self.assertIn("invalid annotation", second["skipped"]["7"])

    def test_dataset_revision_cannot_mix_into_an_existing_export(self):
        self.run_export()
        self.source.source_record = lambda: {"repo_id": "test", "resolved_revision": "different"}
        with self.assertRaisesRegex(SystemExit, "different graph contract"):
            self.run_export()


if __name__ == "__main__":
    unittest.main()
