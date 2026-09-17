"""Artifacts are current only when everything up their chain is: an unchanged input made from stale inputs is stale."""

from __future__ import annotations

import copy
import os
import tempfile
import time
import unittest
from types import SimpleNamespace

import numpy as np

from ..common import load_configs, set_by_path, stable_hash, write_json
from ..preprocessing.artifacts import file_digest, source_video_digest
from ..preprocessing.freshness import ArtifactChain
from ..preprocessing.prepare_videos import index_path, prepared_status, prepared_video_path, video_settings
from ..preprocessing.track_objects import tracks_inputs
from . import synthetic as syn
from .test_annotator import Annotator
from .test_annotator import Scripted as ScriptedGemini
from .test_prompts import valid_bins


def write_bytes(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)


class Source:
    """Two episodes, two cameras, files under one temporary root."""

    mode = "full_episode"

    def __init__(self, root, configs):
        self.root = root
        self.dataset_cfg = configs["dataset"]
        self.source_root = os.path.join(root, "source")
        self.spec = syn.graph_spec()

    def lengths(self):
        return {0: syn.N, 1: syn.N}

    def fps(self):
        return 15.0

    def video_path(self, episode, camera):
        return os.path.join(self.source_root, f"episode_{episode}_{camera}.mp4")

    def source_record(self):
        raise FileNotFoundError("no source.json")

    def annotation_path(self, episode):
        return os.path.join(self.root, "annotations", f"episode_{episode}.json")

    def tracks_path(self, episode):
        return os.path.join(self.root, "tracks", f"episode_{episode}.npz")

    def geometry_path(self, episode):
        return os.path.join(self.root, "geometry", f"episode_{episode}.npz")


class Geometry:
    """The parts of the geometry stage the chain reads, with records the test controls."""

    def __init__(self, root):
        self.root = root
        self.cfg = {"checkpoint": os.path.join(root, "depth_pro.pt")}
        self.alignment_record_current = True

    def camera_geometry_path(self):
        return os.path.join(self.root, "camera_geometry.json")

    def depth_path(self, episode):
        return os.path.join(self.root, "depth", f"episode_{episode}.npz")

    def depth_inputs(self, episode, focal_px):
        return {"video": f"video {episode}", "focal_px": focal_px}

    def alignment_path(self):
        return os.path.join(self.root, "camera_alignment.json")

    def alignment_current(self):
        return self.alignment_record_current, "current" if self.alignment_record_current else "inputs changed: tracks"

    def camera_status(self, episode):
        return True, "fixed"

    def geometry_inputs(self, episode):
        return {"annotation": "a", "alignment": "hash"}


class Chain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        self.configs = load_configs(["dataset", "annotation", "graph"])
        set_by_path(self.configs["dataset"], "paths.videos", os.path.join(root, "videos"))
        self.source = Source(root, self.configs)
        self.spec = self.source.spec
        bins = valid_bins(self.spec)
        self.bins = {"bins": bins, "bins_hash": stable_hash(bins)}
        self.annotation = syn.annotation(self.spec)
        for episode in (0, 1):
            self.prepare_videos(episode)
            self.save_annotation(episode)
            self.save_tracks(episode)
        self.geometry = Geometry(root)
        write_bytes(self.geometry.cfg["checkpoint"], b"weights")
        write_json(self.geometry.camera_geometry_path(), {"focal_px": 500.0,
                                                          "checkpoint": file_digest(self.geometry.cfg["checkpoint"])})
        for episode in (0, 1):
            self.save_npz(self.geometry.depth_path(episode), self.geometry.depth_inputs(episode, 500.0))
        write_json(self.geometry.alignment_path(), {"inputs": {"episodes": [1]}, "hash": "hash"})
        self.save_npz(self.source.geometry_path(0), self.geometry.geometry_inputs(0))

    def tearDown(self):
        self.tmp.cleanup()

    # ------------------------------------------------------------ fixtures
    def chain(self, configs=None):
        configs = configs or self.configs
        chain = ArtifactChain(configs, self.source)
        chain._geometry = self.geometry
        chain._annotator = Annotator(configs, mode="full_episode", client=ScriptedGemini(None),
                                     source=self.source, bins=self.bins)
        return chain

    def prepare_videos(self, episode):
        record = {"rows": syn.N, "fps": 15.0, "settings_hash": stable_hash(video_settings(self.configs["annotation"]
                                                                                           ["videos"])), "cameras": {}}
        for camera in self.spec.cameras:
            write_bytes(self.source.video_path(episode, camera), f"source {episode} {camera}".encode())
            copy_path = prepared_video_path(self.configs["dataset"], episode, camera)
            write_bytes(copy_path, f"labelled {episode} {camera}".encode())
            record["cameras"][camera] = {"source_sha256": source_video_digest(self.source, episode, camera),
                                         "sha256": file_digest(copy_path)}
        write_json(index_path(self.configs["dataset"], episode), record)

    def save_annotation(self, episode):
        annotator = Annotator(self.configs, mode="full_episode", client=ScriptedGemini(None), source=self.source,
                              bins=self.bins)
        prepared, _ = prepared_status(self.source, episode, self.configs["annotation"]["videos"])
        annotation = copy.deepcopy(self.annotation)
        annotation.episode_index = episode
        annotation.input_identity = annotator.input_identity(episode, prepared=prepared)
        write_json(self.source.annotation_path(episode), annotation.to_json(self.spec))

    def save_tracks(self, episode):
        self.save_npz(self.source.tracks_path(episode),
                      tracks_inputs(self.source, episode, self.configs["annotation"]["tracking"]))

    @staticmethod
    def save_npz(path, inputs):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, values=np.arange(3))
        write_json(path[:-4] + ".json", {"inputs": inputs})

    # --------------------------------------------------------------- tests
    def test_a_current_chain_has_no_problems(self):
        chain = self.chain()
        self.assertEqual(chain.annotation(0), [])
        self.assertEqual(chain.tracks(0), [])
        self.assertEqual(chain.geometry_chain(0), [])

    def test_unchanged_tracks_from_a_stale_annotation_are_stale(self):
        changed = copy.deepcopy(self.configs)
        set_by_path(changed["annotation"], "annotation.anchors.max_gap_frames", 10)
        chain = self.chain(changed)
        self.assertTrue(any("validation" in p for p in chain.annotation(0)), chain.annotation(0))
        # The tracks' own record still matches the annotation file they were made from ...
        tracks = chain.tracks(0)
        self.assertTrue(tracks and all(p.startswith("annotation:") for p in tracks), tracks)
        # ... and nothing made from them is current either.
        self.assertTrue(chain.geometry_chain(0))

    def test_changed_videos_make_the_annotation_stale(self):
        time.sleep(0.01)
        write_bytes(prepared_video_path(self.configs["dataset"], 0, "high"), b"re-encoded, another length")
        self.assertIn("videos Gemini watched are not current", self.chain().annotation(0)[0])

    def test_an_alignment_fitted_on_stale_inputs_is_not_used(self):
        # The alignment was fitted on episode 1. Its record still matches the tracks and depth files, but the
        # videos behind episode 1's annotation changed, so those tracks are stale -- and so is the alignment.
        time.sleep(0.01)
        write_bytes(prepared_video_path(self.configs["dataset"], 1, "high"), b"re-encoded, another length")
        chain = self.chain()
        self.assertEqual(chain.tracks(0), [])
        self.assertTrue(chain.tracks(1))
        measurement = chain.measurement_inputs(0)
        self.assertTrue(measurement)
        self.assertTrue(all(p.startswith("alignment: alignment episode 1:") for p in measurement), measurement)
        # Episode 0's own geometry record is unchanged, yet it is not current.
        self.assertTrue(chain.geometry_chain(0))
        self.geometry.alignment_record_current = False
        self.assertIn("stale", self.chain().alignment()[0])

    def test_scales_need_the_same_current_artifacts_they_were_fitted_on(self):
        digests = {"annotation": file_digest(self.source.annotation_path(0)),
                   "geometry": file_digest(self.source.geometry_path(0))}
        scales = SimpleNamespace(provenance={"episodes": [0], "inputs": {"0": digests}})
        self.assertEqual(self.chain().scales(scales), {})
        time.sleep(0.01)
        self.save_npz(self.source.geometry_path(0), self.geometry.geometry_inputs(0))
        with open(self.source.geometry_path(0), "ab") as handle:
            handle.write(b"remeasured")
        problems = self.chain().scales(scales)
        self.assertIn(0, problems)
        self.assertTrue(any("fitted on another version" in p for p in problems[0]))


if __name__ == "__main__":
    unittest.main()
