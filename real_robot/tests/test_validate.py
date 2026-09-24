"""What validation accepts, what it refuses, and what a repair may change."""

from __future__ import annotations

import copy
import unittest

import numpy as np

from ..graphs.pack import interpolate_track, pack_episode
from ..graphs.vocabulary import build_vocab
from ..graphs.validate import (
    EpisodeAnnotation,
    Scope,
    build_annotation,
    fact_ids,
    merge_answer,
    repair_scope,
)
from . import synthetic

N = 90


def build(spec, answer, n=N):
    return build_annotation(spec, episode_index=7, n_frames=n, fps=30.0, answer=answer,
                            settings=synthetic.settings())


def codes(annotation, part=None):
    return sorted(i.code for i in annotation.issues if part is None or i.part == part)


class Facts(unittest.TestCase):
    def setUp(self):
        self.spec = synthetic.spec("banana_pot_lid")
        self.answer = synthetic.complete_answer(self.spec, N)

    def test_a_complete_answer_is_valid(self):
        annotation = build(self.spec, self.answer)
        self.assertEqual(annotation.issues, [])
        self.assertTrue(all(label is not None for series in annotation.absolute for label in series))

    def test_a_gap_is_a_coverage_issue_on_that_fact_and_those_frames(self):
        self.answer["facts"][3]["absolute"][1]["start"] = 40
        annotation = build(self.spec, self.answer)
        issue = annotation.issues[0]
        self.assertEqual((issue.code, issue.fact, issue.frames), ("missing_coverage", "F03", [(30, 39)]))

    def test_overlapping_labels_conflict(self):
        entry = self.answer["facts"][0]
        entry["absolute"].append({"start": 10, "end": 12, "label": entry["absolute"][1]["label"]})
        self.assertEqual(codes(build(self.spec, self.answer)), ["conflict"])

    def test_an_illegal_label_is_refused(self):
        self.answer["facts"][0]["absolute"][0]["label"] = "far"
        self.assertIn("illegal_label", codes(build(self.spec, self.answer)))

    def test_an_interval_past_the_end_is_clipped_with_a_warning(self):
        self.answer["facts"][0]["absolute"][1]["end"] = N + 3
        annotation = build(self.spec, self.answer)
        self.assertTrue(annotation.valid)
        self.assertTrue(any("clipped" in w for w in annotation.warnings))

    def test_temporal_labels_before_the_window_are_dropped(self):
        temporal = next(f for f in self.answer["facts"] if f["temporal"])
        temporal["temporal"][0]["start"] = 0
        annotation = build(self.spec, self.answer)
        self.assertTrue(annotation.valid)
        index = fact_ids(self.spec).index(temporal["fact"])
        self.assertIsNone(annotation.temporal[index][self.spec.temporal_window - 1])

    def test_an_unknown_fact_id_is_an_issue(self):
        self.answer["facts"].append({"fact": "F99", "absolute": [], "temporal": []})
        self.assertIn("unknown_fact", codes(build(self.spec, self.answer)))

    def test_explicit_unknown_is_covered_but_never_fabricated_in_packed_graph(self):
        self.answer["facts"][0]["absolute"][0]["label"] = None
        annotation = build(self.spec, self.answer)
        self.assertTrue(annotation.valid)
        arrays, complete = pack_episode(self.spec, build_vocab(synthetic.graph_config()), annotation)
        self.assertFalse(complete[0])
        self.assertTrue(complete[-1])
        self.assertEqual(np.count_nonzero(arrays["graph_edge_rel"][0]), len(self.spec.facts) - 1)
        self.assertIsNone(EpisodeAnnotation.from_json(self.spec, annotation.to_json(self.spec)).absolute[0][0])

    def test_unknown_and_known_overlapping_intervals_conflict(self):
        self.answer["facts"][0]["absolute"].append({"start": 0, "end": 3, "label": None})
        self.assertIn("conflict", codes(build(self.spec, self.answer)))


class Target(unittest.TestCase):
    def test_a_target_outside_the_task_and_missing_frames_are_issues(self):
        spec = synthetic.spec("blue_on_red")
        answer = synthetic.complete_answer(spec, N)
        answer["active_target"] = [{"start": 0, "end": 40, "object": "blue_cube"},
                                   {"start": 41, "end": N - 1, "object": "red_cube"}]
        self.assertEqual(codes(build(spec, answer), "target"), ["bad_target", "target_coverage"])


class Boxes(unittest.TestCase):
    def setUp(self):
        self.spec = synthetic.spec("blue_on_red")
        self.answer = synthetic.complete_answer(self.spec, N)

    def test_a_missing_entity_camera_pair_is_an_issue(self):
        self.answer["boxes"] = [b for b in self.answer["boxes"] if (b["entity"], b["camera"]) != ("red_cube", "wrist")]
        annotation = build(self.spec, self.answer)
        self.assertEqual([(i.code, i.slot) for i in annotation.issues], [("missing_boxes", ("red_cube", "wrist"))])

    def test_keyframes_too_far_apart_are_an_issue(self):
        entry = self.answer["boxes"][0]
        entry["keyframes"] = [k for k in entry["keyframes"] if k["frame"] not in (15, 30)]
        issue = build(self.spec, self.answer).issues[0]
        self.assertEqual((issue.code, issue.frames), ("box_gap", [(0, 45)]))

    def test_first_and_last_frame_boxes_are_required(self):
        for frame in (0, N - 1):
            answer = copy.deepcopy(self.answer)
            answer["boxes"][0]["keyframes"] = [k for k in answer["boxes"][0]["keyframes"] if k["frame"] != frame]
            self.assertIn("box_gap", codes(build(self.spec, answer)))

    def test_out_of_range_coordinates_and_string_visibility_are_rejected(self):
        self.answer["boxes"][0]["keyframes"][0]["box_2d"] = [-100, 0, 1200, 1000]
        self.assertIn("bad_box", codes(build(self.spec, self.answer)))
        self.answer["boxes"][0]["keyframes"][0]["visible"] = "false"
        self.assertIn("bad_keyframe", codes(build(self.spec, self.answer)))

    def test_a_malformed_visible_box_is_an_issue_and_a_hidden_one_is_not(self):
        entry = self.answer["boxes"][0]
        entry["keyframes"][1] = {"frame": 15, "visible": True, "box_2d": [500, 500, 400, 600]}
        entry["keyframes"][2] = {"frame": 30, "visible": False, "box_2d": [0, 0, 0, 0]}
        bad = [i for i in build(self.spec, self.answer).issues if i.code == "bad_box"]
        self.assertEqual([i.frames for i in bad], [[(15, 15)]])


class Interpolation(unittest.TestCase):
    def test_boxes_move_linearly_hold_into_a_hidden_keyframe_and_stay_hidden(self):
        a, b = [0.1, 0.3, 0.2, 0.4], [0.3, 0.5, 0.2, 0.4]
        keyframes = [{"frame": 0, "visible": True, "box": a}, {"frame": 10, "visible": True, "box": b},
                     {"frame": 20, "visible": False, "box": None}, {"frame": 30, "visible": True, "box": a}]
        boxes, visible = interpolate_track(keyframes, 35)
        np.testing.assert_allclose(boxes[5], [0.2, 0.4, 0.2, 0.4], atol=1e-6)
        np.testing.assert_allclose(boxes[19], b)
        self.assertTrue(visible[:20].all())
        self.assertFalse(visible[20:30].any())
        self.assertTrue(visible[30:].all())
        np.testing.assert_allclose(boxes[34], a)


class Repairs(unittest.TestCase):
    def setUp(self):
        self.spec = synthetic.spec("cubes_in_cup")
        self.answer = synthetic.complete_answer(self.spec, N)

    def test_the_scope_names_exactly_what_the_issues_name(self):
        broken = copy.deepcopy(self.answer)
        broken["facts"] = [f for f in broken["facts"] if f["fact"] != "F04"]
        broken["boxes"] = [b for b in broken["boxes"] if (b["entity"], b["camera"]) != ("cup", "top")]
        broken["active_target"] = broken["active_target"][:1]
        scope = repair_scope(build(self.spec, broken), self.spec)
        self.assertEqual((scope.target, scope.facts, scope.boxes), (True, ["F04"], [("cup", "top")]))

    def test_a_patch_replaces_only_its_scope(self):
        scope = Scope(facts=["F02"])
        patch = {"facts": [synthetic.fact_entry(self.spec, "F02", N), synthetic.fact_entry(self.spec, "F05", N)],
                 "active_target": [{"start": 0, "end": N - 1, "object": "red_cube"}]}
        patch["facts"][0]["absolute"] = [{"start": 0, "end": N - 1, "label": "holds"}]
        merged, rejected = merge_answer(self.spec, self.answer, patch, scope)
        self.assertEqual(len(rejected), 2)
        self.assertEqual(merged["active_target"], self.answer["active_target"])
        f02 = [f for f in merged["facts"] if f["fact"] == "F02"]
        self.assertEqual(f02, [patch["facts"][0]])
        self.assertEqual(len(merged["facts"]), len(self.answer["facts"]))


class Serialisation(unittest.TestCase):
    def test_a_saved_annotation_reads_back_identically(self):
        spec = synthetic.spec("banana_pot_lid")
        annotation = build(spec, synthetic.complete_answer(spec, N))
        again = EpisodeAnnotation.from_json(spec, annotation.to_json(spec))
        self.assertEqual(again.absolute, annotation.absolute)
        self.assertEqual(again.temporal, annotation.temporal)
        self.assertEqual(again.active_target, annotation.active_target)
        self.assertEqual(again.boxes, annotation.boxes)

    def test_another_tasks_annotation_is_refused(self):
        spec = synthetic.spec("banana_pot_lid")
        data = build(spec, synthetic.complete_answer(spec, N)).to_json(spec)
        with self.assertRaises(ValueError):
            EpisodeAnnotation.from_json(synthetic.spec("blue_on_red"), data)


if __name__ == "__main__":
    unittest.main()
