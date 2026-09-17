"""Interval expansion, anchor coverage, consistency checks and scoped repairs of Gemini's answers."""

from __future__ import annotations

import copy
import unittest

from ..graphs.validate import (
    EpisodeAnnotation,
    assemble_past_only,
    build_annotation,
    coverage_issues,
    expand_intervals,
    fact_id_of,
    gemini_box_to_normalized,
    gemini_point_to_normalized,
    merge_keyframes,
    relations_to_intervals,
    repair_plan,
    replace_fact_range,
)
from . import synthetic as syn


def interval(relation, src, dst, label, start, end):
    return {"relation": relation, "src": src, "dst": dst, "label": label, "start_frame": start, "end_frame": end}


def build(spec, events, absolute, temporal, n=syn.N, **kwargs):
    return build_annotation(spec, episode_index=0, n_frames=n, fps=15.0, mode="full_episode", events_raw=events,
                            absolute_raw={"intervals": absolute}, temporal_raw={"intervals": temporal},
                            provenance={}, spec_identity={}, **kwargs)


def codes(annotation):
    return sorted({issue.code for issue in annotation.issues})


class Expansion(unittest.TestCase):
    def setUp(self):
        self.spec = syn.graph_spec()

    def test_reversed_pair_is_stored_canonically_with_the_mirrored_label(self):
        labels, issues, _ = expand_intervals(
            self.spec, [interval("height-offset", "pot", "banana", "far-above", 0, 9)], 10, "absolute")
        self.assertFalse(issues)
        index = self.spec.fact_index("banana", "pot", "height-offset")
        self.assertEqual(labels[index], ["far-below"] * 10)

    def test_aliases_resolve(self):
        labels, issues, _ = expand_intervals(
            self.spec, [interval("grasp", "right gripper", "Banana", "holds", 2, 3)], 5, "absolute")
        self.assertFalse(issues)
        self.assertEqual(labels[self.spec.fact_index("ee", "banana", "grasp")], [None, None, "holds", "holds", None])

    def test_conflicting_intervals_are_an_issue(self):
        _, issues, _ = expand_intervals(self.spec, [
            interval("grasp", "ee", "banana", "holds", 0, 5),
            interval("grasp", "ee", "banana", "not-holds", 4, 9)], 10, "absolute")
        self.assertEqual([i.code for i in issues], ["conflict"])
        self.assertEqual(issues[0].frames, [(4, 5)])
        self.assertEqual(issues[0].stage, "relations")

    def test_illegal_label_is_an_issue(self):
        _, issues, _ = expand_intervals(self.spec, [interval("contain", "banana", "pot", "holds", 0, 9)], 10,
                                        "absolute")
        self.assertEqual(issues[0].code, "illegal_label")

    def test_frames_outside_the_episode_are_an_issue(self):
        _, issues, _ = expand_intervals(self.spec, [interval("grasp", "ee", "banana", "holds", 0, 10)], 10, "absolute")
        self.assertEqual(issues[0].code, "out_of_range")

    def test_gaps_are_reported_with_their_frames(self):
        labels, _, _ = expand_intervals(self.spec, [interval("grasp", "ee", "banana", "holds", 0, 3),
                                                    interval("grasp", "ee", "banana", "holds", 7, 9)], 10, "absolute")
        issues = [i for i in coverage_issues(self.spec, labels, 10, "absolute")
                  if i.fact == ("ee", "banana", "grasp")]
        self.assertEqual(issues[0].frames, [(4, 6)])

    def test_temporal_labels_before_the_window_are_removed(self):
        K = self.spec.temporal_window
        labels, issues, warnings = expand_intervals(
            self.spec, [interval("planar-distance", "ee", "banana", "stable", 0, 20)], 21, "temporal")
        self.assertFalse(issues)
        series = labels[self.spec.fact_index("ee", "banana", "planar-distance")]
        self.assertEqual(series[:K], [None] * K)
        self.assertEqual(series[K], "stable")
        self.assertTrue(warnings)

    def test_temporal_label_on_a_physical_relation_is_an_issue(self):
        _, issues, _ = expand_intervals(self.spec, [interval("grasp", "ee", "banana", "stable", 5, 9)], 10, "temporal")
        self.assertEqual(issues[0].code, "temporal_not_applicable")

    def test_relations_by_fact_id_keep_absolute_and_temporal_boundaries_independent(self):
        fid = fact_id_of(self.spec, ("ee", "banana", "planar-distance"))
        raw = {"facts": [{"fact": fid, "absolute": [{"start": 0, "end": 19, "label": "near"}],
                          "temporal": [{"start": 5, "end": 11, "label": "decrease-fast"},
                                       {"start": 12, "end": 19, "label": "stable"}]},
                         {"fact": "F99", "absolute": [], "temporal": []}],
               "pass1_disagreements": [{"kind": "event", "start_frame": 3, "end_frame": 6,
                                        "description": "the grasp happens at frame 6"}]}
        absolute, temporal, issues, disagreements = relations_to_intervals(self.spec, raw)
        self.assertEqual([i.code for i in issues], ["unknown_fact"])
        self.assertEqual(len(absolute), 1)
        self.assertEqual([t["start_frame"] for t in temporal], [5, 12])
        self.assertEqual(disagreements[0]["kind"], "event")


class Consistency(unittest.TestCase):
    """Contradictions between answers are issues that ask for a correction, not warnings."""

    def setUp(self):
        self.spec = syn.graph_spec()
        self.events, self.absolute, self.temporal = syn.annotation(self.spec, raw=True)

    def replaced(self, relation, src, dst, labels):
        kept = [i for i in self.absolute if not (i["relation"] == relation and {i["src"], i["dst"]} == {src, dst})]
        return kept + labels

    def test_the_synthetic_episode_is_consistent(self):
        annotation = build(self.spec, self.events, self.absolute, self.temporal)
        self.assertTrue(annotation.valid, [i.message for i in annotation.issues])

    def test_success_without_the_banana_ever_in_the_pot_is_an_issue_to_reconcile(self):
        absolute = self.replaced("contain", "banana", "pot",
                                 [interval("contain", "banana", "pot", "not-holds", 0, syn.N - 1)])
        annotation = build(self.spec, self.events, absolute, self.temporal)
        self.assertFalse(annotation.valid)
        plan = repair_plan(annotation)
        self.assertIn("consistency", plan)
        self.assertIn(["banana", "pot", "contain"], plan["consistency"]["facts"])
        self.assertIn("completion_contain", codes(annotation))
        self.assertIn("place_label", codes(annotation))

    def test_a_grasp_event_far_from_the_grasp_label_is_an_issue(self):
        events = copy.deepcopy(self.events)
        for event in events["events"]:
            if event["type"] == "grasp" and event["object"] == "banana":
                event["frame"] = syn.GRASP_BANANA + 20
        events["keyframes"] = syn.anchors(self.spec, syn.N, [e["frame"] for e in events["events"]])
        annotation = build(self.spec, events, self.absolute, self.temporal)
        self.assertIn("grasp_event_label", codes(annotation))
        self.assertIn("grasp_label_event", codes(annotation))
        self.assertEqual(repair_plan(annotation)["consistency"]["facts"], [["ee", "banana", "grasp"]])

    def test_the_target_returns_to_the_banana_when_it_leaves_the_pot(self):
        # The banana falls back out at frame 90 and never returns; the target wrongly stays on the lid.
        absolute = self.replaced("contain", "banana", "pot", [
            interval("contain", "banana", "pot", "not-holds", 0, syn.IN_POT - 1),
            interval("contain", "banana", "pot", "dst-holds", syn.IN_POT, 89),
            interval("contain", "banana", "pot", "not-holds", 90, syn.N - 1)])
        annotation = build(self.spec, self.events, absolute, self.temporal)
        target = [i for i in annotation.issues if i.code == "target_labels"]
        self.assertTrue(target)
        self.assertEqual(target[0].frames[0][0], 90 + 5 + 1)

    def test_a_completion_without_its_event_is_sent_back_to_the_events_pass(self):
        events = copy.deepcopy(self.events)
        events["events"] = [e for e in events["events"] if e["type"] != "task_complete"]
        events["keyframes"] = syn.anchors(self.spec, syn.N, [e["frame"] for e in events["events"]])
        for annotation in (build(self.spec, events, self.absolute, self.temporal),
                           build_annotation(self.spec, episode_index=0, n_frames=syn.N, fps=15.0,
                                            mode="full_episode", events_raw=events, provenance={},
                                            spec_identity={}, events_only=True)):
            self.assertEqual(codes(annotation), ["completion_event"])
            self.assertEqual(list(repair_plan(annotation)), ["events"])

    def test_the_events_pass_alone_validates_without_relations(self):
        annotation = build_annotation(self.spec, episode_index=0, n_frames=syn.N, fps=15.0, mode="full_episode",
                                      events_raw=self.events, provenance={}, spec_identity={}, events_only=True)
        self.assertTrue(annotation.valid, [i.message for i in annotation.issues])

    def test_a_reported_disagreement_is_an_issue(self):
        annotation = build(self.spec, self.events, self.absolute, self.temporal,
                           extra_disagreements=[{"kind": "outcome", "start_frame": 150, "end_frame": 160,
                                                 "description": "the lid is not seated"}])
        self.assertEqual(codes(annotation), ["relations_disagreement"])


class Anchors(unittest.TestCase):
    def setUp(self):
        self.spec = syn.graph_spec()
        self.events, self.absolute, self.temporal = syn.annotation(self.spec, raw=True)

    def with_keyframes(self, keyframes):
        return build(self.spec, {**self.events, "keyframes": keyframes}, self.absolute, self.temporal)

    def test_one_high_box_per_entity_without_points_is_not_enough(self):
        keyframes = [{"frame": 0, "camera": "high", "object": e.id, "visible": True, "box_2d": [100, 100, 300, 300],
                      "points": []} for e in self.spec.entities]
        annotation = self.with_keyframes(keyframes)
        self.assertFalse(annotation.valid)
        self.assertEqual({"anchor_initial", "anchor_points", "anchor_event", "anchor_gap", "anchor_table_plane"}
                         - set(codes(annotation)), set())
        self.assertIn("anchors", repair_plan(annotation))

    def test_hidden_points_count_as_listed(self):
        keyframes = syn.anchors(self.spec, syn.N, [e["frame"] for e in self.events["events"]])
        for key in keyframes:
            if key["object"] == "banana":
                key["points"] = [{"name": p, "visible": False, "point": [0, 0]} for p in self.spec.points["banana"]]
        annotation = self.with_keyframes(keyframes)
        self.assertTrue(annotation.valid, [i.message for i in annotation.issues])
        banana = next(k for k in annotation.keyframes if k["object"] == "banana")
        self.assertEqual(banana["points"], {})
        self.assertEqual(banana["hidden_points"], sorted(self.spec.points["banana"]))

    def test_table_points_must_span_a_plane(self):
        keyframes = syn.anchors(self.spec, syn.N, [e["frame"] for e in self.events["events"]])
        for key in keyframes:
            if key["object"] == "table":
                key["points"] = [{"name": p["name"], "visible": True, "point": [800, 100 + 200 * i]}
                                 for i, p in enumerate(key["points"])]     # all on one image row
        self.assertIn("anchor_table_plane", codes(self.with_keyframes(keyframes)))
        for key in keyframes:
            if key["object"] == "table":
                key["points"] = [dict(p, point=[800, 500]) for p in key["points"]]    # all on one pixel
        self.assertIn("anchor_table_plane", codes(self.with_keyframes(keyframes)))

    def test_a_carried_object_needs_anchors_in_every_camera(self):
        keyframes = syn.anchors(self.spec, syn.N, [e["frame"] for e in self.events["events"]])
        carried = [k for k in keyframes if not (k["object"] == "banana" and k["camera"] == "high"
                                                and syn.GRASP_BANANA < k["frame"] < syn.IN_POT)]
        annotation = self.with_keyframes(carried)
        gaps = [i for i in annotation.issues if i.code == "anchor_gap"]
        self.assertTrue(gaps)
        self.assertEqual((gaps[0].anchors[0]["object"], gaps[0].anchors[0]["camera"]), ("banana", "high"))


class Repairs(unittest.TestCase):
    def setUp(self):
        self.spec = syn.graph_spec()
        self.events, self.absolute, self.temporal = syn.annotation(self.spec, raw=True)

    def test_a_missing_fact_is_asked_again_alone(self):
        broken = [i for i in self.absolute if not (i["relation"] == "grasp" and i["dst"] == "banana")]
        annotation = build(self.spec, self.events, broken, self.temporal)
        plan = repair_plan(annotation)
        self.assertEqual(list(plan), ["relations"])
        self.assertEqual(plan["relations"]["facts"], [["ee", "banana", "grasp"]])

    def test_patch_entries_outside_the_request_are_rejected(self):
        facts = [("ee", "banana", "grasp")]
        patch = [interval("grasp", "ee", "banana", "holds", 20, 40),              # straddles: cut to 25-40
                 interval("grasp", "ee", "banana", "not-holds", 60, 70),          # outside the frames
                 interval("contain", "banana", "pot", "dst-holds", 25, 40)]       # another fact
        merged, rejected = replace_fact_range(self.absolute, patch, self.spec, facts, (25, 40))
        self.assertEqual(len(rejected), 2)
        grasp = [i for i in merged if i["relation"] == "grasp" and i["dst"] == "banana"]
        spans = sorted((i["start_frame"], i["end_frame"], i["label"]) for i in grasp)
        self.assertIn((25, 40, "holds"), spans)
        self.assertIn((0, 24, "not-holds"), spans)
        labels, issues, _ = expand_intervals(self.spec, merged, syn.N, "absolute")
        self.assertFalse(issues)

    def test_keyframe_patches_only_fill_requested_slots(self):
        previous = [{"frame": 0, "camera": "high", "object": "banana", "visible": True, "box_2d": [1, 1, 2, 2],
                     "points": []}]
        patch = [{"frame": 0, "camera": "high", "object": "banana", "visible": True, "box_2d": [5, 5, 9, 9],
                  "points": []},
                 {"frame": 30, "camera": "high", "object": "lid", "visible": True, "box_2d": [5, 5, 9, 9],
                  "points": []}]
        merged, rejected = merge_keyframes(previous, patch, [{"object": "banana", "camera": "high",
                                                              "frames": [[0, 0]]}], self.spec)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["box_2d"], [5, 5, 9, 9])
        self.assertEqual(len(rejected), 1)


class WholeEpisode(unittest.TestCase):
    def setUp(self):
        self.spec = syn.graph_spec()

    def test_synthetic_episode_is_valid_and_round_trips(self):
        annotation = syn.annotation(self.spec)
        self.assertTrue(annotation.valid, [i.message for i in annotation.issues])
        back = EpisodeAnnotation.from_json(self.spec, annotation.to_json(self.spec))
        self.assertEqual(back.absolute, annotation.absolute)
        self.assertEqual(back.temporal, annotation.temporal)
        self.assertEqual(back.active_target, annotation.active_target)
        self.assertEqual(back.keyframes, annotation.keyframes)

    def test_an_earlier_annotation_format_is_refused(self):
        data = syn.annotation(self.spec).to_json(self.spec)
        data["format"] = "real_robot/kitchen-annotation-v1"
        with self.assertRaises(ValueError):
            EpisodeAnnotation.from_json(self.spec, data)

    def test_gemini_boxes_become_repository_boxes(self):
        self.assertEqual(gemini_box_to_normalized([100, 200, 300, 400]), [0.2, 0.4, 0.1, 0.3])
        self.assertEqual(gemini_point_to_normalized([250, 500]), [0.5, 0.25])
        self.assertIsNone(gemini_point_to_normalized([1200, 5]))


class PastOnly(unittest.TestCase):
    def test_labels_hold_between_updates_and_never_reach_backwards(self):
        spec = syn.graph_spec()
        facts = [{"relation": f.relation, "src": f.src, "dst": f.dst,
                  "label": spec.legal_labels(f.relation)[0], "temporal_label": "none"} for f in spec.facts]
        grasped = [dict(f, label="holds") if (f["relation"], f["dst"]) == ("grasp", "banana") else f for f in facts]
        updates = [
            {"frame": 0, "active_target": "banana", "facts": facts, "objects": [], "events_so_far": [],
             "task_complete": False},
            {"frame": 5, "active_target": "banana", "facts": grasped, "objects": [],
             "events_so_far": [{"type": "grasp", "object": "banana", "frame": 4}], "task_complete": False},
            {"frame": 10, "active_target": "lid", "facts": grasped, "objects": [], "events_so_far": [],
             "task_complete": True},
        ]
        events, absolute, _ = assemble_past_only(spec, updates, 14)
        labels, issues, _ = expand_intervals(spec, absolute["intervals"], 14, "absolute")
        self.assertFalse(issues)
        grasp = labels[spec.fact_index("ee", "banana", "grasp")]
        self.assertEqual(grasp[:5], ["not-holds"] * 5)       # the grasp seen at update 5 is not applied earlier
        self.assertEqual(grasp[5:], ["holds"] * 9)
        self.assertEqual(events["outcome"]["completion_frame"], 10)
        self.assertEqual(events["active_target"][-1], {"object": "lid", "start_frame": 10, "end_frame": 13})


if __name__ == "__main__":
    unittest.main()
