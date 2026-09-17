"""The annotation flow with a scripted Gemini: two main calls, repairs in order, nothing stale, nothing out of scope."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest

from ..common import load_configs, set_by_path, stable_hash
from ..graphs.validate import fact_id_of, fact_ids
from ..preprocessing.annotate_episode import EpisodeAnnotator, merge_anchor_requests, reuse_decision, usage_totals
from ..preprocessing.gemini_client import GeminiTruncated, TextPart
from . import synthetic as syn
from .test_gemini_client import USAGE
from .test_gemini_client import Scripted as ScriptedClient
from .test_gemini_client import settings as client_settings
from .test_prompts import valid_bins

TOKENS = {"input_tokens": 1000, "output_tokens": 50, "cached_tokens": 800, "thought_tokens": 0}


class Source:
    def __init__(self, root, n):
        self.root, self.n = root, n

    def lengths(self):
        return {0: self.n}

    def fps(self):
        return 15.0

    def annotation_path(self, episode):
        return os.path.join(self.root, "annotations", f"episode_{episode:06d}.json")


class Scripted:
    backend, model = "generate_content", "scripted"

    def __init__(self, respond):
        self.respond = respond
        self.names, self.texts = [], []

    def settings(self):
        return {"backend": self.backend, "model": self.model, "temperature": 0.0, "max_output_tokens": 100,
                "media_resolution": "default"}

    def generate_json(self, parts, schema, label):
        name = label.split("/")[-1]
        self.names.append(name)
        self.texts.append(parts[-1].text)
        answer = self.respond(name, schema)
        if isinstance(answer, BaseException):
            raise answer
        return answer, {"key": name, "cached": False, "usage_normalized": dict(TOKENS)}


class Annotator(EpisodeAnnotator):
    def video_parts(self, episode, start=None, end=None):
        return [TextPart(f"[videos {start}-{end}]")]


class Flow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.spec = syn.graph_spec()
        self.events, self.absolute, self.temporal = syn.annotation(self.spec, raw=True)
        self.good = syn.relations_answer(self.spec, self.absolute, self.temporal)
        self.by_id = {item["fact"]: item for item in self.good["facts"]}
        self.configs = load_configs(["dataset", "annotation", "graph"])
        set_by_path(self.configs["dataset"], "paths.annotations", os.path.join(self.tmp.name, "annotations"))
        bins = valid_bins(self.spec)
        self.bins = {"bins": bins, "bins_hash": stable_hash(bins)}

    def tearDown(self):
        self.tmp.cleanup()

    def annotator(self, respond):
        client = Scripted(respond)
        return Annotator(self.configs, mode="full_episode", client=client, source=Source(self.tmp.name, syn.N),
                         bins=self.bins), client

    def facts_for(self, schema, override=None):
        ids = schema["properties"]["facts"]["items"]["properties"]["fact"]["enum"]
        facts = [dict(self.by_id[fid]) for fid in ids]
        for fact in facts:
            if override and fact["fact"] in override:
                fact.update(override[fact["fact"]])
        return facts

    def id_of(self, *key):
        return fact_id_of(self.spec, key)

    def test_valid_answers_take_two_calls(self):
        def respond(name, schema):
            return self.events if name == "events" else {"facts": self.facts_for(schema), "pass1_disagreements": []}

        annotator, client = self.annotator(respond)
        annotation = annotator.annotate_full(0)
        self.assertTrue(annotation.valid, [i.message for i in annotation.issues])
        self.assertEqual(client.names, ["events", "relations"])
        self.assertEqual(annotation.provenance["usage"]["calls"], 2)
        # The shared specification comes first; the pass text last.
        self.assertIn("Your job: events", client.texts[0])

    def broken_events(self):
        """The outcome is a success, but no task_complete event is reported: wrong inside the events pass alone."""
        broken = copy.deepcopy(self.events)
        broken["events"] = [e for e in broken["events"] if e["type"] != "task_complete"]
        broken["keyframes"] = syn.anchors(self.spec, syn.N, [e["frame"] for e in broken["events"]])
        return broken

    def test_the_events_pass_is_repaired_before_relations_are_asked_for(self):
        broken = self.broken_events()

        def respond(name, schema):
            if name == "events":
                return broken
            if name == "events_repair1":
                return self.events
            if name == "relations":
                return {"facts": self.facts_for(schema), "pass1_disagreements": []}
            raise AssertionError(name)

        annotator, client = self.annotator(respond)
        annotation = annotator.annotate_full(0)
        self.assertTrue(annotation.valid, [i.message for i in annotation.issues])
        self.assertEqual(client.names, ["events", "events_repair1", "relations"])
        self.assertIn("no task_complete event", client.texts[1])
        # The relations pass is shown the corrected events, not the ones that were wrong.
        self.assertIn('"type": "task_complete"', client.texts[2])

    def test_relations_are_not_requested_while_the_events_pass_stays_invalid(self):
        broken = self.broken_events()

        def respond(name, schema):
            if name.startswith("events"):
                return broken
            raise AssertionError(name)

        annotator, client = self.annotator(respond)
        annotation = annotator.annotate_full(0)
        self.assertFalse(annotation.valid)
        self.assertEqual(client.names, ["events", "events_repair1", "events_repair2"])
        self.assertIn("relations_not_requested", {i.code for i in annotation.issues})

    def test_a_contradiction_is_resolved_by_reconciliation_from_the_video(self):
        contain = self.id_of("banana", "pot", "contain")
        never_inside = {contain: {"absolute": [{"start": 0, "end": syn.N - 1, "label": "not-holds"}]}}

        def respond(name, schema):
            if name == "events":
                return self.events
            if name == "relations":
                return {"facts": self.facts_for(schema, never_inside), "pass1_disagreements": []}
            if name == "consistency_repair1":
                return {k: self.events[k] for k in ("events", "active_target", "outcome")} | {
                    "facts": self.facts_for(schema)}
            raise AssertionError(name)

        annotator, client = self.annotator(respond)
        annotation = annotator.annotate_full(0)
        self.assertTrue(annotation.valid, [i.message for i in annotation.issues])
        self.assertEqual(client.names, ["events", "relations", "consistency_repair1"])
        self.assertIn("contain(pot holds banana) does not hold there", client.texts[-1])

    def test_repairs_see_the_answers_corrected_before_them(self):
        grasp, contain = self.id_of("ee", "banana", "grasp"), self.id_of("banana", "pot", "contain")
        never_inside = {contain: {"absolute": [{"start": 0, "end": syn.N - 1, "label": "not-holds"}]}}

        def respond(name, schema):
            if name == "events":
                return self.events
            if name == "relations":
                facts = [f for f in self.facts_for(schema, never_inside) if f["fact"] != grasp]
                return {"facts": facts, "pass1_disagreements": []}
            if name.startswith("relations_repair"):
                return {"facts": self.facts_for(schema), "pass1_disagreements": []}
            if name.startswith("consistency_repair"):
                return {k: self.events[k] for k in ("events", "active_target", "outcome")} | {
                    "facts": self.facts_for(schema)}
            raise AssertionError(name)

        annotator, client = self.annotator(respond)
        annotation = annotator.annotate_full(0)
        self.assertTrue(annotation.valid, [i.message for i in annotation.issues])
        self.assertEqual(client.names, ["events", "relations", "relations_repair1", "consistency_repair2"])
        # The reconciliation is shown the grasp labels the relations repair supplied, not the gap before it.
        self.assertIn(f"{syn.GRASP_BANANA}-{syn.RELEASE_BANANA - 1} holds", client.texts[-1])

    def test_patch_entries_outside_the_request_are_not_applied(self):
        grasp, contain = self.id_of("ee", "banana", "grasp"), self.id_of("banana", "pot", "contain")

        def respond(name, schema):
            if name == "events":
                return self.events
            if name == "relations":
                return {"facts": [f for f in self.facts_for(schema) if f["fact"] != grasp], "pass1_disagreements": []}
            wrong = {"fact": contain, "absolute": [{"start": 0, "end": syn.N - 1, "label": "not-holds"}],
                     "temporal": []}
            return {"facts": self.facts_for(schema) + [wrong], "pass1_disagreements": []}

        annotator, client = self.annotator(respond)
        annotation = annotator.annotate_full(0)
        self.assertTrue(annotation.valid, [i.message for i in annotation.issues])
        self.assertEqual(client.names, ["events", "relations", "relations_repair1"])
        self.assertGreaterEqual(annotation.repair_log[-1]["rejected_out_of_scope"], 1)

    def test_a_cut_off_answer_is_asked_for_in_halves(self):
        def respond(name, schema):
            if name == "events":
                return self.events
            if name == "relations":
                return GeminiTruncated("too long", [{"attempt": 1, "outcome": "truncated",
                                                     "usage_normalized": dict(TOKENS)}])
            return {"facts": self.facts_for(schema), "pass1_disagreements": []}

        annotator, client = self.annotator(respond)
        annotation = annotator.annotate_full(0)
        self.assertTrue(annotation.valid, [i.message for i in annotation.issues])
        self.assertEqual(client.names, ["events", "relations", "relations_a", "relations_b"])
        usage = annotation.provenance["usage"]
        self.assertEqual(usage["truncated_calls"], 1)
        # The cut-off answer is billed like the three that were used.
        self.assertEqual(usage["input_tokens"], 4000)
        self.assertEqual((usage["attempts"], usage["failed_attempts"]), (4, 1))

    def test_the_cost_of_an_episode_includes_failed_attempts(self):
        # A real client: the events answer is malformed once, then both passes answer.
        responses = [("{not json", USAGE, "STOP"), (json.dumps(self.events), USAGE, "STOP"),
                     (json.dumps(self.good), USAGE, "STOP")]
        client = ScriptedClient(client_settings(os.path.join(self.tmp.name, "cache")), responses)
        annotator = Annotator(self.configs, mode="full_episode", client=client, source=Source(self.tmp.name, syn.N),
                              bins=self.bins)
        annotation = annotator.annotate_full(0)
        self.assertTrue(annotation.valid, [i.message for i in annotation.issues])
        usage = annotation.provenance["usage"]
        self.assertEqual((usage["calls"], usage["attempts"], usage["failed_attempts"]), (2, 3, 1))
        self.assertEqual(usage["input_tokens"], 3 * USAGE["prompt_token_count"])

    def test_usage_totals_leave_cached_answers_out(self):
        totals = usage_totals([
            {"label": "a", "cached": False, "attempts": 2, "usage": {"input_tokens": 2400}},
            {"label": "b", "failed": "GeminiMalformed", "attempts": 2, "usage": {"input_tokens": 2400}},
            {"label": "c", "cached": True, "usage": {"input_tokens": 1200}},
        ])
        self.assertEqual(totals["input_tokens"], 4800)
        self.assertEqual((totals["calls"], totals["cached_calls"], totals["failed_calls"]), (3, 1, 1))
        self.assertEqual((totals["attempts"], totals["failed_attempts"]), (4, 3))

    def test_a_saved_annotation_is_reused_only_for_the_same_inputs(self):
        def respond(name, schema):
            return self.events if name == "events" else {"facts": self.facts_for(schema), "pass1_disagreements": []}

        annotator, _ = self.annotator(respond)
        identity = {"graph": "g", "bins": self.bins["bins_hash"], "prompt_version": "p", "videos": {"high": "v"}}
        path = annotator.save(annotator.annotate_full(0), identity)
        self.assertEqual(reuse_decision(path, identity, 2)[0], True)
        changed = {**identity, "bins": "other"}
        reuse, reason = reuse_decision(path, changed, 2)
        self.assertFalse(reuse)
        self.assertIn("bins", reason)
        self.assertEqual(reuse_decision(os.path.join(self.tmp.name, "missing.json"), identity, 2)[0], False)

    def test_event_anchor_requests_allow_the_event_tolerance(self):
        requests = merge_anchor_requests([{"object": "banana", "camera": "high", "frames": [[30, 30]], "rule": "event"},
                                          {"object": "banana", "camera": "high", "frames": [[31, 40]], "rule": "gap"}],
                                         syn.N, 1)
        self.assertEqual(requests[0]["frames"], [[29, 40]])
        self.assertEqual(len(fact_ids(self.spec)), len(self.spec.facts))


if __name__ == "__main__":
    unittest.main()
