"""The annotation flow against a scripted Gemini."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from ..graphs.validate import Scope, fact_ids, full_scope
from ..preprocessing.annotate_episode import EpisodeAnnotator, answer_schema, labels_text
from ..preprocessing.gemini_client import GeminiClient, TextPart, VideoPart
from . import synthetic

N = 90
USAGE = {"prompt_token_count": 1000, "candidates_token_count": 100}


class FakeSource:
    def __init__(self, task: str):
        self.graph = synthetic.graph_config()
        self.task = task

    def fps(self) -> float:
        return 30.0

    def spec(self, episode):
        return self.graph.spec(self.task)


class Scripted(GeminiClient):
    def __init__(self, cfg, responses):
        super().__init__(cfg)
        self.responses = list(responses)
        self.schemas = []
        self.parts = []

    def _generate_content(self, parts, schema):
        self.parts.append(parts)
        self.schemas.append(schema)
        return self.responses.pop(0)(schema)


def answering(answer, extra_facts=()):
    """A response to whatever the request's schema asks for, from ``answer``."""
    def respond(schema):
        props = schema["properties"]
        out = {"notes": ""}
        if "active_target" in props:
            out["active_target"] = answer.get("active_target", [])
        if "facts" in props:
            variants = props["facts"]["items"]["anyOf"]
            wanted = [variant["properties"]["fact"]["enum"][0] for variant in variants]
            out["facts"] = [f for f in answer["facts"] if f["fact"] in wanted] + list(extra_facts)
        if "boxes" in props:
            variants = props["boxes"]["items"]["anyOf"]
            wanted = {(variant["properties"]["entity"]["enum"][0],
                       variant["properties"]["camera"]["enum"][0]) for variant in variants}
            out["boxes"] = [b for b in answer["boxes"]
                            if (b["entity"], b["camera"]) in wanted]
        return json.dumps(out), USAGE, "STOP"
    return respond


def truncated(schema):
    return '{"facts": [', USAGE, "FinishReason.MAX_TOKENS"


class Flow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.configs = synthetic.configs()
        gemini = self.configs["annotation"]["gemini"]
        gemini.update(cache_dir=self.tmp.name, retry_base_seconds=0.0)
        cameras = {}
        for camera in ("top", "wrist"):
            path = os.path.join(self.tmp.name, f"{camera}.mp4")
            with open(path, "wb") as handle:
                handle.write(camera.encode())
            cameras[camera] = {"path": path, "sha256": camera}
        self.prepared = {"rows": N, "fps": 10.0, "stride": 3, "shown": 30, "cameras": cameras}

    def tearDown(self):
        self.tmp.cleanup()

    def annotator(self, task, responses):
        client = Scripted(self.configs["annotation"]["gemini"], responses)
        return EpisodeAnnotator(self.configs, FakeSource(task), client=client), client

    def test_a_complete_answer_takes_one_request_over_both_videos(self):
        spec = synthetic.spec("blue_on_red")
        annotator, client = self.annotator("blue_on_red", [answering(synthetic.complete_answer(spec, N))])
        annotation = annotator.annotate(3, self.prepared)
        self.assertTrue(annotation.valid, [i.message for i in annotation.issues])
        self.assertEqual(len(client.schemas), 1)
        kinds = [type(p) for p in client.parts[0]]
        self.assertEqual(kinds, [TextPart, VideoPart, VideoPart, TextPart])
        self.assertEqual([p.fps for p in client.parts[0][1:3]], [10.0, 10.0])
        self.assertEqual(set(client.schemas[0]["properties"]), {"active_target", "facts", "boxes", "notes"})
        self.assertEqual(annotation.usage["calls"], 1)

    def test_a_cut_off_answer_is_asked_for_in_pieces(self):
        spec = synthetic.spec("cubes_in_cup")
        answer = synthetic.complete_answer(spec, N)
        annotator, client = self.annotator("cubes_in_cup", [truncated, answering(answer), answering(answer)])
        annotation = annotator.annotate(3, self.prepared)
        self.assertTrue(annotation.valid, [i.message for i in annotation.issues])
        self.assertEqual([sorted(s["properties"]) for s in client.schemas[1:]],
                         [["active_target", "boxes", "notes"], ["facts", "notes"]])

    def test_repairs_ask_only_for_what_is_missing_and_reject_the_rest(self):
        spec = synthetic.spec("banana_pot_lid")
        complete = synthetic.complete_answer(spec, N)
        first = dict(complete, facts=[f for f in complete["facts"] if f["fact"] != "F07"],
                     boxes=[b for b in complete["boxes"] if (b["entity"], b["camera"]) != ("lid", "wrist")])
        stray = synthetic.fact_entry(spec, "F00", N)
        annotator, client = self.annotator("banana_pot_lid",
                                           [answering(first), answering(complete, extra_facts=[stray])])
        annotation = annotator.annotate(3, self.prepared)
        self.assertTrue(annotation.valid, [i.message for i in annotation.issues])
        repair = client.schemas[1]["properties"]
        fact_variant = repair["facts"]["items"]["anyOf"][0]
        box_variant = repair["boxes"]["items"]["anyOf"][0]
        self.assertEqual(fact_variant["properties"]["fact"]["enum"], ["F07"])
        self.assertEqual(box_variant["properties"]["entity"]["enum"], ["lid"])
        self.assertEqual(box_variant["properties"]["camera"]["enum"], ["wrist"])
        self.assertNotIn("active_target", repair)
        self.assertEqual(annotation.repair_log[0]["issues_after"], 0)
        self.assertEqual(len(annotation.repair_log[0]["rejected"]), 1)
        self.assertIn("F07", client.parts[1][-1].text)

    def test_every_prompt_is_fully_filled(self):
        for task in synthetic.TASKS:
            spec = synthetic.spec(task)
            annotator, _ = self.annotator(task, [])
            texts = [annotator.spec_text(spec), annotator.episode_text(spec, 3, self.prepared),
                     annotator.job_text(spec, full_scope(spec), N)]
            for text in texts:
                self.assertNotIn("{{", text)
            self.assertIn(fact_ids(spec)[-1], texts[0])

    def test_identity_changes_when_prompt_values_change(self):
        annotator, _ = self.annotator("blue_on_red", [])
        spec = annotator.source.spec(3)
        before = annotator.input_identity(spec, self.prepared)
        spec.target_rule += " changed"
        self.assertNotEqual(before, annotator.input_identity(spec, self.prepared))
        before = annotator.input_identity(spec, self.prepared)
        annotator.dataset_cfg["source"]["camera_descriptions"]["top"] = "changed view"
        self.assertNotEqual(before, annotator.input_identity(spec, self.prepared))

    def test_label_definitions_cover_every_relation_in_use(self):
        labels = self.configs["labels"]
        for task in synthetic.TASKS:
            spec = synthetic.spec(task)
            text = labels_text(spec, labels)
            for relation in spec.relations_in_use:
                self.assertIn(f"`{relation}`", text, f"{task}: {relation}")

    def test_prompt_requires_visual_measurement_release_and_final_frame_audit(self):
        annotator, _ = self.annotator("cubes_in_cup", [])
        text = annotator.spec_text(annotator.source.spec(3))
        for required in (
            "Do not create smooth ramps",
            "The jaws need not return to their fully open pose",
            "including failed attempts, slips, releases, retries and reversals",
            "Explicitly inspect both views of the final shown frame",
            "not the name of the action phase",
            "Containment alone does not prove",
        ):
            self.assertIn(required, text)

    def test_schema_uses_exact_fact_vocabularies_and_entity_camera_slots(self):
        spec = synthetic.spec("banana_pot_lid")
        ids = fact_ids(spec)
        scope = Scope(facts=[ids[0], ids[2]], boxes=[("lid", "top"), ("pot", "wrist")])
        schema = answer_schema(spec, scope)
        self.assertFalse(schema["additionalProperties"])

        variants = schema["properties"]["facts"]["items"]["anyOf"]
        by_id = {v["properties"]["fact"]["enum"][0]: v for v in variants}
        for fid in scope.facts:
            fact = spec.facts[ids.index(fid)]
            labels = by_id[fid]["properties"]["absolute"]["items"]["properties"]["label"]["enum"]
            self.assertEqual(labels, [*spec.legal_labels(fact.relation), None])

        box_variants = schema["properties"]["boxes"]["items"]["anyOf"]
        exact_slots = {(v["properties"]["entity"]["enum"][0],
                        v["properties"]["camera"]["enum"][0]) for v in box_variants}
        self.assertEqual(exact_slots, set(scope.boxes))


if __name__ == "__main__":
    unittest.main()
