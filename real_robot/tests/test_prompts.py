"""Prompt templates, response schemas and the bin specification validator."""

from __future__ import annotations

import os
import re
import unittest

from scenegraph.core.relation_rules import CHANGE_LABELS, COMPAT_LABELS, SPATIAL_LABELS

from ..common import PROMPT_DIR
from ..graphs.validate import fact_ids
from ..preprocessing.annotate_episode import (
    anchors_schema,
    events_schema,
    past_only_schema,
    reconcile_schema,
    relations_schema,
)
from ..preprocessing.define_bins import (
    bin_schema,
    cameras_text,
    entities_text,
    facts_text,
    load_sections,
    points_text,
    reference_points_text,
    relations_text,
    render_bin_spec,
    render_template,
    scopes_in_use,
    unobserved_text,
    validate_bins,
)
from ..preprocessing.gemini_client import parse_json_text
from . import synthetic as syn

SUPPORTED = {"type", "properties", "required", "items", "enum", "minItems", "maxItems", "description",
             "minimum", "maximum"}


def keywords(schema, found=None):
    found = set() if found is None else found
    if isinstance(schema, dict):
        for key, value in schema.items():
            found.add(key)
            if key == "properties":
                for child in value.values():
                    keywords(child, found)
            elif isinstance(value, (dict, list)):
                keywords(value, found)
    elif isinstance(schema, list):
        for item in schema:
            keywords(item, found)
    return found


def valid_bins(spec):
    bins = {"reference_objects": [{"object": o, "dimension": "size", "approx_cm": 10, "visual_cue": "x"}
                                  for o in spec.object_ids],
            "spatial_bins": [], "physical_rules": [], "compatibility_bins": [], "temporal_bins": [],
            "general_rules": ["be consistent"]}
    edges = {"planar-distance": [0, 3, 8, 20, 40, 1000], "height-offset": [-1000, -10, -2, 2, 10, 1000]}
    for relation, scopes in scopes_in_use(spec).items():
        if relation in SPATIAL_LABELS:
            for scope in scopes:
                for i, label in enumerate(SPATIAL_LABELS[relation]):
                    bins["spatial_bins"].append({"relation": relation, "scope": scope, "label": label, "meaning": "m",
                                                 "lower_cm": edges[relation][i], "upper_cm": edges[relation][i + 1]})
                change = [-1000, -3, -0.5, 0.5, 3, 1000]
                for i, label in enumerate(CHANGE_LABELS):
                    bins["temporal_bins"].append({"family": relation, "scope": scope, "label": label, "meaning": "m",
                                                  "lower_cm_per_window": change[i],
                                                  "upper_cm_per_window": change[i + 1]})
        elif relation.endswith("compatibility"):
            bins["compatibility_bins"] += [{"relation": relation, "label": l, "meaning": "m"} for l in COMPAT_LABELS]
        else:
            from scenegraph.core.relation_rules import ABS_LABELS
            bins["physical_rules"] += [{"relation": relation, "label": l, "meaning": "m", "evidence": "e"}
                                       for l in ABS_LABELS[relation]]
    bins["temporal_bins"] += [{"family": "compatibility", "scope": "any", "label": l, "meaning": "m",
                               "lower_cm_per_window": 0, "upper_cm_per_window": 0} for l in CHANGE_LABELS]
    return bins


class Schemas(unittest.TestCase):
    def test_only_structured_output_keywords_are_used(self):
        spec = syn.graph_spec()
        ids = fact_ids(spec)
        for schema in (events_schema(spec), anchors_schema(spec), relations_schema(spec),
                       relations_schema(spec, ids[:3]), reconcile_schema(spec, ids[:2]), reconcile_schema(spec, []),
                       past_only_schema(spec), bin_schema(spec)):
            self.assertLessEqual(keywords(schema) - SUPPORTED - {"object", "string", "integer", "number",
                                                                   "boolean", "array"}, set(), schema)

    def test_a_repair_schema_names_only_the_requested_facts(self):
        spec = syn.graph_spec()
        ids = fact_ids(spec)[4:6]
        schema = relations_schema(spec, ids)
        self.assertEqual(schema["properties"]["facts"]["items"]["properties"]["fact"]["enum"], ids)
        self.assertNotIn("facts", reconcile_schema(spec, [])["properties"])

    def test_points_are_listed_as_visible_or_hidden(self):
        item = events_schema(syn.graph_spec())["properties"]["keyframes"]["items"]
        self.assertIn("visible", item["properties"]["points"]["items"]["properties"])

    def test_unobserved_is_offered_only_where_the_vocabulary_has_it(self):
        text = unobserved_text(syn.graph_spec())
        allowed = next(line for line in text.splitlines() if "may be `unobserved`" in line)
        refused = next(line for line in text.splitlines() if "has no `unobserved`" in line)
        self.assertNotIn("`contain`", allowed)
        self.assertIn("`contain`", refused)


class Templates(unittest.TestCase):
    def test_every_placeholder_is_filled(self):
        spec = syn.graph_spec()
        sections = load_sections(os.path.join(PROMPT_DIR, "episode_annotation.md"))
        self.assertEqual(set(sections), {"SPEC", "EPISODE", "PASS events", "PASS relations", "REPAIR events",
                                         "REPAIR anchors", "REPAIR relations", "RECONCILE", "PASS past_only"})
        bins = render_bin_spec(valid_bins(spec))
        values = {"EPISODE": 3, "N_FRAMES": 200, "LAST_FRAME": 199, "FPS": 15, "TASK": "task",
                  "CAMERAS": cameras_text(spec), "ENTITIES": entities_text(spec), "FACTS": facts_text(spec),
                  "REFERENCE_POINTS": reference_points_text(spec), "UNOBSERVED": unobserved_text(spec),
                  "BIN_SPEC": bins, "K": 5, "K_MINUS_ONE": 4, "K_SECONDS": "0.33", "POINTS": points_text(spec),
                  "KEYFRAME_EVERY": 15, "TEMPORAL_FACTS": "x", "CONTEXT": "c", "FRAME": 10, "FRAME_MINUS_K": 5,
                  "PREVIOUS_FRAME": 5, "PREVIOUS": "p", "ISSUES": "i", "VIDEO_START": 0, "VIDEO_END": 199,
                  "FACT_SCOPE": "every fact", "ANCHOR_REQUESTS": "a", "FACT_LIST": "`F00`", "RANGE_START": 0,
                  "RANGE_END": 199, "TEMPORAL_START": 5, "VIDEO_FPS": 5}
        for name, text in sections.items():
            rendered = render_template(text, values)
            self.assertIsNone(re.search(r"\{\{[A-Z_]+\}\}", rendered), name)
        with open(os.path.join(PROMPT_DIR, "bin_definition.md"), encoding="utf-8") as handle:
            rendered = render_template(handle.read(), {**values, "N_EPISODES": 3, "EPISODES": "e",
                                                        "RELATIONS": relations_text(spec)})
        self.assertIsNone(re.search(r"\{\{[A-Z_]+\}\}", rendered))

    def test_the_shared_specification_comes_first_and_names_no_episode(self):
        sections = load_sections(os.path.join(PROMPT_DIR, "episode_annotation.md"))
        self.assertNotIn("{{EPISODE}}", sections["SPEC"])
        self.assertNotIn("{{LAST_FRAME}}", sections["SPEC"])
        self.assertIn("{{BIN_SPEC}}", sections["SPEC"])

    def test_json_replies_with_fences(self):
        self.assertEqual(parse_json_text('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_json_text('Here it is: {"a": [1, 2]} done'), {"a": [1, 2]})


class Bins(unittest.TestCase):
    def test_a_complete_specification_validates(self):
        spec = syn.graph_spec()
        self.assertEqual(validate_bins(valid_bins(spec), spec), [])

    def test_gaps_missing_labels_and_off_centre_level_are_caught(self):
        spec = syn.graph_spec()
        bins = valid_bins(spec)
        planar = [b for b in bins["spatial_bins"] if b["relation"] == "planar-distance" and b["scope"] == "ee-object"]
        planar[2]["lower_cm"] = 9
        bins["spatial_bins"].remove(next(b for b in bins["spatial_bins"]
                                         if b["relation"] == "height-offset" and b["label"] == "far-above"
                                         and b["scope"] == "object-object"))
        level = next(b for b in bins["spatial_bins"] if b["label"] == "level" and b["scope"] == "ee-object")
        level["lower_cm"], level["upper_cm"] = 1, 3
        problems = validate_bins(bins, spec)
        self.assertTrue(any("starts at 9" in p for p in problems), problems)
        self.assertTrue(any("missing labels ['far-above']" in p for p in problems), problems)
        self.assertTrue(any("level must straddle" in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main()
