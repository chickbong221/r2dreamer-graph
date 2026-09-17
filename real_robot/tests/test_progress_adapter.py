"""The kitchen progress schedule compiles against the graph configuration."""

from __future__ import annotations

import copy
import json
import unittest

from scenegraph.core.schedule import ScheduleError, compile_schedule

from ..common import repo_path
from ..graphs.validate import holder_label
from ..graphs.vocabulary import build_vocab
from ..models.progress_adapter import compile_kitchen_schedule, schedule_asset_view
from . import synthetic as syn

SCHEDULE = "real_robot/configs/kitchen_schedule.json"


class AssetView(unittest.TestCase):
    def setUp(self):
        self.spec = syn.graph_spec()
        self.objects, self.members, self.bins = schedule_asset_view(self.spec)

    def test_tokens_follow_the_configured_facts(self):
        banana = self.members["actor:banana"]["interaction_types"]
        self.assertIn("grasp", banana)
        self.assertIn("contain", banana)
        self.assertNotIn("grasp", self.members["actor:pot"]["interaction_types"])

    def test_components_follow_the_compatibility_facts(self):
        self.assertIn("grasp_components", self.objects["actor:banana"])
        self.assertIn("contain_components", self.objects["actor:pot"])
        self.assertIn("key_components", self.objects["actor:banana"])
        self.assertIn("support_components", self.objects["actor:pot"])
        self.assertIn("bottom_components", self.objects["actor:lid"])

    def test_bins_exist_for_both_scopes_in_use(self):
        self.assertIn("ee-object-planar-distance", self.bins)
        self.assertIn("object-object-planar-distance", self.bins)
        self.assertIn("ee-manipuland-height-offset", self.bins)


class Compilation(unittest.TestCase):
    def setUp(self):
        self.spec = syn.graph_spec()
        self.vocab = build_vocab(self.spec)
        self.schedule = compile_kitchen_schedule(self.spec, self.vocab, SCHEDULE)

    def test_phase_weights_are_a_partition_of_one(self):
        self.assertAlmostEqual(sum(p.weight for p in self.schedule.phases), 1.0, places=6)
        for phase in self.schedule.phases:
            self.assertAlmostEqual(sum(c.weight for c in phase.clauses), phase.weight, places=5)

    def test_holder_clauses_resolve_to_the_stored_direction(self):
        place = next(p for p in self.schedule.phases if p.name == "place_banana")
        clause = place.clauses[0]
        self.assertEqual((clause.src_key, clause.dst_key), ("actor:banana", "actor:pot"))
        self.assertEqual(clause.labels, (holder_label(self.spec, "pot", "banana"),))
        settle = next(p for p in self.schedule.phases if p.name == "settle_lid")
        self.assertEqual(settle.clauses[0].labels, (holder_label(self.spec, "pot", "lid"),))

    def test_lid_phases_are_gated_on_the_banana_being_in_the_pot(self):
        for name in ("approach_lid", "grasp_lid", "seat_lid", "settle_lid"):
            phase = next(p for p in self.schedule.phases if p.name == name)
            gates = [(c.relation, c.labels) for c in phase.requires]
            self.assertIn(("contain", (holder_label(self.spec, "pot", "banana"),)), gates, name)

    def test_every_slot_names_entities_the_packer_writes(self):
        ids = set(self.vocab.entity.token_to_id.values())
        for slot in self.schedule.slots:
            self.assertIn(slot[1], ids)
            self.assertIn(slot[2], ids)

    def test_a_clause_the_graph_cannot_score_is_refused(self):
        with open(repo_path(SCHEDULE), encoding="utf-8") as handle:
            raw = json.load(handle)
        broken = copy.deepcopy(raw)
        broken["phases"][0]["clauses"][0] = {"relation": "grasp", "src": "ee", "dst": "pot",
                                             "labels": ["holds"], "weight": 0.015}
        objects, members, bins = schedule_asset_view(self.spec)
        with self.assertRaises(ScheduleError):
            compile_schedule(broken, objects, members, bins, self.vocab.entity, sites={}, structural=set())


if __name__ == "__main__":
    unittest.main()
