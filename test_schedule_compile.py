"""Schedule compilation: the translation layer between a written schedule and
what the runtime actually emits.

Every test here is a way a schedule can look right and score nothing.
"""

import unittest

from scenegraph.adapters.graph_vocab import EntityVocab
from scenegraph.core.schedule import (
    ScheduleError,
    compile_schedule,
    scorable_relations,
)

OBJECTS = {
    "actor:cubeA": {"grasp_components": [{}], "contact_components": [{}],
                    "bottom_components": [{}]},
    "actor:cubeB": {"contact_components": [{}], "support_components": [{}],
                    "bottom_components": [{}]},
    "actor:table": {"contact_components": [{}], "support_components": [{}]},
}
MEMBERS = {
    "actor:cubeA": {"interaction_types": ["contact", "grasp", "support"]},
    "actor:cubeB": {"interaction_types": ["contact", "support"]},
    "actor:table": {"interaction_types": ["contact", "support"]},
}
BINS = {"planar-distance": [0.1, 0.2, 0.3, 0.4],
        "height-offset": [-0.2, -0.1, 0.1, 0.2]}
VOCAB = EntityVocab(token_to_id={"<pad>": 0, "ee": 1, "actor:cubeA": 2,
                                 "actor:cubeB": 3, "actor:table": 4})


def _clause(relation, src, dst, labels=("holds",), weight=1.0, **kw):
    return {"relation": relation, "src": src, "dst": dst,
            "labels": list(labels), "weight": weight, **kw}


def _schedule(phases, roles=None):
    return {
        "_schema_version": 1, "env_id": "T-v1",
        "roles": roles or {"movable": "actor:cubeA",
                           "destination": "actor:cubeB"},
        "phases": phases,
    }


def _done(relation, src, dst, labels=("holds",), **kw):
    """A completion clause. Carries labels like any other -- 'this phase is
    done' is a fact about the graph, not a special case. ``labels=None`` is for
    directional relations, whose label the holder decides."""
    out = {"relation": relation, "src": src, "dst": dst, **kw}
    if labels is not None:
        out["labels"] = list(labels)
    return out


def _phase(name, clauses, completion, weight=1.0):
    return {"name": name, "weight": weight, "clauses": clauses,
            "completion": completion}


def _compile(raw, objects=None, members=None):
    return compile_schedule(raw, objects or OBJECTS, members or MEMBERS,
                            BINS, VOCAB)


class DirectionalLabelTest(unittest.TestCase):
    """Which of src-holds/dst-holds a holder resolves to is a property of the
    key strings, not the physics. Hardcoding it inverts on a rename."""

    def _settle(self, holder):
        raw = _schedule([_phase(
            "settle",
            [{"relation": "support", "src": "movable", "dst": "destination",
              "holder": holder, "weight": 1.0}],
            _done("support", "movable", "destination", labels=None,
                  holder=holder))])
        return _compile(raw).phases[0].clauses[0]

    def test_the_later_sorting_key_holding_gives_dst_holds(self):
        # cubeA sorts first, so "cubeB supports cubeA" is dst-holds.
        self.assertEqual(self._settle("destination").labels, ("dst-holds",))

    def test_the_earlier_sorting_key_holding_gives_src_holds(self):
        self.assertEqual(self._settle("movable").labels, ("src-holds",))

    def test_a_directional_relation_without_a_holder_is_rejected(self):
        raw = _schedule([_phase(
            "settle",
            [{"relation": "support", "src": "movable", "dst": "destination",
              "weight": 1.0}],
            _done("support", "movable", "destination", labels=None,
                  holder="destination"))])
        with self.assertRaises(ScheduleError) as cm:
            _compile(raw)
        self.assertIn("holder", str(cm.exception))

    def test_a_holder_on_a_symmetric_relation_is_rejected(self):
        """It would be silently ignored, and the author clearly meant
        something by writing it."""
        raw = _schedule([_phase(
            "reach",
            [_clause("planar-distance", "movable", "destination",
                     labels=("near",), holder="movable")],
            _done("contact", "movable", "destination"))])
        with self.assertRaises(ScheduleError) as cm:
            _compile(raw)
        self.assertIn("not directional", str(cm.exception))


class PairOrientationTest(unittest.TestCase):
    def test_object_pairs_are_stored_in_sorted_key_order(self):
        raw = _schedule([_phase(
            "reach",
            [_clause("planar-distance", "destination", "movable",
                     labels=("near",))],
            _done("contact", "movable", "destination"))])
        clause = _compile(raw).phases[0].clauses[0]
        self.assertEqual((clause.src_key, clause.dst_key),
                         ("actor:cubeA", "actor:cubeB"))

    def test_end_effector_edges_are_not_sorted(self):
        """ee-object edges are always emitted ee -> object, whatever the key
        would sort to."""
        raw = _schedule([_phase(
            "grasp", [_clause("grasp", "ee", "movable")],
            _done("grasp", "ee", "movable"))])
        clause = _compile(raw).phases[0].clauses[0]
        self.assertEqual((clause.src_key, clause.dst_key), ("ee", "actor:cubeA"))

    def test_swapping_endpoints_mirrors_an_antisymmetric_label(self):
        """height-offset written destination->movable is stored the other way
        round, and 'above' becomes 'below' with it."""
        raw = _schedule([_phase(
            "carry",
            [_clause("height-offset", "destination", "movable",
                     labels=("above",))],
            _done("contact", "movable", "destination"))])
        clause = _compile(raw).phases[0].clauses[0]
        self.assertEqual(clause.labels, ("below",))

    def test_a_label_that_does_not_swap_is_left_alone(self):
        raw = _schedule([_phase(
            "carry",
            [_clause("height-offset", "movable", "destination",
                     labels=("above",))],
            _done("contact", "movable", "destination"))])
        self.assertEqual(_compile(raw).phases[0].clauses[0].labels, ("above",))


class ScorabilityTest(unittest.TestCase):
    def test_a_clause_with_no_components_behind_it_is_rejected(self):
        """The runtime emits it as 'unobserved' -- the same label a pair too
        far apart gets -- so nothing downstream could tell it apart."""
        objects = dict(OBJECTS)
        objects["actor:cubeA"] = {"contact_components": [{}]}   # no grasp
        raw = _schedule([_phase(
            "reach",
            [_clause("grasp-compatibility", "ee", "movable",
                     labels=("match",))],
            _done("contact", "ee", "movable"))])
        with self.assertRaises(ScheduleError) as cm:
            _compile(raw, objects=objects)
        self.assertIn("not scorable", str(cm.exception))
        self.assertIn("zero for the whole episode", str(cm.exception))

    def test_a_clause_needing_absent_bins_is_rejected(self):
        raw = _schedule([_phase(
            "carry",
            [_clause("planar-distance", "movable", "destination",
                     labels=("near",))],
            _done("contact", "movable", "destination"))])
        with self.assertRaises(ScheduleError):
            compile_schedule(raw, OBJECTS, MEMBERS, {}, VOCAB)

    def test_contain_without_mined_components_is_rejected(self):
        raw = _schedule([_phase(
            "settle",
            [{"relation": "contain", "src": "movable", "dst": "destination",
              "holder": "destination", "weight": 1.0}],
            _done("contain", "movable", "destination", labels=None, holder="destination"))])
        with self.assertRaises(ScheduleError) as cm:
            _compile(raw)
        self.assertIn("contain", str(cm.exception))


class WeightTest(unittest.TestCase):
    def test_phase_weights_must_sum_to_one(self):
        raw = _schedule([_phase(
            "a", [_clause("grasp", "ee", "movable", weight=0.5)],
            _done("grasp", "ee", "movable"), weight=0.5)])
        with self.assertRaises(ScheduleError) as cm:
            _compile(raw)
        self.assertIn("sum to", str(cm.exception))

    def test_clause_weights_must_reach_the_phase_weight(self):
        """A phase that cannot reach its own weight can never be completed,
        and the potential silently stops being bounded by one."""
        raw = _schedule([_phase(
            "a", [_clause("grasp", "ee", "movable", weight=0.4)],
            _done("grasp", "ee", "movable"), weight=1.0)])
        with self.assertRaises(ScheduleError) as cm:
            _compile(raw)
        self.assertIn("never be completed", str(cm.exception))

    def test_a_valid_schedule_keeps_its_weights(self):
        raw = _schedule([_phase(
            "a", [_clause("grasp", "ee", "movable", weight=1.0)],
            _done("grasp", "ee", "movable"))])
        compiled = _compile(raw)
        self.assertAlmostEqual(sum(p.weight for p in compiled.phases), 1.0)


class RoleTest(unittest.TestCase):
    def test_two_roles_sharing_an_entity_are_rejected(self):
        raw = _schedule(
            [_phase("a", [_clause("grasp", "ee", "movable")],
                    _done("grasp", "ee", "movable"))],
            roles={"movable": "actor:cubeA", "destination": "actor:cubeA"})
        with self.assertRaises(ScheduleError) as cm:
            _compile(raw)
        self.assertIn("cannot tell them apart", str(cm.exception))

    def test_a_role_naming_a_non_member_is_rejected(self):
        raw = _schedule(
            [_phase("a", [_clause("grasp", "ee", "movable")],
                    _done("grasp", "ee", "movable"))],
            roles={"movable": "actor:ghost"})
        with self.assertRaises(ScheduleError) as cm:
            _compile(raw)
        self.assertIn("not a whitelist member", str(cm.exception))

    def test_roles_resolve_to_entity_vocabulary_ids(self):
        raw = _schedule([_phase(
            "a", [_clause("grasp", "ee", "movable", weight=1.0)],
            _done("grasp", "ee", "movable"))])
        self.assertEqual(_compile(raw).role_entity_ids,
                         {"movable": 2, "destination": 3})

    def test_an_unknown_role_in_a_clause_is_rejected(self):
        raw = _schedule([_phase(
            "a", [_clause("grasp", "ee", "gripper")],
            _done("grasp", "ee", "movable"))])
        with self.assertRaises(ScheduleError) as cm:
            _compile(raw)
        self.assertIn("unknown role", str(cm.exception))

    def test_a_clause_relating_an_entity_to_itself_is_rejected(self):
        raw = _schedule([_phase(
            "a", [_clause("contact", "movable", "movable")],
            _done("contact", "ee", "movable"))])
        with self.assertRaises(ScheduleError):
            _compile(raw)


class InventoryTest(unittest.TestCase):
    def test_support_compatibility_needs_a_surface_and_a_bottom(self):
        scorable = scorable_relations(OBJECTS, MEMBERS, BINS)
        self.assertTrue(
            scorable["actor:cubeA / actor:cubeB"]["support-compatibility"])
        objects = dict(OBJECTS)
        objects["actor:cubeB"] = {"contact_components": [{}]}
        scorable = scorable_relations(objects, MEMBERS, BINS)
        self.assertFalse(
            scorable["actor:cubeA / actor:cubeB"]["support-compatibility"])

    def test_a_member_with_no_interaction_types_scores_only_spatially(self):
        members = dict(MEMBERS)
        members["actor:goal"] = {"interaction_types": []}
        scorable = scorable_relations(OBJECTS, members, BINS)
        pair = scorable["actor:cubeA / actor:goal"]
        self.assertTrue(pair["planar-distance"])
        self.assertFalse(pair["contact"])


if __name__ == "__main__":
    unittest.main()


class ShippedScheduleTest(unittest.TestCase):
    """The six real schedules against the six real mined assets.

    This is the gate the plan calls compile-time validation: a schedule that
    passes here can be scored every frame, and one that does not never reaches
    training.
    """

    CONFIGS = "scenegraph/configs"
    SCHEDULES = "scenegraph/configs/schedules"

    def _tasks(self):
        import os
        return sorted(n[:-5] for n in os.listdir(self.SCHEDULES)
                      if n.endswith(".json"))

    def _compile(self, env_id):
        import os
        from scenegraph.adapters.graph_vocab import build_entity_vocab
        from scenegraph.core.schedule import compile_from_files
        vocab = build_entity_vocab(
            os.path.join(self.CONFIGS, "subtask_whitelists", env_id))
        return compile_from_files(env_id, self.SCHEDULES, self.CONFIGS, vocab)

    def test_every_shipped_schedule_compiles(self):
        for env_id in self._tasks():
            with self.subTest(task=env_id):
                self._compile(env_id)

    def test_weights_are_bounded_by_one(self):
        for env_id in self._tasks():
            with self.subTest(task=env_id):
                compiled = self._compile(env_id)
                self.assertAlmostEqual(
                    sum(p.weight for p in compiled.phases), 1.0, places=9)

    def test_directional_labels_resolve_per_task(self):
        """Four tasks, four different answers -- which is why the schedules
        name a holder role instead of a literal src-holds/dst-holds."""
        expected = {
            "StackCube-v1": ("support", "dst-holds"),
            "PlaceSphere-v1": ("support", "src-holds"),
            "PegInsertionSide-v1": ("contain", "src-holds"),
            "PlugCharger-v1": ("contain", "dst-holds"),
        }
        for env_id, (relation, label) in expected.items():
            with self.subTest(task=env_id):
                settle = self._compile(env_id).phases[-1].completion
                self.assertEqual(settle.relation, relation)
                self.assertEqual(settle.labels, (label,))

    def test_a_clause_written_backwards_is_stored_canonically(self):
        """PullCubeTool's reach_cube is written tool -> cube; object pairs are
        stored in sorted key order, so it must come back cube -> tool."""
        phase = next(p for p in self._compile("PullCubeTool-v1").phases
                     if p.name == "reach_cube")
        for clause in phase.clauses:
            self.assertEqual(clause.src_key, "actor:cube")
            self.assertEqual(clause.dst_key, "actor:l_shape_tool")
