"""Schedule compilation: the translation layer between a written schedule and
what the runtime actually emits.

Every test here is a way a schedule can look right and score nothing.
"""

import unittest

from scenegraph.adapters.graph_vocab import (
    EE_TOKEN,
    PAD_TOKEN,
    EntityVocab,
)
from scenegraph.core.spatial_metrics import (
    SPATIAL_SCOPES,
    spatial_bin_key,
)
from scenegraph.core.schedule import (
    ScheduleError,
    compile_schedule,
    scorable_relations,
)
from scenegraph.core.sites import parse_site_declarations
from scenegraph.adapters.site_providers import PULL_SUCCESS_RADIUS
from scenegraph.core.spatial_metrics import (
    OBJECT_REGION_PLANAR_KEY as _REGION_PD,
    OBJECT_SITE_HEIGHT_KEY as _SITE_HO,
    OBJECT_SITE_PLANAR_KEY as _SITE_PD,
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
BINS = {
    k: ([0.1, 0.2, 0.3, 0.4] if k.endswith("planar-distance")
        else [-0.2, -0.1, 0.1, 0.2])
    for scope in SPATIAL_SCOPES
    for k in (spatial_bin_key(scope, "planar-distance"),
              spatial_bin_key(scope, "height-offset"))
}
VOCAB = EntityVocab(token_to_id={PAD_TOKEN: 0, EE_TOKEN: 1, "actor:cubeA": 2,
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


class CompletionConjunctionTest(unittest.TestCase):
    def _completion(self):
        return {"all_of": [
            _done("contact", "movable", "destination"),
            _done("contact", "movable", "surface"),
        ]}

    def _compiled(self):
        raw = _schedule(
            [
                _phase(
                    "approach",
                    [_clause("contact", "ee", "movable", weight=0.5)],
                    _done("contact", "ee", "movable"),
                    weight=0.5,
                ),
                _phase(
                    "settle",
                    [
                        _clause(
                            "contact", "movable", "destination", weight=0.25),
                        _clause("contact", "movable", "surface", weight=0.25),
                    ],
                    self._completion(),
                    weight=0.5,
                ),
            ],
            roles={
                "movable": "actor:cubeA",
                "destination": "actor:cubeB",
                "surface": "actor:table",
            },
        )
        return _compile(raw)

    def test_all_of_compiles_every_completion_clause(self):
        phase = self._compiled().phases[-1]
        self.assertEqual(len(phase.completions), 2)
        self.assertIs(phase.completion, phase.completions[0])

    def test_all_of_must_be_a_non_empty_list(self):
        raw = _schedule([_phase(
            "a", [_clause("grasp", "ee", "movable")],
            {"all_of": []},
        )])
        with self.assertRaisesRegex(ScheduleError, "non-empty list"):
            _compile(raw)


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

    def test_the_four_reviewed_phase_sequences_are_pinned(self):
        expected = {
            "PickCube-v1": ["approach", "grasp", "carry_to_goal"],
            "PlaceSphere-v1": ["approach", "grasp", "transport", "settle"],
            "PegInsertionSide-v1": [
                "approach", "grasp", "reach_hole_mouth",
                "containment_alignment", "contain",
            ],
            "PullCubeTool-v1": [
                "approach_tool", "grasp_tool", "reach_cube", "pull",
            ],
        }
        for env_id, names in expected.items():
            with self.subTest(task=env_id):
                self.assertEqual(
                    [p.name for p in self._compile(env_id).phases], names)

    def test_peg_alignment_is_gated_by_live_mouth_and_contact(self):
        compiled = self._compile("PegInsertionSide-v1")
        self.assertEqual(compiled.roles["site"], "spatial:hole_site")
        reach = next(p for p in compiled.phases
                     if p.name == "reach_hole_mouth")
        self.assertEqual(
            {c.relation for c in reach.completions}, {"reached", "contact"})
        alignment = next(p for p in compiled.phases
                         if p.name == "containment_alignment")
        self.assertEqual(
            {c.relation for c in alignment.requires}, {"reached", "contact"})
        self.assertTrue(all(c.weight == 0.0 for c in alignment.requires))

    def test_pull_uses_the_live_region_and_exact_completion(self):
        compiled = self._compile("PullCubeTool-v1")
        self.assertEqual(
            compiled.roles["reference"], "spatial:pull_goal_region")
        self.assertNotIn("actor:table-workspace", compiled.roles.values())
        pull = compiled.phases[-1]
        self.assertEqual(pull.completion.relation, "reached")
        self.assertEqual(
            [c.relation for c in pull.clauses].count("planar-distance"), 4)
        self.assertEqual(
            [c.relation for c in pull.clauses].count("reached"), 1)


# --------------------------------------------------------------------------- #
# goal sites
# --------------------------------------------------------------------------- #
SITE_MEMBERS = dict(MEMBERS, **{"actor:goal": {"interaction_types": []}})
SITE_VOCAB = EntityVocab(token_to_id={
    PAD_TOKEN: 0, EE_TOKEN: 1, "actor:cubeA": 2, "actor:cubeB": 3,
    "actor:table": 4, "actor:goal": 5,
})
SITES = parse_site_declarations({"actor:goal": {
    "site_type": "point", "subject": "actor:cubeA", "metric": "euclidean",
    "source": "origin", "provider": "pick_cube_goal",
    "provenance": "T-v1.evaluate: is_obj_placed",
}})
SITE_ROLES = {"movable": "actor:cubeA", "destination": "actor:goal"}

# A pair with the mined components a containment ladder needs, so the gate
# tests exercise the gate rather than tripping over scorability first.
CONTAIN_OBJECTS = dict(
    OBJECTS,
    **{
        "actor:cubeA": dict(OBJECTS["actor:cubeA"], key_components=[{}]),
        "actor:cubeB": dict(OBJECTS["actor:cubeB"], contain_components=[{}]),
    },
)
CONTAIN_MEMBERS = dict(
    MEMBERS,
    **{
        "actor:cubeA": {"interaction_types": ["contact", "contain"]},
        "actor:cubeB": {"interaction_types": ["contact", "contain"]},
    },
)


def _compile_sites(raw, sites=SITES, members=None):
    return compile_schedule(raw, OBJECTS, members or SITE_MEMBERS, BINS,
                            SITE_VOCAB, sites=sites)


def _compile_contain(raw):
    return compile_schedule(raw, CONTAIN_OBJECTS, CONTAIN_MEMBERS, BINS, VOCAB)


class ReachedScorabilityTest(unittest.TestCase):
    """``reached`` is the only relation whose threshold is not mined, so it is
    scorable exactly where an asset declared a goal -- never as a generic
    proximity alias any two objects would satisfy."""

    def _reached_schedule(self, src="movable", dst="destination"):
        return _schedule([_phase(
            "goal",
            [_clause("reached", src, dst, labels=("holds",), weight=1.0)],
            _done("reached", src, dst),
        )], roles=SITE_ROLES)

    def test_a_declared_pair_is_scorable(self):
        phase = _compile_sites(self._reached_schedule()).phases[0]
        self.assertEqual(phase.clauses[0].relation, "reached")

    def test_without_a_declaration_reached_is_rejected(self):
        with self.assertRaises(ScheduleError) as ctx:
            _compile_sites(self._reached_schedule(), sites={})
        self.assertIn("reached", str(ctx.exception))

    def test_an_undeclared_pair_is_rejected(self):
        """A task may declare one goal and still not have licensed every other
        pair in the scene to claim one."""
        raw = _schedule([_phase(
            "goal",
            [_clause("reached", "movable", "other", labels=("holds",),
                     weight=1.0)],
            _done("reached", "movable", "other"),
        )], roles={"movable": "actor:cubeA", "other": "actor:cubeB"})
        with self.assertRaises(ScheduleError):
            _compile_sites(raw)

    def test_scorable_table_marks_only_the_declared_pair(self):
        table = scorable_relations(OBJECTS, SITE_MEMBERS, BINS, SITES)
        self.assertTrue(table["actor:cubeA / actor:goal"]["reached"])
        self.assertFalse(table["actor:cubeA / actor:cubeB"]["reached"])
        self.assertFalse(table["ee / actor:cubeA"]["reached"])


class RequiresGateTest(unittest.TestCase):
    """A phase's gate decides when its rungs pay. It is weightless, it is read
    from the current frame only, and the facts it names have to be facts the
    frame actually carries."""

    def _gated(self, requires, weight=1.0):
        phase = _phase(
            "aligned",
            [_clause("contain-compatibility", "movable", "destination",
                     labels=("match",), weight=weight)],
            _done("contact", "movable", "destination"),
        )
        phase["requires"] = requires
        return _schedule([phase], roles={"movable": "actor:cubeA",
                                         "destination": "actor:cubeB"})

    def _ok_gate(self):
        return {"all_of": [
            _clause("contact", "movable", "destination", labels=("holds",),
                    weight=0.0),
        ]}

    def test_a_gate_compiles_into_the_phase(self):
        phase = _compile_contain(self._gated(self._ok_gate())).phases[0]
        self.assertEqual(len(phase.requires), 1)
        self.assertEqual(phase.requires[0].relation, "contact")

    def test_a_phase_without_a_gate_has_none(self):
        raw = _schedule([_phase(
            "grasp",
            [_clause("grasp", "ee", "movable", labels=("holds",), weight=1.0)],
            _done("grasp", "ee", "movable"),
        )])
        self.assertEqual(_compile(raw).phases[0].requires, ())

    def test_a_gate_carries_no_weight(self):
        """Paying for the gate as well would double-count the milestone and
        break the phase's own weight sum."""
        raw = self._gated({"all_of": [
            _clause("contact", "movable", "destination", labels=("holds",),
                    weight=0.5),
        ]})
        with self.assertRaises(ScheduleError) as ctx:
            _compile_contain(raw)
        self.assertIn("weight", str(ctx.exception))

    def test_a_gate_does_not_count_toward_the_phase_weight(self):
        """The clause sum must still equal the phase weight on its own."""
        schedule = _compile_contain(self._gated(self._ok_gate(), weight=1.0))
        self.assertAlmostEqual(schedule.phases[0].weight, 1.0)
        self.assertAlmostEqual(
            sum(c.weight for c in schedule.phases[0].clauses), 1.0)
        self.assertEqual(
            [c.weight for c in schedule.phases[0].requires], [0.0])

    def test_gate_slots_are_looked_up_by_the_runtime(self):
        """A gate the frame cannot read has to invalidate the phase, which
        means the runtime must be resolving its slot in the first place."""
        schedule = _compile_contain(self._gated(self._ok_gate()))
        self.assertIn(schedule.phases[0].requires[0].slot, schedule.slots)

    def test_a_malformed_gate_is_rejected(self):
        for bad in ({"any_of": []}, {"all_of": []}, {"all_of": {}},
                    {"all_of": [1]}, {"all_of": [], "extra": 1}):
            with self.assertRaises(ScheduleError):
                _compile_contain(self._gated(bad))

    def test_a_gate_naming_an_unscorable_relation_is_rejected(self):
        """Same bar as a rung: a gate on a fact the assets cannot produce
        would hold the phase shut for the whole episode."""
        raw = self._gated({"all_of": [
            {"relation": "support", "src": "movable", "dst": "destination",
             "holder": "destination", "weight": 0.0},
        ]})
        with self.assertRaises(ScheduleError):
            _compile_contain(raw)


class ReachedImpliesFinestRungTest(unittest.TestCase):
    """``reached`` sits in the last phase of PickCube and PullCubeTool, so
    there is no later phase to backfill their credit. The phase pays in full
    only if *every* one of its clauses is satisfied at success -- which means
    the environment's own tolerance has to imply the finest rung of each
    ladder beside it. If it does not, a successful episode tops out below 1.0
    and the shaping term never closes.

    The tolerance is read live at runtime; the documented default is used here
    so the arithmetic is checked against the mined bins on every run, and the
    live value is covered by the server integration pass.
    """

    import json as _json
    import os as _os

    CONFIGS = _os.path.join("scenegraph", "configs")
    # PickCubeEnv.goal_thresh
    PICK_CUBE_GOAL_THRESH = 0.025

    def _assets(self, env_id):
        from scenegraph.core.schedule import load_assets
        return load_assets(env_id, self.CONFIGS)

    def _schedule(self, env_id):
        path = self._os.path.join(self.CONFIGS, "schedules", f"{env_id}.json")
        with open(path) as handle:
            return self._json.load(handle)

    def test_pick_cube_reached_implies_very_near(self):
        _, _, bins, _, _ = self._assets("PickCube-v1")
        edges = bins[spatial_bin_key(SPATIAL_SCOPES[1], "planar-distance")]
        very_near_below = edges[0]
        # reached bounds the 3-D distance, so it bounds the planar one too.
        self.assertLess(
            self.PICK_CUBE_GOAL_THRESH, very_near_below,
            "reached can fire while planar-distance is still 'near', so a "
            "successful placement would leave the finest planar rung unpaid "
            "and the potential would terminate below 1.0",
        )

    def test_pick_cube_reached_implies_level(self):
        _, _, bins, _, _ = self._assets("PickCube-v1")
        edges = bins[spatial_bin_key(SPATIAL_SCOPES[1], "height-offset")]
        level_upper = edges[2]
        self.assertLess(self.PICK_CUBE_GOAL_THRESH, level_upper)

    def test_the_reached_phase_is_the_last_one(self):
        """If it were not, cumulative credit would paper over a mismatch and
        this whole check would be moot -- and silently so."""
        raw = self._schedule("PickCube-v1")
        with_reached = [
            i for i, phase in enumerate(raw["phases"])
            if any(c["relation"] == "reached" for c in phase["clauses"])
        ]
        self.assertEqual(with_reached, [len(raw["phases"]) - 1])

    def test_every_shipped_reached_clause_sits_beside_its_completion(self):
        """A weighted ``reached`` that is not also the phase's completion would
        pay for the goal without ending the phase on it."""
        import glob
        for path in sorted(glob.glob(
                self._os.path.join(self.CONFIGS, "schedules", "*.json"))):
            with open(path) as handle:
                raw = self._json.load(handle)
            for phase in raw["phases"]:
                weighted = [c for c in phase["clauses"]
                            if c["relation"] == "reached" and c["weight"] > 0]
                if not weighted:
                    continue
                completion = phase.get("completion") or {}
                items = completion.get("all_of", [completion])
                self.assertIn(
                    "reached", [i.get("relation") for i in items],
                    f"{path}: phase {phase['name']!r} pays for 'reached' but "
                    "does not complete on it",
                )


class FinalPhaseImplicationTest(unittest.TestCase):
    """No mined clause in a final phase may be stricter than its exact
    ``reached`` condition.

    ``reached`` sits in the last phase, so no later phase backfills its
    credit: the phase pays in full only when every clause in it is satisfied
    at success. A ladder rung binned by an equal-population quantile can drift
    tighter than the environment's fixed tolerance as the demos concentrate
    near the goal, and then a successful episode leaves that rung unpaid and
    the potential terminates below 1.0 -- silently, and only visible as a
    training curve that plateaus.

    The tolerances are the environments' own. PickCube's is read live from
    ``goal_thresh``; the documented default is used here so the arithmetic is
    re-checked against the mined bins on every run. PegInsertionSide is absent
    on purpose: its ``reached`` is not in its final phase, and its final phase
    is a physical ``contain`` milestone with no mined threshold at all.
    """

    import json as _json
    import os as _os

    CONFIGS = _os.path.join("scenegraph", "configs")
    TOLERANCE = {
        "PickCube-v1": 0.025,                    # PickCubeEnv.goal_thresh
        "PullCubeTool-v1": PULL_SUCCESS_RADIUS,  # evaluate: dist < 0.6
    }

    # Best label first: how tight each relation's ladder gets.
    ORDER = {
        "planar-distance": ["very-near", "near", "medium", "far", "very-far"],
        "height-offset": ["level", "below", "above", "far-below", "far-above"],
    }

    def _schedule(self, env_id):
        path = self._os.path.join(self.CONFIGS, "schedules", f"{env_id}.json")
        if not self._os.path.isfile(path):
            return None
        with open(path) as handle:
            return self._json.load(handle)

    def _assets(self, env_id):
        from scenegraph.core.schedule import load_assets
        return load_assets(env_id, self.CONFIGS)

    def _bin_key(self, relation, sites, src, dst):
        """Which mined scale labels this pair, mirroring the runtime rule."""
        from scenegraph.core.sites import SITE_PREFIX, SITE_REGION
        for key in (src, dst):
            entry = (sites or {}).get(key)
            if entry is None or not key.startswith(SITE_PREFIX):
                continue
            if entry.site_type == SITE_REGION:
                return _REGION_PD if relation == "planar-distance" else None
            return (_SITE_PD if relation == "planar-distance" else _SITE_HO)
        return f"object-object-{relation}"

    def _satisfied_interval(self, relation, labels, edges):
        """``(low, high)`` of the measurement values this clause accepts.

        Bins are upper-exclusive and ascending, so a label's interval is
        ``[edges[i-1], edges[i])`` with open ends at the extremes.
        """
        names = {"planar-distance": ["very-near", "near", "medium", "far",
                                     "very-far"],
                 "height-offset": ["far-below", "below", "level", "above",
                                   "far-above"]}[relation]
        indices = sorted(names.index(l) for l in labels)
        lo = -float("inf") if indices[0] == 0 else edges[indices[0] - 1]
        hi = float("inf") if indices[-1] == len(names) - 1 else edges[indices[-1]]
        return lo, hi

    def test_no_final_phase_clause_is_stricter_than_reached(self):
        for env_id, tolerance in sorted(self.TOLERANCE.items()):
            raw = self._schedule(env_id)
            if raw is None:
                continue                      # schedule not written yet
            final = raw["phases"][-1]
            if not any(c["relation"] == "reached" for c in final["clauses"]):
                continue
            _, _, bins, sites, _ = self._assets(env_id)
            roles = raw["roles"]
            for clause in final["clauses"]:
                relation = clause["relation"]
                if relation not in self.ORDER:
                    continue
                src = roles.get(clause["src"], clause["src"])
                dst = roles.get(clause["dst"], clause["dst"])
                key = self._bin_key(relation, sites, src, dst)
                edges = bins.get(key)
                if not edges:
                    continue
                lo, hi = self._satisfied_interval(
                    relation, clause["labels"], edges)
                with self.subTest(env=env_id, relation=relation,
                                  labels=clause["labels"]):
                    # reached bounds the 3-D distance, so it bounds both the
                    # planar distance and the absolute height offset.
                    self.assertLess(
                        tolerance, hi,
                        f"{env_id}/{final['name']}: 'reached' fires at "
                        f"{tolerance} but {relation} {clause['labels']} needs "
                        f"< {hi:.4f}. A successful episode would leave this "
                        "rung unpaid and the potential would top out below 1.0.",
                    )
                    if relation == "height-offset":
                        self.assertLessEqual(
                            lo, -tolerance,
                            f"{env_id}/{final['name']}: {relation} "
                            f"{clause['labels']} excludes a cube {tolerance}m "
                            "*below* the goal, which 'reached' accepts.",
                        )

    def test_pick_cube_has_no_quantile_rung_at_the_goal(self):
        """The 'very-near'-only planar rung was removed: it fired at a mined
        quantile within 10% of goal_thresh, and 'reached' already is that rung,
        exactly and in 3-D."""
        raw = self._schedule("PickCube-v1")
        final = raw["phases"][-1]
        planar = [c["labels"] for c in final["clauses"]
                  if c["relation"] == "planar-distance"]
        self.assertNotIn(["very-near"], planar)
        self.assertTrue(planar, "the coarser planar ladder must remain")

    def test_the_tolerances_are_the_environments_own(self):
        """Not numbers chosen here. PullCubeTool's is the constant the site
        provider uses, so the two cannot drift apart."""
        from scenegraph.adapters.site_providers import PULL_SUCCESS_RADIUS as R
        self.assertEqual(self.TOLERANCE["PullCubeTool-v1"], R)
        self.assertEqual(R, 0.6)
