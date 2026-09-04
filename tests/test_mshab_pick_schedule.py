"""The shipped tidy_house Pick schedule, scored against constructed frames.

Approach the object, acquire the grasp, return to rest while still holding it.
Completion is ``grasp AND reached`` -- both, on the same frame -- and the three
ways that pair can be half-satisfied are the cases worth testing, because each
one is a trajectory the environment would not call a success and the shaping
must not either:

* the gripper sits at rest holding nothing (it starts there);
* the object is grasped somewhere far from rest;
* the object is dropped on the way back.

Nothing here adjusts a bin, a threshold, a weight or a completion rule to make
a case pass. A schedule that cannot separate those trajectories is a schedule
to report, not to retune.
"""

import json
import unittest
from pathlib import Path

from scenegraph.adapters.graph_vocab import (
    EE_TOKEN,
    PAD_TOKEN,
    EntityVocab,
    build_absolute_vocab,
    build_relation_vocab,
)
from scenegraph.core.schedule import (
    ACTIVE_TARGET_ENTITY_ID,
    ACTIVE_TARGET_ROLE,
    ACTIVE_TARGET_ROW,
    compile_schedule,
)
from scenegraph.core.sites import SITE_EE_REST, parse_site_declarations

SCHEDULE = Path("scenegraph/configs/schedules/tidy_house/pick.json")
SITES = Path("scenegraph/configs/sites/tidy_house/pick.json")
TARGET = "actor:004_sugar_box"

# Rows, matching the packer's reserved layout.
ROW_EE, ROW_TARGET, ROW_SITE = 0, ACTIVE_TARGET_ROW, 2
VOCAB = EntityVocab(token_to_id={
    PAD_TOKEN: 0, EE_TOKEN: 1, TARGET: 2, SITE_EE_REST: 3})

MEMBERS = {
    TARGET: {"interaction_types": ["contact", "grasp"], "family": "manipuland"},
    SITE_EE_REST: {"interaction_types": [], "kind": "spatial"},
}
OBJECTS = {TARGET: {"grasp_components": [{}], "contact_components": [{}]}}
# Shaped like the pilot's mined asset: ee-object and per-family scales, plus
# the dedicated end-effector-to-site pair.
BINS = {
    "ee-object-planar-distance": [0.15, 0.30, 0.45, 0.60],
    "ee-manipuland-height-offset": [-0.12, -0.03, 0.03, 0.12],
    "ee-site-planar-distance": [0.15, 0.30, 0.45, 0.60],
    "ee-site-height-offset": [-0.45, -0.12, 0.12, 0.45],
    "grasp-compatibility": [1 / 3, 2 / 3],
    "contact-compatibility": [1 / 3, 2 / 3],
    "support-compatibility": [1 / 3, 2 / 3],
    "contain-compatibility": [1 / 3, 2 / 3],
}


def _schedule():
    with open(SCHEDULE) as handle:
        raw = json.load(handle)
    with open(SITES) as handle:
        sites = parse_site_declarations(
            json.load(handle)["sites"], where="tidy_house/pick")
    return compile_schedule(raw, OBJECTS, MEMBERS, BINS, VOCAB, sites=sites)


class ShapeTest(unittest.TestCase):
    """The frozen structure, asserted so a later edit cannot drift it."""

    def setUp(self):
        with open(SCHEDULE) as handle:
            self.raw = json.load(handle)

    def test_three_phases_in_order(self):
        self.assertEqual([p["name"] for p in self.raw["phases"]],
                         ["approach", "acquire", "return_to_rest"])

    def test_the_agreed_weights(self):
        self.assertEqual([p["weight"] for p in self.raw["phases"]],
                         [0.35, 0.20, 0.45])

    def test_the_weights_sum_to_one(self):
        self.assertAlmostEqual(
            sum(p["weight"] for p in self.raw["phases"]), 1.0)

    def test_each_phase_reaches_its_own_weight(self):
        """A phase whose clauses cannot sum to its weight can never complete."""
        for phase in self.raw["phases"]:
            with self.subTest(phase=phase["name"]):
                self.assertAlmostEqual(
                    sum(c["weight"] for c in phase["clauses"]),
                    phase["weight"])

    def test_the_target_is_the_dynamic_role(self):
        self.assertEqual(self.raw["roles"]["target"], ACTIVE_TARGET_ROLE)

    def test_the_end_effector_is_not_a_role(self):
        """It is a reserved token; the gripper is not a whitelist member."""
        self.assertNotIn("ee", self.raw["roles"].values())

    def test_contact_appears_nowhere(self):
        """A grasped object is also in contact, so scoring both pays twice for
        one event, and a brush on the way in would earn approach credit."""
        self.assertNotIn('"contact"', json.dumps(self.raw["phases"]))

    def test_the_return_phase_is_gated_on_the_grasp(self):
        """The gripper starts at rest. Without the gate, an episode would open
        with the terminal phase already paying."""
        gates = self.raw["phases"][2]["requires"]["all_of"]
        self.assertEqual([g["relation"] for g in gates], ["grasp"])
        self.assertEqual([g["weight"] for g in gates], [0.0])

    def test_completion_needs_both_facts(self):
        completion = self.raw["phases"][2]["completion"]
        self.assertEqual(
            sorted(c["relation"] for c in completion["all_of"]),
            ["grasp", "reached"])

    def test_the_terminal_phase_carries_no_finest_spatial_rung(self):
        """``reached`` is exact -- norm(tcp - rest) <= ee_rest_thresh, read
        live -- while the finest rungs are mined quantiles that can tighten
        below it. A successful return would then leave a weighted rung unpaid
        and the potential would top out under 1.0."""
        for clause in self.raw["phases"][2]["clauses"]:
            with self.subTest(relation=clause["relation"], labels=clause["labels"]):
                self.assertNotEqual(clause["labels"], ["very-near"])
                self.assertNotEqual(clause["labels"], ["level"])

    def test_the_approach_phase_may_use_its_finest_rungs(self):
        """It completes on grasp, not on a distance, so no implication rule
        applies to it."""
        labels = [c["labels"] for c in self.raw["phases"][0]["clauses"]]
        self.assertIn(["very-near"], labels)
        self.assertIn(["level"], labels)


class CompilationTest(unittest.TestCase):

    def test_it_compiles_against_a_pilot_shaped_asset(self):
        self.assertEqual(len(_schedule().phases), 3)

    def test_the_target_role_compiles_to_the_sentinel(self):
        self.assertEqual(_schedule().role_entity_ids["target"],
                         ACTIVE_TARGET_ENTITY_ID)

    def test_the_rest_site_role_compiles_to_a_real_entity(self):
        self.assertEqual(_schedule().role_entity_ids["rest_site"],
                         VOCAB.encode(SITE_EE_REST))

    def test_every_fact_it_reads_is_one_the_runtime_emits(self):
        """Compilation refuses a clause the mined assets cannot score, so
        reaching here at all is the check."""
        slots = _schedule().slots
        self.assertTrue(slots)
        relation = build_relation_vocab()
        by_id = {i: name for name, i in relation.token_to_id.items()}
        named = {by_id[slot[0]] for slot in slots}
        self.assertEqual(
            named,
            {"planar-distance", "height-offset", "grasp-compatibility",
             "grasp", "reached"})


def _torch():
    try:
        import torch
    except ImportError:
        raise unittest.SkipTest("torch is not installed")
    return torch


class TrajectoryTest(unittest.TestCase):
    """Scoring constructed frames through the real replay potential."""

    @classmethod
    def setUpClass(cls):
        _torch()

    def _score(self, facts):
        """``facts`` maps (relation, dst_row) -> absolute label."""
        import torch

        from progress import TaskScheduleReplayPotential

        absolute, relation = build_absolute_vocab(), build_relation_vocab()
        scorer = TaskScheduleReplayPotential(_schedule(), len(absolute))
        node_ent = [0] * 8
        node_ent[ROW_EE] = VOCAB.ee_id
        node_ent[ROW_TARGET] = VOCAB.encode(TARGET)
        node_ent[ROW_SITE] = VOCAB.encode(SITE_EE_REST)

        rel, abs_, src, dst = [], [], [], []
        for (name, dst_row), label in facts.items():
            rel.append(relation.encode(name))
            abs_.append(absolute.encode(label))
            src.append(ROW_EE)
            dst.append(dst_row)
        value, valid = scorer(
            torch.tensor([node_ent]),
            torch.tensor(rel, dtype=torch.long),
            torch.tensor(abs_, dtype=torch.long),
            torch.tensor(src, dtype=torch.long),
            torch.tensor(dst, dtype=torch.long),
            torch.zeros(len(rel), dtype=torch.long), 1,
        )
        return float(value.item()), bool(valid.item())

    def _frame(self, *, grasp, reached, near_target=True, near_rest=True):
        """Every fact the schedule reads, so the frame stays readable."""
        return {
            ("planar-distance", ROW_TARGET): "very-near" if near_target else "very-far",
            ("height-offset", ROW_TARGET): "level" if near_target else "far-above",
            ("grasp-compatibility", ROW_TARGET): "match" if near_target else "unobserved",
            ("grasp", ROW_TARGET): "holds" if grasp else "not-holds",
            ("planar-distance", ROW_SITE): "very-near" if near_rest else "very-far",
            ("height-offset", ROW_SITE): "level" if near_rest else "far-above",
            ("reached", ROW_SITE): "holds" if reached else "not-holds",
        }

    # ---- the trajectory the environment calls a success ----------------- #
    def test_a_completed_pick_scores_exactly_one(self):
        value, valid = self._score(self._frame(grasp=True, reached=True))
        self.assertTrue(valid)
        self.assertAlmostEqual(value, 1.0, places=5)

    def test_every_frame_is_readable(self):
        for grasp in (True, False):
            for reached in (True, False):
                with self.subTest(grasp=grasp, reached=reached):
                    _v, valid = self._score(
                        self._frame(grasp=grasp, reached=reached))
                    self.assertTrue(valid)

    def test_the_potential_never_exceeds_one(self):
        for grasp in (True, False):
            for reached in (True, False):
                for near in (True, False):
                    with self.subTest(grasp=grasp, reached=reached, near=near):
                        value, _ = self._score(self._frame(
                            grasp=grasp, reached=reached,
                            near_target=near, near_rest=near))
                        self.assertLessEqual(value, 1.0 + 1e-6)

    # ---- negative case 1: at rest, holding nothing ---------------------- #
    def test_resting_without_a_grasp_does_not_complete(self):
        """The gripper *starts* at rest, so this is frame 0 of every episode.
        Without the gate it would open with the terminal phase paid."""
        value, _ = self._score(self._frame(grasp=False, reached=True))
        self.assertLess(value, 1.0)

    def test_the_opening_frame_scores_well_under_one(self):
        value, _ = self._score(self._frame(
            grasp=False, reached=True, near_target=False))
        self.assertLess(value, 0.5)

    def test_the_gate_withholds_the_return_phase_entirely(self):
        """Not merely the reached rung: no part of the terminal phase pays
        while nothing is held."""
        held = self._score(self._frame(grasp=True, reached=True))[0]
        empty = self._score(self._frame(grasp=False, reached=True))[0]
        self.assertGreaterEqual(held - empty, 0.45 - 1e-6)

    # ---- negative case 2: grasped, but away from rest ------------------- #
    def test_grasping_away_from_rest_does_not_complete(self):
        value, _ = self._score(self._frame(
            grasp=True, reached=False, near_rest=False))
        self.assertLess(value, 1.0)

    def test_it_still_scores_above_an_empty_gripper(self):
        """Having the object is real progress, and the shaping has to say so
        or there is no gradient toward grasping."""
        grasped = self._score(self._frame(
            grasp=True, reached=False, near_rest=False))[0]
        empty = self._score(self._frame(
            grasp=False, reached=False, near_target=False, near_rest=False))[0]
        self.assertGreater(grasped, empty)

    def test_approaching_rest_while_holding_increases_the_potential(self):
        far = self._score(self._frame(
            grasp=True, reached=False, near_rest=False))[0]
        near = self._score(self._frame(grasp=True, reached=False))[0]
        self.assertGreater(near, far)

    # ---- negative case 3: dropped during the return --------------------- #
    def test_dropping_the_target_at_rest_does_not_complete(self):
        """``reached`` alone is where the gripper already was at frame 0."""
        value, _ = self._score(self._frame(grasp=False, reached=True))
        self.assertLess(value, 1.0)

    def test_a_drop_costs_the_potential_it_had(self):
        """The potential is a function of the current frame, so it falls back
        rather than latching -- which is what keeps the shaping telescoping."""
        holding = self._score(self._frame(grasp=True, reached=True))[0]
        dropped = self._score(self._frame(grasp=False, reached=True))[0]
        self.assertLess(dropped, holding)

    def test_only_both_facts_together_reach_one(self):
        for grasp, reached in ((True, False), (False, True), (False, False)):
            with self.subTest(grasp=grasp, reached=reached):
                value, _ = self._score(
                    self._frame(grasp=grasp, reached=reached))
                self.assertLess(value, 1.0 - 1e-6)


class TelescopingTest(unittest.TestCase):
    """The shaping term over a whole trajectory sums to the endpoint
    difference, which is what makes it policy-invariant."""

    @classmethod
    def setUpClass(cls):
        _torch()

    def test_a_pick_trajectory_telescopes(self):
        import torch

        from progress import potential_shaping

        scorer = TrajectoryTest()
        stages = [
            scorer._frame(grasp=False, reached=True, near_target=False),
            scorer._frame(grasp=False, reached=False, near_target=False,
                          near_rest=False),
            scorer._frame(grasp=False, reached=False, near_rest=False),
            scorer._frame(grasp=True, reached=False, near_rest=False),
            scorer._frame(grasp=True, reached=False),
            scorer._frame(grasp=True, reached=True),
        ]
        values = [scorer._score(f)[0] for f in stages]
        phi = torch.tensor(values, dtype=torch.float32).reshape(1, -1, 1)
        shaped = potential_shaping(phi, torch.ones_like(phi), discount=1.0)
        self.assertAlmostEqual(float(shaped.sum().item()),
                               values[-1] - values[0], places=4)

    def test_the_trajectory_ends_at_one_and_starts_below_it(self):
        scorer = TrajectoryTest()
        start = scorer._score(scorer._frame(
            grasp=False, reached=True, near_target=False))[0]
        end = scorer._score(scorer._frame(grasp=True, reached=True))[0]
        self.assertLess(start, 1.0)
        self.assertAlmostEqual(end, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
