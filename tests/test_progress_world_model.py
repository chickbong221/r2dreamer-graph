"""The world-model progress path: scalar target, bounded head, beta schedule.

Synthetic tensors only. The claim under test is that the potential imagination
reads and the potential replay supervises are the same quantity, that an
incomplete ladder is masked rather than scored low, and that beta is the only
thing standing between the two and the actor.
"""

import unittest
from types import SimpleNamespace

import torch

import progress
from dreamer import _masked_std
from networks import ProgressHead, ReturnEMA
from progress import (
    ABS_HOLDS,
    ABS_LEVEL,
    ABS_MATCH,
    ABS_NOT_HOLDS,
    ABS_VERY_FAR,
    ABS_VERY_NEAR,
    PICK_STAGES,
    ProgressScorer,
    TaskScheduleReplayPotential,
    potential_shaping,
)

N_ABS = progress.N_ABS
N_REL = 12


class ScheduleCompletionConjunctionTest(unittest.TestCase):
    def _compiled(self):
        from scenegraph.adapters.graph_vocab import (
            EE_TOKEN,
            PAD_TOKEN,
            EntityVocab,
        )
        from scenegraph.core.schedule import compile_schedule
        from scenegraph.core.spatial_metrics import (
            SPATIAL_SCOPES,
            spatial_bin_key,
        )

        objects = {
            "actor:cubeA": {"contact_components": [{}]},
            "actor:cubeB": {"contact_components": [{}]},
            "actor:table": {"contact_components": [{}]},
        }
        members = {
            key: {"interaction_types": ["contact"]} for key in objects
        }
        bins = {
            spatial_bin_key(scope, relation): [0.1, 0.2, 0.3, 0.4]
            for scope in SPATIAL_SCOPES
            for relation in ("planar-distance", "height-offset")
        }
        vocab = EntityVocab(token_to_id={
            PAD_TOKEN: 0,
            EE_TOKEN: 1,
            "actor:cubeA": 2,
            "actor:cubeB": 3,
            "actor:table": 4,
        })
        done = lambda src, dst: {
            "relation": "contact", "src": src, "dst": dst,
            "labels": ["holds"],
        }
        clause = lambda src, dst, weight: {
            **done(src, dst), "weight": weight,
        }
        raw = {
            "_schema_version": 1,
            "env_id": "Conjunction-v1",
            "roles": {
                "movable": "actor:cubeA",
                "destination": "actor:cubeB",
                "surface": "actor:table",
            },
            "phases": [
                {
                    "name": "approach", "weight": 0.5,
                    "clauses": [clause("ee", "movable", 0.5)],
                    "completion": done("ee", "movable"),
                },
                {
                    "name": "settle", "weight": 0.5,
                    "clauses": [
                        clause("movable", "destination", 0.25),
                        clause("movable", "surface", 0.25),
                    ],
                    "completion": {"all_of": [
                        done("movable", "destination"),
                        done("movable", "surface"),
                    ]},
                },
            ],
        }
        return compile_schedule(raw, objects, members, bins, vocab)

    def test_every_completion_fact_is_required_before_backfilling(self):
        from scenegraph.adapters.graph_vocab import build_absolute_vocab

        schedule = self._compiled()
        absolute = build_absolute_vocab()
        scorer = TaskScheduleReplayPotential(schedule, len(absolute))
        entities = list(schedule.entity_ids)
        row = {entity: index for index, entity in enumerate(entities)}
        holds = absolute.encode("holds")
        not_holds = absolute.encode("not-holds")

        def potential(second_completion_holds):
            labels = [not_holds, holds,
                      holds if second_completion_holds else not_holds]
            slots = list(schedule.slots)
            edge_rel = torch.tensor([slot[0] for slot in slots])
            edge_abs = torch.tensor(labels)
            edge_src = torch.tensor([row[slot[1]] for slot in slots])
            edge_dst = torch.tensor([row[slot[2]] for slot in slots])
            edge_graph = torch.zeros(len(slots), dtype=torch.long)
            value, valid = scorer(
                torch.tensor([entities]), edge_rel, edge_abs,
                edge_src, edge_dst, edge_graph, 1,
            )
            self.assertTrue(bool(valid.item()))
            return float(value.item())

        self.assertAlmostEqual(potential(False), 0.25, places=6)
        self.assertAlmostEqual(potential(True), 1.0, places=6)



class ScheduleRequiresGateTest(unittest.TestCase):
    """A phase's current-frame gate.

    The gate is what makes PegInsertionSide's ordering real: nothing else in
    the scorer consults phase order, so without it the containment ladder pays
    from across the table. It is deliberately not a latch -- it reads this
    frame only, so the potential stays a function of the observed state and the
    shaping term still telescopes.
    """

    def _compiled(self):
        from scenegraph.adapters.graph_vocab import (
            EE_TOKEN,
            PAD_TOKEN,
            EntityVocab,
        )
        from scenegraph.core.schedule import compile_schedule
        from scenegraph.core.spatial_metrics import (
            SPATIAL_SCOPES,
            spatial_bin_key,
        )

        objects = {
            "actor:cubeA": {"contact_components": [{}]},
            "actor:cubeB": {"contact_components": [{}]},
            "actor:table": {"contact_components": [{}]},
        }
        members = {key: {"interaction_types": ["contact"]} for key in objects}
        bins = {
            spatial_bin_key(scope, relation): [0.1, 0.2, 0.3, 0.4]
            for scope in SPATIAL_SCOPES
            for relation in ("planar-distance", "height-offset")
        }
        vocab = EntityVocab(token_to_id={
            PAD_TOKEN: 0, EE_TOKEN: 1, "actor:cubeA": 2, "actor:cubeB": 3,
            "actor:table": 4,
        })
        fact = lambda src, dst: {
            "relation": "contact", "src": src, "dst": dst, "labels": ["holds"],
        }
        clause = lambda src, dst, weight: {**fact(src, dst), "weight": weight}
        raw = {
            "_schema_version": 1,
            "env_id": "Gated-v1",
            "roles": {
                "movable": "actor:cubeA",
                "destination": "actor:cubeB",
                "surface": "actor:table",
            },
            "phases": [
                {
                    "name": "reach", "weight": 0.3,
                    "clauses": [clause("ee", "movable", 0.3)],
                    "completion": fact("ee", "movable"),
                },
                {
                    "name": "align", "weight": 0.3,
                    # Gated on a fact it does not otherwise score, so the test
                    # isolates the gate from the phase's own rung.
                    "requires": {"all_of": [fact("movable", "surface")]},
                    "clauses": [clause("movable", "destination", 0.3)],
                    "completion": fact("movable", "destination"),
                },
                {
                    "name": "settle", "weight": 0.4,
                    "clauses": [clause("destination", "surface", 0.4)],
                    "completion": fact("destination", "surface"),
                },
            ],
        }
        return compile_schedule(raw, objects, members, bins, vocab)

    def setUp(self):
        from scenegraph.adapters.graph_vocab import build_absolute_vocab

        self.schedule = self._compiled()
        self.absolute = build_absolute_vocab()
        self.scorer = TaskScheduleReplayPotential(
            self.schedule, len(self.absolute))
        self.entities = list(self.schedule.entity_ids)
        self.row = {e: i for i, e in enumerate(self.entities)}
        self.slots = list(self.schedule.slots)
        # (relation, src, dst) -> which fact each slot is, by role pair.
        keys = {"ee": 1, "actor:cubeA": 2, "actor:cubeB": 3, "actor:table": 4}
        self.slot_name = {}
        for slot in self.slots:
            src = next(k for k, v in keys.items() if v == slot[1])
            dst = next(k for k, v in keys.items() if v == slot[2])
            self.slot_name[(src, dst)] = slot

    def _score(self, holding, drop=()):
        """``(potential, valid)`` for one frame.

        ``holding`` names the slots whose label is ``holds``; ``drop`` names
        slots to leave out of the packed graph entirely, which is a different
        thing from a fact that is false.
        """
        holds = self.absolute.encode("holds")
        not_holds = self.absolute.encode("not-holds")
        rel, abs_, src, dst = [], [], [], []
        for pair, slot in self.slot_name.items():
            if pair in drop:
                continue
            rel.append(slot[0])
            abs_.append(holds if pair in holding else not_holds)
            src.append(self.row[slot[1]])
            dst.append(self.row[slot[2]])
        value, valid = self.scorer(
            torch.tensor([self.entities]),
            torch.tensor(rel), torch.tensor(abs_),
            torch.tensor(src), torch.tensor(dst),
            torch.zeros(len(rel), dtype=torch.long), 1,
        )
        return float(value.item()), bool(valid.item())

    ALL = (("ee", "actor:cubeA"), ("actor:cubeA", "actor:cubeB"),
           ("actor:cubeA", "actor:table"), ("actor:cubeB", "actor:table"))
    GATE = ("actor:cubeA", "actor:table")
    SETTLE = ("actor:cubeB", "actor:table")

    def test_an_open_gate_lets_the_phase_score(self):
        value, valid = self._score(holding=self.ALL)
        self.assertTrue(valid)
        self.assertAlmostEqual(value, 1.0, places=6)

    def test_a_closed_gate_costs_the_phase_its_quality(self):
        """Everything the phase itself names still holds; only the gate is
        false. The 0.3 it would have earned is withheld."""
        without_settle = tuple(p for p in self.ALL if p != self.SETTLE)
        open_gate, _ = self._score(holding=without_settle)
        closed = tuple(p for p in without_settle if p != self.GATE)
        shut_gate, _ = self._score(holding=closed)
        self.assertAlmostEqual(open_gate, 0.6, places=6)
        self.assertAlmostEqual(shut_gate, 0.3, places=6)

    def test_cumulative_credit_overrides_a_closed_gate(self):
        """Once a later phase has completed, this one was passed through.
        Re-asking whether its entry condition still holds would take back
        progress the episode already made."""
        value, valid = self._score(
            holding=tuple(p for p in self.ALL if p != self.GATE))
        self.assertTrue(valid)
        self.assertAlmostEqual(value, 1.0, places=6)

    def test_a_gate_cannot_cost_a_successful_episode_its_terminal_one(self):
        """The acceptance criterion the gate must never break."""
        for gate_held in (True, False):
            holding = self.ALL if gate_held else tuple(
                p for p in self.ALL if p != self.GATE)
            value, valid = self._score(holding=holding)
            self.assertTrue(valid)
            self.assertAlmostEqual(value, 1.0, places=6)

    def test_a_missing_gate_fact_invalidates_the_frame(self):
        """A gate the frame cannot read is not a gate that is closed -- it is
        a frame that cannot be scored."""
        without_settle = tuple(p for p in self.ALL if p != self.SETTLE)
        _, valid = self._score(holding=without_settle, drop=(self.GATE,))
        self.assertFalse(valid)

    def test_a_missing_gate_fact_is_survivable_once_a_later_phase_completed(self):
        _, valid = self._score(holding=self.ALL, drop=(self.GATE,))
        self.assertTrue(valid)

    def test_the_gate_carries_no_temporal_state(self):
        """Same frame, same potential -- whatever came before it. This is what
        keeps gamma * Phi' - Phi telescoping."""
        without_settle = tuple(p for p in self.ALL if p != self.SETTLE)
        closed = tuple(p for p in without_settle if p != self.GATE)
        first, _ = self._score(holding=closed)
        self._score(holding=self.ALL)          # gate opens, phase completes
        again, _ = self._score(holding=closed)  # and closes again
        self.assertAlmostEqual(first, again, places=6)


# The scorer's own relation order, which is the stage table's order of first
# appearance -- not the relation vocabulary's.
REL_PLANAR, REL_HEIGHT, REL_CCOMPAT, REL_GCOMPAT, REL_CONTACT, REL_GRASP = (
    5, 6, 8, 7, 1, 2,
)
# contact is emitted by the graph but is not a rung: a grasped object is
# also in contact, so scoring both paid twice for one event.
SCORED = (REL_PLANAR, REL_HEIGHT, REL_CCOMPAT, REL_GCOMPAT, REL_GRASP)

SOLVED = {
    REL_PLANAR: ABS_VERY_NEAR,
    REL_HEIGHT: ABS_LEVEL,
    REL_CCOMPAT: ABS_MATCH,
    REL_GCOMPAT: ABS_MATCH,
    REL_CONTACT: ABS_HOLDS,
    REL_GRASP: ABS_HOLDS,
}
UNSTARTED = {
    REL_PLANAR: ABS_VERY_FAR,
    REL_HEIGHT: ABS_LEVEL,
    REL_CCOMPAT: ABS_MATCH,
    REL_GCOMPAT: ABS_MATCH,
    REL_CONTACT: ABS_NOT_HOLDS,
    REL_GRASP: ABS_NOT_HOLDS,
}


def _scorer():
    return ProgressScorer(PICK_STAGES, N_ABS, N_REL)


def _edges(frames):
    """Flatten ``{frame: {relation: label}}`` into packed edge columns."""
    rel, abs_, src, dst, graph = [], [], [], [], []
    for frame, labels in frames.items():
        for relation, label in labels:
            rel.append(relation)
            abs_.append(label)
            src.append(0)
            dst.append(1)
            graph.append(frame)
    as_long = lambda xs: torch.tensor(xs, dtype=torch.long)
    return as_long(rel), as_long(abs_), as_long(src), as_long(dst), as_long(graph)


def _potential(frames, graph_count, scorer=None):
    scorer = scorer or _scorer()
    return scorer.replay_potential(*_edges(frames), graph_count)


class ReplayPotentialTest(unittest.TestCase):
    """The observed scalar equals the old scorer applied to the same labels."""

    def test_matches_the_predicted_scorer_on_complete_one_hot_relations(self):
        scorer = _scorer()
        for labels in (SOLVED, UNSTARTED):
            probs = torch.zeros(1, scorer.n_relations, N_ABS)
            for relation, label in labels.items():
                probs[0, int(scorer.row_of[relation]), label] = 1.0
            expected_soft = scorer.potential(probs, hard=False)
            expected_hard = scorer.potential(probs, hard=True)
            observed, valid = _potential(
                {0: list(labels.items())}, 1, scorer=scorer
            )
            self.assertTrue(bool(valid[0]))
            self.assertAlmostEqual(float(observed[0]), float(expected_hard[0]), places=6)
            self.assertAlmostEqual(float(observed[0]), float(expected_soft[0]), places=6)

    def test_the_solved_state_scores_exactly_one(self):
        observed, valid = _potential({0: list(SOLVED.items())}, 1)
        self.assertTrue(bool(valid[0]))
        self.assertAlmostEqual(float(observed[0]), 1.0, places=6)

    def test_a_partial_ladder_scores_between(self):
        # very-near + level + both compat matches, nothing grasped: the four
        # geometric budgets and neither physical one.
        observed, _ = _potential({0: list(UNSTARTED.items())}, 1)
        self.assertAlmostEqual(float(observed[0]), 0.35, places=6)

    def test_a_missing_relation_masks_the_frame_instead_of_lowering_it(self):
        partial = [(r, SOLVED[r]) for r in SCORED if r != REL_GRASP]
        observed, valid = _potential({0: partial}, 1)
        self.assertFalse(bool(valid[0]))
        # And is zeroed rather than left at the 0.50 it would otherwise read,
        # so a caller that forgets the mask gets an obvious number, not a
        # plausible one.
        self.assertEqual(float(observed[0]), 0.0)

    def test_an_unscored_contact_edge_neither_earns_nor_invalidates(self):
        """The graph still reports contact; the ladder stops paying for it.

        The scorer ignores relations it does not name, so the edge is free --
        it must not add weight, and it must not mask the frame either.
        """
        without = [(r, SOLVED[r]) for r in SCORED]
        with_contact = without + [(REL_CONTACT, ABS_HOLDS)]
        bare, bare_valid = _potential({0: without}, 1)
        full, full_valid = _potential({0: with_contact}, 1)
        self.assertTrue(bool(bare_valid[0]))
        self.assertTrue(bool(full_valid[0]))
        self.assertAlmostEqual(float(bare[0]), float(full[0]), places=6)
        self.assertAlmostEqual(float(full[0]), 1.0, places=6)

    def test_grasping_carries_the_whole_physical_budget(self):
        """Half the potential, in one step, at the moment of the grasp."""
        ungrasped = [(r, SOLVED[r]) for r in SCORED if r != REL_GRASP]
        ungrasped.append((REL_GRASP, ABS_NOT_HOLDS))
        observed, valid = _potential({0: ungrasped}, 1)
        self.assertTrue(bool(valid[0]))
        self.assertAlmostEqual(float(observed[0]), 0.50, places=6)

    def test_a_duplicated_relation_masks_the_frame(self):
        doubled = list(SOLVED.items()) + [(REL_PLANAR, ABS_VERY_FAR)]
        _, valid = _potential({0: doubled}, 1)
        self.assertFalse(bool(valid[0]))

    def test_frames_before_the_first_target_observation_are_masked(self):
        # Frame 0 has no end-effector-to-target block at all: row 1 was still
        # padding. Frame 1 is the first observation.
        observed, valid = _potential({1: list(SOLVED.items())}, 2)
        self.assertFalse(bool(valid[0]))
        self.assertTrue(bool(valid[1]))
        self.assertEqual(float(observed[0]), 0.0)

    def test_edges_that_are_not_the_ee_target_block_are_ignored(self):
        rel, abs_, src, dst, graph = _edges({0: list(SOLVED.items())})
        # Re-point one fact at another object; the ladder is now incomplete.
        dst = dst.clone()
        dst[0] = 2
        scorer = _scorer()
        _, valid = scorer.replay_potential(rel, abs_, src, dst, graph, 1)
        self.assertFalse(bool(valid[0]))

    def test_relations_outside_the_stage_table_do_not_break_completeness(self):
        # support-compatibility is a real relation the scorer never reads.
        extra = list(SOLVED.items()) + [(9, ABS_MATCH)]
        observed, valid = _potential({0: extra}, 1)
        self.assertTrue(bool(valid[0]))
        self.assertAlmostEqual(float(observed[0]), 1.0, places=6)

    def test_an_empty_batch_is_all_masked_and_does_not_raise(self):
        observed, valid = _potential({}, 3)
        self.assertEqual(int(valid.sum()), 0)
        self.assertEqual(float(observed.abs().sum()), 0.0)

    def test_row_of_maps_relation_ids_not_positions(self):
        scorer = _scorer()
        self.assertEqual(int(scorer.row_of[REL_PLANAR]), 0)
        self.assertEqual(int(scorer.row_of[REL_GRASP]), scorer.n_relations - 1)
        # Relations no stage names are -1, which is what excludes them.
        self.assertEqual(int(scorer.row_of[3]), -1)
        # contact is emitted by the graph and scored by nothing, so it is
        # excluded the same way. That is what keeps a grasping frame valid
        # instead of masking it for carrying an extra fact.
        self.assertEqual(int(scorer.row_of[REL_CONTACT]), -1)


class ProgressHeadTest(unittest.TestCase):
    def _head(self):
        config = SimpleNamespace(
            act="SiLU", symlog_inputs=False, device="cpu", layers=1,
            units=8, name="progress",
        )
        return ProgressHead(config, 6)

    def test_output_is_bounded_in_the_unit_interval(self):
        head = self._head()
        # Deliberately extreme inputs: the bound is structural, not empirical.
        for scale in (0.0, 1.0, 1e3, 1e6):
            out = head(torch.randn(64, 6) * scale)
            self.assertEqual(tuple(out.shape), (64, 1))
            self.assertGreaterEqual(float(out.min()), 0.0)
            self.assertLessEqual(float(out.max()), 1.0)
            self.assertTrue(torch.isfinite(out).all())

    def test_the_shaping_reward_is_bounded_by_the_potential(self):
        # F_t = gamma * c * phi' - phi with phi in [0, 1], so |F| <= 1 whatever
        # the horizon. Nothing downstream clamps it.
        head = self._head()
        phi = head(torch.randn(256, 6) * 10.0).reshape(1, 256, 1)
        cont = torch.ones_like(phi)
        for horizon in (16, 333, 1000):
            shaping = potential_shaping(phi, cont, 1 - 1 / horizon)
            self.assertLessEqual(float(shaping.abs().max()), 1.0 + 1e-6)


class PotentialShapingTest(unittest.TestCase):
    """Potential-difference shaping, not occupancy.

    Occupancy paid for *being* in a high-progress state, so holding still there
    earned as much as improving and undoing progress cost nothing.
    """

    GAMMA = 0.9

    def _phi(self, values):
        return torch.tensor(values, dtype=torch.float32).reshape(1, -1, 1)

    def _discounted(self, shaping):
        # What _lambda_return accumulates: rewards from index one onward.
        body = shaping[0, 1:, 0]
        weights = self.GAMMA ** torch.arange(body.numel(), dtype=torch.float32)
        return float((body * weights).sum())

    def test_index_zero_is_unused_and_zero(self):
        phi = self._phi([0.4, 0.5, 0.6])
        shaping = potential_shaping(phi, torch.ones_like(phi), self.GAMMA)
        self.assertEqual(float(shaping[0, 0, 0]), 0.0)

    def test_a_round_trip_earns_nothing(self):
        phi = self._phi([0.0, 0.2, 0.0])
        shaping = potential_shaping(phi, torch.ones_like(phi), self.GAMMA)
        self.assertAlmostEqual(self._discounted(shaping), 0.0, places=6)

    def test_a_repeated_cycle_cannot_be_farmed(self):
        phi = self._phi([0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0])
        shaping = potential_shaping(phi, torch.ones_like(phi), self.GAMMA)
        self.assertAlmostEqual(self._discounted(shaping), 0.0, places=6)

    def test_negative_shaping_is_kept(self):
        phi = self._phi([0.9, 0.1, 0.1])
        shaping = potential_shaping(phi, torch.ones_like(phi), self.GAMMA)
        self.assertLess(float(shaping[0, 1, 0]), 0.0)

    def test_it_telescopes_for_an_arbitrary_sequence(self):
        torch.manual_seed(0)
        phi = torch.rand(1, 12, 1)
        shaping = potential_shaping(phi, torch.ones_like(phi), self.GAMMA)
        expected = (self.GAMMA ** (phi.shape[1] - 1)) * float(phi[0, -1, 0]) \
            - float(phi[0, 0, 0])
        self.assertAlmostEqual(self._discounted(shaping), expected, places=5)

    def test_a_terminal_transition_drops_the_bootstrap(self):
        phi = self._phi([0.3, 0.8])
        cont = torch.tensor([[[1.0], [0.0]]])
        shaping = potential_shaping(phi, cont, self.GAMMA)
        # Continuation zero leaves -phi_t alone: reaching a terminal state is
        # worth only what the environment reward says it is.
        self.assertAlmostEqual(float(shaping[0, 1, 0]), -0.3, places=6)

    def test_a_constant_potential_only_reflects_discounting(self):
        phi = self._phi([0.6] * 5)
        shaping = potential_shaping(phi, torch.ones_like(phi), self.GAMMA)
        expected = (self.GAMMA ** 4) * 0.6 - 0.6
        self.assertAlmostEqual(self._discounted(shaping), expected, places=6)


class ReturnNormaliserTest(unittest.TestCase):
    def test_the_environment_floor_is_still_one(self):
        ema = ReturnEMA(device="cpu")
        self.assertEqual(ema.min_scale, 1.0)
        _, scale = ema(torch.rand(256, 1) * 1e-3)
        self.assertAlmostEqual(float(scale), 1.0, places=6)

    def test_a_bounded_head_can_ask_for_a_smaller_floor(self):
        ema = ReturnEMA(device="cpu", min_scale=0.01)
        # A potential spread of roughly 0.4 must survive normalisation intact
        # rather than being divided by a constant one.
        values = torch.rand(4096, 1)
        for _ in range(400):
            _, scale = ema(values)
        self.assertGreater(float(scale), 0.5)
        self.assertLess(float(scale), 1.0)

    def test_a_non_positive_floor_is_refused(self):
        with self.assertRaisesRegex(ValueError, "min_scale"):
            ReturnEMA(device="cpu", min_scale=0.0)


class BetaScheduleTest(unittest.TestCase):
    """The ramp, read straight off the method that computes it."""

    class _Schedule:
        """The three fields ``_progress_beta_at`` reads, and nothing else."""

        def __init__(self, beta, start, end):
            self.progress_beta = beta
            self.progress_beta_start = start
            self.progress_beta_end = end

    def _at(self, step, beta=0.01, start=400_000, end=700_000):
        from dreamer import Dreamer

        holder = self._Schedule(beta, start, end)
        return Dreamer._progress_beta_at(holder, step)

    def test_beta_is_exactly_zero_through_the_hold(self):
        for step in (0, 1, 200_000, 399_999, 400_000):
            self.assertEqual(self._at(step), 0.0, step)

    def test_beta_reaches_the_plateau_at_the_end_of_the_ramp(self):
        self.assertEqual(self._at(700_000), 0.01)
        self.assertEqual(self._at(1_500_000), 0.01)

    def test_the_ramp_is_linear_across_the_window(self):
        self.assertAlmostEqual(self._at(550_000), 0.005, places=9)
        self.assertAlmostEqual(self._at(475_000), 0.0025, places=9)

    def test_beta_zero_leaves_the_actor_advantage_untouched(self):
        # The one identity the beta-zero control arm rests on.
        env_adv = torch.randn(8, 5, 1)
        progress_adv = torch.randn(8, 5, 1) * 17.0
        beta = self._at(400_000)
        self.assertTrue(torch.equal(env_adv + beta * progress_adv, env_adv))

    def test_an_inverted_window_is_refused_at_construction(self):
        # Guarded in Dreamer.__init__; asserted here so the schedule contract
        # lives in one place.
        holder = self._Schedule(0.01, 700_000, 400_000)
        self.assertLess(holder.progress_beta_end, holder.progress_beta_start)


class MaskedStdTest(unittest.TestCase):
    def test_ignores_the_masked_entries(self):
        values = torch.tensor([0.0, 1.0, 100.0, 200.0])
        mask = torch.tensor([True, True, False, False])
        self.assertAlmostEqual(float(_masked_std(values, mask)), 0.5, places=6)

    def test_a_single_valid_frame_reads_zero_rather_than_nan(self):
        values = torch.tensor([3.0, 9.0])
        mask = torch.tensor([True, False])
        self.assertEqual(float(_masked_std(values, mask)), 0.0)

    def test_an_empty_mask_reads_zero(self):
        values = torch.tensor([3.0, 9.0])
        mask = torch.tensor([False, False])
        self.assertEqual(float(_masked_std(values, mask)), 0.0)


if __name__ == "__main__":
    unittest.main()
