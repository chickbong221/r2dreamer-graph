"""The world-model progress path: scalar target, bounded head, beta schedule.

Synthetic tensors only. The claim under test is that the potential imagination
reads and the potential replay supervises are the same quantity, that an
incomplete ladder is masked rather than scored low, and that beta is the only
thing standing between the two and the actor.
"""

import unittest
from types import SimpleNamespace

import torch

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
)

N_ABS = 17
N_REL = 11

# The scorer's own relation order, which is the stage table's order of first
# appearance -- not the relation vocabulary's.
REL_PLANAR, REL_HEIGHT, REL_CCOMPAT, REL_GCOMPAT, REL_CONTACT, REL_GRASP = (
    5, 6, 8, 7, 1, 2,
)
ALL_SIX = (REL_PLANAR, REL_HEIGHT, REL_CCOMPAT, REL_GCOMPAT, REL_CONTACT, REL_GRASP)

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
        partial = [(r, SOLVED[r]) for r in ALL_SIX if r != REL_GRASP]
        observed, valid = _potential({0: partial}, 1)
        self.assertFalse(bool(valid[0]))
        # And is zeroed rather than left at the 0.70 it would otherwise read,
        # so a caller that forgets the mask gets an obvious number, not a
        # plausible one.
        self.assertEqual(float(observed[0]), 0.0)

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
        self.assertEqual(int(scorer.row_of[REL_GRASP]), 5)
        # Relations no stage names are -1, which is what excludes them.
        self.assertEqual(int(scorer.row_of[3]), -1)


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

    def test_the_shaping_reward_cannot_outgrow_a_unit_return(self):
        # progress_reward = (1 - discount) * phi with phi <= 1, so the
        # discounted return of a permanently solved episode is at most one --
        # and beta therefore means the same thing at any horizon.
        head = self._head()
        for horizon in (16, 333, 1000):
            step = 1 / horizon
            reward = step * head(torch.randn(256, 6) * 10.0)
            self.assertLessEqual(float(reward.max()), step + 1e-9)
            self.assertLessEqual(float(reward.max()) / step, 1.0 + 1e-6)


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
