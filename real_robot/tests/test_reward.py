"""The dense kitchen reward on a scripted episode and its counterfactuals."""

from __future__ import annotations

import unittest

import numpy as np

from ..rewards.kitchen import (
    RewardScales,
    check_failure_penalty,
    closeness,
    compute_rewards,
    critical_failures,
    fallback_report,
    fit_scales,
    gripper_closure,
    inputs_from_artifacts,
    reward_checks,
    trailing_all,
)
from . import synthetic as syn


class Stages(unittest.TestCase):
    def setUp(self):
        self.cfg = syn.reward_config()
        self.scales = RewardScales.from_defaults(self.cfg)

    def run_inputs(self, **overrides):
        return compute_rewards(syn.reward_inputs(**overrides), self.scales, self.cfg)

    def test_successful_episode_walks_every_stage_in_order(self):
        result = self.run_inputs()
        firsts = [int(np.argmax(result.stage == k)) for k in range(6)]
        self.assertEqual(firsts, sorted(firsts))
        self.assertTrue(np.all(np.diff(result.stage) >= 0))
        self.assertEqual(result.completion_frame, syn.EXPECTED_COMPLETION)

    def test_completion_pays_once_and_terminates(self):
        result = self.run_inputs()
        self.assertEqual(int(result.done.sum()), 1)
        t = int(np.flatnonzero(result.done)[0])
        self.assertEqual(t + 1, result.completion_frame)
        self.assertEqual(result.reward[t], 1.0)
        self.assertFalse(result.reward_valid[t + 1:].any())

    def test_other_rewards_are_S_of_the_arrival_minus_one(self):
        result = self.run_inputs()
        for t in range(result.completion_frame - 1):
            self.assertAlmostEqual(result.reward[t], result.S[t + 1] - 1.0)
        self.assertTrue(np.all(result.reward[: result.completion_frame - 1] < 0))

    def test_grasp_label_without_a_closing_gripper_unlocks_no_transport(self):
        result = self.run_inputs(gripper_closure=np.zeros(syn.N))
        self.assertFalse(np.any(result.stage[syn.GRASP_BANANA: syn.IN_POT] == 1))

    def test_dropping_the_banana_returns_to_reaching(self):
        labels = syn.reward_inputs().banana_grasp_label.copy()
        labels[45:] = False
        result = self.run_inputs(banana_grasp_label=labels)
        self.assertEqual(result.stage[44], 1)
        self.assertEqual(result.stage[45], 0)
        self.assertLess(result.S[45], result.S[44])

    def test_lifting_the_lid_off_again_lowers_the_stage(self):
        on_pot = syn.reward_inputs().lid_on_pot_label.copy()
        on_pot[147:150] = False
        result = self.run_inputs(lid_on_pot_label=on_pot)
        self.assertEqual(result.stage[146], 5)
        self.assertEqual(result.stage[148], 3)

    def test_closing_an_empty_pot_is_not_completion(self):
        result = self.run_inputs(banana_in_pot_label=np.zeros(syn.N, dtype=bool))
        self.assertEqual(result.completion_frame, -1)
        self.assertLess(int(result.stage.max()), 3)

    def test_completion_never_precedes_the_observed_event(self):
        result = self.run_inputs(observed_completion_frame=syn.EXPECTED_COMPLETION + 6)
        self.assertEqual(result.completion_frame, syn.EXPECTED_COMPLETION + 6)

    def test_gemini_reporting_failure_blocks_completion(self):
        result = self.run_inputs(observed_success=False, observed_completion_frame=-1)
        self.assertEqual(result.completion_frame, -1)
        self.assertTrue(result.notes)

    def test_truncated_recording_is_not_terminal(self):
        result = compute_rewards(syn.reward_inputs(n=120), self.scales, self.cfg)
        self.assertEqual(result.completion_frame, -1)
        self.assertFalse(result.done.any())
        self.assertEqual(int(result.reward_valid.sum()), 119)

    def test_reward_ignores_episode_length(self):
        full = self.run_inputs()
        prefix = compute_rewards(syn.reward_inputs(n=100), self.scales, self.cfg)
        np.testing.assert_array_equal(prefix.stage, full.stage[:100])
        np.testing.assert_allclose(prefix.reward[:99], full.reward[:99])

    def test_missing_geometry_holds_the_score_and_is_flagged(self):
        inputs = syn.reward_inputs()
        distance = inputs.d_gripper_banana.copy()
        distance[10:15] = np.nan
        result = self.run_inputs(d_gripper_banana=distance)
        self.assertTrue(result.terms["q_imputed"][10:15].all())
        self.assertEqual(result.q[12], result.q[9])

    def test_unknown_lid_geometry_follows_the_stated_rule(self):
        lateral = syn.reward_inputs().lid_lateral_error.copy()
        lateral[140:] = np.nan
        strict = self.run_inputs(lid_lateral_error=lateral)
        self.assertEqual(strict.completion_frame, -1)
        self.assertFalse(strict.terms["lid_seated"][150])
        lenient = compute_rewards(syn.reward_inputs(lid_lateral_error=lateral), self.scales,
                                  syn.reward_config(**{"lid_seated.unknown_geometry": "defer_to_label"}))
        self.assertTrue(lenient.terms["lid_seated"][150])
        with self.assertRaises(ValueError):
            compute_rewards(syn.reward_inputs(), self.scales,
                            syn.reward_config(**{"lid_seated.unknown_geometry": "maybe"}))

    def test_a_settled_banana_hidden_by_the_lid_stays_settled(self):
        speed = syn.reward_inputs().banana_speed.copy()
        speed[100:] = np.nan                      # settled at frame 82, then out of sight
        result = self.run_inputs(banana_speed=speed)
        self.assertEqual(result.completion_frame, syn.EXPECTED_COMPLETION)
        self.assertTrue(result.terms["banana_settled"][100:].all())
        self.assertFalse(result.terms["q_imputed"].any())

    def test_a_banana_never_seen_to_settle_follows_the_stated_rule(self):
        speed = syn.reward_inputs().banana_speed.copy()
        speed[syn.IN_POT:] = np.nan               # hidden the moment it enters the pot
        strict = self.run_inputs(banana_speed=speed)
        self.assertEqual(strict.completion_frame, -1)
        self.assertFalse(strict.terms["banana_settled"].any())
        # Settling could not be judged once the banana was released, and only then.
        self.assertTrue(strict.terms["q_imputed"][syn.RELEASE_BANANA:].all())
        self.assertFalse(strict.terms["q_imputed"][:syn.RELEASE_BANANA].any())
        lenient = compute_rewards(syn.reward_inputs(banana_speed=speed), self.scales,
                                  syn.reward_config(**{"settle.unknown_speed": "defer_to_label"}))
        self.assertEqual(lenient.completion_frame, syn.EXPECTED_COMPLETION)
        with self.assertRaises(ValueError):
            compute_rewards(syn.reward_inputs(), self.scales, syn.reward_config(**{"settle.unknown_speed": "maybe"}))

    def test_a_settled_banana_measured_moving_is_no_longer_settled(self):
        speed = syn.reward_inputs().banana_speed.copy()
        speed[100] = 0.2
        result = self.run_inputs(banana_speed=speed)
        self.assertTrue(result.terms["banana_settled"][99])
        self.assertFalse(result.terms["banana_settled"][100:108].any())
        self.assertEqual(result.stage[100], 2)
        self.assertTrue(result.terms["banana_settled"][108])

    def test_long_stretches_without_distances_are_counted_and_critical(self):
        distance = syn.reward_inputs().d_gripper_banana.copy()
        distance[:28] = np.nan
        inputs = syn.reward_inputs(d_gripper_banana=distance)
        result = compute_rewards(inputs, self.scales, self.cfg)
        report = fallback_report(result)
        self.assertEqual(report["imputed_frames"], 28)
        self.assertEqual(report["longest_imputed_run"], 28)
        cfg = syn.reward_config(**{"fallbacks.max_run_frames": 20})
        names = [c["name"] for c in critical_failures(reward_checks(inputs, result, self.scales, cfg))]
        self.assertEqual(names, ["distance_fallbacks_within_limit"])

    def test_gemini_success_without_verified_completion_is_critical(self):
        on_pot = np.zeros(syn.N, dtype=bool)
        inputs = syn.reward_inputs(lid_on_pot_label=on_pot)
        result = compute_rewards(inputs, self.scales, self.cfg)
        names = [c["name"] for c in critical_failures(reward_checks(inputs, result, self.scales, self.cfg))]
        self.assertIn("observed_outcome_agrees", names)

    def test_every_check_passes_on_the_scripted_episode(self):
        inputs = syn.reward_inputs()
        result = compute_rewards(inputs, self.scales, self.cfg)
        failed = [c for c in reward_checks(inputs, result, self.scales, self.cfg) if c["passed"] is False]
        self.assertEqual(failed, [])


class Pieces(unittest.TestCase):
    def test_closeness_keeps_missing_distances_missing(self):
        values = closeness(np.array([0.0, 0.1, np.nan]), 0.1)
        self.assertEqual(values[0], 1.0)
        self.assertAlmostEqual(values[1], 1.0 - np.tanh(1.0))
        self.assertTrue(np.isnan(values[2]))

    def test_trailing_window(self):
        np.testing.assert_array_equal(trailing_all(np.array([1, 1, 0, 1, 1, 1], bool), 2),
                                      np.array([0, 1, 0, 0, 1, 1], bool))

    def test_gripper_closure_in_either_direction(self):
        np.testing.assert_allclose(gripper_closure(np.array([1.6, 0.6, 1.1]), 1.6, 0.6), [0.0, 1.0, 0.5])
        np.testing.assert_allclose(gripper_closure(np.array([0.0, 1.0]), 0.0, 1.0), [0.0, 1.0])

    def test_failure_penalty_must_not_beat_continuing(self):
        check_failure_penalty({"gamma": 0.99, "failure_penalty": -100.0})
        with self.assertRaises(ValueError):
            check_failure_penalty({"gamma": 0.99, "failure_penalty": -10.0})

    def test_scales_score_the_median_entry_distance_at_the_target(self):
        cfg = syn.reward_config()
        scales = fit_scales([syn.reward_inputs()] * cfg["scales"]["min_samples"], cfg, range(5))
        self.assertAlmostEqual(closeness(np.array([0.30]), scales.reach_banana)[0],
                               cfg["scales"]["target_entry_closeness"], places=6)

    def test_inputs_from_a_saved_annotation(self):
        spec = syn.graph_spec()
        annotation = syn.annotation(spec)
        gripper = {"open_value": 1.6, "closed_value": 0.6}
        values = 1.6 - syn.closure() * 1.0
        inputs = inputs_from_artifacts(spec, annotation, syn.geometry(), values, gripper)
        expected = syn.reward_inputs()
        np.testing.assert_array_equal(inputs.banana_in_pot_label, expected.banana_in_pot_label)
        np.testing.assert_array_equal(inputs.lid_on_pot_label, expected.lid_on_pot_label)
        np.testing.assert_array_equal(inputs.banana_grasp_label, expected.banana_grasp_label)
        np.testing.assert_allclose(inputs.gripper_closure, expected.gripper_closure)


if __name__ == "__main__":
    unittest.main()
