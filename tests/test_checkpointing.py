"""One rolling-best checkpoint, and a load that refuses to guess.

The policy is narrow and every clause of it is a decision someone could
plausibly undo by accident: nothing before 8M, one file, no interrupt or
milestone copies, and an earlier strong evaluation that must not stop the
first eligible one from saving.

The selection metric is deliberately absent. A run with checkpointing on and
no metric refuses to start, because picking one silently decides which model
every later number is reported from.

Saving needs torch; the policy arithmetic does not, and is tested without it.
"""

import os
import tempfile
import unittest

from checkpointing import (
    CheckpointConfig,
    CheckpointError,
    Checkpointer,
    identity_mismatches,
)

IDENTITY = {"entity_vocab": "v18:abc", "relation_vocab": "r12:def",
            "absolute_vocab": "a19:ghi", "graph_schema": "n8e168",
            "schedule": "tidy_house/pick:1", "assets": "sha:123"}


def _config(**over):
    base = dict(enabled=True, start_step=8_000_000, metric="eval/success_once",
                tiebreak="eval/success_at_end", mode="max")
    base.update(over)
    return CheckpointConfig(**base)


class MetricIsUnresolvedTest(unittest.TestCase):
    """It is an experiment decision, not a default."""

    def test_the_shipped_default_names_no_metric(self):
        self.assertEqual(CheckpointConfig().metric, "")

    def test_checkpointing_is_off_by_default(self):
        self.assertFalse(CheckpointConfig().enabled)

    def test_enabling_without_a_metric_refuses(self):
        with self.assertRaises(CheckpointError) as ctx:
            CheckpointConfig(enabled=True).validate()
        self.assertIn("metric", str(ctx.exception))

    def test_a_disabled_run_needs_no_metric(self):
        CheckpointConfig(enabled=False).validate()

    def test_an_unknown_mode_refuses(self):
        with self.assertRaises(CheckpointError):
            _config(mode="highest").validate()

    def test_a_metric_absent_from_the_evaluation_refuses(self):
        """Selecting on a number that was never measured is worse than not
        selecting."""
        keeper = Checkpointer(_config(), "/tmp/x", dict(IDENTITY),
                              save_fn=lambda payload, path: None)
        with self.assertRaises(CheckpointError) as ctx:
            keeper.maybe_save(9_000_000, {"eval/score": 1.0}, dict)
        self.assertIn("never measured", str(ctx.exception))


class EligibilityTest(unittest.TestCase):
    """Nothing before ``start_step``, and nothing before it counts either."""

    def setUp(self):
        self.keeper = Checkpointer(_config(), "/tmp/x", dict(IDENTITY),
                                   save_fn=lambda payload, path: None)

    def test_earlier_steps_are_ineligible(self):
        for step in (0, 1_000_000, 7_999_999):
            with self.subTest(step=step):
                self.assertFalse(self.keeper.eligible(step))

    def test_the_boundary_step_is_eligible(self):
        self.assertTrue(self.keeper.eligible(8_000_000))

    def test_an_early_evaluation_saves_nothing(self):
        saved = self.keeper.maybe_save(
            2_000_000, {"eval/success_once": 0.9}, self._explode)
        self.assertFalse(saved)

    def test_an_early_evaluation_does_not_become_the_incumbent(self):
        """A strong result at 2M must not stop the first eligible evaluation
        from saving -- which is what makes 'best *eligible*' the rule."""
        self.keeper.maybe_save(2_000_000, {"eval/success_once": 0.99},
                               self._explode)
        self.assertIsNone(self.keeper.best)

    @staticmethod
    def _explode():
        raise AssertionError("state was serialized for an ineligible step")


class SelectionTest(unittest.TestCase):
    """Which evaluation claims the file."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.written = []
        self.keeper = Checkpointer(
            _config(), self.tmp, dict(IDENTITY),
            save_fn=lambda payload, path: self.written.append(path))
        self.saves = []

    def _save(self, step, value, tiebreak=0.0):
        return self.keeper.maybe_save(
            step, {"eval/success_once": value,
                   "eval/success_at_end": tiebreak},
            lambda: self.saves.append(step) or {"model": step})

    def test_the_first_eligible_evaluation_saves(self):
        self.assertTrue(self._save(8_000_000, 0.10))

    def test_an_improvement_replaces_it(self):
        self._save(8_000_000, 0.10)
        self.assertTrue(self._save(8_500_000, 0.40))
        self.assertEqual(self.keeper.best_step, 8_500_000)

    def test_a_worse_evaluation_does_not(self):
        self._save(8_000_000, 0.40)
        self.assertFalse(self._save(8_500_000, 0.10))
        self.assertEqual(self.keeper.best_step, 8_000_000)

    def test_an_exact_tie_is_broken_by_the_second_metric(self):
        self._save(8_000_000, 0.40, tiebreak=0.10)
        self.assertTrue(self._save(8_500_000, 0.40, tiebreak=0.30))

    def test_a_tie_on_both_keeps_the_incumbent(self):
        self._save(8_000_000, 0.40, tiebreak=0.10)
        self.assertFalse(self._save(8_500_000, 0.40, tiebreak=0.10))
        self.assertEqual(self.keeper.best_step, 8_000_000)

    def test_the_final_evaluation_competes_rather_than_wins(self):
        """A 10M model that is worse than the 9M one is not the best model."""
        self._save(9_000_000, 0.60)
        self.assertFalse(self._save(10_000_000, 0.55))
        self.assertEqual(self.keeper.best_step, 9_000_000)

    def test_only_improvements_serialize(self):
        self._save(8_000_000, 0.10)
        self._save(8_500_000, 0.05)
        self._save(9_000_000, 0.50)
        self.assertEqual(self.saves, [8_000_000, 9_000_000])

    def test_a_minimising_metric_selects_the_other_way(self):
        keeper = Checkpointer(_config(mode="min"), self.tmp, dict(IDENTITY),
                              save_fn=lambda payload, path: None)
        self.assertTrue(keeper.maybe_save(
            8_000_000, {"eval/success_once": 0.9}, dict))
        self.assertTrue(keeper.maybe_save(
            9_000_000, {"eval/success_once": 0.1}, dict))
        self.assertFalse(keeper.maybe_save(
            9_500_000, {"eval/success_once": 0.5}, dict))


class IdentityTest(unittest.TestCase):
    """A checkpoint whose vocabulary moved is a different model."""

    def test_matching_identity_has_no_mismatches(self):
        self.assertEqual(identity_mismatches(dict(IDENTITY), dict(IDENTITY)),
                         [])

    def test_every_field_is_compared(self):
        for field in IDENTITY:
            with self.subTest(field=field):
                stored = dict(IDENTITY)
                stored[field] = "moved"
                problems = identity_mismatches(stored, dict(IDENTITY))
                self.assertEqual(len(problems), 1)
                self.assertIn(field, problems[0])

    def test_a_missing_field_on_one_side_is_a_mismatch(self):
        stored = dict(IDENTITY)
        del stored["schedule"]
        self.assertTrue(identity_mismatches(stored, dict(IDENTITY)))

    def test_both_absent_is_not_a_mismatch(self):
        self.assertEqual(identity_mismatches({}, {}), [])


def _torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("torch is not installed")


class SaveLoadTest(unittest.TestCase):
    """Round trip, and the refusal."""

    @classmethod
    def setUpClass(cls):
        _torch()

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.keeper = Checkpointer(_config(), self.tmp, dict(IDENTITY))

    def _save(self, step=8_000_000, value=0.5):
        return self.keeper.maybe_save(
            step, {"eval/success_once": value, "eval/success_at_end": 0.4},
            lambda: {"model": {"w": 1}, "optim": {"lr": 3e-4}, "step": step})

    def test_one_file_and_no_others(self):
        self._save(8_000_000, 0.1)
        self._save(9_000_000, 0.5)
        self.assertEqual(sorted(os.listdir(self.tmp)), ["checkpoint_best.pt"])

    def test_it_round_trips(self):
        from checkpointing import load_checkpoint
        self._save()
        payload = load_checkpoint(self.keeper.path, dict(IDENTITY))
        self.assertEqual(payload["model"], {"w": 1})
        self.assertEqual(payload["optim"], {"lr": 3e-4})

    def test_the_selection_metadata_travels(self):
        from checkpointing import load_checkpoint
        self._save(9_000_000, 0.62)
        block = load_checkpoint(self.keeper.path, dict(IDENTITY))["checkpoint"]
        self.assertEqual(block["step"], 9_000_000)
        self.assertEqual(block["metric"], "eval/success_once")
        self.assertAlmostEqual(block["value"], 0.62)

    def test_loading_under_a_moved_vocabulary_raises(self):
        from checkpointing import load_checkpoint
        self._save()
        moved = dict(IDENTITY, entity_vocab="v19:zzz")
        with self.assertRaises(CheckpointError) as ctx:
            load_checkpoint(self.keeper.path, moved)
        self.assertIn("entity_vocab", str(ctx.exception))

    def test_the_refusal_names_every_field_that_moved(self):
        from checkpointing import load_checkpoint
        self._save()
        moved = dict(IDENTITY, entity_vocab="x", schedule="y")
        with self.assertRaises(CheckpointError) as ctx:
            load_checkpoint(self.keeper.path, moved)
        message = str(ctx.exception)
        self.assertIn("entity_vocab", message)
        self.assertIn("schedule", message)

    def test_a_missing_checkpoint_raises(self):
        from checkpointing import load_checkpoint
        with self.assertRaises(CheckpointError):
            load_checkpoint(os.path.join(self.tmp, "absent.pt"),
                            dict(IDENTITY))

    def test_a_failed_write_leaves_the_previous_file_intact(self):
        """Atomic replace: an interrupted save must not destroy the best the
        run had already earned."""
        from checkpointing import atomic_save
        self._save(8_000_000, 0.5)
        before = open(self.keeper.path, "rb").read()

        class Unpicklable:
            def __reduce__(self):
                raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            atomic_save({"bad": Unpicklable()}, self.keeper.path)
        self.assertEqual(open(self.keeper.path, "rb").read(), before)
        self.assertEqual(sorted(os.listdir(self.tmp)), ["checkpoint_best.pt"])


if __name__ == "__main__":
    unittest.main()
