"""Where the rolling-best checkpoint has to attach, and what must stay true.

``test_checkpointing`` proves the policy arithmetic. This proves the things
about the *training path* that the arithmetic assumes, and which nothing else
would notice if they drifted:

* the metric a checkpoint is selected on is one evaluation actually measures.
  Selecting on a key the logger never writes is worse than not selecting: the
  run would raise at the first eligible evaluation, 8M steps in;
* evaluation runs on a schedule the loop controls, so "eligible at 8M" is a
  property of that schedule and not of a step counter someone reads elsewhere;
* nothing in the training path writes a checkpoint unconditionally. One
  rolling best means no final save, no interrupt save, no periodic copy --
  and the cheapest way for one to appear is for someone to add ``torch.save``
  beside the logger and not think of it as a checkpoint.

Read from source rather than by importing: ``trainer`` pulls in torch, and
these are claims about what the file says.
"""

import ast
import unittest
from pathlib import Path

import yaml

from checkpointing import CheckpointConfig, CheckpointError, Checkpointer

TRAINER = Path("trainer.py").read_text(encoding="utf-8")
TRAIN = Path("train.py").read_text(encoding="utf-8")
MANISKILL = Path("envs/maniskill.py").read_text(encoding="utf-8")

# What ``Trainer.eval`` turns each ``log_*`` gauge into: it strips the prefix
# and namespaces the rest under ``eval/``.
CANDIDATE_METRICS = ("eval/success_once", "eval/success_at_end")


class TheCandidateMetricsAreRealTest(unittest.TestCase):
    """A metric has to be measured before it can be selected on."""

    def test_the_env_emits_the_gauges(self):
        for metric in CANDIDATE_METRICS:
            gauge = f'"log_{metric.split("/", 1)[1]}"'
            with self.subTest(metric=metric):
                self.assertIn(gauge, MANISKILL)

    def test_evaluation_republishes_them_under_eval(self):
        """``self.logger.scalar(f"eval/{key[4:]}", ...)`` -- the four-character
        slice is what turns ``log_success_once`` into ``eval/success_once``."""
        self.assertIn('f"eval/{key[4:]}"', TRAINER)

    def test_evaluation_also_writes_a_score(self):
        self.assertIn('"eval/score"', TRAINER)

    def test_the_metric_is_still_unset(self):
        """Choosing it is an experiment decision, not a default."""
        with open("configs/configs.yaml") as handle:
            block = yaml.safe_load(handle)["checkpoint"]
        self.assertEqual(block["metric"], "")
        self.assertFalse(block["enabled"])

    def test_enabling_without_one_refuses_before_training(self):
        with self.assertRaises(CheckpointError):
            CheckpointConfig(enabled=True).validate()

    def test_a_metric_the_evaluation_never_writes_refuses(self):
        keeper = Checkpointer(
            CheckpointConfig(enabled=True, metric="eval/not_measured",
                             start_step=8_000_000),
            "/tmp/x", {}, save_fn=lambda payload, path: None)
        with self.assertRaises(CheckpointError) as ctx:
            keeper.maybe_save(9_000_000, {"eval/score": 1.0}, dict)
        self.assertIn("never measured", str(ctx.exception))


class EvaluationIsWhereEligibilityIsDecidedTest(unittest.TestCase):
    """8M is a step count read at an evaluation, so there has to be one."""

    def test_the_loop_evaluates_on_its_own_schedule(self):
        self.assertIn("self._should_eval(step)", TRAINER)
        self.assertIn("self.eval(agent, step)", TRAINER)

    def test_the_evaluation_is_guarded_by_an_episode_count(self):
        """A run with no eval episodes never evaluates, so it can never
        become eligible -- which is correct, and worth pinning so nobody
        makes the checkpoint fall back to a training-step trigger."""
        self.assertIn("self.eval_episode_num > 0", TRAINER)

    def test_the_boundary_step_is_eligible_and_the_one_before_is_not(self):
        keeper = Checkpointer(
            CheckpointConfig(enabled=True, metric=CANDIDATE_METRICS[0],
                             start_step=8_000_000),
            "/tmp/x", {}, save_fn=lambda payload, path: None)
        self.assertFalse(keeper.eligible(7_999_999))
        self.assertTrue(keeper.eligible(8_000_000))


class NothingSavesUnconditionallyTest(unittest.TestCase):
    """One rolling best, and no second file by any other name."""

    def _calls(self, source):
        return {
            node.func.attr
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }

    def test_the_training_loop_writes_no_checkpoint_of_its_own(self):
        for name, source in (("trainer.py", TRAINER), ("train.py", TRAIN)):
            with self.subTest(module=name):
                self.assertNotIn("save", self._calls(source))

    def test_no_interrupt_or_exit_handler_saves(self):
        """``train.py`` registers ``atexit`` hooks; they close the log file
        and the logger, and must not grow into a save."""
        self.assertNotIn("torch.save", TRAIN)
        self.assertNotIn("KeyboardInterrupt", TRAIN)
        self.assertNotIn("KeyboardInterrupt", TRAINER)

    def test_the_config_names_one_path_and_no_milestone_settings(self):
        with open("configs/configs.yaml") as handle:
            block = yaml.safe_load(handle)["checkpoint"]
        self.assertEqual(block["path"], "checkpoint_best.pt")
        for absent in ("save_latest", "save_final", "save_periodically",
                       "save_on_interrupt", "keep_last"):
            with self.subTest(key=absent):
                self.assertNotIn(absent, block)


class TheModuleIsNotYetAttachedTest(unittest.TestCase):
    """Recording the state this audit found, so the gap is not mistaken for
    a working integration.

    ``checkpointing`` has no caller in the training path. That is not a
    defect in the module -- it is the reason a run today produces no
    checkpoint at all -- and wiring it needs the metric first, because the
    call site has to pass one and an unset metric refuses to start.
    """

    def test_the_trainer_does_not_import_it_yet(self):
        self.assertNotIn("checkpointing", TRAINER)
        self.assertNotIn("checkpointing", TRAIN)

    def test_evaluation_returns_nothing_to_select_on_yet(self):
        """``eval`` writes to the logger and returns None, so the wiring will
        have to hand the metrics back rather than re-read them."""
        tree = ast.parse(TRAINER)
        evals = [n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "eval"]
        self.assertEqual(len(evals), 1)
        returns = [n for n in ast.walk(evals[0])
                   if isinstance(n, ast.Return) and n.value is not None]
        self.assertEqual(returns, [])


if __name__ == "__main__":
    unittest.main()
