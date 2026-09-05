"""The rolling-best checkpoint, driven the way the training loop drives it.

``test_checkpointing`` proves the selection arithmetic in isolation. This
proves the parts that only exist once it is attached to a run: that every
evaluation is offered and only eligible ones can claim the file, that a run
which stops still holds the best it earned, and that the identity a checkpoint
carries is about the model contract rather than about the scene it was
evaluated in -- which is what lets Experiment C load Experiment B's weights
under different lighting.

The metric used here is deliberately a made-up one. Which metric selects the
production checkpoint is an experiment decision that has not been made, and a
test that quietly picked one would make it look decided.
"""

import ast
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from checkpointing import (
    CheckpointConfig,
    CheckpointError,
    Checkpointer,
    identity_mismatches,
    run_identity,
)

TRAINER = Path("trainer.py").read_text(encoding="utf-8")
TRAIN = Path("train.py").read_text(encoding="utf-8")
MANISKILL = Path("envs/maniskill.py").read_text(encoding="utf-8")

# Not the experiment metric. See the module docstring.
TEST_METRIC = "eval/test_metric"

IDENTITY = dict(
    whitelist_dir="scenegraph/configs/subtask_whitelists/tidy_house",
    affordance_path="scenegraph/configs/affordances/tidy_house.json",
    schedule_path="scenegraph/configs/schedules/tidy_house/pick.json",
    schedule_label="tidy_house/pick", n_max=8, e_max=168, n_cams=2,
    entity_ids={"<pad>": 0, "<ee>": 1, "actor:004_sugar_box": 2},
    relation_ids={"grasp": 1, "reached": 2},
    absolute_ids={"holds": 1, "not-holds": 2},
    temporal_ids={"stable": 1, "closer": 2},
)


class _Loop:
    """The evaluation loop, minus the simulator.

    Mirrors ``OnlineTrainer._maybe_checkpoint``: every evaluation is offered,
    the state closure is called only when a save happens, and nothing else in
    the loop writes anything.
    """

    def __init__(self, tmp, **over):
        config = CheckpointConfig(
            enabled=True, start_step=8_000_000, metric=TEST_METRIC,
            tiebreak="eval/tiebreak", mode="max")
        for key, value in over.items():
            setattr(config, key, value)
        self.serialised = []
        self.keeper = Checkpointer(config, tmp, {"assets": "x"},
                                   save_fn=self._write)

    def _write(self, payload, path):
        with open(path, "w") as handle:
            handle.write(str(payload["step"]))

    def evaluate(self, step, value, tiebreak=0.0):
        metrics = {"eval/score": value, TEST_METRIC: value,
                   "eval/tiebreak": tiebreak}
        return self.keeper.maybe_save(
            step, metrics,
            lambda: self.serialised.append(step) or {"step": step})


class EligibilityInTheLoopTest(unittest.TestCase):
    """Before 8M an evaluation still runs; it just cannot claim anything."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.loop = _Loop(self.tmp)

    def test_early_evaluations_save_nothing(self):
        for step in (0, 2_000_000, 7_999_999):
            with self.subTest(step=step):
                self.assertFalse(self.loop.evaluate(step, 0.99))
        self.assertEqual(os.listdir(self.tmp), [])

    def test_a_strong_early_result_does_not_become_the_incumbent(self):
        """Otherwise the first eligible evaluation could never write."""
        self.loop.evaluate(2_000_000, 0.99)
        self.assertIsNone(self.loop.keeper.best)
        self.assertTrue(self.loop.evaluate(8_000_000, 0.10))

    def test_nothing_is_serialised_for_an_ineligible_evaluation(self):
        self.loop.evaluate(2_000_000, 0.99)
        self.assertEqual(self.loop.serialised, [])

    def test_only_improvements_replace_the_file(self):
        self.loop.evaluate(8_000_000, 0.10)
        self.loop.evaluate(8_500_000, 0.05)
        self.loop.evaluate(9_000_000, 0.50)
        self.assertEqual(self.loop.serialised, [8_000_000, 9_000_000])
        self.assertEqual(sorted(os.listdir(self.tmp)), ["checkpoint_best.pt"])

    def test_a_run_that_stops_keeps_the_best_it_earned(self):
        """Cancellation writes nothing and destroys nothing."""
        self.loop.evaluate(8_000_000, 0.40)
        self.loop.evaluate(9_000_000, 0.20)          # then the run is killed
        self.assertEqual(sorted(os.listdir(self.tmp)), ["checkpoint_best.pt"])
        with open(os.path.join(self.tmp, "checkpoint_best.pt")) as handle:
            self.assertEqual(handle.read(), "8000000")

    def test_a_disabled_run_offers_evaluations_and_saves_none(self):
        loop = _Loop(self.tmp, enabled=False)
        self.assertFalse(loop.evaluate(9_000_000, 0.9))
        self.assertEqual(loop.serialised, [])


class IdentityIsAboutTheModelNotTheSceneTest(unittest.TestCase):
    """Experiment C reuses B's weights under different lighting."""

    def _identity(self, **over):
        return run_identity(**dict(IDENTITY, **over))

    def test_the_same_assets_give_the_same_identity(self):
        self.assertEqual(identity_mismatches(self._identity(),
                                             self._identity()), [])

    def test_every_field_is_populated(self):
        identity = self._identity()
        self.assertEqual(sorted(identity),
                         ["absolute_vocab", "assets", "entity_vocab",
                          "graph_schema", "relation_vocab", "schedule", "temporal_vocab"])
        self.assertNotIn("", identity.values())

    def test_vocabulary_ids_matter_but_dictionary_order_does_not(self):
        for field in ("entity", "relation", "absolute", "temporal"):
            name = f"{field}_ids"
            original = IDENTITY[name]
            with self.subTest(vocabulary=field):
                reversed_items = dict(reversed(list(original.items())))
                self.assertEqual(self._identity(**{name: reversed_items}), self._identity())
                moved = dict(original)
                first, second = list(moved)[:2]
                moved[first], moved[second] = moved[second], moved[first]
                problems = identity_mismatches(self._identity(**{name: moved}), self._identity())
                self.assertEqual(len(problems), 1)
                self.assertIn(f"{field}_vocab", problems[0])

    def test_a_moved_capacity_is_a_mismatch(self):
        """Packed shapes and the enabled relation set are part of the contract."""
        for override in ({"n_max": 11}, {"disable_object_object_relations": True},
                         {"protected_pick_fifo": True}):
            with self.subTest(override=override):
                moved = self._identity(**override)
                self.assertTrue(identity_mismatches(moved, self._identity()))

    def test_a_different_schedule_is_a_mismatch(self):
        moved = self._identity(schedule_label="tidy_house/place")
        self.assertTrue(identity_mismatches(moved, self._identity()))

    def test_all_resolved_assets_are_bound_but_their_location_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            whitelist = root / "whitelists"
            whitelist.mkdir()
            union = whitelist / "pick_all.json"
            target = whitelist / "pick_004_sugar_box.json"
            affordance = root / "tidy_house.json"
            for path in (union, target, affordance):
                path.write_text("{}", encoding="utf-8")
            paths = dict(whitelist_dir=str(whitelist), affordance_path=str(affordance))
            original = self._identity(**paths)
            relocated = root / "relocated"
            shutil.copytree(whitelist, relocated)
            self.assertEqual(original, self._identity(**dict(paths, whitelist_dir=str(relocated))))
            for path in (union, target, affordance):
                with self.subTest(asset=path.name):
                    path.write_text(json.dumps({"changed": True}), encoding="utf-8")
                    differences = identity_mismatches(original, self._identity(**paths))
                    self.assertEqual(len(differences), 1)
                    self.assertIn("assets", differences[0])
                    path.write_text("{}", encoding="utf-8")
            with self.assertRaises(CheckpointError):
                self._identity(**dict(paths, affordance_path=str(root / "missing.json")))
            with self.assertRaises(CheckpointError):
                self._identity(**dict(paths, whitelist_dir=str(root / "missing")))


class TheMetricIsStillTheUsersToChooseTest(unittest.TestCase):

    def test_the_shipped_config_names_none_and_is_off(self):
        with open("configs/configs.yaml") as handle:
            block = yaml.safe_load(handle)["checkpoint"]
        self.assertEqual(block["metric"], "")
        self.assertFalse(block["enabled"])
        self.assertEqual(float(block["start_step"]), 8e6)
        self.assertEqual(block["path"], "checkpoint_best.pt")

    def test_enabling_without_one_refuses(self):
        with self.assertRaises(CheckpointError):
            CheckpointConfig(enabled=True).validate()

    def test_an_invalid_mode_refuses(self):
        with self.assertRaises(CheckpointError):
            CheckpointConfig(enabled=True, metric=TEST_METRIC,
                             mode="highest").validate()

    def test_the_refusal_happens_before_the_envs_are_built(self):
        """Seconds, not a scene build and eight million steps."""
        tree = ast.parse(TRAIN)
        body = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
        order = [ast.dump(stmt) for stmt in body.body]
        validate = next(i for i, s in enumerate(order) if "validate" in s)
        envs = next(i for i, s in enumerate(order) if "make_envs" in s)
        self.assertLess(validate, envs)

    def test_a_metric_the_evaluation_never_writes_refuses(self):
        keeper = Checkpointer(
            CheckpointConfig(enabled=True, metric="eval/not_measured",
                             start_step=8_000_000),
            "/tmp/x", {}, save_fn=lambda payload, path: None)
        with self.assertRaises(CheckpointError) as ctx:
            keeper.maybe_save(9_000_000, {"eval/score": 1.0}, dict)
        self.assertIn("never measured", str(ctx.exception))

    def test_the_candidate_metrics_are_ones_the_env_measures(self):
        """Not a choice, a menu: these are what evaluation can select on."""
        for gauge in ("log_success_once", "log_success_at_end"):
            with self.subTest(gauge=gauge):
                self.assertIn(f'"{gauge}"', MANISKILL)


class NothingSavesUnconditionallyTest(unittest.TestCase):
    """One rolling best, and no second file by any other name."""

    def _calls(self, source):
        return {node.func.attr for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)}

    def test_the_loop_writes_no_checkpoint_of_its_own(self):
        """``maybe_save`` is the only route, and it applies the policy."""
        for name, source in (("trainer.py", TRAINER), ("train.py", TRAIN)):
            with self.subTest(module=name):
                self.assertNotIn("save", self._calls(source))
        self.assertIn("maybe_save", self._calls(TRAINER))

    def test_no_interrupt_or_exit_handler_saves(self):
        self.assertNotIn("torch.save", TRAIN)
        self.assertNotIn("KeyboardInterrupt", TRAIN)
        self.assertNotIn("KeyboardInterrupt", TRAINER)

    def test_the_config_carries_no_milestone_settings(self):
        with open("configs/configs.yaml") as handle:
            block = yaml.safe_load(handle)["checkpoint"]
        for absent in ("save_latest", "save_final", "save_periodically",
                       "save_on_interrupt", "keep_last"):
            with self.subTest(key=absent):
                self.assertNotIn(absent, block)


def _torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("torch is not installed")


class TrainerHookTest(unittest.TestCase):
    """The hook itself, on a trainer built without a simulator."""

    @classmethod
    def setUpClass(cls):
        _torch()

    def _trainer(self, checkpointer):
        from trainer import OnlineTrainer
        trainer = OnlineTrainer.__new__(OnlineTrainer)
        trainer.checkpointer = checkpointer
        return trainer

    def test_an_evaluation_is_offered_to_the_checkpointer(self):
        import torch

        tmp = tempfile.mkdtemp()
        loop = _Loop(tmp)
        agent = torch.nn.Linear(2, 2)
        self._trainer(loop.keeper)._maybe_checkpoint(
            agent, 9_000_000, {TEST_METRIC: 0.5, "eval/tiebreak": 0.0})
        self.assertEqual(sorted(os.listdir(tmp)), ["checkpoint_best.pt"])

    def test_checkpointing_off_is_a_no_op(self):
        import torch

        self._trainer(None)._maybe_checkpoint(
            torch.nn.Linear(2, 2), 9_000_000, {TEST_METRIC: 0.5})

    def test_the_model_state_round_trips(self):
        """Weights and identity, enough to evaluate and to fine-tune from."""
        import torch

        from checkpointing import load_checkpoint

        tmp = tempfile.mkdtemp()
        identity = run_identity(**IDENTITY)
        keeper = Checkpointer(
            CheckpointConfig(enabled=True, metric=TEST_METRIC,
                             start_step=8_000_000),
            tmp, identity)
        agent = torch.nn.Linear(2, 2)
        self._trainer(keeper)._maybe_checkpoint(
            agent, 9_000_000, {TEST_METRIC: 0.5})
        payload = load_checkpoint(keeper.path, identity)
        self.assertEqual(sorted(payload["model"]), ["bias", "weight"])
        self.assertEqual(payload["checkpoint"]["step"], 9_000_000)

    def test_a_moved_contract_refuses_to_load(self):
        import torch

        from checkpointing import load_checkpoint

        tmp = tempfile.mkdtemp()
        identity = run_identity(**IDENTITY)
        keeper = Checkpointer(
            CheckpointConfig(enabled=True, metric=TEST_METRIC,
                             start_step=8_000_000),
            tmp, identity)
        self._trainer(keeper)._maybe_checkpoint(
            torch.nn.Linear(2, 2), 9_000_000, {TEST_METRIC: 0.5})
        moved = run_identity(**dict(IDENTITY, n_max=11))
        with self.assertRaises(CheckpointError) as ctx:
            load_checkpoint(keeper.path, moved)
        self.assertIn("graph_schema", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
