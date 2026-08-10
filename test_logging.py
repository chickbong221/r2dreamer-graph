import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools import Logger, wandb_scalars


class _FakeRun:
    def __init__(self):
        self.defined = []
        self.logged = []
        self.finished = False

    def define_metric(self, *args, **kwargs):
        self.defined.append((args, kwargs))

    def log(self, values):
        self.logged.append(dict(values))

    def finish(self):
        self.finished = True


class LoggingTest(unittest.TestCase):
    def test_filter_keeps_only_comparison_metrics(self):
        selected = wandb_scalars(
            [
                ("episode/score", 1.0),
                ("train/loss/dyn", 2.0),
                ("train/semdyn_raw", 3.0),
                ("fps/fps", 4.0),
                ("train/action_mean", 5.0),
                ("train/ret_replay_mean", 6.0),
            ]
        )
        self.assertEqual(
            set(selected),
            {"episode/score", "train/loss/dyn", "train/semdyn_raw", "fps/fps"},
        )

    def test_wandb_backend_sends_only_filtered_scalars(self):
        run = _FakeRun()
        fake_wandb = SimpleNamespace(init=lambda **kwargs: run)
        config = SimpleNamespace(
            enabled=True,
            project="test-project",
            entity="",
            name="test-run",
            group="test-group",
            mode="offline",
        )
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
                logger = Logger(Path(directory), wandb_config=config)
                logger.scalar("train/loss/dyn", 1.0)
                logger.scalar("train/action_mean", 2.0)
                logger.write(10, fps=False)
                logger.close()
        self.assertEqual(run.logged[0]["env_step"], 10)
        self.assertEqual(run.logged[0]["train/loss/dyn"], 1.0)
        self.assertNotIn("train/action_mean", run.logged[0])
        self.assertTrue(run.finished)


if __name__ == "__main__":
    unittest.main()
