"""Progress must arrive during the backlog, not only after it finishes."""

import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sim_vla.training.online_logging import train_updates
from sim_vla.training.wandb_logger import RunLogger, start_run


class FakeRun:
    url = "https://example.invalid/run"
    id = "test-run"

    def __init__(self):
        self.logged = []
        self.defined = []

    def log(self, values):
        self.logged.append(dict(values))

    def define_metric(self, name, **kwargs):
        self.defined.append((name, kwargs))


class TestOnlineReporting(unittest.TestCase):
    def test_backlog_is_announced_then_every_update_is_logged(self):
        now = [0.0]
        seen = []

        class Trainer:
            config = SimpleNamespace(train_ratio=64, batch_size=16, sequence_length=64)
            env_steps = 19200
            updates = 0
            replay = [None] * 128

            def update(self, *, on_progress):
                # The preceding update must reach the sink BEFORE this one starts.
                completed = [m["updates"] for m in seen if "updates" in m]
                self_test.assertEqual(completed, list(range(1, self.updates + 1)))
                self_test.assertEqual(seen[0]["online_progress/pending_updates"], 3)
                on_progress("world_forward")
                now[0] += 20.0
                on_progress("imagine", microbatch=2, microbatches=33,
                            imagination_starts=1040, actor_training=True)
                now[0] += 10.0
                self.updates += 1
                return {"world_loss": 2.0, "actor_loss": -1.0}

        self_test = self
        trainer = Trainer()
        output = io.StringIO()
        with patch("sim_vla.training.online_logging.time.monotonic", side_effect=lambda: now[0]), \
                contextlib.redirect_stdout(output):
            last = train_updates(trainer, 3, seen.append)
        completed = [m for m in seen if "updates" in m]
        self.assertEqual([m["updates"] for m in completed], [1, 2, 3])
        self.assertEqual([m["env_steps"] for m in completed], [19200] * 3)
        self.assertEqual([m["online_progress/backlog_eta_s"] for m in completed], [60, 30, 0])
        self.assertEqual(last["online_progress/pending_updates"], 0)
        self.assertIn("3 updates due", output.getvalue())
        self.assertIn("microbatch=2/33", output.getvalue())
        self.assertIn("trained 3/3", output.getvalue())

    def test_zero_backlog_does_not_train_or_log(self):
        with patch("builtins.print") as output:
            self.assertEqual(train_updates(object(), 0, self.fail), {})
        output.assert_not_called()

    def test_repeated_fast_phase_events_are_throttled(self):
        now = [0.0]
        seen = []

        class Trainer:
            config = SimpleNamespace(train_ratio=1, batch_size=1, sequence_length=1)
            env_steps = 10
            updates = 2
            replay = []

            def update(self, *, on_progress):
                for _ in range(100):
                    on_progress("actor_backward")
                now[0] = 16.0
                on_progress("actor_backward")
                self.updates += 1
                return {"world_loss": 1.0}

        with patch("sim_vla.training.online_logging.time.monotonic", side_effect=lambda: now[0]), \
                contextlib.redirect_stdout(io.StringIO()):
            train_updates(Trainer(), 1, seen.append)
        phases = [m for m in seen if "online_progress/current_update" in m]
        self.assertEqual(len(phases), 1)


class TestWandbUpdateAxes(unittest.TestCase):
    def test_catchup_losses_keep_an_advancing_update_axis(self):
        run = FakeRun()
        logger = RunLogger(run)
        for update in (1, 2):
            logger.log({"env_steps": 19200, "updates": update,
                        "world_loss": 2.0, "actor_loss": -0.1,
                        "online_progress/pending_updates": 1200 - update}, stage="online")
        self.assertEqual([m["online_train/updates"] for m in run.logged], [1, 2])
        self.assertEqual([m["online/env_steps"] for m in run.logged], [19200, 19200])
        self.assertEqual([m["online_progress/event"] for m in run.logged], [1, 2])
        self.assertEqual(run.logged[0]["online_train/actor_loss"], -0.1)
        self.assertEqual(run.logged[0]["online/actor_loss"], -0.1)

    def test_episode_and_heartbeat_rows_do_not_copy_stale_training_losses(self):
        run = FakeRun()
        logger = RunLogger(run)
        logger.log({"env_steps": 19200, "episode/score": 9.314}, stage="online")
        logger.log({"env_steps": 19200, "online_progress/phase": 7}, stage="online")
        self.assertIn("episode/score", run.logged[0])
        self.assertFalse(any(k.startswith("online_train/") for m in run.logged for k in m))

    def test_metric_definitions_bind_both_new_axes(self):
        run = FakeRun()
        fake_wandb = SimpleNamespace(init=lambda **kwargs: run)
        with patch.dict("sys.modules", {"wandb": fake_wandb}), \
                contextlib.redirect_stdout(io.StringIO()):
            start_run({"wandb": {"enabled": True}})
        self.assertIn(("online_train/*", {"step_metric": "online_train/updates"}), run.defined)
        self.assertIn(("online_progress/*", {"step_metric": "online_progress/event"}), run.defined)
        self.assertIn(("episode/*", {"step_metric": "online/env_steps"}), run.defined)


if __name__ == "__main__":
    unittest.main()
