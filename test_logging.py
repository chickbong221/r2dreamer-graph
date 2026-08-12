import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tools import FPS, Logger, prepare_video, wandb_scalars


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


class _FakeVideo:
    def __init__(self, value, fps, format):
        self.value = value
        self.fps = fps
        self.format = format


class LoggingTest(unittest.TestCase):
    def test_filter_keeps_only_comparison_metrics(self):
        selected = wandb_scalars(
            [
                ("episode/score", 1.0),
                ("train/loss/dyn", 2.0),
                ("train/semdyn_raw", 3.0),
                ("train/node_target_acc", 3.5),
                ("train/node_target_frac", 3.75),
                ("fps/policy", 4.0),
                ("fps/train", 5.0),
                ("fps/fps", 6.0),
                ("train/action_mean", 5.0),
                ("train/ret_replay_mean", 6.0),
            ]
        )
        self.assertEqual(
            set(selected),
            {
                "episode/score", "train/loss/dyn", "train/semdyn_raw",
                "train/node_target_acc", "train/node_target_frac",
                "fps/policy", "fps/train",
            },
        )

    def test_fps_counts_items_over_each_logging_interval(self):
        clock = iter((10.0, 12.0, 15.0)).__next__
        fps = FPS(clock)
        fps.step(100)
        self.assertEqual(fps.result(), 50.0)
        fps.step(60)
        self.assertEqual(fps.result(), 20.0)

    def test_video_is_tiled_into_wandb_layout(self):
        video = np.zeros((2, 3, 4, 5, 3), np.uint8)
        self.assertEqual(prepare_video(video).shape, (3, 3, 4, 10))

    def test_wandb_backend_sends_only_filtered_scalars(self):
        run = _FakeRun()
        fake_wandb = SimpleNamespace(init=lambda **kwargs: run, Video=_FakeVideo)
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
                logger.write(10)
                logger.close()
        self.assertEqual(run.logged[0]["env_step"], 10)
        self.assertEqual(run.logged[0]["train/loss/dyn"], 1.0)
        self.assertNotIn("train/action_mean", run.logged[0])
        self.assertTrue(run.finished)

    def test_wandb_video_uses_mp4(self):
        run = _FakeRun()
        fake_wandb = SimpleNamespace(init=lambda **kwargs: run, Video=_FakeVideo)
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
                with mock.patch.object(logger._writer, "add_video"):
                    logger.video(
                        "train_video",
                        np.zeros((1, 2, 4, 5, 3), np.uint8),
                        fps=15,
                    )
                    logger.write(10)
                logger.close()
        video = run.logged[0]["videos/train_video"]
        self.assertEqual(video.format, "mp4")
        self.assertEqual(video.fps, 15)


if __name__ == "__main__":
    unittest.main()
