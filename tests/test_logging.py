import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tools import (
    FPS, Logger, StageTimer, prepare_video, process_memory_stats, wandb_scalars,
)


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
                ("train/graph_align_cos", 3.0),
                ("train/node_target_acc", 3.5),
                ("train/node_target_frac", 3.75),
                ("fps/policy", 4.0),
                ("fps/train", 5.0),
                ("fps/fps", 6.0),
                ("system/process_ram_gib", 7.0),
                ("system/process_peak_ram_gib", 8.0),
                ("system/machine_ram_gib", 9.0),
                ("train/action_mean", 5.0),
                ("train/ret_replay_mean", 6.0),
            ]
        )
        self.assertEqual(
            set(selected),
            {
                "episode/score", "train/loss/dyn", "train/graph_align_cos",
                "train/node_target_acc", "train/node_target_frac",
                "fps/policy", "fps/train", "system/process_ram_gib",
                "system/process_peak_ram_gib",
            },
        )

    @mock.patch("tools._peak_process_rss_bytes", return_value=6 * 1024 ** 3)
    @mock.patch("tools._linux_process_rss_bytes", return_value=4 * 1024 ** 3)
    def test_process_memory_reports_current_and_peak_gib(self, _, __):
        self.assertEqual(
            process_memory_stats(),
            {
                "system/process_ram_gib": 4.0,
                "system/process_peak_ram_gib": 6.0,
            },
        )

    def test_fps_counts_items_over_each_logging_interval(self):
        clock = iter((10.0, 12.0, 15.0)).__next__
        fps = FPS(clock)
        fps.step(100)
        self.assertEqual(fps.result(), 50.0)
        fps.step(60)
        self.assertEqual(fps.result(), 20.0)

    def test_console_line_is_compact_while_json_keeps_every_metric(self):
        values = {
            "train/loss/dyn": 11.2,
            "train/graph_align_mse": 0.61,
            "train/opt/updates": 375.0,
            "train/graph_real_edges": 20.7,
            "eval/success_once": 0.25,
            "eval_scene/held_out/success_once": 0.5,
            "eval_scene/held_out/graph_cache_entries": 3.0,
            "eval_light/dim/reset_rgb_mae": 0.00034,
            "eval_light/dim/success_delta_vs_nominal": -0.125,
            "eval_light/dim/intensity_multiplier": 0.4,
            "fps/policy": 5.5,
        }
        with tempfile.TemporaryDirectory() as directory:
            logger = Logger(Path(directory))
            for name, value in values.items():
                logger.scalar(name, value)
            with mock.patch("builtins.print") as printed:
                logger.write(1764)
            logger.close()
            record = json.loads(
                (Path(directory) / "metrics.jsonl").read_text().strip())
        line = printed.call_args_list[0][0][1]
        self.assertEqual(record["step"], 1764)
        self.assertEqual({key: record[key] for key in values}, values)
        for kept in ("train/opt/updates", "train/graph_real_edges", "fps/policy",
                     "eval/success_once", "eval_scene/held_out/success_once",
                     "eval_light/dim/success_delta_vs_nominal"):
            self.assertIn(kept, line)
        for dropped in ("train/loss/dyn", "train/graph_align_mse",
                        "eval_scene/held_out/graph_cache_entries",
                        "eval_light/dim/intensity_multiplier"):
            self.assertNotIn(dropped, line)
        # The old one-decimal console printed this reading as 0.0.
        self.assertIn("eval_light/dim/reset_rgb_mae 0.00034", line)

    def test_disabled_timing_never_synchronises_or_reports(self):
        calls = []
        timer = StageTimer(enabled=False, device="cuda")
        with mock.patch("torch.cuda.synchronize",
                        side_effect=lambda *args, **kwargs: calls.append(1)):
            with timer.measure("update"):
                pass
        self.assertEqual(calls, [])
        self.assertEqual(timer.metrics(), {})

    def test_enabled_timing_averages_each_stage_then_clears_it(self):
        clock = iter((0.0, 0.5, 2.0, 2.5)).__next__
        timer = StageTimer(enabled=True, device="cpu", clock=clock)
        for _ in range(2):
            with timer.measure("update"):
                pass
        self.assertEqual(timer.metrics("eval_timing"),
                         {"eval_timing/update_ms": 500.0})
        self.assertEqual(timer.metrics(), {})

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
