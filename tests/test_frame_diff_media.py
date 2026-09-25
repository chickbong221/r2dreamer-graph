"""Pixel-difference chart and video: the metric, the CSV, the figures, the 2x2 layout."""

import csv
import importlib.util
import pathlib
import shutil
import tempfile
import unittest

import numpy as np

from scenegraph.figures.diff_writer import DiffEpisodeWriter
from scenegraph.tools.plot_frame_diff import (
    CSV_FIELDS, diff_metrics, draw_episode, draw_panels, episode_rows,
    first_success_step, load_manifest, shared_tops, write_csv,
)
from scenegraph.tools.render_diff_video import (
    BLOCK, PANELS, GridCanvas, episode_frames, panel_keys, video_records,
    write_video,
)

H, W = 10, 14


def _write_episode(root, name="Task-v1_seed0000", steps=4, *, save_frames=True,
                   title="Task One"):
    writer = DiffEpisodeWriter(root, name, roles=["head", "wrist"],
                               sensor_size=[H, W], save_human=False,
                               save_frames=save_frames)
    writer.open()
    for t in range(steps):
        writer.write_step(
            step=t,
            sensors={"head": np.full((H, W, 3), 10 + 3 * t, np.uint8),
                     "wrist": np.full((H, W, 3), 10 + 5 * t, np.uint8)},
            extra={"reward": None if t == 0 else 0.1 * t, "success": t >= 2},
        )
    writer.write_amplified(percentile=99.5)
    return writer.commit({"title": title, "attempt": {"first_success_step": 2}})


class TempRoot(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)


class MetricTest(unittest.TestCase):
    def test_mean_is_over_pixels_and_channels_and_scaled_to_one(self):
        diff = np.zeros((4, 4, 3), np.uint8)
        diff[:2, :, 0] = 51
        mad, changed = diff_metrics(diff, threshold=10)
        self.assertAlmostEqual(mad, 51 / (2 * 3 * 255))
        self.assertAlmostEqual(changed, 0.5)
        self.assertEqual(diff_metrics(diff, threshold=51)[1], 0.0)


class ChartTest(TempRoot):
    def test_rows_come_from_the_raw_differences(self):
        episode = _write_episode(self.root)
        manifest = load_manifest(episode)
        rows = episode_rows(episode, manifest, threshold=4)
        self.assertEqual([r["step"] for r in rows], [1, 2, 3])
        self.assertAlmostEqual(rows[0]["head_mad"], 3 / 255)
        self.assertAlmostEqual(rows[0]["wrist_mad"], 5 / 255)
        self.assertEqual(rows[0]["head_changed"], 0.0)
        self.assertEqual(rows[0]["wrist_changed"], 1.0)
        self.assertEqual([r["success"] for r in rows], [0, 1, 1])
        self.assertEqual(first_success_step(manifest), 2)

    def test_csv_carries_both_metrics(self):
        episode = _write_episode(self.root)
        rows = episode_rows(episode, load_manifest(episode))
        path = write_csv(rows, self.root / "out" / "x.csv")
        with open(path, newline="") as handle:
            read = list(csv.DictReader(handle))
        self.assertEqual(tuple(read[0]), CSV_FIELDS)
        self.assertEqual(len(read), 3)

    def test_figures_are_written_as_png_and_pdf(self):
        a = _write_episode(self.root, "A-v1_seed0000")
        b = _write_episode(self.root, "B-v1_seed0000", steps=6, title="Task Two")
        c = _write_episode(self.root, "C-v1_seed0000", steps=3, title="Task Three")
        rows = {p: episode_rows(p, load_manifest(p)) for p in (a, b, c)}
        written = draw_episode(rows[a], self.root / "out" / "A", title="Task One",
                               first_success=2)
        self.assertEqual([p.suffix for p in written], [".png", ".pdf"])
        panels = [("One", rows[a], 2), ("Two", rows[b], None), ("Three", rows[c], 2)]
        for sharey in ("all", "row", "none"):
            written = draw_panels(panels, self.root / "out" / f"panels_{sharey}",
                                  cols=2, sharey=sharey, metric="changed")
            self.assertTrue(all(p.stat().st_size > 0 for p in written))


class SharedTopsTest(unittest.TestCase):
    def test_every_panel_fits_the_largest_value_of_its_shared_group(self):
        peaks = [0.08, 0.12, 0.10, 0.02, 0.07]
        self.assertEqual(shared_tops(peaks, 3, "all"), [0.12] * 5)
        self.assertEqual(shared_tops(peaks, 3, "row"),
                         [0.12, 0.12, 0.12, 0.07, 0.07])
        self.assertEqual(shared_tops(peaks, 3, "none"), peaks)


class CanvasTest(unittest.TestCase):
    def test_panels_land_pixel_for_pixel_in_a_block_aligned_frame(self):
        canvas = GridCanvas((H, W), title="Task", labels=[l for _, l in PANELS])
        try:
            colours = [(200, 0, 0), (0, 200, 0), (0, 0, 200), (90, 90, 90)]
            panels = [np.tile(np.array(c, np.uint8), (H, W, 1)) for c in colours]
            panels[0][0, 0] = (1, 2, 3)
            frame = canvas.render(panels, "Step 1 / 1")
            height, width = canvas.size
            self.assertEqual(frame.shape, (height, width, 3))
            self.assertEqual((height % BLOCK, width % BLOCK), (0, 0))
            for (top, left), panel in zip(canvas.slots, panels):
                np.testing.assert_array_equal(
                    frame[top:top + H, left:left + W], panel)
            with self.assertRaises(ValueError):
                canvas.render([np.zeros((H + 1, W, 3), np.uint8)] * 4)
        finally:
            canvas.close()


class VideoTest(TempRoot):
    def test_diffs_only_episode_has_nothing_to_show(self):
        episode = _write_episode(self.root, save_frames=False)
        with self.assertRaises(SystemExit):
            video_records(load_manifest(episode), panel_keys(False))

    def test_frames_start_at_the_first_difference(self):
        episode = _write_episode(self.root)
        keys = panel_keys(False)
        records = video_records(load_manifest(episode), keys)
        self.assertEqual([r["step"] for r in records], [1, 2, 3])
        canvas = GridCanvas((H, W), title="Task", labels=[l for _, l in PANELS])
        try:
            frames = list(episode_frames(episode, canvas, records, keys))
        finally:
            canvas.close()
        self.assertEqual(len(frames), 3)

    def test_mp4_is_written(self):
        missing = [m for m in ("imageio", "imageio_ffmpeg")
                   if importlib.util.find_spec(m) is None]
        if missing:
            self.skipTest(f"SKIPPED LOUDLY: {missing} not installed; the mp4 "
                          "writer is untested here")
        frames = (np.full((32, 48, 3), 40 * i, np.uint8) for i in range(3))
        target = self.root / "v" / "x.mp4"
        self.assertEqual(write_video(frames, target, fps=5, crf=18), 3)
        self.assertGreater(target.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
