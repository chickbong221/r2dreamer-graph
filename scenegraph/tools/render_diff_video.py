"""A 2x2 video per episode: head frame, wrist frame, and their frame differences.

    python -m scenegraph.tools.render_diff_video \
        data/paper_figures/CloseSubtaskTrain-v0_fridge_seed0000 \
        --out data/paper_figures/videos --fps 10

Writes <out>/<name>.mp4 (H.264, yuv420p). The difference panels show diff_vis/,
the episode-wide gain the figure uses; --raw-diff shows the unscaled diff/.
Needs imageio with imageio-ffmpeg.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np

from scenegraph.figures.diff_writer import load_png
from scenegraph.tools.demo_motionplanning_reward import INK, MUTED
from scenegraph.tools.plot_frame_diff import house_pyplot, load_manifest

PANELS: Tuple[Tuple[str, str], ...] = (
    ("head", "Head camera"),
    ("wrist", "Wrist camera"),
    ("head_diff", "Head difference"),
    ("wrist_diff", "Wrist difference"),
)
DPI = 100
BLOCK = 16


def _round_up(value: int, block: int = BLOCK) -> int:
    return int(-(-int(value) // block) * block)


class GridCanvas:
    """Four equal panels in a 2x2 grid under a title bar, placed pixel for pixel."""

    def __init__(self, panel_hw: Sequence[int], *, title: str,
                 labels: Sequence[str], margin: int = 24, gap: int = 20,
                 title_height: int = 70, label_height: int = 46,
                 title_pt: float = 24.0, label_pt: float = 18.0):
        plt = house_pyplot()
        h, w = (int(v) for v in panel_hw)
        width = _round_up(2 * w + gap + 2 * margin)
        height = _round_up(title_height + 2 * (label_height + h) + gap + margin)
        left = (width - (2 * w + gap)) // 2
        self.size = (height, width)
        self.fig = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI,
                              facecolor="white")
        self.slots: List[Tuple[int, int]] = []
        for row in range(2):
            top = title_height + row * (label_height + h + gap) + label_height
            for col in range(2):
                self.slots.append((top, left + col * (w + gap)))
        self._images = [
            self.fig.figimage(np.zeros((h, w, 3), np.uint8), xo=x,
                              yo=height - top - h, origin="upper")
            for top, x in self.slots
        ]
        for (top, x), label in zip(self.slots, labels):
            self._text(x + w / 2, top - label_height / 2, label, label_pt,
                       ha="center")
        self._text(left, title_height / 2, title, title_pt, ha="left")
        self._counter = self._text(left + 2 * w + gap, title_height / 2, "",
                                   label_pt, ha="right", color=MUTED)
        self.panel_hw = (h, w)

    def _text(self, x: float, y: float, text: str, size: float, *, ha: str,
              color: str = INK):
        height, width = self.size
        return self.fig.text(x / width, 1.0 - y / height, text, fontsize=size,
                             color=color, ha=ha, va="center")

    def render(self, panels: Sequence[np.ndarray], counter: str = "") -> np.ndarray:
        for image, panel in zip(self._images, panels):
            arr = np.asarray(panel)
            if arr.shape[:2] != self.panel_hw:
                raise ValueError(f"panel is {arr.shape[:2]}, expected {self.panel_hw}")
            image.set_data(arr[..., :3])
        self._counter.set_text(counter)
        self.fig.canvas.draw()
        return np.array(np.asarray(self.fig.canvas.buffer_rgba())[..., :3])

    def close(self) -> None:
        import matplotlib.pyplot as plt

        plt.close(self.fig)


def panel_keys(raw_diff: bool) -> Dict[str, str]:
    suffix = "" if raw_diff else "_vis"
    return {"head": "head", "wrist": "wrist",
            "head_diff": f"head_diff{suffix}", "wrist_diff": f"wrist_diff{suffix}"}


def video_records(manifest: dict, keys: Dict[str, str]) -> List[dict]:
    records = [r for r in manifest.get("steps", [])
               if all(keys[slot] in r for slot, _ in PANELS)]
    if not records:
        missing = [keys[slot] for slot, _ in PANELS]
        raise SystemExit(f"{manifest.get('name')}: no step carries all of "
                         f"{missing}; were the frames exported (not --diffs-only)?")
    return records


def episode_frames(episode: Path, canvas: GridCanvas, records: Sequence[dict],
                   keys: Dict[str, str]) -> Iterator[np.ndarray]:
    last = records[-1]["step"]
    for record in records:
        panels = [load_png(episode / record[keys[slot]]) for slot, _ in PANELS]
        yield canvas.render(panels, f"Step {record['step']} / {last}")


def write_video(frames: Iterator[np.ndarray], path: Path, *, fps: float,
                crf: int) -> int:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with imageio.get_writer(str(path), fps=float(fps), codec="libx264",
                            quality=None, pixelformat="yuv420p",
                            macro_block_size=BLOCK, ffmpeg_log_level="error",
                            ffmpeg_params=["-crf", str(int(crf))]) as writer:
        for frame in frames:
            writer.append_data(frame)
            count += 1
    return count


def run(args) -> int:
    out = Path(args.out)
    keys = panel_keys(args.raw_diff)
    for episode in (Path(e) for e in args.episodes):
        manifest = load_manifest(episode)
        name = str(manifest.get("name") or episode.name)
        records = video_records(manifest, keys)
        first = load_png(episode / records[0][keys["head"]])
        canvas = GridCanvas(first.shape[:2],
                            title=str(manifest.get("title") or name).replace("_", " "),
                            labels=[label for _, label in PANELS])
        try:
            target = out / f"{name}.mp4"
            count = write_video(episode_frames(episode, canvas, records, keys),
                                target, fps=args.fps, crf=args.crf)
        finally:
            canvas.close()
        height, width = canvas.size
        print(f"wrote {target} ({count} frames, {width}x{height}, "
              f"{count / args.fps:.1f}s at {args.fps:g} fps)", flush=True)
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Render a 2x2 head/wrist frame and difference video per "
                    "exported paper-frame episode")
    p.add_argument("episodes", nargs="+")
    p.add_argument("--out", default="data/paper_figures/videos")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--crf", type=int, default=18,
                   help="x264 quality; lower is better and larger")
    p.add_argument("--raw-diff", action="store_true",
                   help="show the unscaled diff/ instead of diff_vis/")
    return p.parse_args(argv)


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
