"""Per-step pixel change of the head and wrist cameras, in the reward figure's style.

    python -m scenegraph.tools.plot_frame_diff \
        data/paper_figures/CloseSubtaskTrain-v0_fridge_seed0000 \
        data/paper_figures/PullCubeTool-v1_seed0000 \
        --out data/paper_figures/pixel_diff

Per episode: <out>/<name>.png, .pdf and .csv. With two or more episodes, also
<out>/pixel_diff_panels.png and .pdf, one panel per episode in the given order.

--metric mad (default): mean absolute difference between consecutive frames,
sum |I_t - I_{t-1}| / (255 * H * W * 3), from the raw diff/ PNGs.
--metric changed: fraction of pixels whose largest channel difference exceeds
--threshold (out of 255). The CSV always carries both.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from scenegraph.figures.diff_writer import MANIFEST, load_png
from scenegraph.tools.demo_motionplanning_reward import (
    CURVE_WIDTH, FONT_STACK, GRID, INK, LABEL_SIZE, RULE_WIDTH, SUCCESS_COLOUR,
    TICK_SIZE, TITLE_SIZE, percent_ticks,
)

ROLES = ("head", "wrist")
ROLE_LABELS = {"head": "Head camera", "wrist": "Wrist camera"}
# The first two of draw/plot_success_curves' validated set.
ROLE_COLOURS = {"head": "#1F77B4", "wrist": "#C2700A"}
METRIC_LABELS = {"mad": r"Mean $|\Delta I|$", "changed": "Changed pixels"}
DEFAULT_THRESHOLD = 10
CSV_FIELDS = ("step", "head_mad", "wrist_mad", "head_changed", "wrist_changed",
              "reward", "success")

# Small-multiples sizes, from draw/plot_success_curves.
PANEL_WIDTH = 2.15
PANEL_HEIGHT = 2.2
PANEL_LINEWIDTH = 2.0
PANEL_TITLE_SIZE = 11.5
PANEL_TICK_SIZE = 9.5
PANEL_LABEL_SIZE = 11.0
PANEL_LEGEND_SIZE = 9.5


def house_pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": FONT_STACK,
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
    })
    return plt


def load_manifest(episode: Path) -> Dict[str, Any]:
    path = Path(episode) / MANIFEST
    if not path.exists():
        raise SystemExit(f"no {MANIFEST} in {episode}")
    return json.loads(path.read_text(encoding="utf-8"))


def diff_metrics(diff: np.ndarray, threshold: int) -> Tuple[float, float]:
    """``(mean |d| / 255, fraction of pixels with max-channel |d| > threshold)``."""
    arr = np.asarray(diff)
    return (float(arr.mean()) / 255.0,
            float((arr.max(axis=-1) > int(threshold)).mean()))


def episode_rows(episode: Path, manifest: Dict[str, Any],
                 threshold: int = DEFAULT_THRESHOLD) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in manifest.get("steps", []):
        if not all(f"{role}_diff" in record for role in ROLES):
            continue
        row: Dict[str, Any] = {
            "step": int(record["step"]),
            "reward": record.get("reward"),
            "success": int(bool(record.get("success"))),
        }
        for role in ROLES:
            mad, changed = diff_metrics(
                load_png(Path(episode) / record[f"{role}_diff"]), threshold)
            row[f"{role}_mad"], row[f"{role}_changed"] = mad, changed
        rows.append(row)
    if not rows:
        raise SystemExit(f"{episode} has no head and wrist differences")
    return rows


def first_success_step(manifest: Dict[str, Any]) -> Optional[int]:
    value = (manifest.get("attempt") or {}).get("first_success_step")
    return None if value is None else int(value)


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_FIELDS})
    return path


def _style_axes(ax, *, labelsize: float, length: float, width: float,
                pad: float) -> None:
    ax.grid(True, which="major", color=GRID, linewidth=0.6, linestyle="-",
            zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(labelsize=labelsize, labelcolor=INK, color=GRID,
                   length=length, width=width, pad=pad)


def _save(fig, path: Path, dpi: int) -> List[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for target in (path.with_suffix(".png"), path.with_suffix(".pdf")):
        fig.savefig(str(target), dpi=dpi, bbox_inches="tight", pad_inches=0.02,
                    facecolor="white")
        written.append(target)
    return written


def draw_episode(rows: Sequence[Dict[str, Any]], path: Path, *, title: str,
                 first_success: Optional[int], metric: str = "mad",
                 dpi: int = 300) -> List[Path]:
    """One episode, sized and styled like ``draw_reward_figure``."""
    plt = house_pyplot()
    steps = [row["step"] for row in rows]
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    peak = 0.0
    for role in ROLES:
        values = [row[f"{role}_{metric}"] for row in rows]
        peak = max(peak, max(values))
        ax.plot(steps, values, color=ROLE_COLOURS[role], linewidth=CURVE_WIDTH,
                solid_capstyle="round", solid_joinstyle="round",
                label=ROLE_LABELS[role], zorder=3)
    if first_success is not None and steps[0] <= first_success <= steps[-1]:
        ax.axvline(first_success, color=SUCCESS_COLOUR, linewidth=RULE_WIDTH,
                   linestyle=(0, (4, 3)), zorder=2)
    positions, labels = percent_ticks(steps)
    ax.set_xlim(steps[0], steps[-1])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    # Headroom so the legend row never sits on a curve.
    ax.set_ylim(0.0, max(peak, 1e-9) * 1.3)
    _style_axes(ax, labelsize=TICK_SIZE, length=4, width=0.9, pad=3)
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=LABEL_SIZE, color=INK,
                  labelpad=5)
    ax.set_xlabel("Steps", fontsize=LABEL_SIZE, color=INK, labelpad=4)
    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, color=INK, pad=8)
    legend = ax.legend(loc="upper right", ncol=2, frameon=True,
                       facecolor="white", edgecolor="none", framealpha=1.0,
                       fontsize=TICK_SIZE, handlelength=1.6, columnspacing=1.2,
                       borderpad=0.2, borderaxespad=0.1)
    legend.set_zorder(4)
    for text in legend.get_texts():
        text.set_color(INK)
    written = _save(fig, path, dpi)
    plt.close(fig)
    return written


def shared_tops(peaks: Sequence[float], cols: int, sharey: str) -> List[float]:
    """The y-axis top each panel needs so no panel in its shared group clips."""
    if sharey == "all":
        return [max(peaks)] * len(peaks)
    if sharey == "row":
        return [max(peaks[(i // cols) * cols:(i // cols + 1) * cols])
                for i in range(len(peaks))]
    return list(peaks)


def draw_panels(episodes: Sequence[Tuple[str, Sequence[Dict[str, Any]], Optional[int]]],
                path: Path, *, metric: str = "mad", cols: int = 3,
                sharey: str = "row", dpi: int = 300) -> List[Path]:
    """Small multiples: one panel per episode, head and wrist in each."""
    from matplotlib.lines import Line2D

    plt = house_pyplot()
    n = len(episodes)
    cols = max(1, min(int(cols) if cols > 0 else n, n))
    rows_n = int(np.ceil(n / cols))
    share = {"all": True, "row": "row", "none": False}[sharey]
    fig, axes = plt.subplots(
        rows_n, cols, squeeze=False, sharey=share,
        figsize=(PANEL_WIDTH * cols + 0.55, PANEL_HEIGHT * rows_n + 0.2),
    )
    peaks = [max(row[f"{role}_{metric}"] for row in rows for role in ROLES)
             for _, rows, _ in episodes]
    tops = shared_tops(peaks, cols, sharey)
    for index, ax in enumerate(axes.flat):
        if index >= n:
            ax.set_visible(False)
            continue
        title, rows, first_success = episodes[index]
        steps = [row["step"] for row in rows]
        for role in ROLES:
            ax.plot(steps, [row[f"{role}_{metric}"] for row in rows],
                    color=ROLE_COLOURS[role], linewidth=PANEL_LINEWIDTH,
                    solid_capstyle="round", solid_joinstyle="round", zorder=3)
        if first_success is not None and steps[0] <= first_success <= steps[-1]:
            ax.axvline(first_success, color=SUCCESS_COLOUR,
                       linewidth=PANEL_LINEWIDTH * RULE_WIDTH / CURVE_WIDTH,
                       linestyle=(0, (4, 3)), zorder=2)
        positions, labels = percent_ticks(steps, count=5)
        ax.set_xlim(steps[0], steps[-1])
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_title(title, fontsize=PANEL_TITLE_SIZE, color=INK, pad=5)
        _style_axes(ax, labelsize=PANEL_TICK_SIZE, length=3, width=0.7, pad=2)
        ax.tick_params(labelleft=True)
        ax.set_ylim(0.0, max(tops[index], 1e-9) * 1.05)
        if index % cols == 0:
            ax.set_ylabel(METRIC_LABELS[metric], fontsize=PANEL_LABEL_SIZE,
                          color=INK, labelpad=4)
    handles = [Line2D([], [], color=ROLE_COLOURS[role], linewidth=PANEL_LINEWIDTH,
                      label=ROLE_LABELS[role]) for role in ROLES]
    legend_band = 0.10 / rows_n
    fig.tight_layout(rect=(0, legend_band, 1, 1), w_pad=1.1, h_pad=1.2)
    bottom = min(ax.get_position().y0 for ax in axes.flat if ax.get_visible())
    scale = PANEL_LABEL_SIZE / PANEL_LEGEND_SIZE
    legend = fig.legend(
        handles=handles, labels=[h.get_label() for h in handles],
        loc="upper center", ncol=len(handles), frameon=False,
        fontsize=PANEL_LEGEND_SIZE, bbox_to_anchor=(0.5, bottom - legend_band),
        columnspacing=2.2 * scale, handlelength=1.6 * scale,
        handletextpad=0.5 * scale, borderpad=0.0, borderaxespad=0.0,
    )
    for text in legend.get_texts():
        text.set_color(INK)
    written = _save(fig, path, dpi)
    plt.close(fig)
    return written


def run(args) -> int:
    out = Path(args.out)
    panels = []
    for episode in (Path(e) for e in args.episodes):
        manifest = load_manifest(episode)
        name = str(manifest.get("name") or episode.name)
        title = str(manifest.get("title") or name).replace("_", " ")
        rows = episode_rows(episode, manifest, args.threshold)
        success = first_success_step(manifest)
        csv_path = write_csv(rows, out / f"{name}.csv")
        figures = draw_episode(rows, out / name, title=title,
                               first_success=success, metric=args.metric,
                               dpi=args.dpi)
        panels.append((title, rows, success))
        means = {role: float(np.mean([r[f"{role}_{args.metric}"] for r in rows]))
                 for role in ROLES}
        print(f"{name}: {len(rows)} differences, episode mean {args.metric} "
              + ", ".join(f"{role}={value:.4f}" for role, value in means.items()),
              flush=True)
        for target in (csv_path, *figures):
            print(f"  wrote {target}", flush=True)
    if len(panels) > 1:
        for target in draw_panels(panels, out / "pixel_diff_panels",
                                  metric=args.metric, cols=args.cols,
                                  sharey=args.sharey, dpi=args.dpi):
            print(f"wrote {target}", flush=True)
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Plot the per-step head and wrist pixel difference of "
                    "exported paper-frame episodes")
    p.add_argument("episodes", nargs="+",
                   help="episode directories holding episode.json and diff/")
    p.add_argument("--out", default="data/paper_figures/pixel_diff")
    p.add_argument("--metric", choices=sorted(METRIC_LABELS), default="mad")
    p.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                   help="per-channel level (0-255) a pixel must exceed to count "
                        "as changed")
    p.add_argument("--cols", type=int, default=3,
                   help="panels per row in the multi-episode figure; 0 is one row")
    p.add_argument("--sharey", choices=("all", "row", "none"), default="row")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args(argv)


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
