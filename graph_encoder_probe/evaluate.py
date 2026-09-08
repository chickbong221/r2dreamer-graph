"""Measuring the pooled token: distances, and the report.

Two numbers per pair. The RMS distance says how far apart the vectors are; the
cosine says whether they point the same way. Neither answers the question alone
-- two tokens can sit at cosine 0.9999 and still differ in length by a factor
the RSSM would see -- so both are always reported.

There is no threshold and no yes/no verdict. The distance is the result, and a
flag derived from it could only ever hide how large it was.

The unchanged control pairs carry the scale: they are the same graph twice, so
whatever distance they show is what "no change at all" costs in this precision
on this hardware. Every other distance is read against that, by eye, from the
same plot -- there is no threshold and no yes/no verdict, because the raw
distance is the result.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

from . import GROUP_LABELS, GROUP_SHORT
from .dataset import GraphFrames
from .pairs import CONTROL_GROUP, PairSet

PROBE_ROWS = "probe_rows.csv"
PROGRESS_ROWS = "progress.csv"


@torch.no_grad()
def encode(model, frames: GraphFrames, indices: Sequence[int], *, device=None, batch_size: int = 64) -> torch.Tensor:
    """Pooled tokens for ``indices``, in inference mode.

    ``eval()`` and ``no_grad`` both, and the training mode is restored by the
    caller through ``probe``: a measurement that left the model in eval, or that
    built a graph the next backward pass could reach, would be changing the run
    it is supposed to be observing.
    """
    idx = np.asarray(indices, dtype=np.int64)
    out = []
    for start in range(0, idx.size, int(batch_size)):
        batch = frames.torch_batch(idx[start:start + int(batch_size)], device)
        out.append(model.token(batch).detach().float().cpu())
    if not out:
        return torch.zeros(0, model.token_dim)
    return torch.cat(out, 0)


def distances(z_a: torch.Tensor, z_b: torch.Tensor, *, eps: float = 1e-12) -> dict[str, np.ndarray]:
    """RMS distance, cosine similarity and both norms, per row."""
    dim = int(z_a.shape[-1])
    rms = torch.sqrt(((z_a - z_b) ** 2).sum(-1) / dim)
    norm_a, norm_b = z_a.norm(dim=-1), z_b.norm(dim=-1)
    cosine = (z_a * z_b).sum(-1) / (norm_a * norm_b + eps)
    return {
        "rms": rms.numpy(),
        "cosine": cosine.numpy(),
        "norm_a": norm_a.numpy(),
        "norm_b": norm_b.numpy(),
    }


@dataclass
class PairMeasurement:
    name: str
    group: str
    rms: float
    cosine: float
    norm_a: float
    norm_b: float
    zero_token: bool
    note: str = ""

    def cosine_text(self) -> str:
        # A near-zero vector has no direction, so its cosine is an artefact of
        # whatever noise is left in it rather than a comparison.
        return "zero-token" if self.zero_token else f"{self.cosine:.6f}"


@dataclass
class ProbeResult:
    update: int
    measurements: list[PairMeasurement]
    token_scale: float
    by_group: dict[str, dict[str, float]] = field(default_factory=dict)

    def group_names(self) -> list[str]:
        seen: list[str] = []
        for item in self.measurements:
            if item.group not in seen:
                seen.append(item.group)
        return seen

    def lookup(self) -> dict[str, PairMeasurement]:
        return {item.name: item for item in self.measurements}

    def summary_line(self, recon: Optional[float] = None) -> str:
        """The one console line per probe."""
        per_group = " ".join(
            f"{GROUP_SHORT.get(name, name[:5])} {self.by_group[name]['mean_rms']:.1e}"
            for name in self.group_names()
        )
        changed = [m for m in self.measurements if m.group != CONTROL_GROUP]
        mean = float(np.mean([m.rms for m in changed])) if changed else 0.0
        loss = "     n/a" if recon is None else f"{recon:8.4f}"
        return f"update {self.update:6d} | recon {loss} | {per_group} | dist {mean:.4e}"


def probe(
    model,
    pool: GraphFrames,
    pairset: PairSet,
    index_a: Sequence[int],
    index_b: Sequence[int],
    *,
    update: int,
    device=None,
    batch_size: int = 64,
    zero_token_eps: float = 1e-6,
) -> ProbeResult:
    """Measure every pair once. Weights are untouched by construction."""
    was_training = model.training
    model.eval()
    try:
        z_a = encode(model, pool, index_a, device=device, batch_size=batch_size)
        z_b = encode(model, pool, index_b, device=device, batch_size=batch_size)
    finally:
        model.train(was_training)

    stats = distances(z_a, z_b)
    # Logged beside the distances because it is what makes them comparable: the
    # same 1e-3 gap means one thing on a token of RMS 1 and another on RMS 100,
    # and this magnitude moves during training.
    token_scale = (
        float(torch.sqrt((torch.cat([z_a, z_b], 0) ** 2).mean())) if z_a.numel() else 0.0
    )

    measurements: list[PairMeasurement] = []
    for i, spec in enumerate(pairset.specs):
        zero = bool(
            stats["norm_a"][i] < zero_token_eps or stats["norm_b"][i] < zero_token_eps
        )
        measurements.append(
            PairMeasurement(
                name=spec.name,
                group=spec.group,
                rms=float(stats["rms"][i]),
                cosine=float(stats["cosine"][i]),
                norm_a=float(stats["norm_a"][i]),
                norm_b=float(stats["norm_b"][i]),
                zero_token=zero,
                note=spec.note,
            )
        )

    result = ProbeResult(int(update), measurements, token_scale)
    for name in result.group_names():
        rows = [m for m in measurements if m.group == name]
        result.by_group[name] = {
            "count": float(len(rows)),
            "mean_rms": float(np.mean([m.rms for m in rows])),
            "mean_cosine": float(np.mean([m.cosine for m in rows])),
        }
    return result


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_probe_rows(path: str, result: ProbeResult) -> None:
    """One row per pair per probe. Written, not printed: 133 rows every 250
    updates is a file, not a console."""
    new = not os.path.isfile(path)
    with open(path, "a", newline="") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(
                ["update", "pair", "group", "rms", "cosine", "norm_a", "norm_b",
                 "zero_token"]
            )
        for item in result.measurements:
            writer.writerow(
                [result.update, item.name, item.group, f"{item.rms:.10e}",
                 f"{item.cosine:.10f}", f"{item.norm_a:.10e}", f"{item.norm_b:.10e}",
                 int(item.zero_token)]
            )


def write_progress_rows(path: str, history: Sequence[Mapping]) -> None:
    if not history:
        return
    columns: list[str] = []
    for row in history:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def final_table(first: ProbeResult, last: ProbeResult, pairset: PairSet, *, per_group: int = 0) -> str:
    """The report's headline table: every pair's distance before and after.

    ``per_group`` truncates each group to its first N rows for the markdown
    body; the full set is always in ``probe_rows.csv``.
    """
    before, after = first.lookup(), last.lookup()
    lines = [
        "| Pair | Change | Distance before | Distance after | After / before | Final cosine |",
        "|---|---|---:|---:|---:|---:|",
    ]
    shown: dict[str, int] = {}
    for spec in pairset.specs:
        shown[spec.group] = shown.get(spec.group, 0) + 1
        if per_group and shown[spec.group] > per_group:
            continue
        start = before.get(spec.name)
        end = after.get(spec.name)
        if end is None:
            continue
        ratio = (
            "n/a" if start is None or start.rms <= 0 else f"{end.rms / start.rms:.2f}x"
        )
        lines.append(
            f"| {spec.name} | {spec.note} | "
            f"{'n/a' if start is None else f'{start.rms:.4e}'} | "
            f"{end.rms:.4e} | {ratio} | {end.cosine_text()} |"
        )
    return "\n".join(lines)


def group_table(first: ProbeResult, last: ProbeResult) -> str:
    """Per group, and what training did to the response.

    The last column is the point of measuring before the first update: a random
    encoder already separates these pairs, so the interesting number is whether
    reconstruction kept that separation, sharpened it, or wore it away.
    """
    lines = [
        "| Group | Pairs | Mean distance before | Mean distance after | After / before | "
        "Mean cosine after |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for name in last.group_names():
        end = last.by_group[name]
        start = first.by_group.get(name, {})
        # Named for the reader, with the identifier kept so a row still points
        # at its pairs (``control-01``) and its CSV column.
        shown = f"{GROUP_LABELS.get(name, name)} (`{name}`)"
        before = float(start.get("mean_rms", float("nan")))
        ratio = end["mean_rms"] / before if before > 0 else float("nan")
        if not math.isfinite(ratio):
            change = "n/a"
        elif ratio > 1.1:
            change = f"{ratio:.2f}x strengthened"
        elif ratio < 0.9:
            change = f"{ratio:.2f}x suppressed"
        else:
            change = f"{ratio:.2f}x preserved"
        lines.append(
            f"| {shown} | {int(end['count'])} | {before:.4e} | {end['mean_rms']:.4e} | "
            f"{change} | {end['mean_cosine']:.6f} |"
        )
    return "\n".join(lines)


def make_plots(history: Sequence[Mapping], out_dir: str, groups: Iterable[str]) -> list[str]:
    """Loss against updates, and latent distance against updates by edit type.

    Missing matplotlib is reported and skipped rather than raised: the numbers
    are already on disk, and a plotting import should not lose a training run.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                               # noqa: BLE001
        print(f"[plots] skipped ({type(exc).__name__}: {exc}); the CSVs hold the numbers")
        return []

    updates = [row["update"] for row in history]
    written: list[str] = []

    fig, ax = plt.subplots(figsize=(6, 4))
    monitor = [row.get("monitor_loss") for row in history]
    if any(v is not None for v in monitor):
        ax.plot(updates, monitor, marker="o", ms=3, label="fixed-subset reconstruction")
    train_loss = [row.get("train_loss") for row in history]
    if any(v is not None for v in train_loss):
        ax.plot(updates, train_loss, marker=".", ms=3, alpha=0.6, label="batch reconstruction")
    ax.set_xlabel("step")
    ax.set_ylabel("reconstruction loss")
    ax.set_title("Reconstruction loss")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "loss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name in groups:
        key = f"mean_rms/{name}"
        values = [row.get(key) for row in history]
        if not any(v is not None for v in values):
            continue
        ax.plot(updates, values, marker="o", ms=3, label=GROUP_LABELS.get(name, name))
    ax.set_yscale("log")
    ax.set_xlabel("step")
    # Each pair is two graphs encoded separately; this is how far apart the one
    # pooled token each of them produces ends up. The legend says what was
    # edited, so the axis does not have to.
    ax.set_ylabel("RMS Distance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "latent_distance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)
    return written


def zero_token_warnings(result: ProbeResult) -> list[str]:
    return [
        f"{item.name}: |z_A|={item.norm_a:.3e} |z_B|={item.norm_b:.3e}"
        for item in result.measurements
        if item.zero_token
    ]


def nan_guard(result: ProbeResult) -> list[str]:
    """Non-finite distances mean the measurement is meaningless, not small."""
    return [
        f"{item.name}: rms={item.rms} cosine={item.cosine}"
        for item in result.measurements
        if not math.isfinite(item.rms) or not math.isfinite(item.cosine)
    ]
