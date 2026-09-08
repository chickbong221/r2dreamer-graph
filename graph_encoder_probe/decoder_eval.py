"""Did the decoder say the right label?

The latent probe answers whether two graphs land on different tokens. This
answers the other half: what comes back out of one token. Every head this
reports is discrete -- entity id, which row is the target, the absolute label
sigma, the temporal change delta -- so the honest readout is a comparison of
argmax against ground truth, not a loss value.

Three things get written: aggregate agreement per head, a confusion matrix per
head (which label gets mistaken for which), and a literal side-by-side of a few
frames' nodes and facts with a tick or a cross beside each.

Every argmax comes from :meth:`GraphProbe.predict`, which taps the decoder's own
output projections while it runs, so what is scored here is what the loss saw.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

from .dataset import GraphFrames

DECODER_ROWS = "decoder_eval.csv"
DECODER_ITEMS = "decoder_predictions.csv"

# The heads whose output is a label rather than a number.
DISCRETE_HEADS = ("node_ent", "node_target", "relabs", "reltemp")


@dataclass
class Agreement:
    """How often one head's argmax was the true label."""

    correct: int = 0
    total: int = 0

    def add(self, correct, total) -> None:
        self.correct += int(correct)
        self.total += int(total)

    @property
    def accuracy(self) -> float:
        return float(self.correct) / float(self.total) if self.total else float("nan")

    def text(self) -> str:
        return f"{self.accuracy:.4f} ({self.correct}/{self.total})"


@dataclass
class DecoderReport:
    update: int
    split: str
    frames: int
    edges: int
    heads: dict[str, Agreement]
    per_relation: dict[str, Agreement]
    bbox_mae: float
    entity_confusion: np.ndarray
    absolute_confusion: np.ndarray
    temporal_confusion: np.ndarray
    losses: dict[str, float] = field(default_factory=dict)

    def summary_line(self) -> str:
        """The one extra console line per probe."""
        parts = " ".join(
            f"{name.replace('node_', '')} {self.heads[name].accuracy:.3f}"
            for name in DISCRETE_HEADS
            if self.heads[name].total
        )
        skipped = [name for name in DISCRETE_HEADS if not self.heads[name].total]
        tail = f" (no {', '.join(skipped)})" if skipped else ""
        return (
            f"       decoder ({self.split}, {self.frames} frames) | {parts} | "
            f"box mae {self.bbox_mae:.4f}{tail}"
        )

    def row(self) -> dict:
        row = {"update": self.update, "split": self.split, "frames": self.frames,
               "edges": self.edges, "bbox_mae": self.bbox_mae}
        for name in DISCRETE_HEADS:
            row[f"acc/{name}"] = self.heads[name].accuracy
            row[f"n/{name}"] = self.heads[name].total
        for name, score in sorted(self.per_relation.items()):
            row[f"acc/relabs/{name}"] = score.accuracy
        row |= {f"loss/{key}": value for key, value in sorted(self.losses.items())}
        return row


def label_maps(dataset_meta: Mapping) -> dict[str, dict[int, str]]:
    """``id -> token`` for every vocabulary, so a confusion axis reads as words.

    Entity tokens come from the cache because they are the collected task's
    whitelist; the other three are the repository's shared tables.
    """
    from .pairs import label_names

    names = label_names()
    entity = dataset_meta.get("entity_tokens") or {}
    names["entity"] = {int(index): str(token) for token, index in entity.items()}
    return names


@torch.no_grad()
def evaluate_decoder(
    model,
    pool: GraphFrames,
    indices: Sequence[int],
    *,
    split: str,
    update: int,
    device=None,
    batch_size: int = 128,
) -> DecoderReport:
    """Agreement and confusions over ``indices``, in inference mode."""
    idx = np.asarray(indices, dtype=np.int64)
    config = model.config
    heads = {name: Agreement() for name in DISCRETE_HEADS}
    per_relation: dict[str, Agreement] = {}
    entity_confusion = np.zeros((config.entity_vocab, config.entity_vocab), np.int64)
    absolute_confusion = np.zeros((config.n_abs, config.n_abs), np.int64)
    temporal_confusion = np.zeros((config.n_temp, config.n_temp), np.int64)
    bbox_error, bbox_count = 0.0, 0
    losses: dict[str, float] = {}
    frames, edges = 0, 0

    for start in range(0, idx.size, int(batch_size)):
        chunk = idx[start:start + int(batch_size)]
        out = model.predict(pool.torch_batch(chunk, device))
        weight = float(chunk.size)
        for key, value in out.losses.items():
            losses[key] = losses.get(key, 0.0) + float(value) * weight
        frames += int(chunk.size)

        valid = out.node_valid
        heads["node_ent"].add(
            (out.node_ent_pred.eq(out.node_ent_true) & valid).sum(), valid.sum()
        )
        _tally(entity_confusion, out.node_ent_true[valid], out.node_ent_pred[valid])

        # Only frames that carry a target can be scored on finding it. Reported
        # as a count of zero rather than an accuracy of zero when the task never
        # names one -- those are different facts.
        has = out.has_target
        heads["node_target"].add(
            (out.target_row_pred.eq(out.target_row_true) & has).sum(), has.sum()
        )

        edges += int(out.edge_rel.numel())
        # Every packed fact carries a legal sigma, so all of them are scored.
        heads["relabs"].add(out.abs_pred.eq(out.abs_true).sum(), out.abs_true.numel())
        _tally(absolute_confusion, out.abs_true, out.abs_pred)
        mask = out.temp_mask
        heads["reltemp"].add(
            (out.temp_pred.eq(out.temp_true) & mask).sum(), mask.sum()
        )
        _tally(temporal_confusion, out.temp_true[mask], out.temp_pred[mask])

        relation = out.edge_rel.cpu().numpy()
        correct = out.abs_pred.eq(out.abs_true).cpu().numpy()
        for rel in np.unique(relation):
            rows = relation == rel
            per_relation.setdefault(int(rel), Agreement()).add(
                correct[rows].sum(), rows.sum()
            )

        error = (out.bbox_pred - out.bbox_true).abs().mean(-1)
        bbox_error += float((error * out.bbox_mask).sum())
        bbox_count += int(out.bbox_mask.sum())

    return DecoderReport(
        update=int(update),
        split=str(split),
        frames=frames,
        edges=edges,
        heads=heads,
        per_relation={str(key): value for key, value in per_relation.items()},
        bbox_mae=bbox_error / max(bbox_count, 1),
        entity_confusion=entity_confusion,
        absolute_confusion=absolute_confusion,
        temporal_confusion=temporal_confusion,
        losses={key: value / max(frames, 1) for key, value in losses.items()},
    )


def _tally(matrix: np.ndarray, true: torch.Tensor, pred: torch.Tensor) -> None:
    if true.numel() == 0:
        return
    np.add.at(matrix, (true.cpu().numpy(), pred.cpu().numpy()), 1)


def name_relations(report: DecoderReport, names: Mapping[str, Mapping[int, str]]) -> DecoderReport:
    """Swap relation ids for their tokens once the vocabulary is known."""
    relation = names["relation"]
    report.per_relation = {
        str(relation.get(int(key), key)): value for key, value in report.per_relation.items()
    }
    return report


# --------------------------------------------------------------------------- #
# Per-item dump
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_predictions(
    model,
    pool: GraphFrames,
    indices: Sequence[int],
    names: Mapping[str, Mapping[int, str]],
    *,
    device=None,
    limit: int = 64,
) -> list[dict]:
    """One row per node and per fact: what it was, what the decoder said.

    This is the readout in its rawest form -- no aggregation, no accuracy, just
    the two labels side by side and whether they agree.
    """
    idx = np.asarray(indices, dtype=np.int64)[: int(limit)]
    if idx.size == 0:
        return []
    out = model.predict(pool.torch_batch(idx, device))
    entity, relation = names["entity"], names["relation"]
    absolute, temporal = names["absolute"], names["temporal"]
    rows: list[dict] = []

    valid = out.node_valid.cpu().numpy()
    ent_true = out.node_ent_true.cpu().numpy()
    ent_pred = out.node_ent_pred.cpu().numpy()
    for frame in range(valid.shape[0]):
        for node in np.flatnonzero(valid[frame]):
            true, pred = int(ent_true[frame, node]), int(ent_pred[frame, node])
            rows.append({
                "pool_index": int(idx[frame]), "kind": "node", "slot": int(node),
                "head": "entity", "subject": f"row {node}",
                "true": entity.get(true, true), "pred": entity.get(pred, pred),
                "match": int(true == pred),
            })

    graph = out.edge_graph.cpu().numpy()
    rel = out.edge_rel.cpu().numpy()
    src, dst = out.edge_src.cpu().numpy(), out.edge_dst.cpu().numpy()
    abs_true, abs_pred = out.abs_true.cpu().numpy(), out.abs_pred.cpu().numpy()
    tmp_true, tmp_pred = out.temp_true.cpu().numpy(), out.temp_pred.cpu().numpy()
    for edge in range(rel.size):
        frame = int(graph[edge])
        subject = (
            f"{entity.get(int(ent_true[frame, src[edge]]), '?')}"
            f" -> {entity.get(int(ent_true[frame, dst[edge]]), '?')}"
        )
        rows.append({
            "pool_index": int(idx[frame]), "kind": "edge", "slot": edge,
            "head": relation.get(int(rel[edge]), int(rel[edge])), "subject": subject,
            "true": absolute.get(int(abs_true[edge]), int(abs_true[edge])),
            "pred": absolute.get(int(abs_pred[edge]), int(abs_pred[edge])),
            "match": int(abs_true[edge] == abs_pred[edge]),
        })
        if int(tmp_true[edge]):
            rows.append({
                "pool_index": int(idx[frame]), "kind": "edge", "slot": edge,
                "head": f"{relation.get(int(rel[edge]), '?')} (delta)", "subject": subject,
                "true": temporal.get(int(tmp_true[edge]), int(tmp_true[edge])),
                "pred": temporal.get(int(tmp_pred[edge]), int(tmp_pred[edge])),
                "match": int(tmp_true[edge] == tmp_pred[edge]),
            })
    return rows


def write_item_rows(path: str, rows: Sequence[Mapping]) -> Optional[str]:
    if not rows:
        return None
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_decoder_rows(path: str, report: DecoderReport) -> None:
    """Append one row per probe.

    The header is fixed by the first row. A relation that appears in no later
    batch would otherwise abort the run mid-training over a missing column, so
    later rows fill the gap rather than raise -- the evaluation set is fixed, so
    that gap should never open.
    """
    row = report.row()
    if os.path.isfile(path):
        with open(path, newline="") as handle:
            header = next(csv.reader(handle), list(row))
    else:
        header = list(row)
        with open(path, "w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=header).writeheader()
    with open(path, "a", newline="") as handle:
        csv.DictWriter(
            handle, fieldnames=header, extrasaction="ignore", restval=""
        ).writerow(row)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:                               # noqa: BLE001
        print(f"[plots] skipped ({type(exc).__name__}: {exc}); the CSVs hold the numbers")
        return None


def plot_confusions(
    report: DecoderReport, names: Mapping[str, Mapping[int, str]], out_dir: str
) -> Optional[str]:
    """One panel per discrete head: true label down, predicted across.

    Padding is dropped from both axes -- index zero is never a real label, so a
    row and column of zeros would only shrink everything else. Cells are counts;
    the colour is row-normalised so a rare label's mistakes stay visible next to
    a common one's.
    """
    plt = _pyplot()
    if plt is None:
        return None
    panels = [
        ("entity", report.entity_confusion, names["entity"]),
        ("absolute (sigma)", report.absolute_confusion, names["absolute"]),
        ("temporal (delta)", report.temporal_confusion, names["temporal"]),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.5))
    for ax, (title, matrix, vocab) in zip(np.atleast_1d(axes), panels):
        counts = matrix[1:, 1:]                            # drop the pad row/column
        labels = [str(vocab.get(i, i)) for i in range(1, matrix.shape[0])]
        if counts.sum() == 0:
            ax.set_title(f"{title}\n(never observed)")
            ax.axis("off")
            continue
        shown = counts / np.clip(counts.sum(1, keepdims=True), 1, None)
        ax.imshow(shown, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        correct = int(np.trace(counts))
        ax.set_title(f"{title}\n{correct}/{int(counts.sum())} exact")
        if len(labels) <= 10:
            for i in range(counts.shape[0]):
                for j in range(counts.shape[1]):
                    if counts[i, j]:
                        ax.text(j, i, int(counts[i, j]), ha="center", va="center",
                                fontsize=7, color="white" if shown[i, j] > 0.5 else "black")
    fig.suptitle(
        f"Decoder label agreement at update {report.update} "
        f"({report.split}, {report.frames} frames)"
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "decoder_confusion.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_accuracy(history: Sequence[Mapping], out_dir: str) -> Optional[str]:
    """Label agreement against updates, one line per head."""
    plt = _pyplot()
    if plt is None:
        return None
    updates = [row["update"] for row in history]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    for name in DISCRETE_HEADS:
        values = [row.get(f"acc/{name}") for row in history]
        if not any(v is not None and np.isfinite(v) for v in values):
            continue
        left.plot(updates, values, marker="o", ms=3, label=name)
    left.set_ylim(0.0, 1.02)
    left.set_xlabel("optimizer updates")
    left.set_ylabel("fraction of labels recovered exactly")
    left.set_title("Discrete heads")
    left.legend(fontsize=8)

    mae = [row.get("bbox_mae") for row in history]
    if any(v is not None for v in mae):
        right.plot(updates, mae, marker="o", ms=3, color="tab:red")
    right.set_xlabel("optimizer updates")
    right.set_ylabel("mean absolute error (normalised box units)")
    right.set_title("Box regression")
    fig.tight_layout()
    path = os.path.join(out_dir, "decoder_accuracy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_examples(rows: Sequence[Mapping], out_dir: str, *, frames: int = 3) -> Optional[str]:
    """A literal side-by-side: every node and fact of a few frames, true vs said.

    Aggregates hide the shape of a failure. Seeing that every ``contact`` came
    back right and every ``planar-distance`` came back one bin low is a different
    diagnosis from the same accuracy spread evenly.
    """
    plt = _pyplot()
    if plt is None or not rows:
        return None
    order: list = []
    for row in rows:
        if row["pool_index"] not in order:
            order.append(row["pool_index"])
    order = order[: int(frames)]
    if not order:
        return None

    groups = [[row for row in rows if row["pool_index"] == index] for index in order]
    height = max(len(group) for group in groups) * 0.22 + 1.8
    fig, axes = plt.subplots(1, len(groups), figsize=(5.6 * len(groups), height))
    columns = ((0.0, "head"), (0.30, "subject"), (0.60, "true"), (0.80, "decoded"))
    for ax, index, group in zip(np.atleast_1d(axes), order, groups):
        ax.axis("off")
        # Padded off the header row: the axes have no frame, so a default title
        # sits directly on top of the column names.
        ax.set_title(f"pool frame {index}", fontsize=10, pad=18)
        agreed = sum(int(item["match"]) for item in group)
        ax.text(0.0, 1.10, f"{agreed}/{len(group)} labels recovered", fontsize=8,
                style="italic", transform=ax.transAxes)
        for x, header in columns:
            ax.text(x, 1.0, header, fontsize=7, weight="bold", family="monospace",
                    transform=ax.transAxes)
        for line, item in enumerate(group):
            y = 0.94 - (line + 1) * 0.94 / (len(group) + 1)
            ok = bool(item["match"])
            colour = "tab:green" if ok else "tab:red"
            ax.text(0.0, y, str(item["head"])[:26], fontsize=7, family="monospace",
                    transform=ax.transAxes)
            ax.text(0.30, y, str(item["subject"])[:26], fontsize=7, family="monospace",
                    transform=ax.transAxes)
            ax.text(0.60, y, str(item["true"])[:16], fontsize=7, family="monospace",
                    transform=ax.transAxes)
            ax.text(0.80, y, str(item["pred"])[:16], fontsize=7, family="monospace",
                    color=colour, transform=ax.transAxes)
            ax.text(0.97, y, "ok" if ok else "x", fontsize=7, family="monospace",
                    color=colour, transform=ax.transAxes)
    fig.suptitle("Decoder output against the true label, per node and per fact")
    fig.tight_layout()
    path = os.path.join(out_dir, "decoder_examples.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def report_table(report: DecoderReport) -> str:
    """Markdown for the run report."""
    lines = [
        f"Evaluated on **{report.frames} {report.split} frames** "
        f"({report.edges} facts) at update {report.update}.",
        "",
        "| Head | Labels recovered exactly |",
        "|---|---:|",
    ]
    for name in DISCRETE_HEADS:
        score = report.heads[name]
        lines.append(
            f"| {name} | {'not scored (no labelled item)' if not score.total else score.text()} |"
        )
    lines += [f"| bbox (regression, MAE) | {report.bbox_mae:.4f} |", ""]
    if report.per_relation:
        lines += [
            "Absolute label, per relation:",
            "",
            "| Relation | Labels recovered exactly |",
            "|---|---:|",
        ]
        for name, score in sorted(report.per_relation.items()):
            lines.append(f"| {name} | {score.text()} |")
        lines.append("")
    return "\n".join(lines)
