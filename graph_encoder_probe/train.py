"""Reconstruction training, with the probe read off at fixed intervals.

There is no train/validation split. Both members of every pair are in the
training pool and the probe set is a measurement set drawn from it: the question
is whether the encoder separates graphs it has been trained on, not whether it
generalises to unseen ones.

Nothing here adds a term that would teach the separation being measured. The
loss is the decoder's own four reconstruction terms and nothing else -- a
contrastive or latent-spread objective would answer the question by assumption.

The measurement at update 0 matters as much as the last one. A randomly
initialised encoder already maps different graphs to different tokens; what the
run shows is whether reconstruction training keeps, sharpens or erodes that.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

from . import EDIT_GROUPS
from .dataset import GraphDataset, GraphFrames
from .evaluate import (
    PROBE_ROWS,
    PROGRESS_ROWS,
    ProbeResult,
    make_plots,
    nan_guard,
    probe,
    write_probe_rows,
    write_progress_rows,
    zero_token_warnings,
)
from .model import LOSS_KEYS, GraphProbe, build_model, save_checkpoint
from .pairs import CONTROL_GROUP, PairSet

INIT_CHECKPOINT = "checkpoint_init.pt"
FINAL_CHECKPOINT = "checkpoint_final.pt"
HISTORY_JSON = "history.json"


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(spec: str) -> torch.device:
    if str(spec) not in ("auto", ""):
        return torch.device(str(spec))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_pool(dataset: GraphDataset, pairset: PairSet) -> tuple[GraphFrames, np.ndarray, np.ndarray]:
    """One table holding the collected frames and every edited copy.

    The edited members are appended rather than kept apart so training samples
    them like any other frame and the probe encodes both members through exactly
    the same indexing path.
    """
    pool = dataset.frames.concat(pairset.edited)
    index_a = pairset.sources
    index_b = np.arange(len(pairset), dtype=np.int64) + len(dataset)
    return pool, index_a, index_b


def monitor_indices(
    pool_size: int, pinned: Sequence[int], count: int, rng: np.random.Generator
) -> np.ndarray:
    """A fixed subset the convergence number is always read on.

    It contains every probe frame plus a random remainder. Comparable between
    checkpoints is the whole point: a fresh sample each time would move the
    number for reasons that have nothing to do with the weights.
    """
    pinned = np.unique(np.asarray(pinned, dtype=np.int64))
    remaining = int(max(0, int(count) - pinned.size))
    if remaining:
        rest = np.setdiff1d(np.arange(pool_size, dtype=np.int64), pinned, assume_unique=False)
        if rest.size:
            extra = rng.choice(rest, size=min(remaining, rest.size), replace=False)
            return np.sort(np.concatenate([pinned, extra]))
    return np.sort(pinned)


@torch.no_grad()
def evaluate_loss(model: GraphProbe, pool: GraphFrames, indices, *, device, batch_size: int) -> dict[str, float]:
    """Mean reconstruction loss over a fixed set of frames.

    Batches are averaged with their sizes as weights so the number does not
    depend on how the subset happens to divide.
    """
    was_training = model.training
    model.eval()
    idx = np.asarray(indices, dtype=np.int64)
    totals: dict[str, float] = {"total": 0.0} | {key: 0.0 for key in LOSS_KEYS}
    seen = 0
    try:
        for start in range(0, idx.size, int(batch_size)):
            chunk = idx[start:start + int(batch_size)]
            out = model(pool.torch_batch(chunk, device))
            totals["total"] += float(out.total) * chunk.size
            for key in LOSS_KEYS:
                totals[key] += float(out.losses[key]) * chunk.size
            seen += int(chunk.size)
    finally:
        model.train(was_training)
    return {key: value / max(seen, 1) for key, value in totals.items()}


@dataclass
class TrainResult:
    run_dir: str
    updates: int
    stop_reason: str
    history: list[dict]
    first: ProbeResult
    last: ProbeResult
    monitor_size: int
    pool_size: int
    device: str
    warnings: list[str] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        """Only a plateau is convergence. Exhausting the budget is not."""
        return self.stop_reason == "plateau"


def train(
    cfg: Mapping,
    dataset: GraphDataset,
    pairset: PairSet,
    run_dir: str,
    *,
    model: Optional[GraphProbe] = None,
) -> TrainResult:
    train_cfg = dict(cfg["train"])
    probe_cfg = dict(cfg["probe"])
    os.makedirs(run_dir, exist_ok=True)

    set_seed(int(train_cfg.get("seed", cfg.get("seed", 0))))
    device = resolve_device(train_cfg.get("device", "auto"))
    torch.set_default_dtype(torch.float32)

    pool, index_a, index_b = build_pool(dataset, pairset)
    if model is None:
        model = build_model(
            cfg["model"], dataset.meta, loss_scales=train_cfg.get("loss_scales"), device=device
        )
    else:
        model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["lr"]))
    rng = np.random.default_rng(int(train_cfg.get("seed", 0)))

    pinned = np.concatenate([index_a, index_b])
    monitor = monitor_indices(len(pool), pinned, int(train_cfg["monitor_frames"]), rng)
    batch_size = min(int(train_cfg["batch_size"]), len(pool))
    probe_every = int(train_cfg["probe_every"])
    max_updates = int(train_cfg["max_updates"])
    min_updates = int(train_cfg["min_updates"])
    plateau_checks = int(train_cfg["plateau_checks"])
    rel_improve = float(train_cfg["plateau_rel_improve"])

    probe_kwargs = dict(
        device=device,
        batch_size=int(probe_cfg.get("batch_size", 64)),
        repeats=int(probe_cfg.get("repeats", 3)),
        control_factor=float(probe_cfg.get("tolerance_control_factor", 5.0)),
        rel_floor=float(probe_cfg.get("tolerance_rel_floor", 1e-3)),
        zero_token_eps=float(probe_cfg.get("zero_token_eps", 1e-6)),
    )

    print(
        f"[train] device={device} pool={len(pool)} frames "
        f"({len(dataset)} collected + {len(pairset)} edited), "
        f"batch={batch_size}, monitor={monitor.size} frames, "
        f"probe every {probe_every} updates, budget {max_updates}"
    )

    history: list[dict] = []
    warnings: list[str] = []
    if not bool(pool.fields["graph_node_target"].any()):
        # Normal ManiSkill names no active subtask object -- the flag is an
        # MS-HAB concept -- so L_nodetgt has only negatives to learn from. The
        # term is kept because the plan's objective names it and the decoder's
        # own masks are left alone, but its number means "nothing is the
        # target here", not "the target was found".
        message = (
            "no frame in the pool carries a target flag; the nodetgt term is "
            "all-negative and its value is not a target-recovery score"
        )
        warnings.append(f"update 0: {message}")
        print(f"[train] note: {message}", flush=True)
    probe_path = os.path.join(run_dir, PROBE_ROWS)
    if os.path.isfile(probe_path):
        os.remove(probe_path)

    def take_probe(update: int, train_loss: Optional[float]) -> ProbeResult:
        result = probe(model, pool, pairset, index_a, index_b, update=update, **probe_kwargs)
        losses = evaluate_loss(
            model, pool, monitor, device=device, batch_size=int(train_cfg["batch_size"])
        )
        write_probe_rows(probe_path, result)
        row: dict = {
            "update": update,
            "train_loss": train_loss,
            "monitor_loss": losses["total"],
            "tolerance": result.tolerance.value,
            "token_scale": result.token_scale,
            "repeat_max": result.tolerance.repeat_max,
            "control_max": result.tolerance.control_max,
        }
        row |= {f"monitor/{key}": losses[key] for key in LOSS_KEYS}
        for name, stats in result.by_group.items():
            row[f"mean_rms/{name}"] = stats["mean_rms"]
            row[f"detected/{name}"] = int(stats["detected"])
        history.append(row)
        print(result.summary_line(losses["total"]), flush=True)
        for message in zero_token_warnings(result) + nan_guard(result):
            warnings.append(f"update {update}: {message}")
            print(f"  [flag] {message}", flush=True)
        return result

    # Before the first update: a randomly initialised encoder is the baseline
    # every later measurement is read against.
    first = take_probe(0, None)
    save_checkpoint(
        os.path.join(run_dir, INIT_CHECKPOINT), model, {"update": 0, "stage": "init"}
    )

    best = float("inf")
    stale = 0
    stop_reason = "budget"
    update = 0
    recent: list[float] = []
    started = time.time()
    last = first

    while update < max_updates:
        order = rng.permutation(len(pool))
        for start in range(0, order.size - batch_size + 1, batch_size):
            batch = pool.torch_batch(order[start:start + batch_size], device)
            out = model(batch)
            optimizer.zero_grad(set_to_none=True)
            out.total.backward()
            optimizer.step()
            update += 1
            recent.append(float(out.total.detach()))

            if update % probe_every == 0 or update >= max_updates:
                mean_recent = float(np.mean(recent[-probe_every:])) if recent else None
                last = take_probe(update, mean_recent)
                monitor_loss = history[-1]["monitor_loss"]
                if monitor_loss < best * (1.0 - rel_improve):
                    best, stale = monitor_loss, 0
                else:
                    stale += 1
                    if update >= min_updates and stale >= plateau_checks:
                        stop_reason = "plateau"
                if stop_reason == "plateau" or update >= max_updates:
                    break
        if stop_reason == "plateau" or update >= max_updates:
            break

    if last.update != update:
        last = take_probe(update, float(np.mean(recent[-probe_every:])) if recent else None)

    save_checkpoint(
        os.path.join(run_dir, FINAL_CHECKPOINT),
        model,
        {"update": update, "stage": "final", "stop_reason": stop_reason},
    )
    write_progress_rows(os.path.join(run_dir, PROGRESS_ROWS), history)
    with open(os.path.join(run_dir, HISTORY_JSON), "w") as handle:
        json.dump(history, handle, indent=2)
    make_plots(history, run_dir, list(EDIT_GROUPS) + [CONTROL_GROUP])

    elapsed = time.time() - started
    print(
        f"[train] stopped after {update} updates ({stop_reason}) in {elapsed / 60:.1f} min; "
        f"{'plateau reached' if stop_reason == 'plateau' else 'budget exhausted -- not convergence'}"
    )
    return TrainResult(
        run_dir=run_dir,
        updates=update,
        stop_reason=stop_reason,
        history=history,
        first=first,
        last=last,
        monitor_size=int(monitor.size),
        pool_size=int(len(pool)),
        device=str(device),
        warnings=warnings,
    )
