"""Checkpoints for offline runs.

The repository's ``checkpointing`` keeps one rolling best selected at online
evaluations -- right for an online run, wrong here, where training is measured
in updates, has no environment to evaluate in, and needs to resume. Instead:

* ``latest.pt`` -- rewritten every ``checkpoint_every`` updates, for resumption;
* ``step_XXXXXXXX.pt`` -- kept every ``snapshot_every`` updates;
* ``final.pt`` -- written once, at the end, and the default output of a run;
* ``<label>.pt`` -- only for a selection rule the run declares, such as
  ``best_diagnostic.pt`` for the lowest diagnostic world-model loss. The label
  and ``<label>.json`` say what it was selected on. Diagnostic episodes are
  training episodes, so such a checkpoint is not evidence of generalisation.
  No policy checkpoint is selected by action error.

Each file carries the model, optimiser and scheduler state, random number
generator states, the step, and the identity of everything the weights depend
on. Loading compares that identity and refuses on any mismatch. Writes go
through the repository's ``atomic_save``.
"""

from __future__ import annotations

import os
import random
import re
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from checkpointing import atomic_save

from ..common import IdentityError, read_json, require_identity, utc_now, write_json

CHECKPOINT_FORMAT = "real_robot/checkpoint-v2"
FIXED_KINDS = ("latest", "final")
SNAPSHOT = re.compile(r"^step_\d{8}$")


def rng_state() -> Dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def snapshot_kind(step: int) -> str:
    return f"step_{int(step):08d}"


class CheckpointManager:
    def __init__(self, run_dir: str, identity: Mapping[str, Any]):
        self.run_dir = run_dir
        self.identity = dict(identity)
        os.makedirs(run_dir, exist_ok=True)

    def path(self, kind: str) -> str:
        if not re.match(r"^[A-Za-z0-9_]+$", kind):
            raise ValueError(f"checkpoint kind {kind!r}")
        return os.path.join(self.run_dir, f"{kind}.pt")

    def save(self, kind: str, step: int, state: Mapping[str, Any], meta: Mapping[str, Any],
             metrics: Optional[Mapping[str, float]] = None, selection: Optional[Mapping[str, Any]] = None) -> str:
        payload = {
            "format": CHECKPOINT_FORMAT,
            "identity": self.identity,
            "meta": dict(meta),
            "state": dict(state),
            "step": int(step),
            "metrics": {k: float(v) for k, v in (metrics or {}).items()},
            "rng": rng_state(),
            "saved": utc_now(),
            "kind": kind,
            "selection": dict(selection) if selection is not None else None,
        }
        atomic_save(payload, self.path(kind))
        return self.path(kind)

    def snapshot(self, step: int, state: Mapping[str, Any], meta: Mapping[str, Any]) -> str:
        return self.save(snapshot_kind(step), step, state, meta)

    def resume(self) -> Optional[Dict[str, Any]]:
        path = self.path("latest")
        if not os.path.isfile(path):
            return None
        return load_checkpoint(path, self.identity)


class CheckpointSelection:
    """A checkpoint chosen by a declared rule, saved under a label that says so."""

    def __init__(self, manager: CheckpointManager, label: str, metric: str, mode: str, note: str):
        if label in FIXED_KINDS or SNAPSHOT.match(label) or label in ("best", ""):
            raise ValueError(f"a selected checkpoint needs a label that names its rule, not {label!r}")
        if not metric:
            raise ValueError("a selection metric must be declared before training starts")
        if mode not in ("min", "max"):
            raise ValueError("mode must be min or max")
        self.manager, self.label, self.metric, self.mode, self.note = manager, label, metric, mode, note
        record = os.path.join(manager.run_dir, f"{label}.json")
        self.value: Optional[float] = read_json(record)["value"] if os.path.isfile(record) else None

    def improved(self, metrics: Mapping[str, float]) -> bool:
        if self.metric not in metrics:
            raise KeyError(f"selection metric {self.metric!r} is not among the reported metrics {sorted(metrics)}")
        value = float(metrics[self.metric])
        if not np.isfinite(value):
            return False
        if self.value is None:
            return True
        return value < self.value if self.mode == "min" else value > self.value

    def update(self, step: int, metrics: Mapping[str, float], state_fn: Callable[[], Mapping[str, Any]],
               meta: Mapping[str, Any]) -> bool:
        if not self.improved(metrics):
            return False
        self.value = float(metrics[self.metric])
        rule = {"label": self.label, "metric": self.metric, "mode": self.mode, "value": self.value,
                "step": int(step), "note": self.note}
        self.manager.save(self.label, step, state_fn(), meta, metrics, selection=rule)
        write_json(os.path.join(self.manager.run_dir, f"{self.label}.json"), {**rule, "saved": utc_now()})
        return True


def load_checkpoint(path: str, expected_identity: Optional[Mapping[str, Any]] = None,
                    fields: Optional[Sequence[str]] = None, map_location: Any = "cpu") -> Dict[str, Any]:
    """Load, or raise naming every identity field that disagrees."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no checkpoint at {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise IdentityError(f"{path} was written by an earlier version of this package "
                            f"(format {payload.get('format')!r}); retrain rather than mix it with current artifacts")
    if expected_identity is not None:
        require_identity(expected_identity, payload.get("identity", {}), path, fields)
    return payload


def existing_checkpoints(run_dir: str) -> Sequence[str]:
    if not os.path.isdir(run_dir):
        return []
    return sorted(name[:-3] for name in os.listdir(run_dir) if name.endswith(".pt"))


__all__ = ["CheckpointManager", "CheckpointSelection", "IdentityError", "existing_checkpoints", "load_checkpoint",
           "restore_rng", "rng_state", "snapshot_kind"]
