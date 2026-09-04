"""One rolling-best checkpoint per run, and a load that refuses to guess.

``train.py`` deliberately wrote nothing, for a reason worth restating: the
relation labels were corrected in place, so the ids stayed the same while
their geometric meaning changed, and resuming across that would have trained
one label to mean two things. Runs were compared from metrics instead.

That hazard is about *loading*, not saving. What makes saving safe now is that
every checkpoint carries the identity of the graph contract it was trained
under -- entity and relation vocabularies, the schedule, the assets -- and
loading refuses on any mismatch rather than warning. A checkpoint whose
vocabulary has moved is not a degraded checkpoint; it is a different model.

The policy, deliberately narrow:

* nothing is written before ``start_step``;
* from the first eligible evaluation onward, the best result so far is kept;
* exactly one file, replaced in place, written atomically;
* no interrupt, latest, milestone or automatic final checkpoint;
* a cancelled run keeps whatever best it had earned.

The selection metric is **not** chosen here. ``metric`` starts empty and a
run with checkpointing enabled and no metric refuses to start, because
silently picking one decides which model the experiment reports.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

# What must match for a checkpoint to be loadable. Every one of these changes
# what a stored weight *means* rather than how good it is.
IDENTITY_FIELDS = (
    "entity_vocab",
    "relation_vocab",
    "absolute_vocab",
    "graph_schema",
    "schedule",
    "assets",
)


class CheckpointError(RuntimeError):
    """A checkpoint that cannot be trusted to mean what the model expects."""


@dataclass
class CheckpointConfig:
    """The agreed policy, with the metric left for a human to fill in."""

    enabled: bool = False
    start_step: int = 8_000_000
    # Empty on purpose. See the module docstring.
    metric: str = ""
    tiebreak: str = ""
    mode: str = "max"
    path: str = "checkpoint_best.pt"

    @classmethod
    def from_mapping(cls, raw: Optional[Mapping[str, Any]]) -> "CheckpointConfig":
        raw = dict(raw or {})
        return cls(
            enabled=bool(raw.get("enabled", False)),
            start_step=int(raw.get("start_step", 8_000_000)),
            metric=str(raw.get("metric", "") or ""),
            tiebreak=str(raw.get("tiebreak", "") or ""),
            mode=str(raw.get("mode", "max") or "max"),
            path=str(raw.get("path", "checkpoint_best.pt")),
        )

    def validate(self) -> None:
        """Refuse a run that would have to invent a selection rule."""
        if not self.enabled:
            return
        if not self.metric:
            raise CheckpointError(
                "checkpoint.enabled is set but checkpoint.metric is empty. "
                "Which evaluation decides the best model is an experiment "
                "decision, not a default: it selects the checkpoint every "
                "later number is reported from. Set it explicitly (for "
                "example 'eval/success_once') or turn checkpointing off."
            )
        if self.mode not in ("max", "min"):
            raise CheckpointError(
                f"checkpoint.mode must be 'max' or 'min', got {self.mode!r}")
        if self.start_step < 0:
            raise CheckpointError("checkpoint.start_step cannot be negative")


def _better(candidate: float, incumbent: Optional[float], mode: str) -> bool:
    if incumbent is None:
        return True
    return candidate > incumbent if mode == "max" else candidate < incumbent


@dataclass
class Checkpointer:
    """Rolling-best saving for one run."""

    config: CheckpointConfig
    logdir: str
    identity: Dict[str, str] = field(default_factory=dict)
    # The writer, injectable so the selection policy can be tested without a
    # serializer -- which rule claims the file is arithmetic, not I/O.
    save_fn: Any = None
    best: Optional[float] = None
    best_tiebreak: Optional[float] = None
    best_step: Optional[int] = None
    n_saved: int = 0

    def __post_init__(self):
        self.config.validate()

    @property
    def path(self) -> str:
        return os.path.join(self.logdir, self.config.path)

    def eligible(self, step: int) -> bool:
        """Whether an evaluation at ``step`` may be saved at all.

        Earlier evaluations still run and still log. They just cannot claim
        the file, and -- the part worth being explicit about -- they do not
        set the incumbent either: a strong result at 2M must not stop the
        first eligible evaluation from saving.
        """
        return self.config.enabled and int(step) >= int(self.config.start_step)

    def maybe_save(self, step: int, metrics: Mapping[str, float],
                   state_fn) -> bool:
        """Save when this is the best eligible evaluation so far.

        ``state_fn`` is called only when a save is actually happening, so an
        ineligible or worse evaluation costs no serialization.
        """
        if not self.eligible(step):
            return False
        if self.config.metric not in metrics:
            raise CheckpointError(
                f"checkpoint.metric {self.config.metric!r} is not among the "
                f"evaluation metrics {sorted(metrics)}. Nothing can be "
                "selected on a number that was never measured."
            )
        value = float(metrics[self.config.metric])
        tiebreak = (float(metrics.get(self.config.tiebreak, float("nan")))
                    if self.config.tiebreak else None)

        if _better(value, self.best, self.config.mode):
            improved = True
        elif value == self.best and self.config.tiebreak:
            improved = _better(tiebreak, self.best_tiebreak, self.config.mode)
        else:
            improved = False
        if not improved:
            return False

        payload = dict(state_fn())
        payload["checkpoint"] = {
            "step": int(step),
            "metric": self.config.metric,
            "value": value,
            "tiebreak": self.config.tiebreak,
            "tiebreak_value": tiebreak,
            "identity": dict(self.identity),
        }
        (self.save_fn or atomic_save)(payload, self.path)
        self.best, self.best_tiebreak, self.best_step = value, tiebreak, int(step)
        self.n_saved += 1
        return True


def atomic_save(payload: Mapping[str, Any], path: str) -> None:
    """Write through a temporary file in the same directory, then rename.

    An interrupted write must not leave a half-written checkpoint where the
    previous good one was: the run would then have neither.
    """
    import torch

    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=directory, suffix=".partial")
    os.close(handle)
    try:
        torch.save(dict(payload), tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def identity_mismatches(stored: Mapping[str, Any],
                        current: Mapping[str, Any]) -> Sequence[str]:
    """Which identity fields disagree, by name."""
    out = []
    for field_name in IDENTITY_FIELDS:
        want, got = current.get(field_name), stored.get(field_name)
        if want is None and got is None:
            continue
        if want != got:
            out.append(f"{field_name}: checkpoint {got!r} vs run {want!r}")
    return out


def _digest(*parts: Any) -> str:
    import hashlib

    text = "\x1f".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _file_digest(path: Optional[str]) -> str:
    if not path or not os.path.isfile(path):
        return "absent"
    with open(path, "rb") as handle:
        return _digest(handle.read().decode("utf-8", "replace"))


def run_identity(*, whitelist_dir: str, schedule_path: str = "",
                 schedule_label: str = "", n_max: int = 0, e_max: int = 0,
                 n_cams: int = 0, entity_tokens: Sequence[str] = (),
                 relation_tokens: Sequence[str] = (),
                 absolute_tokens: Sequence[str] = ()) -> Dict[str, str]:
    """What a checkpoint has to agree with before its weights mean anything.

    Deliberately about the *model contract* and nothing else: which ids the
    embeddings index, how many rows and facts a frame holds, which schedule
    the progress head was supervised against, and which mined assets produced
    all of that. An evaluation is free to change the scene, the lighting or
    the episode count without any of these moving -- which is what lets
    Experiment C load Experiment B's checkpoint.
    """
    union = (os.path.join(whitelist_dir, "pick_all.json")
             if whitelist_dir else "")
    return {
        "entity_vocab": f"{len(entity_tokens)}:{_digest(*entity_tokens)}",
        "relation_vocab":
            f"{len(relation_tokens)}:{_digest(*relation_tokens)}",
        "absolute_vocab":
            f"{len(absolute_tokens)}:{_digest(*absolute_tokens)}",
        "graph_schema": f"n{int(n_max)}e{int(e_max)}c{int(n_cams)}",
        "schedule": f"{schedule_label}:{_file_digest(schedule_path)}",
        "assets": _file_digest(union),
    }


def load_checkpoint(path: str, identity: Mapping[str, Any],
                    map_location: Any = "cpu") -> Dict[str, Any]:
    """Load, or raise naming every field that disagrees.

    Never a warning. A relation id whose meaning moved reads as a perfectly
    valid tensor, and a run that continued past the warning would train one
    label to mean two things -- which is the exact reason nothing used to be
    saved at all.
    """
    import torch

    if not os.path.isfile(path):
        raise CheckpointError(f"no checkpoint at {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    stored = (payload.get("checkpoint") or {}).get("identity") or {}
    problems = identity_mismatches(stored, identity)
    if problems:
        raise CheckpointError(
            f"{path} was trained under a different graph contract:\n  "
            + "\n  ".join(problems)
            + "\nThe stored weights index vocabularies that have moved, so "
              "they do not mean what this run would read them as. Re-mine or "
              "retrain rather than loading."
        )
    return payload
