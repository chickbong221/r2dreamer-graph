"""One set of action and proprioception statistics, shared by both arms.

Fitted once from the demonstrations both arms train on, written to a file, and
loaded from that file thereafter. Refitting per arm would be the easiest way to
make a controlled comparison uncontrolled: the graph arm and the baseline would
be scaling the same actions differently, and every downstream difference would
carry that.

The file records which dataset it was fitted from, by the identity the dataset
itself carries -- task, episode count, controller and the repo revision that
collected it. A normalizer loaded against a different dataset is refused rather
than applied, because the failure it causes otherwise is a quiet loss of
accuracy rather than an error.

On these datasets the action statistics are close to a no-op: ManiSkill's
``pd_joint_pos`` controller sets ``normalize_action=True``, so the recorded
actions already live in [-1, 1]. They are fitted and stored anyway. What they
buy is the check -- an arm whose actions drift outside the controller's range
is clipping, and the statistics are where that shows up first.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

# Percentile bounds, rather than min and max. One outlier frame should not set
# the scale of a whole dimension.
LOW_Q, HIGH_Q = 1.0, 99.0
EPS = 1e-6


@dataclass
class FieldStats:
    """Per-dimension statistics for one field."""

    mean: list
    std: list
    low: list          # the LOW_Q percentile
    high: list         # the HIGH_Q percentile
    minimum: list
    maximum: list
    count: int

    def as_arrays(self):
        return (np.asarray(self.mean, np.float32),
                np.asarray(self.std, np.float32),
                np.asarray(self.low, np.float32),
                np.asarray(self.high, np.float32))


def fit_field(samples: Iterable[np.ndarray]) -> FieldStats:
    stacked = np.concatenate(
        [np.asarray(s, dtype=np.float64).reshape(len(s), -1) for s in samples])
    low, high = np.percentile(stacked, [LOW_Q, HIGH_Q], axis=0)
    return FieldStats(
        mean=stacked.mean(axis=0).tolist(),
        std=stacked.std(axis=0).tolist(),
        low=low.tolist(), high=high.tolist(),
        minimum=stacked.min(axis=0).tolist(),
        maximum=stacked.max(axis=0).tolist(),
        count=int(stacked.shape[0]),
    )


def dataset_identity(metadata: Mapping[str, Any], episodes: int) -> Dict[str, Any]:
    """What makes two datasets the same one for normalization purposes."""
    controller = metadata.get("controller") or {}
    return {
        "env_id": metadata.get("env_id"),
        "episodes": int(episodes),
        "control_mode": controller.get("control_mode"),
        "action_dim": controller.get("action_dim"),
        "proprio_names": list(metadata.get("proprio_names") or []),
        "repo_revision": (metadata.get("versions") or {}).get("repo_revision"),
    }


@dataclass
class Normalizer:
    """Mean/std normalization with the fitted range kept alongside.

    ``mean_std`` is what the model sees. The percentile range is stored rather
    than used so that a value can be checked against the demonstrated range
    without refitting -- which is how clipping gets noticed.
    """

    fields: Dict[str, FieldStats]
    identity: Dict[str, Any]
    mode: str = "mean_std"

    def normalize(self, name: str, values: np.ndarray) -> np.ndarray:
        mean, std, low, high = self.fields[name].as_arrays()
        arr = np.asarray(values, dtype=np.float32)
        if self.mode == "mean_std":
            return (arr - mean) / np.maximum(std, EPS)
        span = np.maximum(high - low, EPS)
        return 2.0 * (arr - low) / span - 1.0

    def denormalize(self, name: str, values: np.ndarray) -> np.ndarray:
        mean, std, low, high = self.fields[name].as_arrays()
        arr = np.asarray(values, dtype=np.float32)
        if self.mode == "mean_std":
            return arr * np.maximum(std, EPS) + mean
        span = np.maximum(high - low, EPS)
        return (arr + 1.0) * 0.5 * span + low

    def out_of_range(self, name: str, values: np.ndarray) -> np.ndarray:
        """Per-row flag: this value sits outside what was demonstrated."""
        _, _, low, high = self.fields[name].as_arrays()
        arr = np.asarray(values, dtype=np.float32)
        return np.any((arr < low) | (arr > high), axis=-1)

    # --------------------------------------------------------------- storage
    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "mode": self.mode,
            "identity": self.identity,
            "fields": {k: asdict(v) for k, v in self.fields.items()},
        }, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path,
             identity: Optional[Mapping[str, Any]] = None) -> "Normalizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        loaded = cls(
            fields={k: FieldStats(**v) for k, v in payload["fields"].items()},
            identity=dict(payload.get("identity") or {}),
            mode=str(payload.get("mode", "mean_std")),
        )
        if identity is not None and dict(identity) != loaded.identity:
            differing = sorted(
                key for key in set(identity) | set(loaded.identity)
                if dict(identity).get(key) != loaded.identity.get(key))
            raise SystemExit(
                f"{path} was fitted on a different dataset; {differing} differ. "
                "Fit once from the demonstrations both arms share, or the two "
                "arms scale the same actions differently.")
        return loaded


def fit_normalizer(data, *, fields: Sequence[str] = ("actions", "proprio"),
                   max_episodes: int = 0) -> Normalizer:
    """Fit from a :class:`~sim_vla.data.dataset.DemoDataset`.

    Reads whole episodes rather than sampled windows: statistics taken from a
    sampler would inherit whatever the sampler over-represents, and the padding
    rows would count twice.
    """
    refs = data.episodes[: max_episodes] if max_episodes else data.episodes
    if not refs:
        raise SystemExit(f"{data.path} holds no episodes to fit from")
    gathered: Dict[str, list] = {name: [] for name in fields}
    for ref in refs:
        block = data.read(ref, 0, ref.steps)
        for name in fields:
            if name not in block:
                raise SystemExit(
                    f"{name!r} is not in this arm's batch fields "
                    f"({sorted(block)}); normalization must be fitted from "
                    "fields both arms share")
            gathered[name].append(block[name])
    return Normalizer(
        fields={name: fit_field(samples) for name, samples in gathered.items()},
        identity=dataset_identity(data.metadata, len(data.episodes)),
    )
