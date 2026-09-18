"""Checkpoints that carry the decisions their weights depend on.

A checkpoint here is not just parameters. It records the graph flag, the model
configuration, the normalization identity, the dataset it was trained from, the
pretrained revision and the training counters -- because every one of those
changes what the weights mean, and none of them is recoverable from the tensors.

Loading checks them. The graph flag in particular is refused rather than
coerced: turning it off is a choice made *before* an arm is trained, and a
graph-trained world model reloaded as a baseline is not a baseline, because the
graph has already reached ``h`` and ``z``. There is no supported conversion
between the arms, and this is where that is enforced.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch


@dataclass
class CheckpointMeta:
    """What has to match for a checkpoint to be loadable into a run."""

    graph_enabled: bool
    stage: str                      # world_model | imitation | online
    env_id: str = ""
    feature_dim: int = 0
    dataset_identity: Dict[str, Any] = field(default_factory=dict)
    normalization_identity: Dict[str, Any] = field(default_factory=dict)
    pretrained_revision: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    step: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


# Differences that make a checkpoint a different object rather than an older
# one. A step count or a dataset episode count may move; these may not.
INCOMPATIBLE = ("graph_enabled", "env_id", "feature_dim", "pretrained_revision")


def save(path: str | Path, meta: CheckpointMeta, modules: Mapping[str, Any],
         optimizers: Optional[Mapping[str, Any]] = None) -> Path:
    """Write parameters, optimizer state and the metadata beside each other."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": asdict(meta),
        "modules": {name: module.state_dict()
                    for name, module in modules.items() if module is not None},
        "optimizers": {name: opt.state_dict()
                       for name, opt in (optimizers or {}).items()
                       if opt is not None},
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)
    # Beside the weights and readable without torch, so a run can be identified
    # from a directory listing.
    target.with_suffix(".json").write_text(
        json.dumps(asdict(meta), indent=2, default=str), encoding="utf-8")
    return target


def check_compatible(stored: Mapping[str, Any], wanted: CheckpointMeta) -> None:
    """Refuse a checkpoint whose decisions differ from this run's."""
    differ = [key for key in INCOMPATIBLE
              if stored.get(key) not in (None, "", 0)
              and stored.get(key) != getattr(wanted, key)]
    if not differ:
        return
    detail = ", ".join(
        f"{key}: checkpoint={stored.get(key)!r} run={getattr(wanted, key)!r}"
        for key in differ)
    if "graph_enabled" in differ:
        raise SystemExit(
            f"refusing to load this checkpoint: {detail}. An arm is chosen "
            "before it is trained -- a graph-trained world model reloaded as a "
            "baseline is not a baseline, because the graph has already reached "
            "h and z. Train the baseline arm from scratch.")
    raise SystemExit(f"refusing to load this checkpoint: {detail}")


def load(path: str | Path, wanted: CheckpointMeta, modules: Mapping[str, Any],
         optimizers: Optional[Mapping[str, Any]] = None,
         *, strict: bool = True) -> CheckpointMeta:
    """Restore a checkpoint into this run, or refuse it."""
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    stored = dict(payload.get("meta") or {})
    check_compatible(stored, wanted)
    for name, module in modules.items():
        if module is None or name not in payload["modules"]:
            continue
        module.load_state_dict(payload["modules"][name], strict=strict)
    for name, opt in (optimizers or {}).items():
        if opt is not None and name in payload.get("optimizers", {}):
            opt.load_state_dict(payload["optimizers"][name])
    return CheckpointMeta(**{k: v for k, v in stored.items()
                             if k in CheckpointMeta.__dataclass_fields__})
