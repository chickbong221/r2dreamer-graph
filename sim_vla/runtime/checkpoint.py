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
#
# Each carries its own rule for "this checkpoint does not record the field",
# because a single ``value not in (None, "", 0)`` test made ``graph_enabled``
# unusable: ``False == 0`` in Python, so a baseline checkpoint read as though
# it had never recorded the flag and loaded into a graph run without complaint
# -- the exact direction this check exists to stop.
INCOMPATIBLE = {
    # Only an absent or None flag is unrecorded. False is a recorded value.
    "graph_enabled": lambda value: value is None,
    "env_id": lambda value: value in (None, ""),
    "feature_dim": lambda value: value in (None, 0),
    "pretrained_revision": lambda value: value in (None, ""),
}


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
    differ = [key for key, unrecorded in INCOMPATIBLE.items()
              if not unrecorded(stored.get(key))
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
            "h and z, and a baseline reloaded as a graph arm has a world model "
            "that was never trained with one. Train each arm from scratch.")
    raise SystemExit(f"refusing to load this checkpoint: {detail}")


def normalization_mode(identity: Optional[Mapping[str, Any]]) -> str:
    """What normalization a recorded identity describes.

    An empty identity means **none**, not "unknown". Treating the two as the
    same is what let a checkpoint fitted with ``mean_std`` load into a run with
    normalization disabled: the comparison ran only when both sides were
    non-empty, so the one side that said nothing was taken to agree. The
    weights then read raw observations and emitted raw actions while having
    been trained on standardised ones, and nothing anywhere said so.

    ``unrecorded`` is for an identity that has content but predates the mode
    being written down; such identities are compared whole.
    """
    identity = dict(identity or {})
    if not identity:
        return "none"
    return str(identity.get("mode") or "unrecorded")


def check_identity(stored: Mapping[str, Any], wanted: CheckpointMeta) -> None:
    """Refuse weights whose interpretation depends on statistics that moved.

    Normalization is not cosmetic: the stored weights read standardised
    observations and emit standardised actions, so a checkpoint fitted on one
    set of statistics and loaded under another is silently mis-scaled at every
    boundary. Mode first, then the fingerprint of the numbers themselves --
    the dataset the statistics came from is a different question and is
    checked separately.
    """
    recorded = dict(stored.get("normalization_identity") or {})
    current = dict(wanted.normalization_identity or {})
    stored_mode = normalization_mode(recorded)
    wanted_mode = normalization_mode(current)
    if stored_mode != wanted_mode:
        raise SystemExit(
            "refusing to load this checkpoint: normalization is "
            f"{stored_mode!r} in the checkpoint and {wanted_mode!r} in this "
            "run. These weights were trained to read one set of units and "
            "would be asked to read another.")
    if stored_mode != "none" and recorded != current:
        differing = sorted(
            key for key in set(recorded) | set(current)
            if recorded.get(key) != current.get(key))
        raise SystemExit(
            f"refusing to load this checkpoint: normalization_identity "
            f"differs in {differing}. checkpoint={recorded!r} run={current!r}. "
            "The saved weights were fitted against those statistics and "
            "cannot be interpreted against these.")

    recorded_data = dict(stored.get("dataset_identity") or {})
    current_data = dict(wanted.dataset_identity or {})
    if recorded_data and current_data and recorded_data != current_data:
        raise SystemExit(
            "refusing to load this checkpoint: dataset_identity differs. "
            f"checkpoint={recorded_data!r} run={current_data!r}.")


def load(path: str | Path, wanted: CheckpointMeta, modules: Mapping[str, Any],
         optimizers: Optional[Mapping[str, Any]] = None,
         *, strict: bool = True,
         explain: Optional[Mapping[str, str]] = None) -> CheckpointMeta:
    """Restore a checkpoint into this run, or refuse it.

    ``explain`` says, per module name, why that module being absent matters
    and what to do instead; it replaces the generic ``strict=False`` advice,
    which is the wrong fix for a module the run cannot do without.
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    stored = dict(payload.get("meta") or {})
    check_compatible(stored, wanted)
    check_identity(stored, wanted)

    requested = {name for name, module in modules.items() if module is not None}
    absent = sorted(requested - set(payload.get("modules") or {}))
    if absent and strict:
        # A strict restore that skipped a requested module left it at its
        # random initialisation and reported success. If a partial restore is
        # what the caller wants, they say so with strict=False.
        notes = " ".join(explain[name] for name in absent
                         if explain and name in explain)
        raise SystemExit(
            f"refusing to load this checkpoint: it has no weights for "
            f"{absent}; it stores {sorted(payload.get('modules') or {})}. "
            + (notes or "Pass strict=False to accept a partial restore."))
    for name, module in modules.items():
        if module is None or name not in payload["modules"]:
            continue
        module.load_state_dict(payload["modules"][name], strict=strict)
    for name, opt in (optimizers or {}).items():
        if opt is not None and name in payload.get("optimizers", {}):
            opt.load_state_dict(payload["optimizers"][name])
    return CheckpointMeta(**{k: v for k, v in stored.items()
                             if k in CheckpointMeta.__dataclass_fields__})
