"""Checkpoints that carry the decisions their weights depend on.

The Dreamer integration learned this the expensive way and the rules are
reused rather than rediscovered. A tensor file is not self-describing: the
same state dict means different things under different action normalization,
against a different dataset, at a different architecture, or with a different
pretrained revision behind the action expert. So all of it is recorded beside
the weights, and loading refuses rather than coerces.

Three refusals in particular:

**Normalization mode.** An *empty* normalization record means ``none``, not
"unknown". Treating the two as the same is what once let weights fitted with
``mean_std`` load into a run with normalization off -- the comparison only ran
when both sides had content, so the side that said nothing was taken to agree.
The weights then read one set of units while emitting another, silently.

**The statistics themselves.** Which dataset the statistics came from and what
the numbers are are two different questions. Two fits over the same episodes
with a different fitter share an identity and produce incompatible weights, so
the numbers get their own fingerprint.

**Architecture.** The widths and depths the model was built at are recorded and
compared. A checkpoint that loads into a differently sized model does not load
at all; one that loads into a same-shaped but differently *configured* model
(a different slot count with the same token dim, a different ensemble size with
the same mlp width) can, and would be wrong.

``smolvla`` is recorded too, because "native TD-MPC2" and "TD-MPC2 with
SmolVLA as its policy" are different agents that share a world model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

STAGES = ("world_model", "imitation", "online")


def fingerprint(payload: Mapping[str, Any]) -> str:
    """A stable digest of a settings block."""
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


@dataclass
class StageMeta:
    """Everything that changes what a set of weights means."""

    backend: str                     # tdmpc2 | sold
    stage: str                       # world_model | imitation | online
    env_id: str = ""
    smolvla: bool = False
    architecture: Dict[str, Any] = field(default_factory=dict)
    normalization: Dict[str, Any] = field(default_factory=dict)
    dataset_identity: Dict[str, Any] = field(default_factory=dict)
    actor: Dict[str, Any] = field(default_factory=dict)
    policy: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    step: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(
                f"stage {self.stage!r} is not one of {list(STAGES)}")

    @property
    def architecture_fingerprint(self) -> str:
        return fingerprint(self.architecture)

    def normalization_mode(self) -> str:
        return normalization_mode(self.normalization)


def normalization_mode(record: Optional[Mapping[str, Any]]) -> str:
    """What a recorded normalization block describes.

    Empty means ``none``. A block with content but no ``mode`` predates the
    field being written and is compared whole.
    """
    record = dict(record or {})
    if not record:
        return "none"
    return str(record.get("mode") or "unrecorded")


def save(path: str | Path, meta: StageMeta, modules: Mapping[str, Any],
         optimizers: Optional[Mapping[str, Any]] = None) -> Path:
    """Weights, optimizer state and the metadata, written atomically."""
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = asdict(meta)
    record["architecture_fingerprint"] = meta.architecture_fingerprint
    payload = {
        "meta": record,
        "modules": {name: module.state_dict()
                    for name, module in modules.items() if module is not None},
        "optimizers": {name: opt.state_dict()
                       for name, opt in (optimizers or {}).items()
                       if opt is not None},
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)
    # Readable without torch, so a run can be identified from a listing.
    target.with_suffix(".json").write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8")
    return target


def check_compatible(stored: Mapping[str, Any], wanted: StageMeta) -> None:
    """Refuse a checkpoint whose decisions differ from this run's."""
    problems = []
    if str(stored.get("backend") or "") != wanted.backend:
        problems.append(
            f"backend: checkpoint={stored.get('backend')!r} "
            f"run={wanted.backend!r}")
    stored_env = str(stored.get("env_id") or "")
    if stored_env and stored_env != wanted.env_id:
        problems.append(
            f"env_id: checkpoint={stored_env!r} run={wanted.env_id!r}")
    if problems:
        raise SystemExit(
            "refusing to load this checkpoint: " + ", ".join(problems))

    stored_arch = dict(stored.get("architecture") or {})
    if stored_arch and stored_arch != wanted.architecture:
        differing = sorted(
            key for key in set(stored_arch) | set(wanted.architecture)
            if stored_arch.get(key) != wanted.architecture.get(key))
        raise SystemExit(
            "refusing to load this checkpoint: the architecture differs in "
            f"{differing}. checkpoint={stored_arch} run={wanted.architecture}. "
            "Weights of one shape may load into another configuration of the "
            "same shape and mean something else.")

    stored_mode = normalization_mode(stored.get("normalization"))
    wanted_mode = wanted.normalization_mode()
    if stored_mode != wanted_mode:
        raise SystemExit(
            "refusing to load this checkpoint: action normalization is "
            f"{stored_mode!r} in the checkpoint and {wanted_mode!r} in this "
            "run. These weights were trained to emit one set of units and "
            "would be read as another.")
    stored_norm = dict(stored.get("normalization") or {})
    if stored_mode != "none" and stored_norm != dict(wanted.normalization):
        differing = sorted(
            key for key in set(stored_norm) | set(wanted.normalization)
            if stored_norm.get(key) != wanted.normalization.get(key))
        raise SystemExit(
            "refusing to load this checkpoint: the action statistics differ "
            f"in {differing}; the saved weights were fitted against those and "
            "cannot be interpreted against these.")

    stored_data = dict(stored.get("dataset_identity") or {})
    wanted_data = dict(wanted.dataset_identity or {})
    if stored_data and wanted_data and stored_data != wanted_data:
        raise SystemExit(
            "refusing to load this checkpoint: dataset_identity differs. "
            f"checkpoint={stored_data} run={wanted_data}.")

    stored_actor = dict(stored.get("actor") or {})
    wanted_actor = dict(wanted.actor or {})
    for key in ("revision", "repo_id", "chunk_size", "action_dim"):
        left, right = stored_actor.get(key), wanted_actor.get(key)
        if left in (None, "", 0) or right in (None, "", 0):
            continue
        if left != right:
            raise SystemExit(
                f"refusing to load this checkpoint: actor.{key} is {left!r} in "
                f"the checkpoint and {right!r} in this run.")

    if bool(stored.get("smolvla", False)) != bool(wanted.smolvla):
        raise SystemExit(
            "refusing to load this checkpoint: it was written with "
            f"smolvla={stored.get('smolvla')} and this run has "
            f"smolvla={wanted.smolvla}. The native Gaussian policy and the "
            "flow policy are different agents over the same world model; the "
            "world-model weights can be shared by loading the world_model "
            "stage explicitly, but the policy weights cannot.")


def load(path: str | Path, wanted: StageMeta, modules: Mapping[str, Any],
         optimizers: Optional[Mapping[str, Any]] = None,
         *, strict: bool = True) -> StageMeta:
    """Restore a checkpoint into this run, or refuse it."""
    import torch

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    stored = dict(payload.get("meta") or {})
    check_compatible(stored, wanted)

    requested = {name for name, module in modules.items() if module is not None}
    stored_modules = dict(payload.get("modules") or {})
    absent = sorted(requested - set(stored_modules))
    if absent and strict:
        raise SystemExit(
            f"refusing to load this checkpoint: it has no weights for "
            f"{absent}; it stores {sorted(stored_modules)}. Pass strict=False "
            "to accept a partial restore.")
    for name, module in modules.items():
        if module is None or name not in stored_modules:
            continue
        module.load_state_dict(stored_modules[name], strict=strict)
    for name, opt in (optimizers or {}).items():
        if opt is not None and name in payload.get("optimizers", {}):
            opt.load_state_dict(payload["optimizers"][name])

    fields = set(StageMeta.__dataclass_fields__)
    return StageMeta(**{k: v for k, v in stored.items() if k in fields})
