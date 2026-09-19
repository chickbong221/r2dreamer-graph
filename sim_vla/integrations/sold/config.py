"""SOLD's own settings, read from SOLD's own config files.

``sold/configs/train_sold.yaml`` and ``sold/configs/autoencoder/savi.yaml`` are
written for Hydra and carry ``_target_`` keys and ``${...}`` interpolations.
They are the source for every architecture number here -- slot count, slot
width, token widths, layer counts, the imagination horizon, the discount, the
lambda -- and this module reads them rather than restating them, so changing
one of those files changes what is built.

Three things are resolved here that Hydra would otherwise do:

* ``${..corrector.slot_dim}`` style interpolations, relative to the node
* ``'???'`` placeholders, which are values the dataset supplies (image size,
  action width)
* ``_target_``, which names the class; the builders in ``model.py`` import
  those classes from the vendored tree and call them directly

The integration's own settings live under ``smolvla`` and ``stages``. Nothing
in them reaches SOLD except through the hook in ``train_sold.py``; with
``smolvla.enabled: false`` the module built here is upstream's.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from ...config import deep_merge
from ..vendor import SOLD_CONFIGS

INTEGRATION_CONFIGS = Path(__file__).resolve().parents[1] / "configs"

# Settings a checkpoint is compared on: they change what the weights are.
ARCHITECTURE_KEYS = (
    "num_slots", "slot_dim", "image_size", "action_dim", "max_episode_steps",
    "num_context", "imagination_horizon", "dynamics", "actor", "critic",
    "reward_predictor", "autoencoder", "adapter_context",
)


def _read_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def resolve(node: Any, root: Mapping[str, Any], here: Sequence[str]) -> Any:
    """Hydra's relative interpolations, and only those.

    ``${..corrector.slot_dim}`` means "up one level, then down". A general
    resolver would be a second implementation of OmegaConf; these files use
    exactly this form and ``${...env.image_size}``.
    """
    if isinstance(node, dict):
        return {key: resolve(value, root, list(here) + [key])
                for key, value in node.items()}
    if isinstance(node, list):
        return [resolve(value, root, here) for value in node]
    if not isinstance(node, str) or not node.startswith("${"):
        return node
    body = node[2:-1]
    ups = len(body) - len(body.lstrip("."))
    path = body[ups:].split(".") if body[ups:] else []
    # One dot is "this node"; each further dot climbs one level.
    base = list(here[:-1])
    for _ in range(max(ups - 1, 0)):
        if base:
            base.pop()
    cursor: Any = root
    for part in base + path:
        if not isinstance(cursor, Mapping) or part not in cursor:
            raise KeyError(
                f"{'.'.join(here)}: {node} does not resolve against this config")
        cursor = cursor[part]
    return resolve(cursor, root, base + path)


def native_defaults() -> Dict[str, Any]:
    """``train_sold.yaml`` plus the SAVi block, with interpolations resolved."""
    sold = _read_yaml(SOLD_CONFIGS / "train_sold.yaml")
    model = dict(sold.get("model") or {})
    model["autoencoder_spec"] = _read_yaml(
        SOLD_CONFIGS / "autoencoder" / "savi.yaml")
    resolved = resolve(model, model, [])
    return resolved


DEFAULT_SMOLVLA: Dict[str, Any] = {
    "enabled": True,
    "pretrained": "lerobot/smolvla_base",
    "revision": "",
    "chunk_size": 0,
    "flow_steps": 0,
    "state_token_mode": "embedding",
    # The actor-side adapter over the causal slot history. Its own widths;
    # 0 for `context` takes the run's num_context, which is the largest value
    # that is the same at every imagined step.
    "adapter": {"token_dim": 256, "hidden_dim": 512, "num_heads": 8,
                "num_layers": 3, "num_mlp_layers": 1, "context": 0},
    "action_normalization": "identity",
    "execute": 1,
    "online_lr": 1.0e-5,
    "max_batch": 256,
    "instruction": "",
}

DEFAULT_STAGES: Dict[str, Any] = {
    "autoencoder": {"steps": 20_000, "batch_size": 8, "sequence_length": 8,
                    "lr": 1.0e-4, "grad_clip": 0.05, "log_every": 200},
    "world_model": {"steps": 50_000, "batch_size": 8, "log_every": 200},
    "imitation": {"steps": 20_000, "batch_size": 8, "lr": 1.0e-4,
                  "grad_clip": 1.0, "log_every": 100, "sequence_length": 16},
    "online": {"steps": 1_000_000},
}


def load(task: str = "pickcube",
         overrides: Optional[Mapping[str, Any]] = None,
         root: Path = INTEGRATION_CONFIGS) -> Dict[str, Any]:
    """The staged run's settings: task, SOLD world model, SmolVLA, stages."""
    from ...config import load_config as load_sim_vla

    base = _read_yaml(root / "sold.yaml")
    merged = deep_merge(base, dict(overrides or {}))
    sim_cfg = load_sim_vla(str(task), "dreamer")
    merged["task"] = deep_merge(dict(sim_cfg["task"]),
                                dict(merged.get("task") or {}))
    merged["world_model"] = deep_merge(native_defaults(),
                                       dict(merged.get("world_model") or {}))
    merged["smolvla"] = deep_merge(DEFAULT_SMOLVLA,
                                   dict(merged.get("smolvla") or {}))
    merged["stages"] = deep_merge(DEFAULT_STAGES, dict(merged.get("stages") or {}))
    if not merged["smolvla"].get("instruction"):
        merged["smolvla"]["instruction"] = str(
            merged["task"].get("instruction") or "")
    merged["backend"] = "sold"
    return merged


def context_bounds(world: Mapping[str, Any]) -> tuple:
    """``(min, max)`` context frames, from ``num_context``."""
    value = world.get("num_context", 3)
    if isinstance(value, (list, tuple)):
        low, high = int(value[0]), int(value[1])
    else:
        low = high = int(value)
    if low > high:
        raise SystemExit(
            f"num_context={value}: the minimum exceeds the maximum")
    return low, high


def architecture(world: Mapping[str, Any], *, num_slots: int, slot_dim: int,
                 action_dim: int, image_size: Sequence[int],
                 max_episode_steps: int, adapter_context: int) -> Dict[str, Any]:
    """The settings a checkpoint has to be compared on."""
    spec = dict(world.get("autoencoder_spec") or {})
    def widths(node):
        node = dict(node or {})
        return {k: v for k, v in node.items()
                if k in ("token_dim", "hidden_dim", "num_layers", "num_heads",
                         "num_mlp_layers", "residual", "teacher_forcing")}

    return {
        "num_slots": int(num_slots),
        "slot_dim": int(slot_dim),
        "action_dim": int(action_dim),
        "image_size": [int(x) for x in image_size],
        "max_episode_steps": int(max_episode_steps),
        "num_context": list(context_bounds(world)),
        "imagination_horizon": int(world.get("imagination_horizon", 15)),
        "adapter_context": int(adapter_context),
        "dynamics": widths(world.get("dynamics_predictor")),
        "actor": widths(world.get("actor")),
        "critic": widths(world.get("critic")),
        "reward_predictor": widths(world.get("reward_predictor")),
        "autoencoder": {
            "corrector": {k: v for k, v in dict(spec.get("corrector") or {}).items()
                          if k != "_target_"},
            "encoder": {k: v for k, v in dict(spec.get("encoder") or {}).items()
                        if k != "_target_"},
            "decoder": {k: v for k, v in dict(spec.get("decoder") or {}).items()
                        if k != "_target_"},
        },
    }
