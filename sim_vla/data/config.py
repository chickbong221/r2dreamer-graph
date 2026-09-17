"""Read the graph and rendering settings out of the training config.

The collector's job is to produce data the online policy could have produced,
which means every setting that changes what a graph or a frame contains has to
be the one the trainer uses. Held as CLI defaults, those numbers agree today
and drift the first time someone raises ``e_max`` in ``configs/model`` -- and a
dataset packed at a different capacity does not announce itself, it just
silently drops facts past the row it ran out of.

So they are read from the same files Hydra reads. Only the handful of keys this
collector needs are resolved, and only the one interpolation form those keys
use (``${model.graph.<key>}``): a general config resolver would be a second
implementation of Hydra, which is a worse thing to maintain than this comment.

Anything missing falls back to the value passed in, and the resolved settings
go into the dataset's metadata either way, so a reader never has to guess which
of the two applied.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_ENV_CONFIG = "configs/env/maniskill.yaml"
DEFAULT_MODEL_CONFIG = "configs/model/size50M_graph_simple.yaml"

_INTERP = re.compile(r"^\$\{model\.graph\.(\w+)\}$")


def _resolve(value: Any, model_graph: Dict[str, Any]) -> Any:
    """Substitute ``${model.graph.x}``; leave anything else alone.

    An unresolved interpolation is returned as the literal string rather than
    guessed at, and the caller treats a string where it wanted a number as a
    missing value -- which is what it is.
    """
    if isinstance(value, str):
        match = _INTERP.match(value.strip())
        if match:
            return model_graph.get(match.group(1), value)
    return value


def load_training_settings(
    env_config: str | Path = DEFAULT_ENV_CONFIG,
    model_config: str | Path = DEFAULT_MODEL_CONFIG,
) -> Dict[str, Any]:
    """The subset of the training config that changes what gets recorded.

    Returns ``{}`` when the files are not readable, which is not an error: a
    caller outside the repo tree still has its own defaults, and the metadata
    records that the config was not consulted.
    """
    try:
        import yaml

        env_cfg = yaml.safe_load(Path(env_config).read_text(encoding="utf-8"))
        model_cfg = yaml.safe_load(Path(model_config).read_text(encoding="utf-8"))
    except Exception:                                      # noqa: BLE001
        return {}

    graph = dict((env_cfg or {}).get("graph") or {})
    model_graph = dict((model_cfg or {}).get("graph") or {})
    size = (env_cfg or {}).get("size") or []
    out: Dict[str, Any] = {
        "_sources": {
            "env_config": str(env_config),
            "model_config": str(model_config),
        },
        "shader": (env_cfg or {}).get("shader_dir"),
        "sensor_size": [int(v) for v in size] if len(size) == 2 else None,
        "visibility_policy": graph.get("visibility_policy"),
        "use_target_flag": graph.get("use_target_flag"),
        "object_object_spatial": graph.get("object_object_spatial"),
        "thresholds_path": graph.get("thresholds_path"),
        "whitelist_dir": graph.get("whitelist_dir"),
    }
    for key in ("n_max", "e_max"):
        value = _resolve(graph.get(key), model_graph)
        out[key] = int(value) if isinstance(value, (int, float)) else None
    return {key: value for key, value in out.items() if value is not None}


def apply_defaults(args, settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Fill the arguments the caller left unset from the training config.

    An explicit flag always wins: argparse leaves these at ``None`` when they
    were not given, and only those are filled. The return value is what was
    actually taken, for the metadata.
    """
    settings = load_training_settings(
        getattr(args, "env_config", DEFAULT_ENV_CONFIG),
        getattr(args, "model_config", DEFAULT_MODEL_CONFIG),
    ) if settings is None else settings

    taken: Dict[str, Any] = {}
    for key in ("shader", "sensor_size", "visibility_policy", "n_max", "e_max",
                "use_target_flag", "object_object_spatial", "thresholds_path",
                "whitelist_dir"):
        if getattr(args, key, None) is None and key in settings:
            setattr(args, key, settings[key])
            taken[key] = settings[key]
    return {"from_config": taken, "sources": settings.get("_sources", {})}
