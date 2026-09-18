"""The simulator's model config, as an attribute-access object.

``sim_vla`` composes the simulator's own network components, and those read
their settings with attribute access (``config.rssm.deter``,
``config.encoder.cnn_keys``). Hydra supplies that in the existing trainer; here
the same yaml files are read directly and wrapped, so the components are
configured by the same numbers rather than by a second copy of them.

The graph capacities and the semantic width are overridden from the sim_vla
config, because those are the two the dataset also records and must agree with.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO / "configs/model/size50M_graph_simple.yaml"
BASE_MODEL = REPO / "configs/model/_base_.yaml"


class Node:
    """Recursive attribute access over a config dict."""

    def __init__(self, data: Mapping[str, Any]):
        object.__setattr__(self, "_data", dict(data))

    def __getattr__(self, name: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if name not in data:
            raise AttributeError(
                f"config has no {name!r}; present: {sorted(data)[:20]}")
        value = data[name]
        return Node(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        object.__getattribute__(self, "_data")[name] = value

    def __contains__(self, name: str) -> bool:
        return name in object.__getattribute__(self, "_data")

    def get(self, name: str, default: Any = None) -> Any:
        data = object.__getattribute__(self, "_data")
        value = data.get(name, default)
        return Node(value) if isinstance(value, dict) else value

    def to_dict(self) -> Dict[str, Any]:
        return dict(object.__getattribute__(self, "_data"))


def _resolve(node: Any, root: Mapping[str, Any]) -> Any:
    """Substitute ``${model.x}`` against the preset's own top-level scalars.

    The simulator's model configs are written for Hydra, where the size preset
    supplies ``deter``, ``units``, ``act`` and the rest and every block refers
    back to them. Reading the yaml without resolving these leaves the literal
    string ``${model.deter}`` where a layer width belongs, and the failure
    lands inside a constructor rather than here.
    """
    if isinstance(node, dict):
        return {k: _resolve(v, root) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve(v, root) for v in node]
    if not isinstance(node, str) or not node.startswith("${model."):
        return node
    key = node[len("${model."):-1]
    if key not in root:
        raise KeyError(
            f"{node} does not resolve; the preset defines {sorted(k for k, v in root.items() if not isinstance(v, dict))}")
    return root[key]


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_model_config(sim_cfg: Mapping[str, Any],
                      model_yaml: Path = DEFAULT_MODEL) -> Node:
    """The simulator model config, with sim_vla's graph settings applied."""
    import yaml

    base = yaml.safe_load(BASE_MODEL.read_text(encoding="utf-8")) or {}
    override = yaml.safe_load(Path(model_yaml).read_text(encoding="utf-8")) or {}
    merged = _merge(base, override)
    merged = _resolve(merged, merged)

    graph_cfg = dict((sim_cfg.get("model") or {}).get("graph") or {})
    merged.setdefault("graph", {})
    merged["graph"] = dict(merged["graph"]) | {
        "enabled": bool(graph_cfg.get("enabled", False)),
        "n_max": int(graph_cfg.get("n_max", merged["graph"].get("n_max", 8))),
        "e_max": int(graph_cfg.get("e_max", merged["graph"].get("e_max", 168))),
    }
    merged["device"] = str(sim_cfg.get("device", merged.get("device", "cuda")))
    return Node(merged)
