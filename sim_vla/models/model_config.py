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
from collections.abc import Mapping
from typing import Any, Dict

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO / "configs/model/size50M_graph_simple.yaml"
BASE_MODEL = REPO / "configs/model/_base_.yaml"
DEFAULT_ENV = REPO / "configs/env/maniskill.yaml"

# sim_vla's own observation contract. The simulator's env config names the
# proprioception key "state"; the collected dataset names it "proprio", so the
# encoder regexes are set here rather than inherited, and a mismatch would
# silently leave the MLP encoder with nothing to read.
OBSERVATION_KEYS = {
    "cnn_keys": "^image_",
    "mlp_keys": "^proprio$",
}


class Node(Mapping):
    """Attribute access *and* the mapping protocol, like Hydra's DictConfig.

    The simulator's networks use config nodes both ways in the same line::

        partial(getattr(dists, str(config.cnn_dist.name)), **config.cnn_dist)

    and ``dreamer.py`` does ``dict(config.loss_scales)``. An object that only
    supported attribute access satisfied the first half of that line and raised
    ``TypeError: argument after ** must be a mapping`` on the second, from
    inside a constructor that never mentions config.

    So ``__getitem__`` / ``__iter__`` / ``__len__`` are implemented and the
    class registers as a Mapping. Item access returns the raw value, because
    what reads it is a keyword-argument unpack feeding a distribution
    constructor; attribute access still wraps nested dicts, because what reads
    *that* is ``config.rssm.deter``.
    """

    def __init__(self, data: Mapping[str, Any]):
        object.__setattr__(self, "_data", dict(data))

    def __getitem__(self, key: str) -> Any:
        return object.__getattribute__(self, "_data")[key]

    def __iter__(self):
        return iter(object.__getattribute__(self, "_data"))

    def __len__(self) -> int:
        return len(object.__getattribute__(self, "_data"))

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


def _lookup(dotted: str, tree: Mapping[str, Any]) -> Any:
    node: Any = tree
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def _resolve(node: Any, model_root: Mapping[str, Any],
             env_root: Mapping[str, Any], path: str = "",
             globals_root: Mapping[str, Any] | None = None) -> Any:
    """Substitute the Hydra interpolations, and refuse to leave one behind.

    The simulator's model configs are written for Hydra: the size preset
    supplies ``deter``, ``units`` and the rest, and the *env* config supplies
    the encoder and decoder key regexes. Both forms appear, and an earlier
    version of this resolved only ``${model.*}``.

    What that cost is worth recording. ``${env.encoder.cnn_keys}`` survived as
    a literal string, matched no observation key, and ``MultiEncoder`` was
    handed an empty shape dict -- which it reports as a bare
    ``NotImplementedError`` from the line that discovers it has no encoders.
    Nothing said "unresolved interpolation" anywhere in that traceback.

    So anything still shaped like ``${...}`` after this is an error here,
    where the name of the setting is still in hand.
    """
    globals_root = globals_root or {}
    if isinstance(node, dict):
        return {k: _resolve(v, model_root, env_root,
                            f"{path}.{k}" if path else k, globals_root)
                for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve(v, model_root, env_root, path, globals_root)
                for v in node]
    if not isinstance(node, str) or "${" not in node:
        return node

    # A bare ${name} is a Hydra global -- device, seed. They come from the
    # sim_vla config, which is the thing that actually knows them here.
    bare = node[2:-1] if node.startswith("${") and node.endswith("}") else ""
    if bare and "." not in bare and ":" not in bare:
        if bare in globals_root:
            return globals_root[bare]
        raise KeyError(
            f"{path or 'config'}: {node} is a global sim_vla does not set; "
            f"it knows {sorted(globals_root)}")

    for prefix, root in (("${model.", model_root), ("${env.", env_root)):
        if node.startswith(prefix) and node.endswith("}"):
            dotted = node[len(prefix):-1]
            try:
                return _lookup(dotted, root)
            except KeyError:
                raise KeyError(
                    f"{path or 'config'}: {node} does not resolve; "
                    f"{prefix[2:-1]} defines {sorted(root)[:20]}") from None
    raise KeyError(
        f"{path or 'config'}: {node} is an interpolation sim_vla does not "
        "resolve. Set it explicitly rather than leaving it to be matched "
        "against as a literal string.")


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_model_config(sim_cfg: Mapping[str, Any],
                      model_yaml: Path = DEFAULT_MODEL,
                      env_yaml: Path = DEFAULT_ENV) -> Node:
    """The simulator model config, with sim_vla's own settings applied."""
    import yaml

    base = yaml.safe_load(BASE_MODEL.read_text(encoding="utf-8")) or {}
    override = yaml.safe_load(Path(model_yaml).read_text(encoding="utf-8")) or {}
    env = yaml.safe_load(Path(env_yaml).read_text(encoding="utf-8")) or {}
    merged = _merge(base, override)

    # sim_vla's observation contract wins over the simulator env's, because the
    # collected dataset is what the encoder will actually be handed.
    keys = dict(OBSERVATION_KEYS) | dict((sim_cfg.get("observation") or {}))
    for block in ("encoder", "decoder"):
        merged.setdefault(block, {})
        merged[block] = dict(merged[block]) | keys

    # progress.mode uses ${oc.select:...}, which is Hydra's and not resolvable
    # here. sim_vla sets it from its own config or drops the key.
    progress = dict(merged.get("progress") or {})
    if isinstance(progress.get("mode"), str) and progress["mode"].startswith("${oc."):
        progress["mode"] = str(((sim_cfg.get("model") or {}).get("progress")
                                or {}).get("mode", "task_schedule"))
        merged["progress"] = progress

    globals_root = {
        "device": str(sim_cfg.get("device", "cuda")),
        "seed": int(sim_cfg.get("seed", (sim_cfg.get("data") or {}).get("seed", 0))),
    }
    merged = _resolve(merged, merged, env, globals_root=globals_root)

    graph_cfg = dict((sim_cfg.get("model") or {}).get("graph") or {})
    merged.setdefault("graph", {})
    merged["graph"] = dict(merged["graph"]) | {
        "enabled": bool(graph_cfg.get("enabled", False)),
        "n_max": int(graph_cfg.get("n_max", merged["graph"].get("n_max", 8))),
        "e_max": int(graph_cfg.get("e_max", merged["graph"].get("e_max", 168))),
    }
    merged["device"] = str(sim_cfg.get("device", merged.get("device", "cuda")))
    return Node(merged)
