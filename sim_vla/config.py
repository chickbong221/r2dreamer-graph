"""Merge base + task + experiment, and refuse the combinations that lie.

Three yaml files describe a run: the shared settings, the task, and the arm.
They are merged in that order, so an experiment file needs to carry only the
switch that makes it an experiment.

Two rules are enforced here rather than left to fail later.

``progress.enabled`` requires ``graph.enabled``. Progress supervision is
derived from the graph schedule; with no graph there is nothing to derive it
from, and the existing trainer already refuses this pairing.

The graph capacities in the config must match the dataset's recorded ones. A
dataset packed at ``e_max=168`` and a model built for 256 agree on every array
shape they can cheaply check, and disagree about which facts were dropped when
the packer ran out of rows.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

CONFIG_ROOT = Path(__file__).resolve().parent / "configs"


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """``override`` wins per leaf, not per block."""
    out = dict(copy.deepcopy(dict(base)))
    for key, value in (override or {}).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read(path: Path) -> Dict[str, Any]:
    import yaml

    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def _interpolate(node: Any, root: Mapping[str, Any]) -> Any:
    """Resolve ``${a.b}`` against the merged tree.

    Only the one form the config files use. A general resolver would be a
    second implementation of Hydra.
    """
    if isinstance(node, Mapping):
        return {k: _interpolate(v, root) for k, v in node.items()}
    if isinstance(node, list):
        return [_interpolate(v, root) for v in node]
    if not isinstance(node, str) or "${" not in node:
        return node
    out = node
    while "${" in out:
        start = out.index("${")
        end = out.index("}", start)
        dotted = out[start + 2:end]
        value: Any = root
        for part in dotted.split("."):
            value = (value or {}).get(part) if isinstance(value, Mapping) else None
        if value is None:
            raise KeyError(f"cannot resolve ${{{dotted}}}")
        out = out[:start] + str(value) + out[end + 1:]
    return out


def load_config(task: str, experiment: str,
                overrides: Optional[Mapping[str, Any]] = None,
                root: Path = CONFIG_ROOT) -> Dict[str, Any]:
    """The resolved settings for one arm on one task."""
    merged = _read(root / "base.yaml")
    merged = deep_merge(merged, _read(root / "tasks" / f"{task}.yaml"))
    merged = deep_merge(merged, _read(root / "experiments" / f"{experiment}.yaml"))
    merged = deep_merge(merged, overrides or {})
    merged = _interpolate(merged, merged)
    merged["experiment"] = {"task": task, "arm": experiment}
    validate(merged)
    return merged


def validate(cfg: Mapping[str, Any]) -> None:
    actor = cfg.get("actor") or {}
    revision = actor.get("revision", "")
    # A commit hash of only decimal digits is a valid hash and an integer in
    # YAML, and PyYAML resolves it to one. It then reaches the hub as a number
    # whose leading zeros are gone, and the error is about a revision that does
    # not exist rather than about quoting.
    if revision not in ("", None) and not isinstance(revision, str):
        raise SystemExit(
            f"actor.revision parsed as {type(revision).__name__} "
            f"({revision}), not a string. Quote it in the config: "
            f'actor.revision: "{revision}"')

    graph = ((cfg.get("model") or {}).get("graph") or {})
    progress = ((cfg.get("model") or {}).get("progress") or {})
    if progress.get("enabled") and not graph.get("enabled"):
        raise SystemExit(
            "model.progress.enabled requires model.graph.enabled: progress "
            "targets are derived from the graph schedule, and there is no "
            "schedule without a graph.")


def check_dataset_compatibility(cfg: Mapping[str, Any],
                                metadata: Mapping[str, Any]) -> None:
    """Refuse a dataset this arm cannot be trained on as configured."""
    graph = ((cfg.get("model") or {}).get("graph") or {})
    if not graph.get("enabled"):
        # A baseline needs nothing from the graph metadata, including its
        # absence: a dataset with graphs is a fine baseline dataset.
        return
    recorded = dict(metadata.get("graph") or {})
    if not recorded.get("relation_tokens"):
        raise SystemExit(
            "model.graph.enabled=true but this dataset records no graph "
            "vocabulary; it cannot train the graph arm.")
    for key in ("n_max", "e_max", "n_cams"):
        wanted, got = graph.get(key), recorded.get(key)
        # 0 is the "take it from the dataset" sentinel, not a capacity of zero.
        if wanted in (None, 0) or got is None:
            continue
        if int(wanted) != int(got):
            raise SystemExit(
                f"graph.{key} is {wanted} but the dataset was packed at {got}. "
                "The arrays would load and describe a different capacity.")
