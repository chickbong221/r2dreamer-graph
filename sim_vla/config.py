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
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

CONFIG_ROOT = Path(__file__).resolve().parent / "configs"

# Settings the online actor-critic no longer has, and why. A stale config or a
# copied override would otherwise resolve, run, and be a different experiment
# from the one its file describes.
REMOVED_ONLINE = {
    "actor_objective":
        "there is one online objective now -- the pathwise return of an "
        "executed chunk -- so there is nothing to select",
    "flow_noise_std":
        "the stochastic flow sampler went with flow_reinforce; collection and "
        "imagination both use the deterministic sampler",
    "flow_noise_schedule":
        "the stochastic flow sampler went with flow_reinforce",
    "actor_transition_microbatch":
        "no flow transition is scored any more; imagination_microbatch bounds "
        "the actor's memory",
    "imagination_batch":
        "imagination starts from every eligible replay state, so there is no "
        "cap to set; imagination_microbatch is the memory control",
    "imag_horizon":
        "an imagined rollout is exactly actor.execute transitions of one "
        "generated chunk",
    "anchor_window_microbatch":
        "the anchor's windows are encoded together; anchor_windows bounds them",
    "anchor_retries":
        "a draw with no eligible anchor row fails the update instead",
    "advantage_scale":
        "the pathwise objective maximises the return itself; no advantage is "
        "formed or normalized",
    "eval_sampler":
        "there is one sampler, so the evaluation already runs the policy "
        "being trained",
    "lam":
        "the executed-chunk return is the lambda = 1 case and is not tunable",
}


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
    pretrain = cfg.get("pretrain") or {}
    for key in ("world_lr", "imitation_lr"):
        value = pretrain.get(key)
        if value is not None and not (math.isfinite(float(value))
                                      and float(value) > 0):
            raise SystemExit(f"pretrain.{key} must be positive when set")

    online = cfg.get("online") or {}
    removed = sorted(key for key in REMOVED_ONLINE if key in online)
    if removed:
        detail = "\n".join(f"  - online.{key}: {REMOVED_ONLINE[key]}"
                           for key in removed)
        raise SystemExit(
            "these online settings no longer exist and would be read by "
            "nothing:\n" + detail + "\nRemove them rather than leaving a "
            "config that describes a different experiment from the one it "
            "runs.")
    ratio = float(online.get("train_ratio", 64))
    if not math.isfinite(ratio) or ratio < 0:
        raise SystemExit("online.train_ratio must be finite and nonnegative")
    if int(online.get("imagination_microbatch", 0)) < 0:
        raise SystemExit("online.imagination_microbatch must be nonnegative")
    if online.get("precision", "bfloat16") not in ("float32", "bfloat16"):
        raise SystemExit("online.precision must be float32 or bfloat16")
    if int(online.get("num_envs", 1) or 0) < 1:
        raise SystemExit("online.num_envs must be at least 1")
    if (online.get("reconfiguration_freq") is not None
            and int(online["reconfiguration_freq"]) < 0):
        raise SystemExit("online.reconfiguration_freq must be nonnegative "
                         "when set")
    if int(online.get("num_envs", 1)) > 1 and float(online.get(
            "train_ratio", 64)) == 0:
        raise SystemExit(
            "online.num_envs > 1 needs a nonzero online.train_ratio: parallel "
            "envs train between steps, and train_ratio=0 is the legacy "
            "per-collection schedule")

    if int(online.get("critic_warmup", 0)) < 0:
        raise SystemExit("online.critic_warmup must be nonnegative")
    anchor = online.get("demo_anchor", 0.0)
    if not (math.isfinite(float(anchor or 0.0)) and float(anchor or 0.0) >= 0):
        raise SystemExit("online.demo_anchor must be finite and >= 0")
    for key in ("anchor_windows", "anchor_rows"):
        if int(online.get(key, 1)) < 1:
            raise SystemExit(f"online.{key} must be at least 1")
    for key in ("anchor_microbatch", "grad_report_every"):
        if int(online.get(key, 0)) < 0:
            raise SystemExit(f"online.{key} must be nonnegative")
    start = online.get("progress_warmup_start")
    end = online.get("progress_warmup_end")
    if (start is None) != (end is None):
        raise SystemExit("online.progress_warmup_start and "
                         "online.progress_warmup_end are set together or not "
                         "at all")
    if start is not None and not 0 <= int(start) < int(end):
        raise SystemExit(f"online.progress_warmup_start={start} must be >= 0 "
                         f"and below online.progress_warmup_end={end}")
    for key in ("actor_lr", "world_lr", "critic_lr", "progress_lr"):
        value = online.get(key)
        if value is not None and not (math.isfinite(float(value))
                                      and float(value) > 0):
            raise SystemExit(f"online.{key} must be positive when set")

    actor = cfg.get("actor") or {}
    # One number decides how many actions a generated chunk contributes, in
    # the environment and in imagination. chunk_size is checked against the
    # loaded checkpoint when the actor is built; this is the part that can be
    # checked from the config alone.
    execute = actor.get("execute", 1)
    if int(execute) < 1:
        raise SystemExit(
            f"actor.execute={execute!r} must be at least 1: a chunk has to "
            "contribute at least one executed action")
    chunk = int(actor.get("chunk_size") or 0)
    if chunk and int(execute) > chunk:
        raise SystemExit(
            f"actor.execute={execute} exceeds actor.chunk_size={chunk}; "
            "executing more actions than the policy predicts would repeat or "
            "invent commands")
    online_chunk = online.get("chunk_size")
    if online_chunk is not None:
        if int(online_chunk) < int(execute):
            raise SystemExit(
                f"online.chunk_size={online_chunk} is below actor.execute="
                f"{execute}; a chunk has to supply every executed action")
        if chunk and int(online_chunk) > chunk:
            raise SystemExit(
                f"online.chunk_size={online_chunk} exceeds actor.chunk_size="
                f"{chunk}; positions past the length Stage 1B imitated were "
                "never supervised")
    if int(actor.get("flow_steps") or 0) < 0:
        raise SystemExit("actor.flow_steps must be nonnegative "
                         "(0 takes the checkpoint's own value)")
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
