"""Check a collected dataset before training on it, from its own metadata.

The dataset on the server was written by whichever revision of the collector
was checked out at the time, and the sidecar is the only reliable statement of
what that produced. So every check here reads the recorded contract -- the
camera keys, the proprioception column names, the controller, the field kinds --
and verifies the arrays against *that*, rather than against what this file
believes the collector does.

It reports; it does not repair, and it never re-collects. A dataset that fails
a check is a decision for whoever collected it.

    python -m sim_vla.data.audit --root data/sim_vla_demos
    python -m sim_vla.data.audit --dataset data/sim_vla_demos/PickCube-v1/demos.h5

Exit status is 1 if any check failed, so it can gate a training launch.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from scenegraph.adapters.graph_pack import GRAPH_KEYS

from .dataset import DIAGNOSTIC_FIELDS, SUPERVISION_FIELDS


@dataclass
class Report:
    """What one dataset's audit found."""

    dataset: str
    passed: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        (self.passed if ok else self.failures).append(
            name if ok or not detail else f"{name}: {detail}")
        return ok

    @property
    def ok(self) -> bool:
        return not self.failures

    def render(self) -> str:
        lines = [f"[audit] {self.dataset}: "
                 f"{len(self.passed)} passed, {len(self.failures)} failed"]
        for key, value in self.stats.items():
            lines.append(f"[audit]   {key}: {value}")
        for note in self.notes:
            lines.append(f"[audit]   note: {note}")
        for failure in self.failures:
            lines.append(f"[audit]   FAIL {failure}")
        return "\n".join(lines)


def _tree_lengths(node, prefix: str = "") -> Dict[str, int]:
    import h5py

    out: Dict[str, int] = {}
    for key in node:
        child = node[key]
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, h5py.Group):
            out |= _tree_lengths(child, name)
        else:
            out[name] = int(child.shape[0])
    return out


def audit_dataset(path: str | Path, *, graph_required: bool = False,
                  sample: int = 0) -> Report:
    """Every check the spec asks for, against the recorded contract."""
    import h5py

    path = Path(path)
    report = Report(dataset=str(path))
    sidecar = path.with_suffix(".json")
    if not report.check("sidecar exists", sidecar.exists(), str(sidecar)):
        return report

    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    meta: Mapping[str, Any] = payload.get("metadata") or {}
    entries: Sequence[Mapping[str, Any]] = payload.get("episodes") or []
    report.stats["episodes_in_sidecar"] = len(entries)

    camera_keys = sorted(str(v) for v in (meta.get("camera_keys") or {}).values())
    proprio_names = list(meta.get("proprio_names") or [])
    controller = dict(meta.get("controller") or {})
    field_kinds = dict(meta.get("field_kinds") or {})
    budget = dict(meta.get("budget") or {})
    image_size = list(meta.get("image_size") or [])

    report.check("metadata names cameras", bool(camera_keys))
    report.check("metadata names proprio columns", bool(proprio_names))
    report.check("metadata records the controller",
                 bool(controller.get("control_mode")) and
                 bool(controller.get("action_dim")))
    report.check("metadata declares field kinds", bool(field_kinds))
    report.check("controller is pd_joint_pos",
                 controller.get("control_mode") == "pd_joint_pos",
                 str(controller.get("control_mode")))
    report.check("control frequency recorded",
                 int(controller.get("control_freq") or 0) > 0)
    report.stats["control"] = (
        f"{controller.get('control_mode')} dim={controller.get('action_dim')} "
        f"@{controller.get('control_freq')}Hz")
    report.stats["cameras"] = f"{camera_keys} at {image_size}"
    report.stats["proprio_dim"] = len(proprio_names)

    graph_meta = dict(meta.get("graph") or {})
    if graph_required:
        report.check("graph vocabulary recorded",
                     bool(graph_meta.get("relation_tokens")))
        report.check("graph capacities recorded",
                     bool(graph_meta.get("n_max")) and bool(graph_meta.get("e_max")))
        report.check("whitelist hashed",
                     graph_meta.get("whitelist_digest") not in (None, "", "missing"))
        report.check("thresholds hashed",
                     graph_meta.get("thresholds_digest") not in (None, "", "default"))

    with h5py.File(path, "r") as handle:
        groups = [key for key in handle if key.startswith("traj_")]
        report.stats["episodes_in_file"] = len(groups)
        report.check("sidecar matches file", len(groups) == len(entries),
                     f"{len(groups)} groups vs {len(entries)} entries")

        incomplete = [key for key in groups
                      if not bool(handle[key].attrs.get("complete", False))]
        report.check("all episodes marked complete", not incomplete,
                     f"{len(incomplete)} without the marker")

        seeds = [entry.get("seed") for entry in entries
                 if entry.get("seed") is not None]
        report.check("seeds are distinct", len(set(seeds)) == len(seeds),
                     f"{len(seeds) - len(set(seeds))} repeats")
        report.stats["distinct_seeds"] = len(set(seeds))

        chosen = groups if not sample else groups[: int(sample)]
        lengths, settled, terminal_at_end, internal_truncation = [], [], 0, []
        for key in chosen:
            group = handle[key]
            steps = int(group["actions"].shape[0])
            lengths.append(steps)

            expected = {name: steps for name in SUPERVISION_FIELDS}
            actual = {name: int(group[name].shape[0])
                      for name in SUPERVISION_FIELDS if name in group}
            if actual != {k: v for k, v in expected.items() if k in actual}:
                report.check(f"{key} transition lengths", False, str(actual))

            obs_lengths = _tree_lengths(group["obs"])
            bad_obs = {k: v for k, v in obs_lengths.items() if v != steps + 1}
            if bad_obs:
                report.check(f"{key} observation lengths", False, str(bad_obs))

            for name in DIAGNOSTIC_FIELDS:
                if name not in group:
                    continue
                bad = {k: v for k, v in _tree_lengths(group[name]).items()
                       if v != steps + 1}
                if bad:
                    report.check(f"{key} {name} lengths", False, str(bad))

            missing_cams = [c for c in camera_keys if c not in group["obs"]]
            if missing_cams:
                report.check(f"{key} cameras", False, str(missing_cams))
            elif camera_keys:
                shape = tuple(group["obs"][camera_keys[0]].shape[1:3])
                if image_size and list(shape) != image_size:
                    report.check(f"{key} image shape", False,
                                 f"{shape} vs metadata {image_size}")

            if "proprio" in group["obs"]:
                width = int(group["obs"]["proprio"].shape[1])
                if proprio_names and width != len(proprio_names):
                    report.check(f"{key} proprio width", False,
                                 f"{width} vs {len(proprio_names)} names")

            if graph_required:
                absent = [k for k in GRAPH_KEYS if k not in group["obs"]]
                if absent:
                    report.check(f"{key} graph arrays", False, str(absent))

            terminated = np.asarray(group["terminated"][()], dtype=bool)
            truncated = np.asarray(group["truncated"][()], dtype=bool)
            success = np.asarray(group["success"][()], dtype=bool)
            terminal_at_end += int(terminated[-1]) if terminated.size else 0
            # A truncation before the final step is a boundary inside the
            # episode, and a loader that honours boundaries would split the
            # demonstration there.
            if truncated[:-1].any():
                internal_truncation.append(key)
            if success.size:
                onset = np.flatnonzero(~success)
                settled.append(int(onset[-1]) + 2 if onset.size else 1)

        report.check("no truncation inside an episode", not internal_truncation,
                     f"{len(internal_truncation)} episodes")
        report.stats["steps"] = (
            f"min={min(lengths)} med={sorted(lengths)[len(lengths) // 2]} "
            f"max={max(lengths)}" if lengths else "none")
        if settled:
            report.stats["settled"] = (
                f"min={min(settled)} med={sorted(settled)[len(settled) // 2]} "
                f"max={max(settled)}")
        report.stats["ends_terminal"] = f"{terminal_at_end}/{len(chosen)}"

        horizon = int(budget.get("max_steps_to_success") or 0)
        if horizon and lengths:
            over = [n for n in lengths if n > horizon]
            report.check("no episode exceeds the budget", not over,
                         f"{len(over)} over {horizon}")

        pad = int(budget.get("pad_after_success") or 0)
        if pad and terminal_at_end == len(chosen):
            report.notes.append(
                f"every episode ends on a terminal step with pad={pad} "
                "recorded after success settled; the loader drops the "
                "post-terminal tail")

    reasons: Dict[str, int] = {}
    for entry in entries:
        reasons[str(entry.get("end_reason") or "?")] = 1 + reasons.get(
            str(entry.get("end_reason") or "?"), 0)
    report.stats["end_reasons"] = reasons
    collection = dict(meta.get("collection") or {})
    if collection:
        report.stats["selection"] = (
            f"{collection.get('attempts')} attempts, "
            f"solve {collection.get('solve_rate', 0):.0%}, "
            f"budget yield {collection.get('budget_yield', 0):.0%}")
    else:
        report.notes.append("no collection summary recorded for this dataset")
    return report


def find_datasets(root: Path, name: str = "demos.h5") -> List[Path]:
    return sorted(p for p in root.glob(f"*/{name}") if p.is_file())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Check collected demonstrations against their own metadata")
    parser.add_argument("--root", default="data/sim_vla_demos")
    parser.add_argument("--dataset", default="",
                        help="one dataset instead of every one under --root")
    parser.add_argument("--name", default="demos.h5")
    parser.add_argument("--graph", action="store_true",
                        help="also require what a graph-enabled run needs")
    parser.add_argument("--sample", type=int, default=0,
                        help="episodes to open; 0 opens all of them")
    parser.add_argument("--out", default="",
                        help="write the reports as JSON here")
    args = parser.parse_args(argv)

    targets = ([Path(args.dataset)] if args.dataset
               else find_datasets(Path(args.root), args.name))
    if not targets:
        raise SystemExit(f"no datasets found under {args.root}")

    reports = [audit_dataset(path, graph_required=args.graph, sample=args.sample)
               for path in targets]
    for report in reports:
        print(report.render(), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(
            [{"dataset": r.dataset, "ok": r.ok, "passed": r.passed,
              "failures": r.failures, "notes": r.notes, "stats": r.stats}
             for r in reports], indent=2, default=str), encoding="utf-8")
        print(f"\n[audit] wrote {args.out}", flush=True)

    failed = [r.dataset for r in reports if not r.ok]
    print(f"\n[audit] {len(reports) - len(failed)}/{len(reports)} datasets passed",
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
