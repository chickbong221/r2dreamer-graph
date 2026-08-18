"""Shorten raw mined whitelists into the runtime assets, without recollecting.

Reads ``subtask_whitelists_raw/<task_group>/`` and writes
``subtask_whitelists/<task_group>/``. The raw files are the expensive artefact
-- hours of GPU rollouts each -- so this stage never writes back into them, and
refuses an ``--out-dir`` that resolves to the input.

Separating pruning from mining is the point: trying a different membership rule
costs one re-prune instead of another collection run, and the evidence that a
rule discarded stays on disk to argue with.

Policies:

``target-supporters`` (default)
    The target plus whatever directly supports it. A rollout contacts whatever
    is in the way, so keeping every contacted entity fills a pick-the-bowl
    graph with the groceries the arm brushed past. Direct support is what makes
    an entity part of the task: it is what the target rests on and must be
    lifted off. Support is never expanded recursively.

``full-evidence``
    Copy membership through unchanged. Useful to measure what the runtime gate
    is actually costing before committing to a narrower rule.

Bin statistics are carried over untouched -- they are mined from per-rollout
value samples, not from membership -- so pruning never moves a relation bin.
The per-group ``<subtask>_all.json`` union is rebuilt from the pruned files at
the end, because that is the file the runtime reads its bins from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from scenegraph.tools.build_subtask_whitelists import (
    MEMBERSHIP_FULL_EVIDENCE,
    MEMBERSHIP_POLICIES,
    MEMBERSHIP_TARGET_SUPPORTERS,
)
from scenegraph.tools.build_union_whitelist import UNION_TARGET, merge


def _sources(raw_dir: Path, subtask: str) -> List[Path]:
    """Per-target raw files for one subtask, excluding the union."""
    return sorted(
        p for p in raw_dir.glob(f"{subtask}_*.json")
        if p.name != f"{subtask}_{UNION_TARGET}.json"
    )


def prune_payload(raw: Dict, policy: str) -> Dict:
    """Apply ``policy`` to one raw whitelist payload.

    Mirrors ``_WhitelistBuilder._admitted`` plus its support-reference
    filtering, so a pruned file is indistinguishable from one the miner would
    have written under the same policy.
    """
    if policy not in MEMBERSHIP_POLICIES:
        raise ValueError(
            f"unknown policy {policy!r}; have {list(MEMBERSHIP_POLICIES)}")

    members = dict(raw.get("members") or {})
    target = str(raw.get("target") or "")

    if policy == MEMBERSHIP_FULL_EVIDENCE:
        keep = set(members)
    else:
        keep = {target} if target else set()
        for key, entry in members.items():
            if target and target in (entry.get("supports") or ()):
                keep.add(key)

    out_members: Dict[str, Dict] = {}
    for key in sorted(keep):
        entry = members.get(key)
        if entry is None:
            continue
        entry = dict(entry)
        # No member may point at a key the pruned file does not contain.
        supports = sorted(set(entry.get("supports") or ()) & keep)
        if supports:
            entry["supports"] = supports
        else:
            entry.pop("supports", None)
        out_members[key] = entry

    out = dict(raw)
    out["members"] = out_members
    out["membership_policy"] = policy
    out["_pruned_from"] = {
        "membership_policy": str(raw.get("membership_policy") or ""),
        "members": len(members),
    }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--raw-dir", required=True,
        help="subtask_whitelists_raw/<task_group>/ to read.",
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="subtask_whitelists/<task_group>/ to write. Must differ from "
             "--raw-dir: raw mining output is never overwritten.",
    )
    parser.add_argument("--subtask", nargs="+", default=["pick"])
    parser.add_argument(
        "--policy", default=MEMBERSHIP_TARGET_SUPPORTERS,
        choices=list(MEMBERSHIP_POLICIES),
    )
    parser.add_argument(
        "--task-group", default=None,
        help="Refuse to prune files recording a different group.",
    )
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not raw_dir.is_dir():
        print(f"[prune] raw dir not found: {raw_dir}", file=sys.stderr)
        return 2
    if out_dir == raw_dir:
        print("[prune] --out-dir must differ from --raw-dir; raw mining output "
              "is the expensive artefact and is never overwritten",
              file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for subtask in args.subtask:
        sources = _sources(raw_dir, subtask)
        if not sources:
            print(f"[prune] no {subtask}_*.json under {raw_dir}",
                  file=sys.stderr)
            continue
        for path in sources:
            raw = json.loads(path.read_text())
            group = str(raw.get("task_group") or "")
            if args.task_group and group != args.task_group:
                print(f"[prune] FAILED {path.name}: records task group "
                      f"{group or '<none>'!r}, expected "
                      f"{args.task_group!r}", file=sys.stderr)
                return 1
            payload = prune_payload(raw, args.policy)
            (out_dir / path.name).write_text(
                json.dumps(payload, indent=2, sort_keys=True))
            print(f"[prune] {path.name}: {len(raw.get('members') or {})} -> "
                  f"{len(payload['members'])} members")
            written += 1

        # The runtime reads its relation bins from the union, so it has to
        # follow the pruned files rather than the raw ones.
        try:
            data = merge(out_dir, subtask)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[prune] FAILED union for {subtask}: {exc}", file=sys.stderr)
            return 1
        union = out_dir / f"{subtask}_{UNION_TARGET}.json"
        union.write_text(json.dumps(data, indent=2, sort_keys=True))
        print(f"[prune] wrote {union.name}: {len(data['members'])} members")

    if not written:
        return 2
    print(f"[prune] {written} whitelist(s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
