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

Whichever policy runs, this is the stage that decides what a training run
loads, so it is also where runtime readiness is enforced: every physical
member that survives has to carry a usable end-effector height family. Raw
evidence is deliberately wider than that -- it keeps the sofa the arm brushed
past, which no family rule reaches and which ``target-supporters`` removes
anyway -- and one that survives the policy stops the prune. It is never
dropped to make the error go away: deleting a member the policy admitted would
change what the graph contains in order to silence a check on it.

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
from typing import Dict, List, Tuple

from scenegraph.core import families as families_rules
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

    # A declared site is task geometry, not mined membership. The policy
    # filters what the *evidence* admitted -- a site supports nothing and
    # touches nothing, so both policies would drop it, and dropping it removes
    # the vocabulary row its runtime node needs: the site then encodes as
    # padding and every fact naming it disappears. MS-HAB Pick is scored on
    # ``reached(ee, spatial:ee_rest_site)``, so that is the whole schedule.
    keep |= set(raw.get("sites") or {})

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

    # Runtime readiness, checked on what survived rather than on what the
    # evidence held. Raw membership is deliberately wider: it keeps the sofa
    # the arm brushed past, which no height-family rule reaches. That member
    # is not a defect in the evidence, and it is gone by this line -- but if
    # one is still here, this file is about to become an asset a run loads,
    # and a member with no scale would be labelled on another family's
    # deadband. Refused rather than dropped: silently deleting a member the
    # policy admitted would change the graph's membership to make an error go
    # away.
    blockers = families_rules.runtime_blockers(out_members)
    if blockers:
        detail = "; ".join(f"{key}: {reason}"
                           for key, reason in sorted(blockers.items()))
        raise ValueError(
            f"{len(blockers)} member(s) admitted by the {policy!r} policy "
            f"cannot be measured -- {detail}"
        )

    out = dict(raw)
    out["members"] = out_members
    out["membership_policy"] = policy
    out.pop("_unresolved_members", None)
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
        # Every payload before any file, for the same reason the miner stages
        # its writes: a prune that fails halfway leaves a directory holding
        # some of the targets and looking like all of them.
        staged: List[Tuple[Path, Dict, int]] = []
        for path in sources:
            raw = json.loads(path.read_text())
            group = str(raw.get("task_group") or "")
            if args.task_group and group != args.task_group:
                print(f"[prune] FAILED {path.name}: records task group "
                      f"{group or '<none>'!r}, expected "
                      f"{args.task_group!r}", file=sys.stderr)
                return 1
            try:
                payload = prune_payload(raw, args.policy)
            except ValueError as exc:
                print(f"[prune] FAILED {path.name}: {exc}", file=sys.stderr)
                print("        A runtime whitelist is what a training run "
                      "loads, so a member it cannot measure stops the prune "
                      "rather than being quietly dropped. The raw evidence is "
                      "untouched; re-mine or fix the affordance asset.",
                      file=sys.stderr)
                return 1
            staged.append(
                (out_dir / path.name, payload, len(raw.get("members") or {})))

        for out_path, payload, seen in staged:
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            print(f"[prune] {out_path.name}: {seen} -> "
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
