"""Mine phase-schedule candidates from successful-episode traces.

Reads the per-episode records the collector writes (interaction spans plus the
environment's own predicate spans) and exports one evidence bundle per task for
a single offline refinement pass. Nothing here decides a schedule: ambiguity is
measured and exported, never resolved by a tie-break.

Two gates, both with the exact counts alongside:

* a relation is a milestone candidate when it appears in ``--min-presence`` of
  successful episodes (default 0.99, not 1.0 -- one flicker of ``eps_force`` in
  one episode should not delete a real milestone);
* an ordering constraint holds when one direction wins ``--min-order`` of the
  episodes containing both (default 0.95).

The environment's predicates are reported beside the detector's milestones and
never overwrite them. Which predicate corresponds to which relation is measured
by span agreement rather than assumed from its name, so no task registry is
needed here either.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from scenegraph.adapters.interaction_events import EE_KEY
from scenegraph.tools.build_maniskill_assets import load_shards

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

BUNDLE_SCHEMA_VERSION = 1
MILESTONE_PRESENCE = 0.99
ORDER_CONFIDENCE = 0.95
# Milestones true from the first frame in nearly every episode describe the
# initial scene, not something the robot achieved.
INITIAL_CONDITION_RATE = 0.95
# Spatial ladders that give a phase dense structure between its endpoints.
SPATIAL_RELATIONS = ("planar-distance", "height-offset")


Milestone = Tuple[str, str, str]


def _key(entry: Sequence) -> Milestone:
    return (str(entry[0]), str(entry[1]), str(entry[2]))


def _spans(record: Dict[str, Any]) -> Dict[Milestone, Tuple[int, int]]:
    """First span per milestone in one episode."""
    out: Dict[Milestone, Tuple[int, int]] = {}
    for entry in record.get("interactions") or ():
        out[_key(entry)] = (int(entry[3]), int(entry[4]))
    return out


# --------------------------------------------------------------------------- #
# presence
# --------------------------------------------------------------------------- #
def milestone_stats(traces: List[Dict[str, Any]],
                    min_presence: float) -> Dict[str, Any]:
    """Per-milestone episode presence, with the episodes that lacked it.

    The missing indices are exported rather than just the rate: a milestone at
    99.8% is either detector noise or a real branch in the task, and only the
    episodes themselves distinguish those.
    """
    total = len(traces)
    seen: Dict[Milestone, List[int]] = defaultdict(list)
    onsets: Dict[Milestone, List[int]] = defaultdict(list)
    releases: Dict[Milestone, List[int]] = defaultdict(list)
    at_zero: Dict[Milestone, int] = defaultdict(int)
    for index, record in enumerate(traces):
        for milestone, (on, off) in _spans(record).items():
            seen[milestone].append(index)
            onsets[milestone].append(on)
            releases[milestone].append(off)
            if on == 0:
                at_zero[milestone] += 1

    out: Dict[str, Any] = {}
    for milestone, episodes in sorted(seen.items()):
        count = len(episodes)
        rate = count / total if total else 0.0
        present = set(episodes)
        missing = [i for i in range(total) if i not in present]
        initial = at_zero[milestone] / count if count else 0.0
        out[" / ".join(milestone)] = {
            "relation": milestone[0], "src": milestone[1], "dst": milestone[2],
            "episodes": count, "of": total, "rate": rate,
            "is_candidate": rate >= min_presence,
            # Truncated: 19 missing episodes out of 1900 is a list worth
            # reading, 1600 is not.
            "missing_episodes": missing[:32],
            "n_missing": len(missing),
            "onset": _summary(onsets[milestone]),
            "release": _summary(releases[milestone]),
            "starts_at_frame_zero_rate": initial,
            "initial_condition": initial >= INITIAL_CONDITION_RATE,
        }
    return out


def _summary(values: Sequence[int]) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()), "max": float(arr.max()),
    }


# --------------------------------------------------------------------------- #
# ordering
# --------------------------------------------------------------------------- #
def ordering_stats(traces: List[Dict[str, Any]],
                   candidates: Sequence[Milestone],
                   min_order: float) -> List[Dict[str, Any]]:
    """Pairwise onset ordering over the episodes containing both.

    Simultaneous onsets are counted as ties and kept in the denominator: two
    milestones that always start on the same frame have no order, and folding
    ties into one side would invent one.
    """
    out: List[Dict[str, Any]] = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            ab = ba = tie = 0
            for record in traces:
                spans = _spans(record)
                if a not in spans or b not in spans:
                    continue
                on_a, on_b = spans[a][0], spans[b][0]
                if on_a < on_b:
                    ab += 1
                elif on_b < on_a:
                    ba += 1
                else:
                    tie += 1
            comparable = ab + ba + tie
            if not comparable:
                continue
            rate_ab, rate_ba = ab / comparable, ba / comparable
            rate_tie = tie / comparable
            if rate_ab >= min_order:
                verdict = "before"
            elif rate_ba >= min_order:
                verdict = "after"
            elif rate_tie >= min_order:
                # Not the same as ambiguous, and the difference decides a
                # phase boundary: these two always begin together, so they
                # belong to one phase. Ambiguous means the order genuinely
                # varies between episodes.
                verdict = "simultaneous"
            else:
                verdict = "ambiguous"
            out.append({
                "a": " / ".join(a), "b": " / ".join(b),
                "a_before_b": ab, "b_before_a": ba, "same_frame": tie,
                "comparable": comparable,
                "rate_a_before_b": rate_ab, "rate_b_before_a": rate_ba,
                "rate_same_frame": rate_tie, "verdict": verdict,
            })
    return out


# --------------------------------------------------------------------------- #
# phases
# --------------------------------------------------------------------------- #
def _pair(milestone: Milestone) -> Tuple[str, str]:
    _, src, dst = milestone
    if src == EE_KEY or dst == EE_KEY:
        return (EE_KEY, dst if src == EE_KEY else src)
    return tuple(sorted((src, dst)))  # type: ignore[return-value]


def propose_phases(traces: List[Dict[str, Any]],
                   stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Group candidate milestones by the pair they act on, ordered by onset.

    A phase boundary is a change of the relevant pair -- 'the gripper works on
    A' then 'A works on B'. Milestones true from frame zero are grouped too but
    flagged: they describe the scene the episode started in.
    """
    grouped: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for name, entry in stats.items():
        if not entry["is_candidate"]:
            continue
        grouped[_pair((entry["relation"], entry["src"], entry["dst"]))].append(name)

    phases = []
    for pair, names in grouped.items():
        onsets = [stats[n]["onset"].get("median", 0.0) for n in names]
        phases.append({
            "pair": list(pair),
            "milestones": sorted(names),
            "median_onset": float(np.median(onsets)) if onsets else 0.0,
            "all_initial_conditions": all(
                stats[n]["initial_condition"] for n in names),
        })
    phases.sort(key=lambda p: p["median_onset"])
    for index, phase in enumerate(phases):
        phase["index"] = index
    return phases


# --------------------------------------------------------------------------- #
# environment predicates
# --------------------------------------------------------------------------- #
def predicate_stats(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-predicate presence and shape, from the environment's own ``info``."""
    total = len(traces)
    count: Dict[str, int] = defaultdict(int)
    onsets: Dict[str, List[int]] = defaultdict(list)
    releases: Dict[str, List[int]] = defaultdict(list)
    runs: Dict[str, List[int]] = defaultdict(list)
    for record in traces:
        for key, spans in (record.get("predicates") or {}).items():
            if not spans:
                continue
            count[key] += 1
            onsets[key].append(int(spans[0][0]))
            releases[key].append(int(spans[-1][1]))
            runs[key].append(len(spans))
    out = {}
    for key in sorted(count):
        n_runs = np.asarray(runs[key], dtype=float)
        out[key] = {
            "episodes": count[key], "of": total,
            "rate": count[key] / total if total else 0.0,
            "first_onset": _summary(onsets[key]),
            "last_release": _summary(releases[key]),
            "median_runs": float(np.median(n_runs)),
            # More than one run means the predicate went false and true again,
            # which is what "this phase can be undone" looks like.
            "toggles": bool(n_runs.max() > 1),
        }
    return out


def scalar_stats(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Continuous ``info`` signals, summarised. These are candidate dense
    ladders inside a phase where the relation bins are too coarse."""
    first: Dict[str, List[float]] = defaultdict(list)
    last: Dict[str, List[float]] = defaultdict(list)
    count: Dict[str, int] = defaultdict(int)
    for record in traces:
        for key, series in (record.get("scalars") or {}).items():
            values = np.asarray(series[1], dtype=float)
            if not values.size:
                continue
            count[key] += 1
            first[key].append(float(values[0]))
            last[key].append(float(values[-1]))
    total = len(traces)
    return {
        key: {
            "episodes": count[key], "of": total,
            "median_first": float(np.median(first[key])),
            "median_last": float(np.median(last[key])),
            # A signal that reliably moves one way over a successful episode is
            # a progress signal; one that does not is bookkeeping.
            "decreases": float(np.median(last[key])) < float(np.median(first[key])),
        }
        for key in sorted(count)
    }


def detector_agreement(traces: List[Dict[str, Any]],
                       candidates: Sequence[Milestone],
                       top_k: int = 3) -> Dict[str, Any]:
    """How each environment predicate lines up with each mined milestone.

    Measured, not assumed from names: for every (milestone, predicate) pair the
    onset delta and span overlap are computed across the episodes holding both,
    and the closest matches are exported. Disagreement is the interesting part
    -- a milestone with no predicate near it is either a real fact the task does
    not track or a detector artefact, and only the refinement pass can say
    which.
    """
    out: Dict[str, Any] = {}
    for milestone in candidates:
        scores = []
        for key in _predicate_keys(traces):
            deltas, ious = [], []
            for record in traces:
                spans = _spans(record)
                predicate = (record.get("predicates") or {}).get(key)
                if milestone not in spans or not predicate:
                    continue
                m_on, m_off = spans[milestone]
                p_on, p_off = int(predicate[0][0]), int(predicate[-1][1])
                deltas.append(p_on - m_on)
                ious.append(_iou((m_on, m_off), (p_on, p_off)))
            if not deltas:
                continue
            arr = np.asarray(deltas, dtype=float)
            scores.append({
                "predicate": key,
                "episodes": len(deltas),
                "median_onset_delta": float(np.median(arr)),
                "onset_within_1_frame": float(np.mean(np.abs(arr) <= 1)),
                "median_span_iou": float(np.median(ious)),
            })
        scores.sort(key=lambda s: -s["median_span_iou"])
        out[" / ".join(milestone)] = scores[:top_k]
    return out


def _predicate_keys(traces: List[Dict[str, Any]]) -> List[str]:
    keys = set()
    for record in traces:
        keys.update((record.get("predicates") or {}).keys())
    return sorted(keys)


def _iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    inter = max(0, hi - lo + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union if union else 0.0


# --------------------------------------------------------------------------- #
# what a schedule may actually name
# --------------------------------------------------------------------------- #
def clause_inventory(env_id: str, configs: Path) -> Dict[str, Any]:
    """Which relations the runtime can actually emit, per pair.

    A schedule naming a relation with no mined components behind it scores zero
    forever, and for the compatibility families the runtime cannot tell that
    apart from 'too far to judge' -- both read ``unobserved``. So the bundle
    carries the inventory and compilation rejects anything outside it, rather
    than the schedule failing silently at training time.
    """
    aff_path = configs / "affordances" / f"{env_id}.json"
    wl_path = configs / "subtask_whitelists" / env_id / "task_all.json"
    for path in (aff_path, wl_path):
        if not path.exists():
            raise SystemExit(
                f"{path} is missing; mine the task's assets before its "
                "schedule, or the bundle cannot say which clauses are scorable"
            )
    with open(aff_path) as handle:
        objects = json.load(handle).get("objects", {})
    with open(wl_path) as handle:
        whitelist = json.load(handle)
    members = whitelist.get("members", {})
    bins = whitelist.get("bin_edges", {})

    def has(key: str, field: str) -> bool:
        return bool((objects.get(key) or {}).get(field))

    def types(key: str) -> set:
        return set((members.get(key) or {}).get("interaction_types") or ())

    pairs: Dict[str, Any] = {}
    keys = sorted(members)
    for key in keys:
        pairs[f"{EE_KEY} / {key}"] = {
            "contact": "contact" in types(key),
            "grasp": "grasp" in types(key),
            "grasp-compatibility": has(key, "grasp_components"),
            "contact-compatibility": has(key, "contact_components"),
            "planar-distance": bool(bins.get("planar-distance")),
            "height-offset": bool(bins.get("height-offset")),
        }
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            both = types(a) & types(b)
            pairs[f"{a} / {b}"] = {
                "contact": "contact" in both,
                "support": "support" in both,
                "contain": "contain" in both,
                "contact-compatibility": has(a, "contact_components")
                                         and has(b, "contact_components"),
                "support-compatibility": (has(a, "support_components")
                                          and has(b, "bottom_components"))
                                         or (has(b, "support_components")
                                             and has(a, "bottom_components")),
                "contain-compatibility": (has(a, "contain_components")
                                          and has(b, "key_components"))
                                         or (has(b, "contain_components")
                                             and has(a, "key_components")),
                "planar-distance": bool(bins.get("planar-distance")),
                "height-offset": bool(bins.get("height-offset")),
            }
    return {
        "members": {k: sorted(types(k)) for k in keys},
        "components": {
            k: sorted(field for field in
                      ("grasp_components", "contact_components",
                       "support_components", "bottom_components",
                       "contain_components", "key_components")
                      if has(k, field))
            for k in keys
        },
        "spatial_bins": {r: bool(bins.get(r)) for r in SPATIAL_RELATIONS},
        "scorable": pairs,
    }


# --------------------------------------------------------------------------- #
# bundle
# --------------------------------------------------------------------------- #
def propose_roles(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Candidate roles from the interaction structure alone.

    ``movable`` is whatever the end effector grasps; ``destination`` is what the
    movable object later contacts, supports against or enters. Proposed, not
    decided -- a task with two grasped objects gets both listed and flagged.
    """
    grasped, partners = [], []
    for entry in stats.values():
        if not entry["is_candidate"]:
            continue
        if entry["relation"] == "grasp" and entry["src"] == EE_KEY:
            grasped.append(entry["dst"])
        elif EE_KEY not in (entry["src"], entry["dst"]):
            # An object the movable one was already touching at frame zero is
            # the surface it started on, not somewhere it was taken.
            if entry["initial_condition"]:
                continue
            partners.append((entry["src"], entry["dst"],
                             entry["onset"].get("median", 0.0)))
    movable = sorted(set(grasped))
    destinations = []
    for src, dst, onset in sorted(partners, key=lambda p: p[2]):
        for key in (src, dst):
            if key not in movable and key not in destinations:
                destinations.append(key)
    return {
        "movable": movable,
        "destination_candidates": destinations,
        "ambiguous": len(movable) != 1,
    }


def build_bundle(env_id: str, merged: Dict[str, Any], configs: Path,
                 min_presence: float, min_order: float,
                 n_examples: int = 3) -> Dict[str, Any]:
    traces = merged["traces"]
    if not traces:
        raise SystemExit(f"{env_id}: shards carry no episode traces")
    stats = milestone_stats(traces, min_presence)
    candidates = [
        (e["relation"], e["src"], e["dst"])
        for e in stats.values() if e["is_candidate"]
    ]
    candidates.sort(key=lambda m: stats[" / ".join(m)]["onset"].get("median", 0.0))
    return {
        "_schema_version": BUNDLE_SCHEMA_VERSION,
        "env_id": env_id,
        "successful_episodes": len(traces),
        "gates": {"milestone_presence": min_presence,
                  "ordering_confidence": min_order},
        "milestones": stats,
        "candidate_order": [" / ".join(m) for m in candidates],
        "ordering": ordering_stats(traces, candidates, min_order),
        "proposed_phases": propose_phases(traces, stats),
        "proposed_roles": propose_roles(stats),
        "environment_predicates": predicate_stats(traces),
        "environment_scalars": scalar_stats(traces),
        "detector_agreement": detector_agreement(traces, candidates),
        "scorable_clauses": clause_inventory(env_id, configs),
        "example_traces": [
            {"interactions": [list(e) for e in (r.get("interactions") or ())],
             "predicates": {k: [list(s) for s in v]
                            for k, v in (r.get("predicates") or {}).items()}}
            for r in traces[:n_examples]
        ],
    }


def report(bundle: Dict[str, Any]) -> None:
    env = bundle["env_id"]
    total = bundle["successful_episodes"]
    print(f"\n=== {env}: {total} successful episodes")
    for name, entry in sorted(bundle["milestones"].items(),
                              key=lambda kv: -kv[1]["rate"]):
        mark = "MILESTONE" if entry["is_candidate"] else "  below  "
        note = "  [initial condition]" if entry["initial_condition"] else ""
        print(f"  {mark}  {entry['episodes']:5d}/{entry['of']:<5d} "
              f"{entry['rate']:6.1%}  {name}{note}")
    by_verdict = defaultdict(list)
    for entry in bundle["ordering"]:
        by_verdict[entry["verdict"]].append(entry)
    ordered = len(by_verdict["before"]) + len(by_verdict["after"])
    print(f"  ordering: {ordered} constraints, "
          f"{len(by_verdict['simultaneous'])} simultaneous, "
          f"{len(by_verdict['ambiguous'])} ambiguous")
    for entry in by_verdict["ambiguous"]:
        print(f"    ?  {entry['rate_a_before_b']:.0%} before / "
              f"{entry['rate_b_before_a']:.0%} after / "
              f"{entry['rate_same_frame']:.0%} same frame"
              f"  {entry['a']}  vs  {entry['b']}")
    roles = bundle["proposed_roles"]
    print(f"  roles: movable={roles['movable']} "
          f"destinations={roles['destination_candidates']}"
          + ("  [AMBIGUOUS]" if roles["ambiguous"] else ""))
    predicates = bundle["environment_predicates"]
    print(f"  predicates: {', '.join(sorted(predicates)) or 'none'}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Mine ManiSkill phase schedules")
    p.add_argument("--env-id", action="append", required=True,
                   help="repeatable; one bundle file per task")
    p.add_argument("--shards", default="data/maniskill_evidence")
    p.add_argument("--configs", default=str(CONFIGS))
    p.add_argument("--out", default="schedule_candidates")
    p.add_argument("--min-presence", type=float, default=MILESTONE_PRESENCE)
    p.add_argument("--min-order", type=float, default=ORDER_CONFIDENCE)
    p.add_argument("--examples", type=int, default=3)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    configs = Path(args.configs)
    manifest = {
        "_schema_version": BUNDLE_SCHEMA_VERSION,
        "gates": {"milestone_presence": args.min_presence,
                  "ordering_confidence": args.min_order},
        "tasks": {},
    }
    for env_id in args.env_id:
        merged = load_shards(env_id, Path(args.shards))
        bundle = build_bundle(env_id, merged, configs,
                              args.min_presence, args.min_order, args.examples)
        report(bundle)
        with open(out / f"{env_id}.json", "w") as handle:
            json.dump(bundle, handle, indent=2)
        manifest["tasks"][env_id] = {
            "file": f"{env_id}.json",
            "successful_episodes": bundle["successful_episodes"],
            "milestones": sum(1 for e in bundle["milestones"].values()
                              if e["is_candidate"]),
            "ambiguous_orderings": sum(1 for o in bundle["ordering"]
                                       if o["verdict"] == "ambiguous"),
            "simultaneous_orderings": sum(1 for o in bundle["ordering"]
                                          if o["verdict"] == "simultaneous"),
            "roles_ambiguous": bundle["proposed_roles"]["ambiguous"],
        }
        print(f"  wrote {out / f'{env_id}.json'}")
    with open(out / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"\nwrote {out / 'manifest.json'} ({len(manifest['tasks'])} tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
