"""Report what a collected pickle actually holds, before the long run starts.

The pilot's acceptance check. A schema-v9 collection records three things that
no amount of re-mining can recover if they are wrong or absent -- the build
configuration each rollout came from, the collision extents of every entity
that can enter ``members``, and the end-effector rest calibration -- and the
last economical moment to catch a mistake in any of them is after two or three
successes, not after the full run.

Reads the pickle only. No simulator, no torch::

    python tests/probes/probe_collection_pickle.py \\
        $MS_ASSET_DIR/data/robot_success_states/fetch/set_table/pick/024_bowl.pkl

    python tests/probes/probe_collection_pickle.py <pkl> --build-config apt_0
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter

REQUIRED_SCHEMA = 9


class Report:
    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail=""):
        self.rows.append((bool(ok), name, detail))

    def note(self, name, detail):
        self.rows.append((None, name, detail))

    def render(self):
        width = max(len(n) for _, n, _ in self.rows)
        failed = 0
        for ok, name, detail in self.rows:
            mark = "    " if ok is None else (" ok " if ok else "FAIL")
            failed += 1 if ok is False else 0
            print(f"  [{mark}] {name:<{width}}  {detail}")
        return failed


def _members(whitelist_path):
    """``{canonical key: family or None}`` from a mined whitelist, or {}."""
    if not whitelist_path:
        return {}
    import json
    with open(whitelist_path) as handle:
        raw = json.load(handle)
    return {k: (e or {}).get("family")
            for k, e in (raw.get("members") or {}).items()
            if not k.startswith("spatial:")}


def analyse(path, wanted_config="", members=None):
    members = members or {}
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    rollouts = payload.get("interaction_rollouts") or []
    rep = Report()
    print(f"\n=== {path}")

    version = int(payload.get("_schema_version", 0))
    rep.check("schema is current", version >= REQUIRED_SCHEMA,
              f"v{version}, need v{REQUIRED_SCHEMA}+")
    rep.check("has rollouts", bool(rollouts), f"{len(rollouts)}")
    if not rollouts:
        return rep.render()

    rep.note("payload build configs",
             ", ".join(payload.get("build_configs") or []) or "<none>")

    # ---- one scene, if one was asked for ------------------------------- #
    configs = Counter(
        (r.get("provenance") or {}).get("build_config_name", "")
        for r in rollouts
    )
    rep.note("rollout build configs", dict(configs))
    if wanted_config:
        rep.check("every rollout is the requested scene",
                  set(configs) == {wanted_config},
                  f"wanted {wanted_config!r}, got {sorted(configs)}")
    else:
        rep.check("rollouts come from one scene", len(configs) == 1,
                  f"{len(configs)} distinct")

    sources = Counter(
        (r.get("provenance") or {}).get("plan_source", "absent")
        for r in rollouts
    )
    rep.check("plan identity was read live, not from the assignment",
              set(sources) == {"live"}, f"{dict(sources)}")

    plans = Counter(
        (r.get("provenance") or {}).get("init_config_name", "")
        for r in rollouts
    )
    rep.note("init configs", dict(plans))
    rep.note("task plan indices", dict(Counter(
        (r.get("provenance") or {}).get("task_plan_index")
        for r in rollouts)))
    envs = sorted({(r.get("provenance") or {}).get("env_idx") for r in rollouts})
    rep.note("contributing env indices", f"{envs}")

    # ---- target resolution --------------------------------------------- #
    resolution = Counter(
        (r.get("provenance") or {}).get("target_resolution", "?")
        for r in rollouts
    )
    rep.check("target resolution never fell back to the merged handle",
              set(resolution) <= {"actual"}, f"{dict(resolution)}")

    # ---- geometry ------------------------------------------------------- #
    status = Counter()
    missing = set()
    for rollout in rollouts:
        for key, entry in (rollout.get("extents") or {}).items():
            state = entry.get("extent_status", "absent")
            status[state] += 1
            if state != "ok":
                missing.add(f"{key} ({state})")
    rep.check("every recorded entity has readable extents",
              bool(status) and set(status) == {"ok"},
              f"{dict(status)}" + (f"  unreadable: {sorted(missing)[:5]}"
                                   if missing else ""))
    raw = [entry for rollout in rollouts
           for entry in (rollout.get("extents") or {}).values()]
    rep.check("raw names travel beside the canonical keys",
              bool(raw) and all("name" in e for e in raw),
              f"{sum('raw_articulation' in e for e in raw)} of {len(raw)} "
              "carry a raw articulation")

    # ---- end-effector rest ---------------------------------------------- #
    samples = [s for r in rollouts for s in (r.get("ee_rest_samples") or [])]
    rep.check("rest-site samples were recorded", bool(samples),
              f"{len(samples)} frames")
    if samples:
        distances = [float(s["euclidean_distance"]) for s in samples]
        finite = all(d == d and abs(d) != float("inf") for d in distances)
        rep.check("rest distances are finite", finite, "")
        rep.check("rest distance varies during the episode",
                  max(distances) - min(distances) > 1e-4,
                  f"[{min(distances):.4f}, {max(distances):.4f}]")
        tolerances = {round(float(s["tolerance"]), 6) for s in samples}
        rep.note("live tolerance", f"{sorted(tolerances)}")
        reached = Counter(bool(s["reached"]) for s in samples)
        rep.check("the rest predicate fires at least once", reached[True] > 0,
                  f"{dict(reached)}")
        # The exact predicate must agree with the distance it was read beside.
        agrees = all(bool(s["reached"]) == (float(s["euclidean_distance"])
                                            <= float(s["tolerance"]))
                     for s in samples)
        rep.check("reached agrees with distance vs tolerance", agrees, "")

    # ---- pose trace ------------------------------------------------------ #
    # The per-family end-effector height scales are reprojected from this
    # trace at mining time, not recorded by the collector, so a member absent
    # from it silently calibrates no scale -- and the graph builder then
    # refuses the whole asset for a bin it cannot find. That is exactly how
    # tidy_house ended up classifying six structural surfaces and calibrating
    # none of them.
    snaps = [s for r in rollouts for s in (r.get("pose_samples") or [])]
    rep.check("pose snapshots were recorded", bool(snaps), f"{len(snaps)}")
    if snaps:
        seen = Counter(
            key for s in snaps for key in (s.get("entities") or {}))
        rep.note("entities in the pose trace",
                 ", ".join(f"{k} x{n}" for k, n in sorted(seen.items())))
        posed = Counter(
            key for s in snaps
            for key, e in (s.get("entities") or {}).items()
            if isinstance(e, dict) and e.get("pose"))
        no_pose = sorted(set(seen) - set(posed))
        rep.check("every traced entity carries a pose", not no_pose,
                  f"missing: {no_pose}" if no_pose else "")
        if members:
            absent = sorted(set(members) - set(seen))
            rep.check(
                "every whitelist member appears in the pose trace",
                not absent,
                f"absent: {absent}" if absent else f"{len(members)} member(s)")
            for key in sorted(members):
                if members[key] and key in seen:
                    rep.note(f"  {key}", f"family={members[key]} "
                                         f"x{seen[key]} snapshot(s)")
                elif members[key]:
                    rep.note(f"  {key}",
                             f"family={members[key]} -- NOT TRACED, so its "
                             "family scale calibrates from nothing")

    keys = Counter(
        key for r in rollouts for key in (r.get("bin_samples") or {})
    )
    for key in ("ee_site_planar_distance", "ee_site_height_offset"):
        rep.check(f"bin stream {key}", keys.get(key, 0) > 0,
                  f"{keys.get(key, 0)} rollout(s)")
    rep.note("all bin streams", ", ".join(sorted(keys)))
    return rep.render()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pickles", nargs="+")
    p.add_argument("--build-config", default="",
                   help="Assert every rollout came from this configuration.")
    p.add_argument("--whitelist", default="",
                   help="A mined whitelist whose members the pose trace must "
                        "cover. Each family scale is reprojected from that "
                        "trace, so a member missing from it calibrates "
                        "nothing and the asset is refused at graph-builder "
                        "startup.")
    args = p.parse_args()
    members = _members(args.whitelist)
    failed = sum(analyse(path, args.build_config, members)
                 for path in args.pickles)
    print(f"\n{'all checks passed' if not failed else str(failed) + ' CHECK(S) FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
