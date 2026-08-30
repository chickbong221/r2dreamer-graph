"""Check a recorded rollout's graphs against the revision's acceptance criteria.

Reads the per-frame ``graph_json`` a ``render_paper_frames`` run writes and
asks what the schedules assume: that a table is measured against its top and
offers no horizontal position, that end-effector heights actually move, that a
goal site is a vertex with the right facts on it, and that ``reached`` is
emitted every frame rather than only when it is true.

Nothing here needs a simulator. It is a reader, so it can run against any
episode already on disk.

    python tests/probes/probe_graph_acceptance.py data/verify/PickCube-v1_seed0000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict


def load_frames(root):
    paths = sorted(glob.glob(os.path.join(root, "graph_json", "*.json")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(root, "*.json")))
    if not paths:
        sys.exit(f"no graph json under {root}")
    return [json.load(open(p)) for p in paths]


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


def edges_of(frame, relation=None, src=None, dst=None):
    out = []
    for e in frame["edges"]:
        if relation and e["relation"] != relation:
            continue
        if src and e["src"] != src:
            continue
        if dst and e["dst"] != dst:
            continue
        out.append(e)
    return out


def analyse(root, n_max=8, e_max=168):
    frames = load_frames(root)
    env_id = frames[0]["env_id"]
    rep = Report()
    print(f"\n=== {env_id}   {len(frames)} frames   {root}")

    keys = {n["node_id"] for n in frames[0]["nodes"]}
    table = next((k for k in keys if k.endswith("table-workspace")), None)
    sites = sorted(k for k in keys if k.startswith("spatial:"))
    rep.note("nodes", ", ".join(sorted(keys)))

    # ---- capacity ------------------------------------------------------- #
    max_nodes = max(len(f["nodes"]) for f in frames)
    max_edges = max(len(f["edges"]) for f in frames)
    rep.check("nodes within n_max", max_nodes <= n_max, f"{max_nodes} <= {n_max}")
    rep.check("edges within e_max", max_edges <= e_max, f"{max_edges} <= {e_max}")

    # ---- structural surface --------------------------------------------- #
    if table:
        planar = [e for f in frames
                  for e in edges_of(f, "planar-distance")
                  if table in (e["src"], e["dst"])]
        rep.check("no planar edge touches the table", not planar,
                  f"{len(planar)} found")

        heights = [e["raw_value"] for f in frames
                   for e in edges_of(f, "height-offset")
                   if e["src"] == "ee" and e["dst"] == table]
        if heights:
            rep.check("ee-table height is surface-relative",
                      max(abs(h) for h in heights) < 0.5,
                      f"|h| max {max(abs(h) for h in heights):.4f} "
                      "(origin-relative would be ~1.0)")

        rest = [e["raw_value"] for f in frames
                for e in edges_of(f, "height-offset")
                if table in (e["src"], e["dst"]) and e["src"] != "ee"]
        if rest:
            rep.note("object-table height range",
                     f"[{min(rest):+.4f}, {max(rest):+.4f}]")

    # ---- end-effector heights actually move ------------------------------ #
    by_dst = defaultdict(list)
    for f in frames:
        for e in edges_of(f, "height-offset", src="ee"):
            by_dst[e["dst"]].append(e["label"])
    for dst, labels in sorted(by_dst.items()):
        distinct = sorted(set(labels))
        rep.check(f"ee-height varies: {dst}", len(distinct) > 1,
                  f"{distinct}")

    # ---- virtual sites --------------------------------------------------- #
    for site in sites:
        ee_edges = [e for f in frames for e in f["edges"]
                    if e["src"] == "ee" and e["dst"] == site]
        rep.check(f"no ee edges to {site}", not ee_edges,
                  f"{len(ee_edges)} found")

    # ---- reached --------------------------------------------------------- #
    reached = [edges_of(f, "reached") for f in frames]
    counts = Counter(len(r) for r in reached)
    if any(counts):
        n = sorted(counts)[0]
        rep.check("exactly one reached per frame",
                  set(counts) == {max(counts)} and max(counts) >= 1,
                  f"per-frame counts {dict(counts)}")
        labels = [e["label"] for r in reached for e in r]
        rep.check("reached emits not-holds when false",
                  "not-holds" in labels, f"{dict(Counter(labels))}")
        rep.check("reached holds on the final frame",
                  bool(reached[-1]) and reached[-1][0]["label"] == "holds",
                  f"final: {[e['label'] for e in reached[-1]]}")
        first_true = next((i for i, r in enumerate(reached)
                           if r and r[0]["label"] == "holds"), None)
        rep.note("reached first true at frame",
                 f"{first_true} of {len(frames)}")
        pair = reached[-1][0] if reached[-1] else None
        if pair:
            rep.note("reached pair", f"{pair['src']} -> {pair['dst']}")

    # ---- bin keys actually used ------------------------------------------ #
    used = Counter(e.get("bin_key") for f in frames for e in f["edges"]
                   if e.get("bin_key"))
    for key, n in sorted(used.items()):
        rep.note("bin_key in use", f"{key}  ({n} edges)")

    # ---- compatibility observability ------------------------------------- #
    for relation in ("contain-compatibility", "support-compatibility"):
        labels = [e["label"] for f in frames for e in edges_of(f, relation)]
        if labels:
            observed = sum(1 for l in labels if l != "unobserved")
            rep.note(f"{relation} observed",
                     f"{observed}/{len(labels)} frames scored")

    failed = rep.render()
    return failed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("roots", nargs="+")
    p.add_argument("--n-max", type=int, default=8)
    p.add_argument("--e-max", type=int, default=168)
    args = p.parse_args()
    failed = sum(analyse(r, args.n_max, args.e_max) for r in args.roots)
    print(f"\n{'all checks passed' if not failed else str(failed) + ' CHECK(S) FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
