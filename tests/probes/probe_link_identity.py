"""Whether one logical articulation is numbered differently across scenes.

A **sampled** audit, not coverage. It reads whatever build configurations the
probe collection happened to visit, which is a handful of the 63 -- so a clean
result here is evidence of absence only for the scenes sampled, and the count
of configurations seen is printed alongside for that reason.

Two questions, with opposite answers:

* One base numbered differently *across* configurations, never coexisting:
  the same counter is ``kitchen_counter-0`` here and ``-1`` there, so a
  whitelist mined in one scene cannot match in the other. Canonicalising would
  help.
* Two indices of one base *inside* one configuration: two real counters stand
  in that apartment, and collapsing them would merge two distinct
  articulations into one vertex. Canonicalising would be wrong.

Names are put through the repository's own ``canonical_scene_name`` before the
terminal index is removed. Doing it with a hand-written regex reads the
``env-0_`` and ``scs-[2,3]_`` prefixes as part of the base and reports
variation that is only per-environment tagging.

This probe decides nothing. Link canonicalisation is not changed on the
strength of a small sample; a finding here is a reason to stop and look.

    python tests/probes/probe_link_identity.py /tmp/probe_assets
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from scenegraph.core.entity_identity import canonical_scene_name  # noqa: E402

# The trailing instance index ``canonical_scene_name`` deliberately preserves.
_INSTANCE = re.compile(r"-(\d+)$")


def audit(root):
    pattern = os.path.join(
        root, "robot_success_states", "fetch", "*", "*", "*.pkl")
    paths = sorted(glob.glob(pattern))
    if not paths:
        sys.exit(f"no rollout pickles under {root}")

    configs = set()
    # base -> {canonical name}, and (config, base) -> {canonical name}
    across = defaultdict(set)
    within = defaultdict(set)
    seen_articulations = 0

    for path in paths:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        for rollout in payload.get("interaction_rollouts") or []:
            config = (rollout.get("provenance") or {}).get(
                "build_config_name", "<unknown>")
            configs.add(config)
            for entry in (rollout.get("extents") or {}).values():
                raw = entry.get("raw_articulation")
                if not raw:
                    continue
                seen_articulations += 1
                # The repository's own rule first: it strips the per-env and
                # scene-config-set prefixes and keeps the instance index.
                name = canonical_scene_name(raw) or raw
                base = _INSTANCE.sub("", name)
                across[base].add(name)
                within[(config, base)].add(name)

    print(f"\n=== {root}")
    print(f"  {len(paths)} pickle(s), {len(configs)} build configuration(s) "
          f"sampled, {seen_articulations} articulation reading(s)")
    print(f"  sampled: {', '.join(sorted(configs))}")
    print("  NOTE: a sample of the 63 configurations, not coverage.")

    varying = {b: n for b, n in across.items() if len(n) > 1}
    colliding = {k: n for k, n in within.items() if len(n) > 1}

    print("\n  --- one base, several indices ACROSS configurations ---")
    if varying:
        for base, names in sorted(varying.items()):
            print(f"    {base}: {sorted(names)}")
    else:
        print("    none: every articulation kept one index in this sample")

    print("\n  --- several indices of one base WITHIN one configuration ---")
    if colliding:
        for (config, base), names in sorted(colliding.items()):
            print(f"    {config} :: {base} -> {sorted(names)}")
    else:
        print("    none: no configuration held two instances of one base")

    print()
    if colliding:
        print("  STOP: collapsing the terminal index would merge two distinct "
              "articulations that stand in one scene at once.")
        return 1
    if varying:
        print("  STOP AND REPORT: one logical articulation is numbered "
              "differently across scenes, so a whitelist mined in one cannot "
              "match in the other. Do not change canonicalisation on this "
              "sample alone.")
        return 1
    print("  clean for the configurations sampled")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots", nargs="+",
        help="Asset root(s) holding robot_success_states/, e.g. /tmp/probe_assets")
    args = parser.parse_args()
    return max(audit(root) for root in args.roots)


if __name__ == "__main__":
    sys.exit(main())
