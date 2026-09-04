"""Is this task group's mined asset tree fit to train on?

Everything a run checks at startup, plus the things it does not check and
should: run without a simulator, a GPU or torch, so the answer arrives in a
second rather than after an env build.

The gates, in the order a run would hit them:

1. every target the experiment names has a runtime whitelist;
2. no member of a runtime asset carries an unresolved-classification mark
   (raw evidence is allowed to; a runtime asset is not);
3. the union loads at all;
4. every key ``required_bin_keys`` names is calibrated -- this is the exact
   check ``GraphBuilder._bind_global_bin_edges`` makes, and the one that
   catches a family classified but never measured;
5. every structural surface has a ``reference_surface`` in the affordance
   asset, because its height is measured against that plane and against
   nothing otherwise;
6. the declared site is both declared and a member, since dropping either
   makes the node encode as padding and every fact naming it disappear;
7. the schedule compiles against the vocabulary the assets produce.

Capacity is reported, not asserted: ``entity_vocab`` is a property of the
assets and is knowable here, while ``n_max``/``e_max`` are properties of what
a scene puts on screen at once and need real frames. See
``scenegraph.tools.audit_graph_capacity``.

    python tests/probes/validate_task_assets.py --task tidy_house \\
        --targets 002_master_chef_can 004_sugar_box ...
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scenegraph.core import families as families_rules  # noqa: E402
from scenegraph.core.affordance import (  # noqa: E402
    load_affordance_set, lookup_reference_surface,
)
from scenegraph.core.relation_rules import required_bin_keys  # noqa: E402
from scenegraph.core.schema import Node  # noqa: E402
from scenegraph.core.sites import SITE_PREFIX  # noqa: E402
from scenegraph.core.whitelist import load_whitelist  # noqa: E402


class Report:
    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail=""):
        self.rows.append((bool(ok), name, detail))

    def note(self, name, detail=""):
        self.rows.append((None, name, detail))

    def render(self) -> int:
        width = max(len(n) for _, n, _ in self.rows)
        failed = 0
        for ok, name, detail in self.rows:
            mark = "    " if ok is None else (" ok " if ok else "FAIL")
            failed += 1 if ok is False else 0
            print(f"  [{mark}] {name:<{width}}  {detail}")
        return failed


def _node(key: str) -> Node:
    return Node(node_id=key, node_type="object", name=key,
                pose_world=[0.0] * 3 + [1.0, 0.0, 0.0, 0.0],
                attributes={"whitelist_key": key, "entity_key": key})


def validate(args) -> int:
    rep = Report()
    runtime = pathlib.Path(args.runtime_dir)
    raw = pathlib.Path(args.raw_dir) if args.raw_dir else None
    union_path = runtime / f"{args.subtask}_all.json"

    # ---- 1. coverage ---------------------------------------------------- #
    present = sorted(p.stem[len(args.subtask) + 1:]
                     for p in runtime.glob(f"{args.subtask}_*.json")
                     if p.name != union_path.name)
    rep.note("runtime assets", f"{len(present)} target(s) in {runtime}")
    if args.targets:
        missing = sorted(set(args.targets) - set(present))
        extra = sorted(set(present) - set(args.targets))
        rep.check("every named target has a whitelist", not missing,
                  f"missing {missing}" if missing else f"{len(args.targets)}")
        rep.check("no target the experiment did not name", not extra,
                  f"unexpected {extra}" if extra else "")

    # ---- 2. nothing unresolved reaches the runtime ----------------------- #
    blocked = {}
    for path in sorted(runtime.glob("*.json")):
        members = json.loads(path.read_text()).get("members") or {}
        found = families_rules.runtime_blockers(members)
        if found:
            blocked[path.name] = found
    rep.check("no runtime member is unmeasurable", not blocked,
              json.dumps({k: sorted(v) for k, v in blocked.items()})
              if blocked else "")
    if raw and raw.is_dir():
        kept = {}
        for path in sorted(raw.glob("*.json")):
            found = families_rules.unresolved_members(
                json.loads(path.read_text()).get("members") or {})
            if found:
                kept[path.name] = sorted(found)
        rep.note("raw evidence keeps, marked",
                 json.dumps(kept) if kept else "nothing unresolved")

    # ---- 3-4. the union, and the bins a run demands of it ---------------- #
    if not union_path.is_file():
        rep.check("the union exists", False, str(union_path))
        return rep.render()
    try:
        union = load_whitelist(str(union_path))
    except ValueError as exc:
        rep.check("the union loads", False, str(exc)[:160])
        return rep.render()
    rep.check("the union loads", True,
              f"{len(union.by_key)} member(s), schema v{union.schema_version}")
    rep.check("the union is not key-migrated", not union.migrated_pre_anchor,
              "carries _bins_migrated_pre_anchor" if union.migrated_pre_anchor
              else "")

    cfg = {"structural_surfaces": set(union.structural_surfaces),
           "families": dict(union.families),
           "site_declarations": dict(union.sites)}
    required = required_bin_keys(cfg)
    absent = [key for key in required if not union.bin_edges.get(key)]
    rep.check("every required bin is calibrated", not absent,
              f"missing {absent}" if absent
              else f"{len(required)} key(s)")
    for key in required:
        rep.note(f"  {key}",
                 "calibrated" if union.bin_edges.get(key) else "MISSING")

    # ---- 5. surfaces have a plane to be measured against ------------------ #
    aff = load_affordance_set(args.affordance)
    surfaces = sorted(union.structural_surfaces)
    planeless = [key for key in surfaces
                 if lookup_reference_surface(aff, _node(key)) is None]
    rep.check("every structural surface has a mined plane", not planeless,
              f"missing {planeless}" if planeless
              else f"{len(surfaces)} surface(s)")

    # ---- 6. the site is declared and is a member -------------------------- #
    declared = sorted(union.sites)
    member_sites = sorted(k for k in union.by_key if k.startswith(SITE_PREFIX))
    rep.check("every declared site is also a member",
              set(declared) == set(member_sites),
              f"declared {declared}, members {member_sites}")
    rep.check("no site was given a height family",
              not any(k.startswith(SITE_PREFIX) for k in union.families),
              "")

    # ---- 7. the schedule compiles ----------------------------------------- #
    try:
        from scenegraph.adapters.graph_vocab import build_entity_vocab
        from scenegraph.core.schedule import (
            compile_from_source, mshab_schedule_source,
        )
        vocab = build_entity_vocab(str(runtime))
        source = mshab_schedule_source(
            args.task, args.subtask, args.configs, args.schedule_dir,
            str(runtime))
        schedule = compile_from_source(source, vocab)
    except Exception as exc:                       # noqa: BLE001
        rep.check("the schedule compiles", False, f"{type(exc).__name__}: {exc}")
        return rep.render()
    rep.check("the schedule compiles", True,
              f"{len(schedule.phases)} phase(s), {len(schedule.slots)} fact(s)")
    rep.note("  phases", ", ".join(
        f"{p.name}@{p.weight:g}" for p in schedule.phases))
    rep.note("  scored entities", ", ".join(str(e) for e in schedule.entity_ids))

    # ---- capacity inputs, reported not asserted --------------------------- #
    rep.note("entity_vocab (asset property)", f"{len(vocab.token_to_id)}")
    rep.note("n_max / e_max",
             "NOT decidable here: they bound simultaneous instances, which "
             "needs real frames -- run audit_graph_capacity")
    return rep.render()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", default="tidy_house")
    p.add_argument("--subtask", default="pick")
    p.add_argument("--targets", nargs="*", default=[],
                   help="Object names the experiment names, e.g. 004_sugar_box.")
    p.add_argument("--configs", default=str(ROOT / "scenegraph" / "configs"))
    p.add_argument("--runtime-dir", default="")
    p.add_argument("--raw-dir", default="")
    p.add_argument("--affordance", default="")
    p.add_argument("--schedule-dir", default="")
    args = p.parse_args(argv)

    configs = pathlib.Path(args.configs)
    args.runtime_dir = args.runtime_dir or str(
        configs / "subtask_whitelists" / args.task)
    args.raw_dir = args.raw_dir or str(
        configs / "subtask_whitelists_raw" / args.task)
    args.affordance = args.affordance or str(
        configs / "affordances" / f"{args.task}.json")
    args.schedule_dir = args.schedule_dir or str(configs / "schedules")

    print(f"\n=== {args.task}/{args.subtask}")
    failed = validate(args)
    print(f"\n{'all checks passed' if not failed else str(failed) + ' CHECK(S) FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
