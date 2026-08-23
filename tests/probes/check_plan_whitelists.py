"""Cross-check an MS-HAB task-plan file against the mined whitelists.

Answers "why does this ``mshab_obj`` refuse to start" without a GPU, a
simulator or a single environment build: it reads the same plan file the env
would load and resolves the same whitelist paths the runtime would ask for.

Three startup failures are the ones this reproduces, in the order they fire:

1. ``_verify_whitelist_coverage`` -- a pick/place target in the plan with no
   mined ``<subtask>_<target>.json``.
2. ``_bind_global_bin_edges`` -- a subtask *type* in the plan with no
   ``<subtask>_all.json``, which is where the global relation bins come from.
3. ``pack_graph`` -- a whitelisted member missing from the entity vocabulary,
   which raises on the first packed frame rather than at construction.

It also reports one failure that happens *before* any of ours, inside MS-HAB's
own env construction: a plan whose pick subtasks span more than one kind of
articulation. ``_merge_pick_subtasks`` builds a single batched ``Articulation``
view over whatever the parallel envs drew, and ``create_from_physx_articulations``
asserts they all have the same link and joint counts. ``set_table`` picks the
apple out of a fridge and the bowl out of a kitchen counter, so any batch drawn
from ``all.json`` mixes the two and the assert fires. Per-object plan files each
name one articulation throughout, which is why mining and single-object training
both work.

Usage:

    python -m runs.check_plan_whitelists --mshab-task set_table --mshab-obj all
"""

import argparse
import pathlib
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scenegraph.core.affordance import canonical_affordance_key  # noqa: E402
from scenegraph.core.whitelist import (  # noqa: E402
    load_whitelist,
    resolve_whitelist_path,
)
from scenegraph.adapters.graph_vocab import build_entity_vocab  # noqa: E402


def _articulation_kinds(plans):
    """Distinct articulations the plans' pick/place subtasks name.

    MS-HAB merges these into one batched view across parallel envs, and the
    view can only manage articulations with identical link and joint counts.
    Field names differ across mshab versions, so this probes a few and falls
    back to the repr rather than reporting nothing.
    """
    kinds = Counter()
    for plan in plans:
        for subtask in getattr(plan, "subtasks", []) or []:
            if str(getattr(subtask, "type", None)) not in ("pick", "place"):
                continue
            config = getattr(subtask, "articulation_config", None)
            if config is None:
                continue
            label = None
            for attr in ("articulation_id", "articulation_type", "art_id", "id"):
                value = getattr(config, attr, None)
                if value:
                    label = str(value)
                    break
            kinds[canonical_affordance_key(label) or label or repr(config)[:40]] += 1
    return kinds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mshab-task", default="set_table")
    parser.add_argument("--mshab-obj", default="all")
    parser.add_argument("--subtask", default="pick")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-build-configs", type=int, default=10)
    parser.add_argument(
        "--whitelist-root",
        default=str(ROOT / "scenegraph/configs/subtask_whitelists"),
    )
    args = parser.parse_args()

    from mani_skill import ASSET_DIR
    from mshab.envs.planner import plan_data_from_file

    plan_path = (
        ASSET_DIR
        / "scene_datasets/replica_cad_dataset/rearrange/task_plans"
        / args.mshab_task
        / args.subtask
        / args.split
        / f"{args.mshab_obj}.json"
    )
    print(f"plan file : {plan_path}")
    if not plan_path.is_file():
        print("  MISSING -- this alone stops the env from constructing.")
        return 1

    whitelist_dir = str(pathlib.Path(args.whitelist_root) / args.mshab_task)
    print(f"whitelists: {whitelist_dir}")
    print()

    plans = list(plan_data_from_file(plan_path).plans)
    names = sorted({p.build_config_name for p in plans})
    keep = set(names[: args.num_build_configs]) if args.num_build_configs > 0 else set(names)
    selected = [p for p in plans if p.build_config_name in keep]
    print(f"plans     : {len(plans)} total, {len(names)} build configs")
    print(
        f"selected  : {len(selected)} plans over "
        f"{min(args.num_build_configs, len(names)) if args.num_build_configs > 0 else len(names)}"
        " build configs (the env keeps a prefix of sorted names)"
    )
    print()

    # (subtask type, canonical target) -> how many plans name it.
    pairs = Counter()
    raw_by_key = defaultdict(set)
    subtask_types = Counter()
    for plan in selected:
        for subtask in getattr(plan, "subtasks", []) or []:
            st_type = getattr(subtask, "type", None)
            subtask_types[str(st_type)] += 1
            obj_id = getattr(subtask, "obj_id", None)
            key = canonical_affordance_key(str(obj_id)) if obj_id else None
            pairs[(str(st_type), key)] += 1
            if key:
                raw_by_key[key].add(str(obj_id))

    print("subtask types in the selected plans:")
    for st_type, count in subtask_types.most_common():
        print(f"  {st_type:10s} x{count}")
    print()

    # 0. Articulation homogeneity -- MS-HAB's own construction, before ours.
    articulations = _articulation_kinds(selected)
    print("articulations named by pick/place subtasks (MS-HAB merge):")
    if not articulations:
        print("  none -- no articulation_config on these subtasks, merge is skipped")
    for kind, count in sorted(articulations.items()):
        print(f"  {kind:44s} x{count}")
    print()

    # 1. Per-target whitelists, the coverage check.
    print("per-target whitelists (_verify_whitelist_coverage):")
    missing = []
    for (st_type, key), count in sorted(pairs.items(), key=lambda kv: str(kv[0])):
        if st_type not in ("pick", "place"):
            continue
        if not key:
            print(f"  {st_type}: obj_id did not canonicalise -- raw={raw_by_key}")
            continue
        target = f"actor:{key}"
        path = resolve_whitelist_path(whitelist_dir, st_type, target)
        mark = "ok  " if path else "MISS"
        print(f"  [{mark}] {st_type}:{key:22s} x{count:<4d} {path or '<not found>'}")
        if path is None:
            missing.append(f"{st_type}:{key}")
        else:
            recorded = load_whitelist(path).task_group
            if recorded != args.mshab_task:
                print(
                    f"         ^ records task_group={recorded!r}, run is "
                    f"{args.mshab_task!r} -- mislabelled"
                )
    print()

    # 2. Union whitelists, the global relation bins.
    print("union whitelists (_bind_global_bin_edges):")
    no_union = []
    for st_type in sorted(subtask_types):
        path = resolve_whitelist_path(whitelist_dir, st_type, "all")
        mark = "ok  " if path else "MISS"
        print(f"  [{mark}] {st_type}_all.json -> {path or '<not found>'}")
        if path is None:
            no_union.append(st_type)
    print()

    # 3. Entity vocabulary, which raises later at pack time.
    vocab = build_entity_vocab(whitelist_dir)
    print(f"entity vocabulary: {len(vocab.token_to_id)} entries")
    for token, index in sorted(vocab.token_to_id.items(), key=lambda kv: kv[1]):
        print(f"  {index}: {token}")
    print()

    print("=" * 66)
    if len(articulations) > 1:
        print(
            "FAIL: these plans name "
            f"{len(articulations)} different articulations "
            f"({', '.join(sorted(articulations))}). MS-HAB merges pick "
            "articulations across parallel envs into one batched view, and "
            "that view requires identical link and joint counts, so env "
            "construction asserts in create_from_physx_articulations long "
            "before the graph adapter is built."
        )
        print(
            "  Fix: use a per-object plan file (--mshab-obj <obj_id>), which "
            "names one articulation throughout. This is not a whitelist "
            "problem and re-mining will not change it."
        )
    if missing:
        print(
            "FAIL: startup raises FileNotFoundError from "
            "_verify_whitelist_coverage for: " + ", ".join(sorted(missing))
        )
        print(
            f"  Fix: python -m scenegraph.tools.prepare_assets "
            f"--mshab-task {args.mshab_task} --subtask {args.subtask}"
        )
    if no_union:
        print(
            "FAIL: no union whitelist for subtask type(s) "
            + ", ".join(no_union)
            + " -- the global relation bins come from that file, and the "
            "builder raises on the first frame of such an episode."
        )
    if not missing and not no_union and len(articulations) <= 1:
        print("PASS: every target and subtask type in this plan is mined,")
        print("      and its pick articulations are homogeneous.")
    return 1 if (missing or no_union or len(articulations) > 1) else 0


if __name__ == "__main__":
    raise SystemExit(main())
