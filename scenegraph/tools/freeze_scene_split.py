"""Freeze Experiment B's scene split from the installed MS-HAB task plans.

The manifest is the experiment's record of which apartments were trained in
and which were held out, so it is written once from the dataset that is
actually installed rather than assumed. Run this on the machine the training
runs on, before the budget is spent:

    python -m scenegraph.tools.freeze_scene_split \
        --task tidy_house --subtask pick --obj 004_sugar_box --split train

It reads the plan file, applies ``envs.scene_manifest.split_scenes`` and
writes ``configs/scenes/mshab_pick_b.json``: five arrangements of the pinned
scene's apartment to train in, and thirty drawn evenly from the apartments
outside it to be held out. Re-running it on the same
dataset rewrites the same file byte for byte; re-running it on a different
dataset is a different experiment and the diff says so.

``--check`` compares the shipped manifest against the installed plans and
changes nothing, which is what the pre-run validation calls.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def _scene_manifest():
    """Import the loader without pulling in ``envs``' torch dependencies."""
    path = os.path.join(REPO_ROOT, "envs", "scene_manifest.py")
    spec = importlib.util.spec_from_file_location("scene_manifest", path)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: the module defines dataclasses, and those
    # resolve their own module out of sys.modules while being built.
    sys.modules["scene_manifest"] = module
    spec.loader.exec_module(module)
    return module


def available_scenes(task: str, subtask: str, obj: str, split: str):
    """Every build configuration the installed task plan actually contains."""
    from mani_skill import ASSET_DIR
    from mshab.envs.planner import plan_data_from_file

    path = (ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
            / "task_plans" / task / subtask / split / f"{obj}.json")
    if not path.is_file():
        raise FileNotFoundError(f"MS-HAB task plan not found: {path}")
    plans = plan_data_from_file(path).plans
    return sorted({str(plan.build_config_name) for plan in plans}), str(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="tidy_house")
    parser.add_argument("--subtask", default="pick")
    parser.add_argument("--obj", default="004_sugar_box")
    parser.add_argument("--split", default="train")
    parser.add_argument("--pinned",
                        default="v3_sc0_staging_00.scene_instance.json",
                        help="the original single training scene; it stays a "
                             "training scene and carries the lighting panel")
    parser.add_argument("--train-scenes", type=int, default=5)
    parser.add_argument("--held-out-scenes", type=int, default=30)
    parser.add_argument("--out", default=os.path.join(
        REPO_ROOT, "configs", "scenes", "mshab_pick_b.json"))
    parser.add_argument("--check", action="store_true",
                        help="verify the shipped manifest against the "
                             "installed plans; write nothing")
    args = parser.parse_args(argv)

    manifest = _scene_manifest()
    names, plan_path = available_scenes(
        args.task, args.subtask, args.obj, args.split)
    print(f"[scenes] {len(names)} build configuration(s) in {plan_path}")
    computed = manifest.split_scenes(
        names, args.pinned, args.train_scenes, args.held_out_scenes)

    if args.check:
        shipped = manifest.load_manifest(args.out)
        problems = []
        for name in shipped.evaluation:
            if name not in names:
                problems.append(f"{name} is not in the installed task plan")
        if shipped.train != computed.train:
            problems.append(f"train scenes differ from the split rule: "
                            f"{shipped.train} vs {computed.train}")
        if shipped.held_out != computed.held_out:
            problems.append("held-out scenes differ from the split rule")
        if problems:
            for line in problems:
                print(f"[scenes] MISMATCH: {line}")
            return 1
        print(f"[scenes] manifest matches the installed dataset: "
              f"{manifest.counts(shipped)}")
        return 0

    payload = {
        "_comment": (
            "Frozen by scenegraph/tools/freeze_scene_split.py. Experiment B "
            "trains in `train` and reports scene generalisation on `held_out`; "
            "`lighting_scene` is the single original training scene the "
            "0.4x/1.0x/2.0x comparison runs on. Edit by re-running the tool, "
            "not by hand."),
        "task": args.task,
        "subtask": args.subtask,
        "object": args.obj,
        "split": args.split,
        "available": len(names),
        "lighting_scene": computed.lighting_scene,
        "train": computed.train,
        "held_out": computed.held_out,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"[scenes] wrote {args.out}: {manifest.counts(computed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
