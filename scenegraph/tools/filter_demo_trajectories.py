"""Keep the motion-planning demos that reach success inside a step budget.

ManiSkill's motion-planning runner records whatever the solver did, and the
solver is written to reach the goal rather than to fit the task's horizon --
PegInsertionSide-v1 ends an episode at 100 steps while its own scripted
solution takes a median of 135 to insert the peg. So a recorded demo set is
mixed: some episodes fit the budget an imitation policy will be evaluated
under and some are twice it, with nothing in the file to tell them apart.

This reads a recorded ``trajectory.h5`` and its ``.json`` sidecar, and writes
the pair back out holding only the episodes that earn their place:

    python -m scenegraph.tools.filter_demo_trajectories \
        demos/PegInsertionSide-v1/motionplanning/trajectory.h5 \
        --max-steps 150

An episode is kept when three things hold, and each rejects a different kind of
useless demo:

* **It ends successful.** The per-step ``success`` flag is true on the last
  recorded step. An episode that reached the goal and then knocked the object
  back out has no step it could be cut at.
* **It was not successful on its first step.** A few seeds spawn the object
  already satisfying the goal -- one PullCubeTool seed in 1900, two PickCube in
  300. They demonstrate nothing while scoring the shortest length in the file,
  so a budget applied naively selects for exactly them.
* **It settles inside the budget.** Not where the flag *first* flickers true:
  PickCube's success asks for a static robot as well as a placed cube, so a
  pause mid-reach can set it early and drop it again. The step that counts is
  where it goes true for the last time.

Kept episodes are trimmed there, plus ``--pad`` steps of margin. The steps
after success are the solver retreating from a goal it has already reached --
2 to 11 steps, on the tasks measured -- and an imitation policy trained on all
of them learns to let go. The margin exists because a replay re-simulates: cut
to the exact step success arrives and a replay that drifts by one frame lands
on a final step that is not a success, and ``replay_trajectory`` discards the
episode.

**Filter after converting the control mode, not before.** ManiSkill's
``from_pd_joint_pos`` conversion steps the target env twice whenever the delta
action it computed had to be clipped, so a demo recorded in ``pd_joint_pos``
can be up to twice as long once it is ``pd_joint_delta_pos``. A budget met in
the recorded file is not a budget met in the converted one, and the converted
file is the one an imitation policy is trained on::

    python -m mani_skill.trajectory.replay_trajectory \
        --traj-path demos/PegInsertionSide-v1/motionplanning/trajectory.h5 \
        --use-first-env-state -c pd_joint_delta_pos -o state --save-traj
    python -m scenegraph.tools.filter_demo_trajectories \
        demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_joint_delta_pos.h5 \
        --max-steps 150

Run it with ``--dry-run`` on the converted file first: that prints the length
distribution without writing anything, which is how much the conversion cost.

Several recordings can be filtered into one file, which is what a run split
across seed ranges or processes produces. Episode ids are then renumbered,
because two files both start at zero and the h5 group name is the id.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def settled_step(success: np.ndarray) -> Optional[int]:
    """Actions needed to reach the success that held to the end.

    ``success[i]`` is the flag after action ``i``, so an onset at index ``k``
    costs ``k + 1`` actions. None means the episode did not end successful,
    which is the case no cut point exists for.
    """
    flags = np.asarray(success, dtype=bool).reshape(-1)
    if flags.size == 0 or not bool(flags[-1]):
        return None
    unsuccessful = np.flatnonzero(~flags)
    onset = int(unsuccessful[-1]) + 1 if unsuccessful.size else 0
    return onset + 1


def copy_trimmed(src, dst, keep: int, total: int) -> None:
    """Recursively copy an episode group, cutting every per-step axis.

    Two lengths appear in a recording and they are off by one: an action array
    has one entry per step, an observation array has that plus the one from the
    reset. Both are cut to match, and an array of neither length is copied
    whole -- it is not indexed by step and guessing at it would corrupt it.
    """
    import h5py

    for name, node in src.items():
        if isinstance(node, h5py.Group):
            copy_trimmed(node, dst.create_group(name), keep, total)
            continue
        arr = node[()]
        length = arr.shape[0] if getattr(arr, "ndim", 0) else 0
        if length == total:
            arr = arr[:keep]
        elif length == total + 1:
            arr = arr[:keep + 1]
        dst.create_dataset(name, data=arr)


def episode_index(meta: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(ep["episode_id"]): ep for ep in meta.get("episodes") or []}


def scan(path: Path, max_steps: int, keep_unsettled: bool) -> Dict[str, Any]:
    """Every episode in one recording, with its verdict and its cut point."""
    import h5py

    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        raise SystemExit(f"no sidecar beside {path}: expected {sidecar}")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    by_id = episode_index(meta)

    rows: List[Dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        for key in handle:
            if not key.startswith("traj_"):
                continue
            episode_id = int(key.split("_", 1)[1])
            group = handle[key]
            total = int(group["actions"].shape[0])
            if "success" not in group:
                # Without a per-step flag there is no cut point to find. The
                # recorded length is reported so the episode is not silently
                # dropped, but it cannot be trimmed.
                rows.append({
                    "episode_id": episode_id, "key": key, "total": total,
                    "settled": None, "keep": False,
                    "reason": "no success flag recorded",
                })
                continue
            flags = np.asarray(group["success"][()], dtype=bool).reshape(-1)
            settled = settled_step(flags)
            reason = ""
            if settled is None:
                reason = "did not end successful"
            elif bool(flags[0]):
                reason = "already successful at the first step"
            elif settled > max_steps:
                reason = f"settles at {settled} > {max_steps}"
            keep = not reason or (keep_unsettled and settled is not None
                                  and settled <= max_steps and not flags[0])
            rows.append({
                "episode_id": episode_id, "key": key, "total": total,
                "settled": settled, "keep": bool(keep), "reason": reason,
                "meta": by_id.get(episode_id),
            })
    rows.sort(key=lambda r: r["episode_id"])
    return {"path": path, "meta": meta, "rows": rows}


def report(name: str, rows: Sequence[Dict[str, Any]]) -> None:
    kept = [r for r in rows if r["keep"]]
    lengths = sorted(r["settled"] for r in kept if r["settled"])
    print(f"[filter] {name}: {len(kept)}/{len(rows)} episodes kept")
    if lengths:
        median = lengths[len(lengths) // 2]
        print(f"[filter]   settled steps kept: min={lengths[0]} "
              f"med={median} max={lengths[-1]}")
    dropped: Dict[str, int] = {}
    for row in rows:
        if not row["keep"]:
            dropped[row["reason"]] = dropped.get(row["reason"], 0) + 1
    for reason, count in sorted(dropped.items(), key=lambda kv: -kv[1]):
        print(f"[filter]   dropped {count}: {reason}")


def filter_trajectories(args) -> int:
    import h5py

    sources = [Path(p) for p in args.traj_path]
    for path in sources:
        if not path.exists():
            raise SystemExit(f"no such trajectory file: {path}")

    scans = [scan(path, args.max_steps, args.keep_unsettled) for path in sources]
    for entry in scans:
        report(entry["path"].name, entry["rows"])

    env_ids = {str((e["meta"].get("env_info") or {}).get("env_id")) for e in scans}
    if len(env_ids) > 1:
        raise SystemExit(f"refusing to merge different tasks: {sorted(env_ids)}")

    total_kept = sum(1 for e in scans for r in e["rows"] if r["keep"])
    if not total_kept:
        print("[filter] nothing to write: no episode met the budget", flush=True)
        return 1
    if args.dry_run:
        print(f"[filter] dry run: {total_kept} episodes would be written",
              flush=True)
        return 0

    out = Path(args.out) if args.out else sources[0].with_name(
        f"{sources[0].stem}.le{args.max_steps}.h5")
    out.parent.mkdir(parents=True, exist_ok=True)

    # The sidecar of the first input carries env_info -- the task, its kwargs
    # and its horizon -- which the replay tool reads to rebuild the env. Kept
    # verbatim: a filtered file describes the same task it was recorded from.
    meta_out = dict(scans[0]["meta"])
    episodes: List[Dict[str, Any]] = []
    written = 0
    with h5py.File(out, "w") as dst:
        for entry in scans:
            with h5py.File(entry["path"], "r") as src:
                for row in entry["rows"]:
                    if not row["keep"]:
                        continue
                    # Renumbered only when several files are merged: the h5
                    # group name is the episode id, and two recordings both
                    # start at zero.
                    new_id = written if len(scans) > 1 else row["episode_id"]
                    keep = row["total"]
                    if args.trim and row["settled"]:
                        keep = min(row["settled"] + args.pad, row["total"])
                    copy_trimmed(src[row["key"]], dst.create_group(f"traj_{new_id}"),
                                 keep, row["total"])
                    episode = dict(row["meta"] or {})
                    episode["episode_id"] = new_id
                    episode["elapsed_steps"] = int(keep)
                    episode["success"] = True
                    # The budget is on the step success settles; what is
                    # written is that plus the replay margin. Both are recorded
                    # rather than inferred: a consumer that wants the untrimmed
                    # demo has to know one was cut, and from which file.
                    episode["settled_steps"] = int(row["settled"] or keep)
                    if args.trim and row["settled"]:
                        episode["trimmed_from"] = int(row["total"])
                        episode["source_file"] = entry["path"].name
                    episodes.append(episode)
                    written += 1
    meta_out["episodes"] = episodes
    out.with_suffix(".json").write_text(json.dumps(meta_out, indent=2),
                                        encoding="utf-8")
    settled = sorted(int(ep["settled_steps"]) for ep in episodes)
    lengths = sorted(int(ep["elapsed_steps"]) for ep in episodes)
    print(f"\n[filter] wrote {written} episodes to {out}")
    print(f"[filter] settled at: min={settled[0]} "
          f"med={settled[len(settled) // 2]} max={settled[-1]} "
          f"(budget {args.max_steps})")
    print(f"[filter] written length: min={lengths[0]} "
          f"med={lengths[len(lengths) // 2]} max={lengths[-1]} "
          f"(+{args.pad} steps of replay margin)")
    print(f"[filter] sidecar: {out.with_suffix('.json')}", flush=True)
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Keep the recorded motion-planning demos that reach "
                    "success within a step budget, trimmed at that step")
    p.add_argument("traj_path", nargs="+",
                   help="recorded trajectory.h5 file(s); several are merged "
                        "and their episode ids renumbered")
    p.add_argument("--max-steps", type=int, default=150,
                   help="steps to settled success a demo may cost")
    p.add_argument("--out", default="",
                   help="output .h5; the default sits beside the first input "
                        "as <name>.le<max-steps>.h5")
    p.add_argument("--pad", type=int, default=5,
                   help="steps kept after success settles, so a replay that "
                        "drifts still ends on a successful step")
    p.add_argument("--no-trim", dest="trim", action="store_false",
                   help="keep every recorded step, including the retreat "
                        "after the goal was already reached")
    p.add_argument("--keep-unsettled", action="store_true",
                   help="also keep episodes whose success flag dropped again "
                        "before the end; they have no honest cut point")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be kept and write nothing")
    return p.parse_args(argv)


def main(argv=None) -> int:
    return filter_trajectories(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
