"""Collect demos until N of them meet a step budget, on disjoint seeds.

Two things about ManiSkill's own motion-planning runner make it the wrong tool
for assembling a fixed-size imitation dataset.

It counts what it *recorded*, not what is usable. ``-n 500`` writes 500
successful episodes, and on PegInsertionSide-v1 about a third of those take
more than 150 control steps to reach success -- so a 500-episode file yields
some 340 demos once a step budget is applied, and the shortfall is only visible
after the fact. This counts the demos that meet the budget and keeps sampling
until it has the number asked for.

And it parallelises onto seed blocks exactly as wide as the successes each
process must return, while ``--only-count-success`` makes a process walk
forward past every seed whose plan failed. Each process runs off the end of its
block and into the next one's, and the merge that follows renumbers episode ids
without ever comparing seeds -- so the duplicate initial states it produces are
invisible in the output. Here a block is wide enough that a process cannot
reach the next even in the worst case, because it is also the cap on how many
attempts that process may make.

    python -m scenegraph.tools.collect_mp_demos \
        --env-id PegInsertionSide-v1 --num-traj 500 --max-steps 150

An episode is kept when it ends successful, was not already successful on its
first step, and settles inside the budget -- the same three rules
``filter_demo_trajectories`` applies, from the same
:func:`~scenegraph.tools.survey_demo_lengths.LengthTrace`, so the filter that
runs afterwards drops nothing this kept. What it does do is trim each episode
at its success step, which cannot happen here: the recorder writes the buffer
it holds, whole.

The budget applies to ``pd_joint_pos``, the control mode the scripted solutions
are written for. Converting a demo to ``pd_joint_delta_pos`` afterwards can
lengthen it by up to half again, so a set collected to 150 here is not a set of
150-step demos there -- measure the conversion and apply the real budget with
``filter_demo_trajectories`` on the converted file.

Observations are not recorded: ``-o none`` is the runner's default and the
right one here, because the observations a policy trains on are whatever
``replay_trajectory`` re-renders later, in the obs mode the policy actually
uses. Rendering them now would pay for the expensive half of every episode
that the budget then throws away.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Tuple

# A process may attempt this many episodes per demo it owes before its seed
# block is exhausted. Sized for the worst task measured -- PegInsertionSide
# solves about nine attempts in ten and clears a 150-step budget on two thirds
# of those, so it needs about 1.6 attempts per demo. Eight is five times that.
DEFAULT_STRIDE_FACTOR = 8


def worker_args(args, proc_id: int) -> Namespace:
    """The settings one process runs under.

    The env is built exactly as ManiSkill's runner builds it -- same obs mode,
    same control mode, same shader on all three camera groups, same backend --
    because a demo recorded under different settings than the reference one is
    not comparable to it, and the sidecar records those settings as the env
    kwargs a replay will rebuild from.
    """
    return Namespace(
        env_id=str(args.env_id),
        obs_mode=str(args.obs_mode),
        sim_backend=str(args.sim_backend),
        shader=str(args.shader),
        sensor_size=tuple(int(v) for v in (args.sensor_size or ())),
        record_dir=str(args.record_dir),
        max_steps=int(args.max_steps),
        log_every=int(args.log_every),
        traj_name=(str(args.traj_name) if int(args.num_procs) == 1
                   else f"{args.traj_name}.{proc_id}"),
    )


def _run_one(args: Namespace, proc_id: int, start_seed: int, target: int,
             max_attempts: int) -> Tuple[str, Dict[str, Any]]:
    """One process: sample seeds forward until ``target`` demos are kept."""
    import gymnasium as gym
    import mani_skill.envs                                 # noqa: F401
    from mani_skill.utils.wrappers.record import RecordEpisode

    from scenegraph.figures.rollout import MotionPlanRunner
    from scenegraph.tools.survey_demo_lengths import LengthTrace

    # Whatever is passed here is what the recorder writes into the sidecar as
    # env_kwargs -- it reads env.spec.kwargs -- and env_kwargs is what a later
    # replay rebuilds the task from. So the camera size a policy trains at
    # belongs here, set once, rather than edited into the sidecar afterwards.
    sensors: Dict[str, Any] = dict(shader_pack=args.shader)
    if args.sensor_size:
        sensors |= dict(width=int(args.sensor_size[1]),
                        height=int(args.sensor_size[0]))
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sensor_configs=sensors,
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        sim_backend=args.sim_backend,
    )
    recorder = RecordEpisode(
        env,
        output_dir=os.path.join(args.record_dir, args.env_id, "motionplanning"),
        trajectory_name=args.traj_name,
        save_video=False,
        source_type="motionplanning",
        source_desc="official motion planning solution from ManiSkill contributors",
        video_fps=30,
        record_reward=False,
        save_on_reset=False,
    )
    out_path = recorder._h5_file.filename
    trace = LengthTrace()
    runner = MotionPlanRunner(recorder, args.env_id,
                              on_reset=trace.on_reset, on_step=trace.on_step)

    kept, attempts, successes = 0, 0, 0
    settled_kept: List[int] = []
    seed = int(start_seed)
    while kept < target and attempts < max_attempts:
        # Cleared here as well as on the solver's reset: a plan that fails
        # before it reaches ``env.reset`` would otherwise be judged on the
        # previous episode's flags.
        trace.on_reset(None)
        attempts += 1
        attempt = runner.attempt(seed)
        seed += 1
        successes += int(attempt.success)
        settled = trace.settled
        keep = bool(
            attempt.success
            and settled is not None
            and not trace.success_at_first_step
            and settled <= args.max_steps
        )
        # Called either way, and this is what advances the recorder's buffer
        # pointer: an episode that is not saved still has to be cleared, or the
        # next one is written with this one's frames in front of it.
        recorder.flush_trajectory(save=keep)
        if keep:
            kept += 1
            settled_kept.append(int(settled))
        if attempts % args.log_every == 0 or kept == target:
            print(f"[proc {proc_id}] {kept}/{target} kept, {attempts} attempts, "
                  f"{successes} solved, seed at {seed}", flush=True)
    recorder.close()

    stats = {
        "proc_id": proc_id,
        "target": int(target),
        "kept": kept,
        "attempts": attempts,
        "successes": successes,
        "first_seed": int(start_seed),
        "last_seed": seed - 1,
        "exhausted": kept < target,
        "settled": settled_kept,
    }
    if stats["exhausted"]:
        print(f"[proc {proc_id}] WARNING: seed block exhausted at "
              f"{kept}/{target} after {attempts} attempts", flush=True)
    return out_path, stats


def verify_seeds(path: Path) -> Dict[str, Any]:
    """Distinct initial states in a merged file, which is the point of it."""
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    episodes = meta.get("episodes") or []
    seeds = [ep.get("episode_seed") for ep in episodes]
    known = [s for s in seeds if s is not None]
    return {
        "episodes": len(episodes),
        "distinct_seeds": len(set(known)),
        "duplicates": len(known) - len(set(known)),
    }


def summarise(stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    settled = sorted(s for entry in stats for s in entry["settled"])
    attempts = sum(entry["attempts"] for entry in stats)
    kept = sum(entry["kept"] for entry in stats)
    successes = sum(entry["successes"] for entry in stats)
    return {
        "kept": kept,
        "attempts": attempts,
        "successes": successes,
        "solve_rate": successes / max(attempts, 1),
        "budget_yield": kept / max(successes, 1),
        "attempts_per_demo": attempts / max(kept, 1),
        "settled_min": settled[0] if settled else 0,
        "settled_median": settled[len(settled) // 2] if settled else 0,
        "settled_max": settled[-1] if settled else 0,
        "exhausted": [e["proc_id"] for e in stats if e["exhausted"]],
    }


def collect(args) -> int:
    from mani_skill.examples.motionplanning.panda.run import MP_SOLUTIONS
    from mani_skill.trajectory.merge_trajectory import merge_trajectories

    if args.env_id not in MP_SOLUTIONS:
        raise SystemExit(
            f"no motion-planning solution for {args.env_id}; "
            f"have {sorted(MP_SOLUTIONS)}")

    procs = max(int(args.num_procs), 1)
    per_proc = -(-int(args.num_traj) // procs)      # ceil, so the total is met
    stride = int(args.seed_stride) or (per_proc * DEFAULT_STRIDE_FACTOR)
    # The attempt cap is the block width, so a process cannot reach the next
    # process's seeds however badly the solver does. Collisions are ruled out
    # by construction rather than by a rate holding up.
    max_attempts = stride
    starts = [int(args.start_seed) + i * stride for i in range(procs)]

    print(f"[collect] {args.env_id}: {procs} processes x {per_proc} demos "
          f"<= {args.max_steps} steps = {procs * per_proc} total", flush=True)
    print(f"[collect] seed blocks {starts[0]}..{starts[-1] + stride - 1}, "
          f"{stride} wide, {max_attempts} attempts per process", flush=True)

    jobs = [(worker_args(args, i), i, starts[i], per_proc, max_attempts)
            for i in range(procs)]
    if procs == 1:
        results = [_run_one(*jobs[0])]
    else:
        pool = mp.Pool(procs)
        results = pool.starmap(_run_one, jobs)
        pool.close()
        pool.join()

    produced = [Path(path) for path, _ in results]
    stats = [entry for _, entry in results]
    out = Path(args.record_dir) / args.env_id / "motionplanning" / f"{args.traj_name}.h5"
    if len(produced) == 1:
        if produced[0] != out:
            produced[0].replace(out)
            produced[0].with_suffix(".json").replace(out.with_suffix(".json"))
    else:
        merge_trajectories(str(out), [str(p) for p in produced])
        for path in produced:
            os.remove(path)
            os.remove(path.with_suffix(".json"))

    totals = summarise(stats)
    seeds = verify_seeds(out)
    print(f"\n[collect] wrote {seeds['episodes']} demos to {out}")
    print(f"[collect] {totals['attempts']} attempts -> "
          f"{totals['successes']} solved ({totals['solve_rate']:.0%}) -> "
          f"{totals['kept']} within {args.max_steps} steps "
          f"({totals['budget_yield']:.0%} of solved), "
          f"{totals['attempts_per_demo']:.1f} attempts per demo")
    print(f"[collect] settled steps: min={totals['settled_min']} "
          f"med={totals['settled_median']} max={totals['settled_max']}")
    if seeds["duplicates"]:
        print(f"[collect] WARNING: {seeds['duplicates']} episodes repeat a "
              f"seed; only {seeds['distinct_seeds']} distinct initial states",
              flush=True)
    else:
        print(f"[collect] {seeds['distinct_seeds']} distinct seeds, no repeats")
    if totals["exhausted"]:
        short = int(args.num_traj) - totals["kept"]
        print(f"[collect] WARNING: processes {totals['exhausted']} ran out of "
              f"seeds; {max(short, 0)} demos short. Re-run with "
              f"--start-seed {starts[-1] + stride} --num-traj {max(short, 1)} "
              f"and pass both files to filter_demo_trajectories.", flush=True)
        return 1
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Collect motion-planning demos until N of them meet a "
                    "step budget, across processes that cannot collide")
    p.add_argument("--env-id", default="PickCube-v1")
    p.add_argument("--num-traj", type=int, default=500,
                   help="demos to keep -- episodes that meet --max-steps, not "
                        "episodes recorded")
    p.add_argument("--max-steps", type=int, default=150,
                   help="control steps a demo may take to reach the success "
                        "that holds to the end")
    p.add_argument("--num-procs", type=int, default=8,
                   help="CPU processes; the sim is single-env by construction "
                        "because the solutions read pose.sp")
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--seed-stride", type=int, default=0,
                   help="seeds per process, and the attempt cap that keeps it "
                        "inside them; 0 picks eight per demo it owes")
    p.add_argument("--record-dir", default="demos")
    p.add_argument("--traj-name", default="mp",
                   help="base name of the merged trajectory file")
    p.add_argument("--obs-mode", default="none",
                   help="'none' is right here: replay re-renders whatever the "
                        "policy trains on, after the budget has been applied")
    p.add_argument("--sensor-size", type=int, nargs=2, default=None,
                   metavar=("H", "W"),
                   help="camera resolution recorded into env_kwargs, which is "
                        "what a later replay renders at; default is the task's")
    p.add_argument("--sim-backend", default="cpu")
    p.add_argument("--shader", default="default")
    p.add_argument("--log-every", type=int, default=25,
                   help="attempts between progress lines, per process")
    return p.parse_args(argv)


def main(argv=None) -> int:
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # Already set by an outer process; the runner asks for the same one.
        pass
    return collect(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
