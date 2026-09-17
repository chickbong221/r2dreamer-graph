"""How long the scripted demos are, and how many of them fit a step budget.

Imitation learning off the motion-planning solutions runs into a number nobody
publishes: the plan's length. The solver is not written to the task's registered
horizon -- it is written to reach the goal -- so PegInsertionSide-v1 ends an
episode at 100 steps while its own scripted solution routinely takes more than
that to insert the peg. A demo longer than the horizon the policy is evaluated
under is a demo of a behaviour the policy is never given time to reproduce.

Three numbers per episode, and they are not the same one:

* **total steps** -- every control step the solver took, including the retreat
  after the goal was reached. What a recorded trajectory file will contain.
* **steps to success** -- the step the task's own success flag first went true.
* **settled steps** -- the step the flag went true *for the last time*, never
  going false again before the episode ended. This is the length to compare
  against a horizon, because it is where a demo can actually be cut. The two
  differ whenever the flag flickers, and it does: PickCube's success asks for a
  static robot as well as a placed cube, so a mid-reach pause sets it early.

The flicker matters more than it sounds. A handful of seeds are flagged
successful at reset -- the goal region happened to contain the object already --
and those episodes demonstrate nothing at all while scoring the shortest length
in the sample. A step budget applied to the first-success step selects for them,
which is why ``success_at_first_step`` is reported and those episodes are
counted out of the yields.

Read from already-mined evidence, no simulator::

    python -m scenegraph.tools.survey_demo_lengths --from-evidence

Or measured live, which is the only way to see what a knob does::

    python -m scenegraph.tools.survey_demo_lengths \
        --env-id PegInsertionSide-v1 --episodes 50 --thresholds 100 150

Two knobs shorten a plan, and both are measured rather than assumed:

* ``--joint-vel-limits`` / ``--joint-acc-limits`` override what the task's own
  solution asked for. The trajectory is time-parameterised, so a solution that
  planned at 0.5 of the joint limits takes roughly twice the steps of one at
  0.9 -- and is that much less likely to shake the peg loose on the way.
* ``--control-freq`` re-times the whole episode. The planner discretises at the
  env's control timestep, so halving the control frequency halves the step count
  for the same motion, at the cost of a coarser target for the PD controller to
  track. A demo collected this way is only usable by a policy running at the
  same control frequency.

Neither knob moves the fixed costs: the gripper open and close are a set number
of control steps each, whatever the arm is doing.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from scenegraph.figures.rollout import MotionPlanRunner
# One definition of "did this episode succeed" for the whole repo: a survey that
# disagreed with the miner would be measuring a different set of episodes.
from scenegraph.tools.collect_maniskill_interactions import success_flag

DEFAULT_EVIDENCE = "data/maniskill_evidence"
DEFAULT_THRESHOLDS = (50, 100, 150, 200, 250)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def describe(values: Sequence[int]) -> Dict[str, Any]:
    """The spread of a length sample, or an empty dict for an empty one."""
    if not values:
        return {}
    arr = np.asarray(sorted(int(v) for v in values))
    p10, p25, p50, p75, p90 = np.percentile(arr, [10, 25, 50, 75, 90])
    return {
        "n": int(arr.size),
        "min": int(arr[0]),
        "p10": int(round(p10)),
        "p25": int(round(p25)),
        "median": float(p50),
        "p75": int(round(p75)),
        "p90": int(round(p90)),
        "max": int(arr[-1]),
        "mean": float(arr.mean()),
    }


def line(label: str, stats: Dict[str, Any]) -> str:
    if not stats:
        return f"  {label:<18} (none)"
    return (f"  {label:<18} min={stats['min']:>5} p10={stats['p10']:>5} "
            f"p25={stats['p25']:>5} med={stats['median']:>7.1f} "
            f"p75={stats['p75']:>5} p90={stats['p90']:>5} "
            f"max={stats['max']:>5}")


def yields(lengths: Sequence[int], thresholds: Sequence[int],
           attempts: int) -> List[Dict[str, Any]]:
    """For each budget: how many demos clear it, and what one costs.

    ``attempts`` is the denominator that matters when planning a collection
    run. A task whose solver succeeds nine times in ten and clears the budget
    one time in three needs roughly four attempts per kept demo, and that is
    the number to multiply by the per-episode wall clock.
    """
    rows = []
    total = max(int(attempts), 1)
    for threshold in thresholds:
        kept = int(sum(1 for v in lengths if v <= int(threshold)))
        rows.append({
            "threshold": int(threshold),
            "kept": kept,
            "of_successes": (kept / len(lengths)) if lengths else 0.0,
            "of_attempts": kept / total,
            "attempts_per_demo": (total / kept) if kept else None,
        })
    return rows


def print_yields(rows: Sequence[Dict[str, Any]]) -> None:
    print(f"  {'budget':>8} {'kept':>8} {'of succ':>9} {'of attempts':>12} "
          f"{'attempts/demo':>14}")
    for row in rows:
        cost = row["attempts_per_demo"]
        print(f"  {row['threshold']:>8} {row['kept']:>8} "
              f"{row['of_successes']:>8.1%} {row['of_attempts']:>11.1%} "
              f"{(f'{cost:.1f}' if cost else '--'):>14}")


# --------------------------------------------------------------------------- #
# Offline: the lengths already mined
# --------------------------------------------------------------------------- #
def from_evidence(root: Path, thresholds: Sequence[int]) -> Dict[str, Any]:
    """Lengths read back from ``collect_maniskill_interactions`` shards.

    Those shards keep one trace per *successful* episode, with the episode's
    frame count and the frames over which the success predicate held. That is
    exactly the two numbers this tool reports, already paid for -- so the
    baseline needs no simulator. What it cannot say is anything about the
    failures, because a failed episode was never committed: read the success
    rate off a live run instead.
    """
    shards = sorted(root.glob("*/*.pkl"))
    if not shards:
        raise SystemExit(f"no evidence shards under {root}")
    out: Dict[str, Any] = {}
    for shard in shards:
        with open(shard, "rb") as handle:
            payload = pickle.load(handle)
        env_id = str(payload.get("env_id") or shard.parent.name)
        totals, firsts, settled, degenerate = [], [], [], 0
        for trace in payload.get("traces") or []:
            frames = int(trace.get("frames") or 0)
            if frames <= 0:
                continue
            totals.append(frames)
            runs = [tuple(int(v) for v in run)
                    for run in ((trace.get("predicates") or {}).get("success") or ())]
            if not runs:
                continue
            # Onset frames are 0-based; a demo cut at one is that many steps
            # plus the one that reached it.
            if runs[0][0] == 0:
                degenerate += 1
                continue
            firsts.append(runs[0][0] + 1)
            # The last run is the settled one only if it ran to the final
            # frame. A solver that reached the goal and then knocked the object
            # back out has no step this demo could be cut at.
            if runs[-1][1] >= frames - 1:
                settled.append(runs[-1][0] + 1)
        if not totals:
            continue
        entry = {
            "source": str(shard),
            "episodes": len(totals),
            "success_at_first_step": degenerate,
            "total_steps": describe(totals),
            "steps_to_success": describe(firsts),
            "settled_steps": describe(settled),
            # Denominator is the successful episodes: the shard holds no others.
            "yields": yields(settled, thresholds, len(totals)),
        }
        out.setdefault(env_id, entry)
        print(f"\n[{env_id}] {len(totals)} successful episodes "
              f"({shard.parent.name})")
        print(line("total steps", entry["total_steps"]))
        print(line("steps to success", entry["steps_to_success"]))
        print(line("settled steps", entry["settled_steps"]))
        if degenerate:
            print(f"  {degenerate} episode(s) were already flagged successful "
                  f"at the first step and demonstrate nothing; excluded")
        print_yields(entry["yields"])
    print("\nnote: these are successful episodes only -- the shards never "
          "committed a failure, so the yields above are per success, not per "
          "attempt. Run the live path for the attempt cost.", flush=True)
    return out


# --------------------------------------------------------------------------- #
# Live: the lengths under a given set of knobs
# --------------------------------------------------------------------------- #
def make_survey_env(env_id: str, *, control_mode: str, sim_backend: str,
                    control_freq: int = 0, max_episode_steps: int = 0):
    """A single-env task wired for the scripted solver and nothing else.

    ``obs_mode="none"`` because this measures step counts: rendering a camera
    per step would multiply the run time of a survey by an order of magnitude
    for pixels nothing here reads. Batch size one on CPU is not a tunable --
    the solutions read ``pose.sp``, which exists only for an unbatched pose.
    """
    import gymnasium as gym
    import mani_skill.envs                                 # noqa: F401

    kwargs: Dict[str, Any] = dict(
        id=env_id,
        obs_mode="none",
        control_mode=control_mode,
        render_mode="rgb_array",
        sim_backend=sim_backend,
    )
    if int(control_freq) > 0:
        kwargs["sim_config"] = dict(control_freq=int(control_freq))
    if int(max_episode_steps) > 0:
        # Raised only so a truncation does not sit in the middle of the trace.
        # The solver ignores the flag either way; what changes is what a
        # recorded trajectory would look like.
        kwargs["max_episode_steps"] = int(max_episode_steps)
    return gym.make(**kwargs)


@contextlib.contextmanager
def planner_limits(vel: float, acc: float):
    """Force the solver's velocity and acceleration limits for this run.

    The scripted solutions construct their own planner and hard-code these --
    PegInsertionSide asks for a slow one because a fast approach shakes the peg
    in the fingers. There is no argument to ``solve`` for it, so the constructor
    is wrapped for the duration of the survey and restored after, rather than
    the solutions being forked.
    """
    if vel <= 0 and acc <= 0:
        yield []
        return
    from mani_skill.examples.motionplanning.panda import motionplanner

    patched = []
    for name, obj in list(vars(motionplanner).items()):
        if not inspect.isclass(obj) or not name.endswith("MotionPlanningSolver"):
            continue
        try:
            params = inspect.signature(obj.__init__).parameters
        except (TypeError, ValueError):                    # noqa: BLE001
            continue
        if "joint_vel_limits" not in params:
            continue

        def wrap(original):
            def __init__(self, *args, **kwargs):
                if vel > 0:
                    kwargs["joint_vel_limits"] = float(vel)
                if acc > 0:
                    kwargs["joint_acc_limits"] = float(acc)
                return original(self, *args, **kwargs)
            return __init__

        patched.append((obj, obj.__init__))
        obj.__init__ = wrap(obj.__init__)
    try:
        yield [cls.__name__ for cls, _ in patched]
    finally:
        for cls, original in patched:
            cls.__init__ = original


class LengthTrace:
    """Step counts for the episode in flight.

    ``_run_start`` is cleared on every unsuccessful step, so whatever it holds
    when the episode ends is the onset of the terminal success run -- and it
    holds nothing at all if the last step was not a success, which is the case
    a demo cannot be cut from.
    """

    def __init__(self) -> None:
        self.on_reset(None)

    def on_reset(self, obs) -> None:
        del obs
        self.steps = 0
        self.first_success: Optional[int] = None
        self.success_at_first_step = False
        self._run_start: Optional[int] = None

    def on_step(self, obs, info) -> None:
        del obs
        self.steps += 1
        if not success_flag(info):
            self._run_start = None
            return
        if self._run_start is None:
            self._run_start = self.steps
        if self.first_success is None:
            self.first_success = self.steps
            self.success_at_first_step = self.steps == 1

    @property
    def settled(self) -> Optional[int]:
        return self._run_start


def episode_horizon(env) -> Optional[int]:
    """The task's own step limit, read back from whichever setting applied."""
    try:
        from mani_skill.utils import gym_utils

        value = gym_utils.find_max_episode_steps_value(env)
    except Exception:                                      # noqa: BLE001
        value = None
    return int(value) if value else None


def survey(args) -> Dict[str, Any]:
    env = make_survey_env(
        args.env_id,
        control_mode=args.control_mode,
        sim_backend=args.sim_backend,
        control_freq=args.control_freq,
        max_episode_steps=args.max_episode_steps,
    )
    horizon = episode_horizon(env)
    control_freq = getattr(env.unwrapped, "control_freq", None)
    trace = LengthTrace()
    rows: List[Dict[str, Any]] = []

    with planner_limits(args.joint_vel_limits, args.joint_acc_limits) as names:
        runner = MotionPlanRunner(
            env, args.env_id, on_reset=trace.on_reset, on_step=trace.on_step)
        print(f"[survey] {args.env_id} horizon={horizon} "
              f"control_freq={control_freq} control_mode={args.control_mode}"
              + (f" limits={args.joint_vel_limits}/{args.joint_acc_limits} "
                 f"on {names}" if names else ""), flush=True)
        try:
            for i in range(int(args.episodes)):
                seed = int(args.seed) + i
                attempt = runner.attempt(seed)
                row = {
                    "seed": seed,
                    "success": bool(attempt.success),
                    "total_steps": int(attempt.steps),
                    "steps_to_success": trace.first_success,
                    "settled_steps": trace.settled,
                    "success_at_first_step": trace.success_at_first_step,
                    "error": attempt.error,
                }
                rows.append(row)
                note = f" ({attempt.error})" if attempt.error else ""
                if trace.success_at_first_step:
                    note += " (already successful at reset)"
                print(f"[survey] {i + 1}/{args.episodes} seed={seed} "
                      f"success={attempt.success} steps={attempt.steps} "
                      f"to_success={trace.first_success} "
                      f"settled={trace.settled}{note}", flush=True)
        except KeyboardInterrupt:
            print("[survey] interrupted", flush=True)
        finally:
            env.close()

    # Usable means: it succeeded, it was still succeeding at the last step, and
    # it was not already succeeding at the first one.
    ok = [r for r in rows if r["success"] and r["settled_steps"]
          and not r["success_at_first_step"]]
    degenerate = sum(1 for r in rows if r["success_at_first_step"])
    result = {
        "env_id": args.env_id,
        "horizon": horizon,
        "control_freq": int(control_freq) if control_freq else None,
        "control_mode": args.control_mode,
        "joint_vel_limits": args.joint_vel_limits or None,
        "joint_acc_limits": args.joint_acc_limits or None,
        "attempts": len(rows),
        "successes": sum(1 for r in rows if r["success"]),
        "usable": len(ok),
        "success_at_first_step": degenerate,
        "total_steps": describe([r["total_steps"] for r in ok]),
        "steps_to_success": describe([r["steps_to_success"] for r in ok]),
        "settled_steps": describe([r["settled_steps"] for r in ok]),
        "yields": yields([r["settled_steps"] for r in ok],
                         args.thresholds, len(rows)),
        "episodes": rows,
    }
    rate = result["successes"] / max(len(rows), 1)
    print(f"\n[survey] {result['successes']}/{len(rows)} attempts succeeded "
          f"({rate:.0%}); {len(ok)} usable as demos")
    print(line("total steps", result["total_steps"]))
    print(line("steps to success", result["steps_to_success"]))
    print(line("settled steps", result["settled_steps"]))
    if degenerate:
        print(f"  {degenerate} attempt(s) were already flagged successful at "
              f"the first step and demonstrate nothing; excluded")
    print_yields(result["yields"])
    if horizon and result["settled_steps"]:
        over = sum(1 for r in ok if r["settled_steps"] > horizon)
        print(f"\n[survey] {over}/{len(ok)} usable demos need more than the "
              f"registered horizon of {horizon} steps", flush=True)
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Measure how long the scripted motion-planning demos are "
                    "and how many of them fit a step budget")
    p.add_argument("--env-id", default="PegInsertionSide-v1")
    p.add_argument("--episodes", type=int, default=50,
                   help="attempts, not successes: the failures are the other "
                        "half of what a collection run costs")
    p.add_argument("--seed", type=int, default=0, help="first seed to try")
    p.add_argument("--thresholds", type=int, nargs="*",
                   default=list(DEFAULT_THRESHOLDS),
                   help="step budgets to report the yield at")
    p.add_argument("--from-evidence", nargs="?", const=DEFAULT_EVIDENCE,
                   default="",
                   help="read lengths out of already-mined interaction shards "
                        "instead of running the simulator")
    p.add_argument("--control-mode", default="pd_joint_pos",
                   help="the scripted solutions are written for pd_joint_pos")
    p.add_argument("--sim-backend", default="cpu",
                   help="the solutions read pose.sp, which needs batch size 1")
    p.add_argument("--control-freq", type=int, default=0,
                   help="0 keeps the task's own; lowering it shortens every "
                        "plan proportionally and coarsens the tracking")
    p.add_argument("--joint-vel-limits", type=float, default=0.0,
                   help="0 keeps what the task's solution asked for")
    p.add_argument("--joint-acc-limits", type=float, default=0.0,
                   help="0 keeps what the task's solution asked for")
    p.add_argument("--max-episode-steps", type=int, default=0,
                   help="0 keeps the task's registered horizon")
    p.add_argument("--out", default="",
                   help="write the per-episode rows and the summary here")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = (from_evidence(Path(args.from_evidence), args.thresholds)
              if args.from_evidence else survey(args))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
