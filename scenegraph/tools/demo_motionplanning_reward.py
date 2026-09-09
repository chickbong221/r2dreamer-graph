"""Motion-planning demo for one ManiSkill task, with the reward logged.

Drives the task's own scripted solution -- the same solver the interaction
miner and the paper figures use -- and records what the task paid at every
control step, keeping the episodes that succeeded. Out comes one successful
episode's per-step table (reward, undiscounted return, discounted return,
success flag) as CSV and JSON, and the return figure drawn from it as a PNG.

    python -m scenegraph.tools.demo_motionplanning_reward \
        --env-id PegInsertionSide-v1 --episodes 1 --out data/reward_demos

The figure is redrawable from the CSV alone, which is what makes the expensive
half of this optional: run the episode once where the simulator lives, then
restyle or resize the PNG anywhere the file has been copied to.

    python -m scenegraph.tools.demo_motionplanning_reward \
        --from-csv data/reward_demos/PegInsertionSide-v1/seed0000_success.csv

Two facts about these numbers, both reported rather than hidden:

* The reward mode is the training one -- ``normalized_dense``, stepping down
  ``--reward-fallback`` exactly as ``envs.maniskill`` does -- so a step's reward
  is on the scale the agent is trained against. The control mode is not: the
  scripted solutions are written for ``pd_joint_pos`` while the policy runs
  ``pd_joint_delta_pos``, so step counts, and therefore returns, demonstrate
  the task's reward shape rather than a baseline to beat.
* A scripted plan routinely runs past the task's registered horizon --
  PegInsertionSide-v1 ends an episode at 100 steps and the plan takes more.
  Those later steps are simulation the task already considers over, so the
  return is reported twice: over the whole trace, and over the horizon a policy
  would actually have been given.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from scenegraph.figures.rollout import Attempt, MotionPlanRunner
# One definition of "did this episode succeed" for the whole repo: a demo that
# disagreed with the miner would be illustrating a different set of episodes.
from scenegraph.tools.collect_maniskill_interactions import success_flag

# 1 - 1/333, the discount implied by ``model.horizon: 333`` in the configs. The
# undiscounted return is reported beside it, so this only sets the second
# number, never the first.
DEFAULT_DISCOUNT = 1.0 - 1.0 / 333.0


def scalar(value, env_idx: int = 0) -> float:
    """One env's entry from whatever the task returned.

    ManiSkill hands back a torch tensor even at batch size one, and a task on
    ``sparse`` may hand back a plain float. Same unwrap ``success_flag`` does,
    for the same reason.
    """
    if value is None:
        return float("nan")
    arr = np.asarray(value.cpu() if hasattr(value, "cpu") else value,
                     dtype=float)
    if arr.ndim == 0:
        return float(arr)
    return float(arr.reshape(-1)[min(env_idx, arr.size - 1)])


@dataclass
class StepRecord:
    """One control step, with the running sums already carried forward."""

    step: int                       # 1-based: step 1 is the first action
    reward: float
    ret: float                      # undiscounted return through this step
    discounted: float               # sum of gamma^(t-1) * r_t through here
    success: bool                   # the task's own flag at this step
    terminated: bool
    truncated: bool

    def row(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "reward": round(self.reward, 6),
            "return": round(self.ret, 6),
            "discounted_return": round(self.discounted, 6),
            "success": int(self.success),
            "terminated": int(self.terminated),
            "truncated": int(self.truncated),
        }


class RewardTrace:
    """Per-step reward for one episode, and the summary of it.

    Deliberately simulator-free: it is handed the four things a step returns
    and knows nothing about how they were produced, so the arithmetic here is
    testable without ManiSkill.
    """

    def __init__(self, discount: float = DEFAULT_DISCOUNT, env_idx: int = 0):
        self.discount = float(discount)
        self.env_idx = int(env_idx)
        self.steps: List[StepRecord] = []
        self._ret = 0.0
        self._disc = 0.0

    def reset(self) -> None:
        self.steps.clear()
        self._ret = 0.0
        self._disc = 0.0

    def observe(self, obs, reward, terminated, truncated, info) -> None:
        """The transition hook's signature. ``obs`` goes unread here and is
        kept so this can be handed straight to :class:`MotionPlanRunner`."""
        del obs
        value = scalar(reward, self.env_idx)
        index = len(self.steps)                 # 0-based, for the discount
        self._ret += value
        self._disc += (self.discount ** index) * value
        self.steps.append(StepRecord(
            step=index + 1,
            reward=value,
            ret=self._ret,
            discounted=self._disc,
            success=success_flag(info, self.env_idx),
            terminated=bool(scalar(terminated, self.env_idx)),
            truncated=bool(scalar(truncated, self.env_idx)),
        ))

    # -------------------------------------------------------------- read-out
    @property
    def rewards(self) -> List[float]:
        return [s.reward for s in self.steps]

    def first_success_step(self) -> Optional[int]:
        for record in self.steps:
            if record.success:
                return record.step
        return None

    def first_flag(self, field: str) -> Optional[int]:
        """Step at which ``terminated`` or ``truncated`` first went true."""
        for record in self.steps:
            if getattr(record, field):
                return record.step
        return None

    def summary(self, horizon: Optional[int] = None) -> Dict[str, Any]:
        """Everything a caller would otherwise recompute from the rows.

        ``horizon`` is the task's episode limit. The plan is free to run past
        it -- see the module docstring -- so both returns are reported and the
        overrun is named rather than left to be inferred from a step count.
        """
        rewards = self.rewards
        n = len(rewards)
        first = self.first_success_step()
        within = n if not horizon else min(n, int(horizon))
        return {
            "steps": n,
            "horizon": None if not horizon else int(horizon),
            "steps_past_horizon": max(0, n - within),
            "discount": self.discount,
            "success": first is not None,
            "first_success_step": first,
            "success_at_last_step": bool(self.steps[-1].success) if n else False,
            "success_within_horizon": bool(
                first is not None and (not horizon or first <= int(horizon))
            ),
            "return": float(sum(rewards)),
            "return_within_horizon": float(sum(rewards[:within])),
            "discounted_return": float(self.steps[-1].discounted) if n else 0.0,
            "return_to_first_success": (
                float(sum(rewards[:first])) if first is not None else None
            ),
            "reward_first": rewards[0] if n else None,
            "reward_last": rewards[-1] if n else None,
            "reward_min": float(min(rewards)) if n else None,
            "reward_max": float(max(rewards)) if n else None,
            "reward_mean": float(sum(rewards) / n) if n else None,
            "first_terminated_step": self.first_flag("terminated"),
            "first_truncated_step": self.first_flag("truncated"),
        }

    def write_csv(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(StepRecord(0, 0.0, 0.0, 0.0, False, False, False).row())
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in self.steps:
                writer.writerow(record.row())
        return path


# --------------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------------- #
def printed_steps(trace: RewardTrace, every: int,
                  horizon: Optional[int] = None) -> List[StepRecord]:
    """Rows worth putting on a terminal.

    Every ``every``-th step, plus the ones carrying information no stride is
    guaranteed to land on: the first, the last, the step the task first calls a
    success, the steps the flags first go true, and the horizon itself.
    """
    if not trace.steps:
        return []
    every = max(1, int(every))
    keep = {1, trace.steps[-1].step}
    first = trace.first_success_step()
    if first is not None:
        keep.add(first)
    for field in ("terminated", "truncated"):
        step = trace.first_flag(field)
        if step is not None:
            keep.add(step)
    if horizon and int(horizon) <= trace.steps[-1].step:
        keep.add(int(horizon))
    return [s for s in trace.steps if s.step % every == 0 or s.step in keep]


def print_trace(trace: RewardTrace, *, every: int,
                horizon: Optional[int] = None) -> None:
    print(f"{'step':>6} {'reward':>10} {'return':>12} {'disc.return':>12} "
          f"{'success':>8}  flags")
    for record in printed_steps(trace, every, horizon):
        flags = " ".join(
            name for name, on in (
                ("term", record.terminated),
                ("trunc", record.truncated),
                ("past-horizon", bool(horizon) and record.step > int(horizon)),
            ) if on
        )
        print(f"{record.step:>6} {record.reward:>10.4f} {record.ret:>12.4f} "
              f"{record.discounted:>12.4f} {str(record.success):>8}  {flags}")


SUMMARY_ORDER = (
    "steps", "horizon", "steps_past_horizon", "success", "first_success_step",
    "success_within_horizon", "success_at_last_step", "return",
    "return_within_horizon", "return_to_first_success", "discounted_return",
    "discount", "reward_first", "reward_last", "reward_min", "reward_max",
    "reward_mean", "first_terminated_step", "first_truncated_step",
)


def print_summary(summary: Dict[str, Any]) -> None:
    width = max(len(key) for key in SUMMARY_ORDER)
    for key in SUMMARY_ORDER:
        value = summary.get(key)
        if isinstance(value, float):
            value = f"{value:.4f}"
        print(f"  {key:<{width}}  {value}")


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #
# House style taken from ``draw/plot_success_curves``: serif, a light solid grid
# in both directions, no top or right spine -- so a return curve printed beside
# those success curves does not read as another paper's figure. Restated rather
# than imported because ``draw/`` is a directory of scripts, not a package. The
# colours are that figure's validated set, which keeps its pairs separable under
# colour-vision deficiency where an obvious blue/green pair would not.
FONT_STACK = [
    "CMU Serif", "Latin Modern Roman", "Computer Modern Roman", "cmr10",
    "STIXGeneral", "DejaVu Serif",
]
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d8d8d8"
RETURN_COLOUR = "#1F77B4"
DISCOUNTED_COLOUR = "#C2700A"
SUCCESS_COLOUR = "#9467BD"
HORIZON_COLOUR = "#D62728"


def figure_title(env_id: str, seed: Optional[int], reward_mode: str) -> str:
    """One title spelling, so a redrawn figure names its episode the way the
    run that produced it did.

    ASCII only: the serif stack falls through to whatever a machine has, and a
    dash or a middot that one font lacks prints as a box.
    """
    parts = [str(env_id)]
    if seed is not None:
        parts.append(f"seed {int(seed)}")
    if reward_mode:
        parts.append(str(reward_mode))
    return ", ".join(parts)


def draw_return_figure(trace: RewardTrace, path: Path, *,
                       horizon: Optional[int] = None, title: str = "",
                       dpi: int = 300) -> Optional[Path]:
    """Return against control step, with the per-step reward beneath it.

    Return is the subject and takes the tall panel; the reward strip below is
    what explains the shape it has -- a dense reward climbing steadily, or a
    sparse one paying once at the end. The two vertical rules carry what a
    return curve alone cannot say: where the task first called the episode a
    success, and where its horizon fell when the plan ran past it.

    Optional in the sense that matplotlib is: a headless run that only wants
    the CSV should not fail for want of a plotting library.
    """
    if not trace.steps:
        print("[warn] no figure: the trace holds no steps", flush=True)
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:                               # noqa: BLE001
        print(f"[warn] no figure ({type(exc).__name__}: {exc})", flush=True)
        return None

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": FONT_STACK,
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        # cmr10 warns unless the tick formatter is mathtext-aware.
        "axes.formatter.use_mathtext": True,
    })

    steps = [s.step for s in trace.steps]
    last = trace.steps[-1]
    first = trace.first_success_step()
    marked = bool(horizon) and int(horizon) < steps[-1]
    fig, (top, low) = plt.subplots(
        2, 1, figsize=(5.6, 3.9), sharex=True,
        gridspec_kw=dict(height_ratios=(2.1, 1.0), hspace=0.14),
    )

    top.plot(steps, [s.ret for s in trace.steps], color=RETURN_COLOUR,
             linewidth=2.0, zorder=3)
    top.plot(steps, [s.discounted for s in trace.steps],
             color=DISCOUNTED_COLOUR, linewidth=2.0, linestyle="--", zorder=3)
    # The two final values ride in the legend rather than as labels pinned to
    # the end of each curve: a return curve ends at its own maximum, in the one
    # corner where a label has neither the curve nor the axis out of its way.
    low.plot(steps, [s.reward for s in trace.steps], color=RETURN_COLOUR,
             linewidth=1.3, zorder=3)

    for ax in (top, low):
        if first is not None:
            ax.axvline(first, color=SUCCESS_COLOUR, linewidth=1.2,
                       linestyle=(0, (4, 3)), zorder=2)
        if marked:
            ax.axvline(int(horizon), color=HORIZON_COLOUR, linewidth=1.2,
                       linestyle=(0, (1, 2)), zorder=2)
        ax.grid(True, which="major", color=GRID, linewidth=0.6,
                linestyle="-", zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(steps[0], steps[-1])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(labelsize=9, labelcolor=INK, color=GRID,
                       length=3, width=0.7, pad=2)

    top.set_ylabel("Return", fontsize=11, color=INK, labelpad=4)
    low.set_ylabel("Reward", fontsize=11, color=INK, labelpad=4)
    low.set_xlabel("Control step", fontsize=11, color=INK, labelpad=3)
    if title:
        top.set_title(title, fontsize=10.5, color=INK, pad=6)

    handles = [
        Line2D([], [], color=RETURN_COLOUR, linewidth=2.0,
               label=f"Return (final {last.ret:.2f})"),
        Line2D([], [], color=DISCOUNTED_COLOUR, linewidth=2.0, linestyle="--",
               label=f"Discounted, $\\gamma$ = {trace.discount:.3f} "
                     f"(final {last.discounted:.2f})"),
    ]
    if first is not None:
        handles.append(Line2D([], [], color=SUCCESS_COLOUR, linewidth=1.2,
                              linestyle=(0, (4, 3)),
                              label=f"First success (step {first})"))
    if marked:
        handles.append(Line2D([], [], color=HORIZON_COLOUR, linewidth=1.2,
                              linestyle=(0, (1, 2)),
                              label=f"Task horizon ({int(horizon)})"))
    # "best", not a fixed corner: which corner is empty depends on the task's
    # reward. A dense return fills the upper right and a sparse one leaves the
    # left flat, and a demo run against an unfamiliar task has to survive both.
    top.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc="best", frameon=False, fontsize=8.5,
               labelcolor=INK, handlelength=1.9, handletextpad=0.6,
               borderaxespad=0.4, labelspacing=0.35)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close(fig)
    return path


def read_csv_trace(path: Path,
                   discount: float = DEFAULT_DISCOUNT) -> RewardTrace:
    """Rebuild a trace from a written CSV, so the figure can be redrawn without
    paying for a second simulator run.

    Every column is read verbatim rather than recomputed. A CSV written under
    one discount would otherwise be redrawn under whatever the redraw happened
    to be invoked with, and the figure would quietly disagree with the table
    sitting next to it.
    """
    trace = RewardTrace(discount=discount)
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            trace.steps.append(StepRecord(
                step=int(row["step"]),
                reward=float(row["reward"]),
                ret=float(row["return"]),
                discounted=float(row["discounted_return"]),
                success=bool(int(row["success"])),
                terminated=bool(int(row["terminated"])),
                truncated=bool(int(row["truncated"])),
            ))
    if trace.steps:
        # Same module, so the running sums are set directly: a trace read back
        # has to be able to keep accumulating from where the file left off.
        trace._ret = trace.steps[-1].ret
        trace._disc = trace.steps[-1].discounted
    return trace


# --------------------------------------------------------------------------- #
# The env
# --------------------------------------------------------------------------- #
def make_demo_env(env_id: str, *, reward_mode: str,
                  reward_fallback: Sequence[str], control_mode: str,
                  sim_backend: str, obs_mode: str,
                  max_episode_steps: int = 0):
    """A single-env task on the training reward, wired for the scripted solver.

    Batch size one on CPU is not a tunable: the scripted solutions read
    ``pose.sp``, which exists only for an unbatched pose -- the same constraint
    ``collect_maniskill_interactions`` runs under.

    Returns ``(env, reward_mode)``, the second being the mode actually
    implemented: ``_make_with_supported_reward`` rewrites the kwargs it is
    handed as it steps down the fallback chain, and the point of a reward demo
    is to say which reward it is a demo of.
    """
    import gymnasium as gym                                # noqa: F401
    import mani_skill.envs                                 # noqa: F401
    # Private by module convention only. Reused rather than reimplemented so
    # this lands on the same reward the trainer would get for the task,
    # including the announced step down to sparse.
    from envs.maniskill import _make_with_supported_reward

    kwargs: Dict[str, Any] = dict(
        id=env_id,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode="rgb_array",
        sim_backend=sim_backend,
        reward_mode=str(reward_mode),
    )
    if int(max_episode_steps) > 0:
        kwargs["max_episode_steps"] = int(max_episode_steps)
    env = _make_with_supported_reward(kwargs, list(reward_fallback or []))
    return env, str(kwargs["reward_mode"])


def episode_horizon(env) -> Optional[int]:
    """The task's own step limit, read back from whichever setting applied."""
    try:
        from mani_skill.utils import gym_utils

        value = gym_utils.find_max_episode_steps_value(env)
    except Exception:                                      # noqa: BLE001
        value = None
    return int(value) if value else None


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def write_episode(trace: RewardTrace, summary: Dict[str, Any],
                  attempt: Attempt, args, out: Path, reward_mode: str,
                  horizon: Optional[int]) -> Dict[str, Path]:
    """CSV of the steps, JSON of the summary, and optionally the plot.

    Named by seed and outcome, so a ``--keep-failures`` run cannot leave a
    failed episode's table looking like a successful one's.
    """
    tag = f"seed{attempt.seed:04d}_{'success' if attempt.success else 'failed'}"
    paths = {"csv": out / f"{tag}.csv", "json": out / f"{tag}.json"}
    trace.write_csv(paths["csv"])
    paths["json"].write_text(json.dumps({
        "env_id": args.env_id,
        "reward_mode": reward_mode,
        "reward_mode_requested": args.reward_mode,
        "control_mode": args.control_mode,
        "sim_backend": args.sim_backend,
        "obs_mode": args.obs_mode,
        "horizon": horizon,
        "attempt": attempt.to_dict(),
        "summary": summary,
        "steps": [record.row() for record in trace.steps],
    }, indent=2), encoding="utf-8")
    if args.plot:
        png = draw_return_figure(
            trace, out / f"{tag}.png", horizon=horizon, dpi=args.dpi,
            title=figure_title(args.env_id, attempt.seed, reward_mode))
        if png is not None:
            paths["figure"] = png
    return paths


def replot(args) -> int:
    """Redraw the figure from an episode CSV: no simulator, no second run.

    The sidecar JSON is read when it is there, because two of the figure's
    facts -- the discount the discounted column was written under, and the
    horizon the rules are drawn at -- are properties of the run that produced
    the CSV and not of the redraw.
    """
    source = Path(args.from_csv)
    if not source.exists():
        raise SystemExit(f"no such episode CSV: {source}")
    sidecar = source.with_suffix(".json")
    meta = (json.loads(sidecar.read_text(encoding="utf-8"))
            if sidecar.exists() else {})
    summary = dict(meta.get("summary") or {})
    discount = float(summary.get("discount", args.discount))
    horizon = summary.get("horizon")
    if horizon is None and args.max_episode_steps > 0:
        horizon = int(args.max_episode_steps)

    trace = read_csv_trace(source, discount)
    if not trace.steps:
        raise SystemExit(f"{source} holds no steps")
    attempt = dict(meta.get("attempt") or {})
    title = figure_title(
        meta.get("env_id", args.env_id), attempt.get("seed"),
        meta.get("reward_mode", ""))
    print(f"redrawing {source} ({len(trace.steps)} steps, "
          f"horizon={horizon}, discount={discount:.4f})", flush=True)
    print_trace(trace, every=args.print_every, horizon=horizon)
    print_summary(trace.summary(horizon))

    target = Path(args.figure) if args.figure else source.with_suffix(".png")
    png = draw_return_figure(trace, target, horizon=horizon, title=title,
                             dpi=args.dpi)
    if png is None:
        return 1
    print(f"wrote figure: {png}", flush=True)
    return 0


def run(args) -> int:
    env, reward_mode = make_demo_env(
        args.env_id,
        reward_mode=args.reward_mode,
        reward_fallback=args.reward_fallback,
        control_mode=args.control_mode,
        sim_backend=args.sim_backend,
        obs_mode=args.obs_mode,
        max_episode_steps=args.max_episode_steps,
    )
    horizon = episode_horizon(env)
    trace = RewardTrace(discount=args.discount)
    runner = MotionPlanRunner(
        env, args.env_id,
        # Cleared on reset rather than before ``solve``: the solutions call
        # ``env.reset(seed=...)`` themselves, and anything stepped before that
        # reset belongs to no episode.
        on_reset=lambda obs: trace.reset(),
        on_transition=trace.observe,
    )

    out = Path(args.out) / args.env_id
    print(f"env={args.env_id} reward_mode={reward_mode} "
          f"control_mode={args.control_mode} horizon={horizon} "
          f"discount={args.discount:.4f}", flush=True)

    kept: List[Path] = []
    written: List[Dict[str, Any]] = []
    seed, attempts = int(args.seed), 0
    try:
        while len(kept) < args.episodes and attempts < args.max_attempts:
            attempts += 1
            attempt = runner.attempt(seed)
            seed += 1
            note = f" ({attempt.error})" if attempt.error else ""
            print(f"\n[{attempts}] seed={attempt.seed} "
                  f"success={attempt.success} steps={attempt.steps}{note}",
                  flush=True)
            if not attempt.success and not args.keep_failures:
                print("  dropped (nothing in this episode succeeded)",
                      flush=True)
                continue
            if not trace.steps:
                print("  dropped (no step reached the reward hook)", flush=True)
                continue
            summary = trace.summary(horizon)
            print_trace(trace, every=args.print_every, horizon=horizon)
            print_summary(summary)
            paths = write_episode(trace, summary, attempt, args, out,
                                  reward_mode, horizon)
            written.append({
                "attempt": attempt.to_dict(),
                "summary": summary,
                "files": {key: str(path) for key, path in paths.items()},
            })
            for label, path in paths.items():
                print(f"  wrote {label}: {path}", flush=True)
            if attempt.success:
                kept.append(paths["csv"])
    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
    finally:
        env.close()

    if written:
        index = out / "episodes.json"
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(json.dumps(written, indent=2), encoding="utf-8")
        print(f"\nindex: {index}", flush=True)
    print(f"{len(kept)}/{args.episodes} successful episodes logged "
          f"in {attempts} attempts", flush=True)
    if not kept:
        print("no episode succeeded; raise --max-attempts, or check that the "
              f"scripted solution for {args.env_id} runs on this install")
        return 1
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run a task's scripted motion-planning solution and log "
                    "the per-step reward and return of a successful episode",
    )
    p.add_argument("--env-id", default="PegInsertionSide-v1")
    p.add_argument("--out", default="data/reward_demos")
    p.add_argument("--episodes", type=int, default=1,
                   help="successful episodes to log")
    p.add_argument("--seed", type=int, default=0, help="first seed to try")
    p.add_argument("--max-attempts", type=int, default=25)
    p.add_argument("--keep-failures", action="store_true",
                   help="also log the attempts that never reached success")

    p.add_argument("--reward-mode", default="normalized_dense",
                   help="what env.reward_mode is set to for training")
    p.add_argument("--reward-fallback", nargs="*", default=["sparse"],
                   help="modes to try, in order, if the task does not "
                        "implement --reward-mode; empty is strict")
    p.add_argument("--discount", type=float, default=DEFAULT_DISCOUNT,
                   help="only affects the discounted-return column")
    p.add_argument("--control-mode", default="pd_joint_pos",
                   help="the scripted solutions are written for pd_joint_pos")
    p.add_argument("--sim-backend", default="cpu",
                   help="the solutions read pose.sp, which needs batch size 1")
    p.add_argument("--obs-mode", default="none",
                   help="nothing here reads the observation; 'none' is fastest")
    p.add_argument("--max-episode-steps", type=int, default=0,
                   help="0 keeps the task's registered horizon; a large value "
                        "lets the whole scripted plan run inside one episode")

    p.add_argument("--print-every", type=int, default=10,
                   help="print every Nth step; the first, last, first success "
                        "and terminal steps are always printed")
    p.add_argument("--no-plot", dest="plot", action="store_false",
                   help="write the CSV and JSON only; the figure is the point "
                        "of the demo, so it is drawn by default")
    p.add_argument("--dpi", type=int, default=300,
                   help="figure resolution; 300 is print size")
    p.add_argument("--from-csv", default="",
                   help="redraw the figure from an episode CSV a previous run "
                        "wrote, instead of running the simulator at all")
    p.add_argument("--figure", default="",
                   help="where --from-csv writes the PNG; the default sits "
                        "beside the CSV it was drawn from")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    # The redraw path touches no simulator, which is the whole reason it
    # exists: the figure can be regenerated wherever the CSV was copied to.
    return replot(args) if args.from_csv else run(args)


if __name__ == "__main__":
    sys.exit(main())
