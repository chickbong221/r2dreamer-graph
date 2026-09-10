"""Reward demo for one MS-HAB checkpoint, on the task it was trained for.

The sibling ``demo_motionplanning_reward`` logs what a scripted plan earns on a
tabletop task. This logs what a *trained policy* earns on an MS-HAB subtask:
same per-step table, same CSV and figure, driven by a released checkpoint
instead of a motion planner.

    python -m scenegraph.tools.demo_mshab_checkpoint_reward \
        --ckpt-dir /root/projects/ReLDreamer/mshab_checkpoints/rl/set_table/close/fridge

Everything about the episode is read from the checkpoint's own ``config.yml``
rather than restated on the command line -- env id, task plan, spawn data,
frame stack, the Fetch action mask, the force-penalty kwargs, and the episode
length. A reward is a property of the environment it was earned in, and an
environment rebuilt from half a config is a different environment: the force
penalty alone (``robot_force_mult``) is a term in the reward.

The wrapper stack is the one ``collect_robot_success_states`` already rolls
these same checkpoints under, and the policy is loaded by
``adapters.policy_loader``, which matches ``mshab/evaluate.py``. What is added
here is only the reward trace.

Two things this refuses to do quietly:

* Roll out a random policy. ``policy_loader`` falls back to random actions when
  a checkpoint will not load, which is right for a data collector and wrong
  for a reward demo -- a reward curve labelled with a checkpoint path has to
  have come from that checkpoint. Pass ``--allow-random`` to override.
* Invent an episode length. ``--max-episode-steps 0`` takes the config's, and
  the run prints which section it came from: the released configs carry two
  (``env`` for training, ``eval_env`` for evaluation) and they differ.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# One definition of the reward table, its summary, its CSV and its figure. This
# tool differs from the motion-planning demo in how an episode is produced, and
# in nothing about how it is recorded -- so a curve from a checkpoint and a
# curve from a scripted plan stay directly comparable, and ``--from-csv`` over
# there redraws what is written here.
from scenegraph.tools.demo_motionplanning_reward import (
    DEFAULT_DISCOUNT, RewardTrace, draw_reward_figure, figure_title,
    print_summary, print_trace, scalar,
)

# The released MS-HAB configs carry both. ``eval_env`` is the one a trained
# policy is evaluated under and the one with the longer horizon; ``env`` is the
# training rollout. Neither is a safe silent default, so the choice is a flag
# and the value that applied is printed.
CONFIG_SECTIONS = ("eval_env", "env")

# What the checkpoint tree looks like on the project server. Both files sit in
# the same directory, which is why one ``--ckpt-dir`` names both.
POLICY_NAME = "policy.pt"
CONFIG_NAME = "config.yml"


def load_ckpt_config(ckpt_dir: Path) -> Dict[str, Any]:
    """The checkpoint's ``config.yml``, as plain data."""
    path = Path(ckpt_dir) / CONFIG_NAME
    if not path.exists():
        raise SystemExit(f"no {CONFIG_NAME} in {ckpt_dir}")
    import yaml

    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def env_section(config: Dict[str, Any], section: str) -> Dict[str, Any]:
    """One of the config's two env blocks, named rather than guessed."""
    block = config.get(section)
    if not isinstance(block, dict):
        available = [key for key in CONFIG_SECTIONS if key in config]
        raise SystemExit(
            f"config has no {section!r} section; it has {available}")
    return dict(block)


def resolve_path(value: Any) -> Path:
    """A config path as the config wrote it, with ``~`` expanded.

    The plan and spawn files are taken from the config rather than rebuilt from
    task/subtask/target: the config names the exact pair the policy was trained
    against, and a rebuilt path is a guess that happens to be right.
    """
    return Path(str(value)).expanduser()


def _unexpected_kwarg(message: str) -> Optional[str]:
    match = re.search(r"unexpected keyword argument '([^']+)'", message)
    return match.group(1) if match else None


def make_dropping_unknown(make, kwargs: Dict[str, Any], limit: int = 8):
    """``make(**kwargs)``, dropping the ones this mshab build does not accept.

    The released configs carry fields that different mshab versions take in
    different places -- ``continuous_task`` is one. Passing everything and
    dropping what is rejected, loudly, beats two worse options: failing a GPU
    run outright, or pre-emptively omitting fields that this build would have
    honoured.
    """
    kwargs = dict(kwargs)
    for _ in range(limit):
        try:
            return make(**kwargs)
        except TypeError as exc:
            name = _unexpected_kwarg(str(exc))
            if name is None or name not in kwargs:
                raise
            print(f"[env] this mshab build does not accept {name!r}; "
                  "dropping it", flush=True)
            kwargs.pop(name)
    return make(**kwargs)


def make_mshab_env(section: Dict[str, Any], *, num_envs: int,
                   max_episode_steps: int, sensor_size: Tuple[int, int]):
    """The wrapper stack the released policies were trained under.

    Copied in shape from ``collect_robot_success_states._build_env``, which
    already rolls these checkpoints out on this project's server, with two
    differences: no contact-data collector (nothing here mines anything), and
    the config's own ``env_kwargs`` are passed through, because
    ``robot_force_mult`` and ``robot_force_penalty_min`` are terms in the
    reward this tool exists to report.
    """
    import gymnasium as gym
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    import mshab.envs                                      # noqa: F401
    from mshab.envs.planner import plan_data_from_file
    from mshab.envs.wrappers import (
        FetchActionWrapper, FetchDepthObservationWrapper, FrameStack,
    )

    plan_fp = resolve_path(section["task_plan_fp"])
    spawn_fp = resolve_path(section["spawn_data_fp"])
    if not plan_fp.exists():
        raise SystemExit(f"missing task plan: {plan_fp}")
    plan_data = plan_data_from_file(plan_fp)
    plans = list(plan_data.plans)
    if not plans:
        raise SystemExit(f"{plan_fp} contained no plans")
    # Cycle the plans when there are fewer than envs, as the collector does.
    task_plans = [plans[i % len(plans)] for i in range(max(1, num_envs))]

    kwargs: Dict[str, Any] = dict(
        id=str(section["env_id"]),
        num_envs=max(1, int(num_envs)),
        obs_mode="rgb+depth+segmentation",
        sim_backend="gpu",
        robot_uids="fetch",
        control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense",
        render_mode="all",
        shader_dir="minimal",
        max_episode_steps=int(max_episode_steps),
        task_plans=task_plans,
        scene_builder_cls=plan_data.dataset,
        spawn_data_fp=spawn_fp,
        require_build_configs_repeated_equally_across_envs=False,
        add_event_tracker_info=True,
        continuous_task=bool(section.get("continuous_task", True)),
        sensor_configs=dict(width=int(sensor_size[1]),
                            height=int(sensor_size[0])),
        **dict(section.get("env_kwargs") or {}),
    )
    env = make_dropping_unknown(gym.make, kwargs)
    env = FetchDepthObservationWrapper(
        env,
        cat_state=bool(section.get("cat_state", True)),
        cat_pixels=bool(section.get("cat_pixels", False)),
    )
    env = FrameStack(
        env,
        num_stack=int(section.get("frame_stack", 3)),
        stacking_keys=["fetch_head_depth", "fetch_hand_depth"],
    )
    env = FetchActionWrapper(
        env,
        stationary_base=bool(section.get("stationary_base", False)),
        stationary_torso=bool(section.get("stationary_torso", False)),
        stationary_head=bool(section.get("stationary_head", True)),
    )
    venv = ManiSkillVectorEnv(
        env, ignore_terminations=True,
        max_episode_steps=int(max_episode_steps),
    )
    return venv, plan_fp


def run_episode(venv, policy, trace: RewardTrace, horizon: int, seed: int,
                env_idx: int = 0) -> RewardTrace:
    """One episode into ``trace``, from ``seed``.

    Terminations are ignored by the vector env -- the MS-HAB subtasks are
    continuous -- so an episode is ``horizon`` steps unless truncation lands
    early, which is what makes the step count here a constant rather than a
    property of the policy.
    """
    obs, _ = venv.reset(seed=int(seed))
    trace.reset()
    for _ in range(int(horizon)):
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = venv.step(action)
        trace.observe(obs, reward, terminated, truncated, info)
        if bool(scalar(truncated, env_idx)):
            break
    return trace


def write_episode(trace: RewardTrace, summary: Dict[str, Any], *, out: Path,
                  seed: int, success: bool, args, section: Dict[str, Any],
                  plan_fp: Path, algo: str, horizon: int,
                  title: str) -> Dict[str, Path]:
    """Same three files the motion-planning demo writes, with the metadata that
    identifies a checkpoint rollout rather than a scripted one."""
    tag = f"seed{int(seed):04d}_{'success' if success else 'failed'}"
    paths = {"csv": out / f"{tag}.csv", "json": out / f"{tag}.json"}
    trace.write_csv(paths["csv"])
    paths["json"].write_text(json.dumps({
        # ``env_id`` keeps the name the sibling tool's --from-csv reads for a
        # redraw's title.
        "env_id": str(section.get("env_id", "")),
        "title": title,
        "checkpoint": str(Path(args.ckpt_dir) / POLICY_NAME),
        "config": str(Path(args.ckpt_dir) / CONFIG_NAME),
        "algo": algo,
        "config_section": args.config_section,
        "task_plan": str(plan_fp),
        "horizon": int(horizon),
        "reward_mode": "normalized_dense",
        "control_mode": "pd_joint_delta_pos",
        "num_envs": int(args.num_envs),
        "env_idx": int(args.env_idx),
        "env_kwargs": dict(section.get("env_kwargs") or {}),
        "attempt": {"seed": int(seed), "success": bool(success),
                    "steps": len(trace.steps), "error": None},
        "summary": summary,
        "steps": [record.row() for record in trace.steps],
    }, indent=2, default=str), encoding="utf-8")
    if args.plot:
        png = draw_reward_figure(trace, out / f"{tag}.png", title=title,
                                 dpi=args.dpi)
        if png is not None:
            paths["figure"] = png
    return paths


def run(args) -> int:
    ckpt_dir = Path(args.ckpt_dir)
    policy_fp = ckpt_dir / POLICY_NAME
    if not policy_fp.exists():
        raise SystemExit(f"no {POLICY_NAME} in {ckpt_dir}")
    config = load_ckpt_config(ckpt_dir)
    section = env_section(config, args.config_section)
    algo = str((config.get("algo") or {}).get("name", "")) or "unknown"

    horizon = int(args.max_episode_steps or section.get("max_episode_steps", 0))
    if horizon <= 0:
        raise SystemExit(
            f"{args.config_section}.max_episode_steps is missing from the "
            "config; pass --max-episode-steps")
    env_id = str(section.get("env_id", ""))
    title = args.title or figure_title(env_id)

    print(f"checkpoint={policy_fp}", flush=True)
    print(f"algo={algo} env_id={env_id} section={args.config_section} "
          f"horizon={horizon} num_envs={args.num_envs} "
          f"discount={args.discount:.4f}", flush=True)

    venv, plan_fp = make_mshab_env(
        section, num_envs=args.num_envs, max_episode_steps=horizon,
        sensor_size=args.sensor_size)
    print(f"task_plan={plan_fp}", flush=True)

    from scenegraph.adapters.policy_loader import load_policy

    # The policy's constructor is shaped from a sample observation, so the env
    # has to have been reset once before it can be built.
    sample_obs, _ = venv.reset(seed=args.seed)
    policy = load_policy(str(ckpt_dir), venv, sample_obs, args.device)
    if policy.kind == "random" and not args.allow_random:
        venv.close()
        raise SystemExit(
            "the checkpoint did not load and policy_loader fell back to random "
            "actions; a reward curve from random actions is not a demo of this "
            "checkpoint. Fix the load (see the traceback above) or pass "
            "--allow-random if random is genuinely what you want")

    out = Path(args.out) / (args.name or f"{env_id}_{plan_fp.stem}")
    trace = RewardTrace(discount=args.discount, env_idx=args.env_idx)
    kept: List[Path] = []
    written: List[Dict[str, Any]] = []
    seed, attempts = int(args.seed), 0
    try:
        while len(kept) < args.episodes and attempts < args.max_attempts:
            attempts += 1
            run_episode(venv, policy, trace, horizon, seed, args.env_idx)
            success = any(record.success for record in trace.steps)
            print(f"\n[{attempts}] seed={seed} success={success} "
                  f"steps={len(trace.steps)}", flush=True)
            if not success and not args.keep_failures:
                print("  dropped (nothing in this episode succeeded)",
                      flush=True)
                seed += 1
                continue
            summary = trace.summary(horizon)
            print_trace(trace, every=args.print_every, horizon=horizon)
            print_summary(summary)
            paths = write_episode(
                trace, summary, out=out, seed=seed, success=success,
                args=args, section=section, plan_fp=plan_fp, algo=algo,
                horizon=horizon, title=title)
            written.append({
                "attempt": {"seed": seed, "success": success,
                            "steps": len(trace.steps)},
                "summary": summary,
                "files": {key: str(path) for key, path in paths.items()},
            })
            for label, path in paths.items():
                print(f"  wrote {label}: {path}", flush=True)
            if success:
                kept.append(paths["csv"])
            seed += 1
    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
    finally:
        venv.close()

    if written:
        index = out / "episodes.json"
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(json.dumps(written, indent=2), encoding="utf-8")
        print(f"\nindex: {index}", flush=True)
    print(f"{len(kept)}/{args.episodes} successful episodes logged "
          f"in {attempts} attempts", flush=True)
    if not kept:
        print("no episode succeeded; raise --max-attempts, or check that this "
              "checkpoint matches the task plan in its own config")
        return 1
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Roll out an MS-HAB checkpoint on its own task and log the "
                    "per-step reward and return of a successful episode",
    )
    p.add_argument("--ckpt-dir", required=True,
                   help=f"directory holding {POLICY_NAME} and {CONFIG_NAME}")
    p.add_argument("--out", default="data/reward_demos")
    p.add_argument("--name", default="",
                   help="output subdirectory; defaults to <env id>_<plan>")
    p.add_argument("--title", default="",
                   help="figure title; defaults to the env id, which for an "
                        "MS-HAB subtask does not name the target")
    p.add_argument("--episodes", type=int, default=1,
                   help="successful episodes to log")
    p.add_argument("--seed", type=int, default=0, help="first seed to try")
    p.add_argument("--max-attempts", type=int, default=10)
    p.add_argument("--keep-failures", action="store_true",
                   help="also log the episodes that never reached success")

    p.add_argument("--config-section", default=CONFIG_SECTIONS[0],
                   choices=list(CONFIG_SECTIONS),
                   help="which env block of the checkpoint's config to rebuild "
                        "from; the released configs give the two different "
                        "episode lengths")
    p.add_argument("--max-episode-steps", type=int, default=0,
                   help="0 takes the config section's own value")
    p.add_argument("--num-envs", type=int, default=1,
                   help="parallel envs; only --env-idx is traced")
    p.add_argument("--env-idx", type=int, default=0,
                   help="which parallel env the trace follows")
    p.add_argument("--sensor-size", type=int, nargs=2, default=[128, 128],
                   metavar=("H", "W"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow-random", action="store_true",
                   help="proceed even if the checkpoint failed to load and the "
                        "actions are random")

    p.add_argument("--discount", type=float, default=DEFAULT_DISCOUNT,
                   help="only affects the discounted-return column")
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--no-plot", dest="plot", action="store_false",
                   help="write the CSV and JSON only")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args(argv)


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
