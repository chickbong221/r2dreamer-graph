"""MS-HAB paper frames: the human view, and what the robot's cameras saw change.

``render_multicamera_paper_frames`` does this for a tabletop ManiSkill task
driven by its scripted solution. An MS-HAB subtask has no scripted solution --
the episode comes from a released RL checkpoint -- and the figure wanted from it
is a different one, so this is a separate tool rather than a flag on that one:

* ``frames/human``   the third-person render camera at figure resolution. What
                     the reader looks at to follow the episode.
* ``diff/head``      ``|f[t] - f[t-1]|`` for the Fetch head camera.
* ``diff/wrist``     the same for the wrist camera.

No graph is built and no whitelist is read: this figure is about pixels.

    python -m scenegraph.tools.render_mshab_paper_frames \
        --ckpt-dir /root/projects/ReLDreamer/mshab_checkpoints/rl/set_table/close/fridge \
        --config-section env --seed 0 --max-frames 60 \
        --out data/paper_figures

``--random-policy`` rolls uniform random actions instead of the checkpoint, in
the same env: ``--ckpt-dir`` is still read for its config, but the weights are
never loaded. The action sampler is seeded with ``--seed``, so the same command
gives the same episode, and the default name gains a ``_random`` suffix so it
sits beside the checkpoint's episode rather than colliding with it.

Why the episode is rolled out twice
-----------------------------------
The released checkpoints consume *depth* from the two Fetch cameras, and the
agent's CNN is constructed from the observation shape before its weights are
loaded -- so the sensors have to stay at the 128px the checkpoint was trained
on, which is far below figure resolution. Rendering the sensors large and
downsampling for the policy would feed it an image it was never trained on and
could quietly change the trajectory.

So the episode is produced once and photographed once. Pass one builds the env
exactly as ``demo_mshab_checkpoint_reward`` does -- same wrapper stack, same
config, same 128px sensors -- and records the actions the checkpoint took. Pass
two rebuilds the same env with the cameras at figure resolution, resets to the
same seed, and replays those actions with no policy in the loop.

That only works if the replay lands on the same trajectory, so it is checked
rather than assumed: pass two's own per-step reward is compared against pass
one's, and a mismatch beyond ``--reward-tol`` aborts and discards the staged
episode. A figure captioned with a reward curve has to come from the episode
that produced it.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from scenegraph.core.mask_extractor import read_unwrapped_rgbs
from scenegraph.figures.diff_writer import DiffEpisodeWriter, episode_path
# One definition of the env, the wrapper stack and the checkpoint's config.
# Re-deriving any of it here would let the figure's episode drift away from the
# reward curve that is printed beside it.
from scenegraph.tools.demo_mshab_checkpoint_reward import (
    CONFIG_NAME, CONFIG_SECTIONS, POLICY_NAME, env_section, load_ckpt_config,
    make_mshab_env, resolve_path,
)
from scenegraph.tools.demo_motionplanning_reward import figure_title, scalar

# 500x500 prints at ~1.7in / 300dpi per panel, so a head and a wrist panel sit
# side by side in one column. Same size the tabletop two-camera figure uses, so
# the two figures in the same paper are the same size.
PAPER_SENSOR_SIZE: Tuple[int, int] = (500, 500)
# 1000x1000 is a full single-column figure, and the size
# ``figures.render_camera`` already renders the human view at.
PAPER_HUMAN_SIZE: Tuple[int, int] = (1000, 1000)
# Not a tunable: the released MS-HAB agents' CNNs are built from this shape.
POLICY_SENSOR_SIZE: Tuple[int, int] = (128, 128)
# Roles are aliases the caller sets, never inferred from the sensor list: which
# camera is "the wrist" is a fact about the robot, and a task that renamed its
# cameras should fail the preflight rather than have one guessed for it.
DEFAULT_HEAD_CAMERA = "fetch_head"
DEFAULT_WRIST_CAMERA = "fetch_hand"
# The MS-HAB configs are written for one traced env; the figure follows row 0.
ENV_IDX = 0


@dataclass
class RecordedEpisode:
    """What pass one saw, and what pass two has to reproduce.

    The actions are stored exactly as they were handed to ``venv.step`` -- that
    is, before ``FetchActionWrapper`` -- so replaying them through the same
    wrapper stack reproduces the same joint targets.
    """

    seed: int
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    successes: List[bool] = field(default_factory=list)
    truncated_at: Optional[int] = None

    @property
    def steps(self) -> int:
        return len(self.actions)

    @property
    def first_success_step(self) -> Optional[int]:
        for index, flag in enumerate(self.successes):
            if flag:
                return index + 1              # step 1 is after the first action
        return None


def record_policy_episode(venv, policy, *, seed: int, steps: int) -> RecordedEpisode:
    """Roll the checkpoint for ``steps`` control steps and keep what it did."""
    episode = RecordedEpisode(seed=int(seed))
    obs, _ = venv.reset(seed=int(seed))
    for index in range(int(steps)):
        action = np.asarray(policy.act(obs))
        obs, reward, _terminated, truncated, info = venv.step(action)
        episode.actions.append(np.array(action, copy=True))
        episode.rewards.append(float(scalar(reward, ENV_IDX)))
        episode.successes.append(
            bool(scalar(info.get("success", 0), ENV_IDX) > 0.5)
        )
        if bool(scalar(truncated, ENV_IDX)):
            episode.truncated_at = index + 1
            break
    return episode


def seeded_random_policy(venv, seed: int):
    """Uniform random actions, drawn from a sampler seeded with ``seed``.

    An unseeded gym space draws from OS entropy, so two runs of the same command
    would roll different episodes under the same name.
    """
    from scenegraph.adapters.policy_loader import _random_policy

    venv.action_space.seed(int(seed))
    return _random_policy(venv)


def preflight(frames: Dict[str, np.ndarray], human: Optional[np.ndarray], *,
              roles: Dict[str, str], sensor_size: Sequence[int],
              human_size: Sequence[int]) -> None:
    """Check the camera contract against pass two's reset observation.

    Every one of these is silent if it is left to fail later: a camera renamed
    by a different mshab build reads as a KeyError forty steps into a replay,
    and a sensor left at its default size produces a figure whose panels are a
    quarter of the intended resolution. All problems are collected so one run
    reports the whole mismatch.
    """
    height, width = (int(v) for v in sensor_size)
    problems: List[str] = []
    if len(set(roles.values())) != len(roles):
        problems.append(
            f"two roles name the same camera ({roles}); the figure needs a "
            "distinct view per role"
        )
    for role, camera in roles.items():
        if camera not in frames:
            problems.append(
                f"{role} camera {camera!r} renders no rgb in this env; "
                f"have {sorted(frames)}"
            )
            continue
        shape = np.asarray(frames[camera]).shape
        if tuple(shape[:2]) != (height, width):
            problems.append(
                f"{role} camera {camera!r} renders {tuple(shape[:2])}, "
                f"expected {(height, width)}"
            )
    if human is None:
        problems.append(
            "the env returned no human render frame; it has to be built with "
            "a render_mode that renders the human cameras"
        )
    else:
        expected = (int(human_size[0]), int(human_size[1]))
        shape = tuple(np.asarray(human).shape[:2])
        if shape != expected:
            problems.append(
                f"the human render camera renders {shape}, expected {expected}"
            )
    if problems:
        raise SystemExit("preflight failed:\n  - " + "\n  - ".join(problems))


def render_human(venv, env_idx: int = ENV_IDX) -> Optional[np.ndarray]:
    """The human render camera as ``[H, W, 3]`` uint8, or None."""
    frame = venv.render()
    if frame is None:
        return None
    arr = frame.detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame)
    if arr.ndim == 4:
        arr = arr[min(env_idx, arr.shape[0] - 1)]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr[..., :3])


def replay_and_export(venv, episode: RecordedEpisode, writer: DiffEpisodeWriter,
                      *, roles: Dict[str, str], args) -> Dict[str, Any]:
    """Replay the recorded actions with the cameras turned up, exporting frames.

    Returns what the replay proved: the largest reward disagreement with pass
    one and whether the success flags matched. The caller decides what to do
    with a disagreement -- this function does not commit anything.
    """
    obs, _ = venv.reset(seed=episode.seed)
    # Pass one reset twice before its episode (once to shape the policy, once to
    # start), and so does this, so the two passes put the same number of resets
    # between ``gym.make`` and the first action.
    obs, _ = venv.reset(seed=episode.seed)

    human = render_human(venv)
    frames = read_unwrapped_rgbs(venv, ENV_IDX)
    preflight(frames, human, roles=roles, sensor_size=args.sensor_size,
              human_size=args.human_size)

    def sensors_by_role(rgbs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {role: rgbs[camera] for role, camera in roles.items()}

    writer.write_step(step=0, human=human, sensors=sensors_by_role(frames),
                      extra={"reward": None, "success": False})

    reward_gap, success_matches = 0.0, True
    for index, action in enumerate(episode.actions):
        if writer.count >= args.max_frames:
            break
        _obs, reward, _terminated, _truncated, info = venv.step(action)
        replay_reward = float(scalar(reward, ENV_IDX))
        replay_success = bool(scalar(info.get("success", 0), ENV_IDX) > 0.5)
        reward_gap = max(reward_gap, abs(replay_reward - episode.rewards[index]))
        success_matches = success_matches and (
            replay_success == episode.successes[index]
        )
        writer.write_step(
            step=index + 1,
            human=render_human(venv),
            sensors=sensors_by_role(read_unwrapped_rgbs(venv, ENV_IDX)),
            extra={"reward": replay_reward, "success": replay_success},
        )
        if (index + 1) % args.print_every == 0:
            print(f"  [{index + 1:4d}/{episode.steps}] reward={replay_reward:.6f} "
                  f"success={replay_success} gap={reward_gap:.2e}", flush=True)
    return {"max_reward_abs_diff": float(reward_gap),
            "success_flags_match": bool(success_matches)}


def run(args) -> int:
    ckpt_dir = Path(args.ckpt_dir)
    if not (ckpt_dir / POLICY_NAME).exists():
        raise SystemExit(f"no {POLICY_NAME} in {ckpt_dir}")
    config = load_ckpt_config(ckpt_dir)
    section = env_section(config, args.config_section)
    algo = str((config.get("algo") or {}).get("name", "")) or "unknown"

    # The horizon is the config's, not the number of frames wanted: truncation
    # at a different step is a different environment, and this figure has to be
    # the episode the reward demo already published.
    horizon = int(args.max_episode_steps or section.get("max_episode_steps", 0))
    if horizon <= 0:
        raise SystemExit(
            f"{args.config_section}.max_episode_steps is missing from the "
            "config; pass --max-episode-steps")
    env_id = str(section.get("env_id", ""))
    plan_fp = resolve_path(section["task_plan_fp"])
    name = args.name or (f"{env_id}_{plan_fp.stem}_seed{int(args.seed):04d}"
                         + ("_random" if args.random_policy else ""))
    roles = {"head": args.head_camera, "wrist": args.wrist_camera}

    root = Path(args.out)
    destination = episode_path(root, name)
    if destination.exists() and not args.overwrite:
        raise SystemExit(
            f"{destination} already exists; pass --overwrite to replace it, or "
            "write to a different --out")
    if args.max_frames < 2:
        raise SystemExit(
            f"--max-frames {args.max_frames} exports no difference at all; a "
            "consecutive-frame difference needs at least two frames")

    print(f"checkpoint={ckpt_dir / POLICY_NAME}"
          + (" (config only; --random-policy)" if args.random_policy else ""),
          flush=True)
    print(f"algo={algo} env_id={env_id} section={args.config_section} "
          f"horizon={horizon} seed={args.seed} frames={args.max_frames}",
          flush=True)

    # ------------------------------------------------- pass one: the policy
    print(f"\n[pass 1/2] rolling "
          f"{'random actions' if args.random_policy else 'the checkpoint'} "
          f"at policy resolution {tuple(args.policy_sensor_size)}", flush=True)
    venv, plan_fp = make_mshab_env(
        section, num_envs=1, max_episode_steps=horizon,
        sensor_size=tuple(args.policy_sensor_size))
    print(f"task_plan={plan_fp}", flush=True)
    try:
        from scenegraph.adapters.policy_loader import load_policy

        # Reset here either way: the replay resets twice before its first
        # action because this pass does, and the two have to agree.
        sample_obs, _ = venv.reset(seed=int(args.seed))
        if args.random_policy:
            policy = seeded_random_policy(venv, int(args.seed))
        else:
            policy = load_policy(str(ckpt_dir), venv, sample_obs, args.device)
        if (policy.kind == "random" and not args.random_policy
                and not args.allow_random):
            raise SystemExit(
                "the checkpoint did not load and policy_loader fell back to "
                "random actions; frames captioned with this checkpoint have to "
                "have come from it. Fix the load (see the traceback above) or "
                "pass --allow-random if random is genuinely what you want")
        episode = record_policy_episode(
            venv, policy, seed=int(args.seed), steps=args.max_frames - 1)
    finally:
        venv.close()
    print(f"[pass 1/2] {episode.steps} actions recorded, "
          f"first success at step {episode.first_success_step}", flush=True)
    if not args.keep_failures and episode.first_success_step is None:
        raise SystemExit(
            f"nothing succeeded in the first {episode.steps} steps of seed "
            f"{args.seed}; try another --seed, raise --max-frames, or pass "
            "--keep-failures if the figure is about the attempt")

    # ------------------------------------------------- pass two: the cameras
    print(f"\n[pass 2/2] replaying at figure resolution "
          f"{tuple(args.sensor_size)} sensors / {tuple(args.human_size)} human",
          flush=True)
    venv, _ = make_mshab_env(
        section, num_envs=1, max_episode_steps=horizon,
        sensor_size=tuple(args.sensor_size),
        # "rgb_array" renders the human cameras, which is what ``venv.render()``
        # has to return here; "all" tiles every sensor into one composite.
        render_mode="rgb_array",
        # The whole-env shader is dropped so the two camera groups can be set
        # separately: the sensors and the human view are the figure now, not a
        # 128px input to a CNN, so neither is left on "minimal".
        shader_dir="",
        sensor_shader=args.sensor_shader,
        human_render_size=tuple(args.human_size),
        human_render_shader=args.human_shader,
    )
    writer = DiffEpisodeWriter(
        root, name, roles=list(roles),
        human_size=tuple(args.human_size), sensor_size=tuple(args.sensor_size),
        save_human=not args.no_human, save_frames=not args.diffs_only,
        max_gain=args.diff_max_gain, overwrite=args.overwrite,
    )
    writer.open()
    try:
        replay = replay_and_export(venv, episode, writer, roles=roles, args=args)
    except BaseException:
        writer.discard()
        raise
    finally:
        venv.close()

    # The replay is only worth photographing if it is the same episode.
    if replay["max_reward_abs_diff"] > args.reward_tol or not replay["success_flags_match"]:
        writer.discard()
        raise SystemExit(
            "the replay diverged from the policy rollout "
            f"(max reward difference {replay['max_reward_abs_diff']:.3e} > "
            f"--reward-tol {args.reward_tol:.3e}; success flags "
            f"{'matched' if replay['success_flags_match'] else 'differed'}). "
            "Nothing was written: these frames would not be the episode the "
            "reward curve came from")

    gains = writer.write_amplified(
        percentile=args.diff_percentile, gain=args.diff_gain,
        invert=args.diff_invert)
    print("\ndifference gains: " + ", ".join(
        f"{role}={value:.2f}x" for role, value in gains.items()), flush=True)

    path = writer.commit({
        "env_id": env_id,
        "title": args.title or figure_title(env_id),
        # A random episode names no checkpoint: nothing in it came from one.
        "policy": policy.kind,
        "checkpoint": None if policy.kind == "random" else str(ckpt_dir / POLICY_NAME),
        "config": str(ckpt_dir / CONFIG_NAME),
        "algo": algo,
        "config_section": args.config_section,
        "task_plan": str(plan_fp),
        "seed": int(args.seed),
        "horizon": horizon,
        "reward_mode": "normalized_dense",
        "control_mode": "pd_joint_delta_pos",
        "robot_uids": "fetch",
        "env_kwargs": dict(section.get("env_kwargs") or {}),
        "frame_cameras": dict(roles, human="human render camera"),
        "human_size": list(args.human_size),
        "sensor_size": list(args.sensor_size),
        "policy_sensor_size": list(args.policy_sensor_size),
        "shaders": {"sensor": args.sensor_shader or "env default",
                    "human": args.human_shader or "env default"},
        # Stated rather than implied: nothing here annotates, crops or resizes,
        # and no scene graph was built.
        "frame_annotations": False,
        "graph": None,
        "diff": {
            "reference": "previous exported frame",
            "percentile": args.diff_percentile,
            "max_gain": args.diff_max_gain,
            "gain_override": args.diff_gain,
            "inverted": bool(args.diff_invert),
            "gains": gains,
        },
        "replay": dict(replay, reward_tol=args.reward_tol,
                       policy_steps=episode.steps),
        "attempt": {
            "seed": int(args.seed),
            "success": episode.first_success_step is not None,
            "first_success_step": episode.first_success_step,
            "steps": episode.steps,
            "truncated_at": episode.truncated_at,
            "error": None,
        },
    })
    print(f"\nwrote {writer.count} frames and {max(0, writer.count - 1)} "
          f"differences per camera to {path}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Export the human view and the head/wrist frame "
                    "differences of one MS-HAB checkpoint episode",
    )
    p.add_argument("--ckpt-dir", required=True,
                   help=f"directory holding {POLICY_NAME} and {CONFIG_NAME}")
    p.add_argument("--out", default="data/paper_figures")
    p.add_argument("--name", default="",
                   help="episode directory name; defaults to "
                        "<env id>_<plan>_seed<NNNN>")
    p.add_argument("--title", default="",
                   help="figure title recorded in the manifest; defaults to "
                        "the env id, which for an MS-HAB subtask does not name "
                        "the target")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true",
                   help="replace an episode directory that already exists; off "
                        "by default so a figure already in a paper cannot be "
                        "rewritten by accident")
    p.add_argument("--keep-failures", action="store_true",
                   help="export even if nothing succeeded within the exported "
                        "frames")

    p.add_argument("--config-section", default="env",
                   choices=list(CONFIG_SECTIONS),
                   help="which env block of the checkpoint's config to rebuild "
                        "from; the released configs give the two different "
                        "episode lengths")
    p.add_argument("--max-episode-steps", type=int, default=0,
                   help="0 takes the config section's own value")
    p.add_argument("--max-frames", type=int, default=60,
                   help="frames to export, counting the reset frame; the "
                        "episode is stepped exactly this many times minus one")

    p.add_argument("--sensor-size", type=int, nargs=2,
                   default=list(PAPER_SENSOR_SIZE), metavar=("H", "W"),
                   help="head and wrist resolution in the replay pass")
    p.add_argument("--human-size", type=int, nargs=2,
                   default=list(PAPER_HUMAN_SIZE), metavar=("H", "W"),
                   help="human render camera resolution")
    p.add_argument("--policy-sensor-size", type=int, nargs=2,
                   default=list(POLICY_SENSOR_SIZE), metavar=("H", "W"),
                   help="sensor size the checkpoint was trained on; changing "
                        "this builds a CNN its weights do not fit")
    p.add_argument("--sensor-shader", default="default",
                   help="shader for the head and wrist cameras in the replay "
                        "pass; empty uses the env default")
    p.add_argument("--human-shader", default="default",
                   help="shader for the human render camera; 'rt' and "
                        "'rt-fast' ray-trace, far prettier and far slower")

    p.add_argument("--head-camera", default=DEFAULT_HEAD_CAMERA)
    p.add_argument("--wrist-camera", default=DEFAULT_WRIST_CAMERA)
    p.add_argument("--no-human", action="store_true",
                   help="skip the human view; export the differences only")
    p.add_argument("--diffs-only", action="store_true",
                   help="do not keep the head and wrist frames the differences "
                        "were computed from")

    p.add_argument("--diff-percentile", type=float, default=99.5,
                   help="difference magnitude mapped to full scale in diff_vis")
    p.add_argument("--diff-gain", type=float, default=0.0,
                   help="fixed diff_vis gain; 0 derives one from the episode")
    p.add_argument("--diff-max-gain", type=float, default=16.0,
                   help="cap on the derived gain, past which a difference "
                        "image is amplified sensor noise")
    p.add_argument("--diff-invert", action="store_true",
                   help="write diff_vis dark-on-white instead of light-on-black")

    p.add_argument("--device", default="cuda")
    p.add_argument("--random-policy", action="store_true",
                   help="roll uniform random actions seeded by --seed instead "
                        "of the checkpoint; --ckpt-dir is still read for the "
                        "env config, and the default name gains '_random'")
    p.add_argument("--allow-random", action="store_true",
                   help="proceed even if the checkpoint failed to load and the "
                        "actions are random")
    p.add_argument("--reward-tol", type=float, default=1e-4,
                   help="largest per-step reward difference the replay may "
                        "show before it is treated as a different episode")
    p.add_argument("--print-every", type=int, default=10)
    return p.parse_args(argv)


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
