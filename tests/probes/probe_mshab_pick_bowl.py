"""Roll out an MS-HAB pick checkpoint and export a high-resolution demo view.

Demonstration only: nothing here trains, and no graph or whitelist asset is
read. It loads a released MS-HAB checkpoint, steps it through one task plan,
and writes frames.

Resolution is split on purpose. ``sensor_configs`` sizes the cameras the policy
consumes and must stay at the size the checkpoint was trained on -- the agent's
CNN is built from the observation shape before its weights are loaded, so a
larger sensor constructs a network the checkpoint cannot populate.
``human_render_camera_configs`` is independent of that, which is why the demo
view can be 1000x1000 while the policy still sees its own 128x128.

Env construction and the wrapper stack follow mshab.envs.make.make_env, as the
released Fetch checkpoints expect it.

Usage:
    python -m tests.probes.probe_mshab_pick_bowl \\
        --ckpt-dir /root/projects/ReLDreamer/mshab_checkpoints/tidy_house/pick/024_bowl \\
        --steps 200 --human-res 1000
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scenegraph.adapters.policy_loader import detect_algo, load_policy  # noqa: E402
from scenegraph.core.mask_extractor import read_unwrapped_rgbs  # noqa: E402

SENSOR_CAMERAS = ("fetch_head", "fetch_hand")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt-dir", required=True,
                   help="dir holding the MS-HAB .pt (policy/latest/best/ckpt)")
    p.add_argument("--task", default="tidy_house",
                   help="MS-HAB task; set_table also ships a bowl plan")
    p.add_argument("--subtask", default="pick")
    p.add_argument("--split", default="train")
    p.add_argument("--obj", default="024_bowl",
                   help="task-plan stem; must match the checkpoint's object")
    p.add_argument("--plan-index", type=int, default=0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")

    p.add_argument("--human-res", type=int, default=1000,
                   help="demo view size; independent of the policy's sensors")
    p.add_argument("--human-shader", default="default",
                   help="'default' rasterises; 'rt' and 'rt-fast' ray-trace, "
                        "far prettier and far slower")
    p.add_argument("--policy-res", type=int, default=128,
                   help="sensor size the checkpoint was trained on; changing "
                        "this will not match its CNN weights")
    p.add_argument("--policy-shader", default="minimal")

    p.add_argument("--also-sensors", action="store_true",
                   help="additionally dump fetch_head/fetch_hand RGB. These "
                        "are the policy's cameras, so they come out at "
                        "--policy-res, not --human-res")
    p.add_argument("--frame-stack", type=int, default=3,
                   help="0 disables; released checkpoints use 3")
    p.add_argument("--free-head", action="store_true",
                   help="release the head joint (checkpoints train it locked)")
    p.add_argument("--save-every", type=int, default=1,
                   help="write every Nth frame")
    p.add_argument("--out", default="outputs/mshab_pick_bowl")
    return p.parse_args(argv)


def _plan_paths(task: str, subtask: str, split: str, obj: str):
    from mani_skill import ASSET_DIR

    root = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
    plan_fp = root / "task_plans" / task / subtask / split / f"{obj}.json"
    if not plan_fp.exists():
        raise FileNotFoundError(
            f"no MS-HAB task plan at {plan_fp}. An object-specific checkpoint "
            f"needs its own plan file; pass --obj all only if you mean the "
            f"aggregate one."
        )
    return plan_fp, root / "spawn_data" / task / subtask / split / "spawn_data.pt"


def build_env(args):
    """One MS-HAB env, wrapped the way the released checkpoints expect."""
    import gymnasium as gym
    import mshab.envs  # noqa: F401  registers the MS-HAB ids
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    from mshab.envs.planner import plan_data_from_file
    from mshab.envs.wrappers import (
        FetchActionWrapper, FetchDepthObservationWrapper, FrameStack,
    )

    plan_fp, spawn_fp = _plan_paths(args.task, args.subtask, args.split, args.obj)
    plan_data = plan_data_from_file(plan_fp)
    if not 0 <= args.plan_index < len(plan_data.plans):
        raise IndexError(
            f"--plan-index {args.plan_index} outside [0, "
            f"{len(plan_data.plans) - 1}] for {plan_fp}"
        )

    env = gym.make(
        f"{args.subtask.capitalize()}SubtaskTrain-v0",
        num_envs=1,
        obs_mode="rgb+depth",
        sim_backend="gpu",
        robot_uids="fetch",
        control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense",
        # rgb_array renders the human cameras only, which is the demo view.
        render_mode="rgb_array",
        max_episode_steps=args.steps,
        task_plans=[plan_data.plans[args.plan_index]],
        scene_builder_cls=plan_data.dataset,
        spawn_data_fp=spawn_fp,
        require_build_configs_repeated_equally_across_envs=False,
        add_event_tracker_info=True,
        sensor_configs=dict(
            width=args.policy_res, height=args.policy_res,
            shader_pack=args.policy_shader,
        ),
        human_render_camera_configs=dict(
            width=args.human_res, height=args.human_res,
            shader_pack=args.human_shader,
        ),
    )
    env = FetchDepthObservationWrapper(env, cat_state=True, cat_pixels=False)
    if args.frame_stack:
        env = FrameStack(
            env, num_stack=args.frame_stack,
            stacking_keys=["fetch_head_depth", "fetch_hand_depth"],
        )
    env = FetchActionWrapper(
        env, stationary_base=False, stationary_torso=False,
        stationary_head=not args.free_head,
    )
    venv = ManiSkillVectorEnv(
        env, ignore_terminations=True, max_episode_steps=args.steps,
    )
    return venv, plan_fp


def rollout(args, venv):
    """Step the checkpoint, writing the demo view each frame."""
    obs, _ = venv.reset(seed=args.seed, options=dict(reconfigure=True))
    policy = load_policy(args.ckpt_dir, venv, obs, device=args.device)
    print(f"[policy] kind={policy.kind} hint={detect_algo(args.ckpt_dir)}")

    human_dir = os.path.join(args.out, "human")
    os.makedirs(human_dir, exist_ok=True)
    sensor_dirs = {}
    if args.also_sensors:
        for cam in SENSOR_CAMERAS:
            sensor_dirs[cam] = os.path.join(args.out, cam)
            os.makedirs(sensor_dirs[cam], exist_ok=True)

    saved, successes, first_success = 0, 0, None
    for step in range(args.steps):
        if step % args.save_every == 0:
            frame = _render_human(venv)
            if frame is not None:
                _save_png(frame, os.path.join(human_dir, f"{step:04d}.png"))
                saved += 1
            for cam, path in sensor_dirs.items():
                rgbs = read_unwrapped_rgbs(venv)
                if cam in rgbs:
                    _save_png(rgbs[cam], os.path.join(path, f"{step:04d}.png"))

        action = policy.act(obs)
        obs, _, _, _, info = venv.step(action)
        if _scalar(info.get("success", 0)) > 0.5:
            successes += 1
            first_success = step if first_success is None else first_success
        if step % 25 == 0:
            print(f"  [{step:4d}/{args.steps}] success_frames={successes}")

    print(f"[out] {saved} demo frames in {human_dir}")
    if first_success is None:
        print("[warn] the checkpoint never reported success on this plan; "
              "check --task/--obj match the checkpoint's directory")
    else:
        print(f"[eval] first success at step {first_success}, "
              f"{successes} success frames")
    return saved, successes, first_success


def _render_human(venv):
    """The human render camera as HxWx3 uint8, or None if unavailable."""
    frame = venv.render()
    if frame is None:
        return None
    arr = np.asarray(_to_numpy(frame))
    if arr.ndim == 4:          # [num_envs, H, W, 3]
        arr = arr[0]
    return arr


def _save_png(rgb: np.ndarray, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg

    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    mpimg.imsave(path, arr)


def _to_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else value


def _scalar(value) -> float:
    arr = np.asarray(_to_numpy(value), dtype=np.float32).reshape(-1)
    return float(arr[0]) if arr.size else 0.0


def main(argv=None) -> int:
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    venv, plan_fp = build_env(args)
    print(f"[env] plan={plan_fp}")
    print(f"[env] policy sensors {args.policy_res}x{args.policy_res} "
          f"({args.policy_shader}) | demo view "
          f"{args.human_res}x{args.human_res} ({args.human_shader})")
    try:
        saved, successes, first = rollout(args, venv)
    finally:
        venv.close()

    with open(os.path.join(args.out, "run.json"), "w") as handle:
        json.dump({
            "ckpt_dir": args.ckpt_dir, "task": args.task,
            "subtask": args.subtask, "split": args.split, "obj": args.obj,
            "plan_index": args.plan_index, "seed": args.seed,
            "steps": args.steps, "human_res": args.human_res,
            "human_shader": args.human_shader, "policy_res": args.policy_res,
            "frames": saved, "success_frames": successes,
            "first_success_step": first,
        }, handle, indent=2)
    print(f"[out] ffmpeg -framerate 20 -i {args.out}/human/%04d.png "
          f"-pix_fmt yuv420p {args.out}/demo.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
