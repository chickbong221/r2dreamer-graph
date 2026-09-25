"""Human view plus head/wrist frame differences from a task's scripted solution.

    python -m scenegraph.tools.render_motionplanning_paper_frames \
        --env-id PullCubeTool-v1 --title "Pull Cube Tool" \
        --max-frames 250 --out data/paper_figures

Writes the layout ``render_mshab_paper_frames`` writes, one directory per kept
episode named ``<env id>_seed<NNNN>``. Only episodes that succeed within the
exported frames are kept. A robot without a wrist camera (plain ``panda``) gets
``panda_wristcam``'s camera mounted on the same hand link, so the robot, and
with it the planned trajectory, stays the one ``demo_motionplanning_reward``
logs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from scenegraph.figures.diff_writer import DiffEpisodeWriter, episode_path
from scenegraph.figures.rollout import MotionPlanRunner
from scenegraph.tools.collect_maniskill_interactions import success_flag
from scenegraph.tools.demo_motionplanning_reward import (
    episode_horizon, figure_title, scalar,
)
from scenegraph.tools.render_mshab_paper_frames import (
    PAPER_HUMAN_SIZE, PAPER_SENSOR_SIZE, preflight, render_human,
)

DEFAULT_HEAD_CAMERA = "base_camera"
DEFAULT_WRIST_CAMERA = "hand_camera"
DEFAULT_MAX_FRAMES = 250
WRIST_MOUNT_LINK = "panda_hand"
# panda_v3.urdf: realsense_joint (panda_hand -> camera_base_link), then camera_link_joint.
WRIST_MOUNT_CHAIN: Tuple[Tuple[Tuple[float, ...], Tuple[float, ...]], ...] = (
    ((0.035, 0.0, 0.036), (0.0, -1.5707, 3.1415)),
    ((0.0, 0.02, 0.0115), (0.0, 0.0, 0.0)),
)
WRIST_FOV = np.pi / 2
ENV_IDX = 0


def quat_mul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_rotate(q, v) -> np.ndarray:
    conj = np.asarray(q, dtype=float) * np.array([1.0, -1.0, -1.0, -1.0])
    return quat_mul(quat_mul(q, np.concatenate([[0.0], v])), conj)[1:]


def rpy_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF rpy (fixed axes, Rz @ Ry @ Rx) as a wxyz quaternion."""
    def about(axis: int, angle: float) -> np.ndarray:
        q = np.zeros(4)
        q[0], q[1 + axis] = np.cos(angle / 2), np.sin(angle / 2)
        return q
    return quat_mul(quat_mul(about(2, yaw), about(1, pitch)), about(0, roll))


def chain_pose(chain) -> Tuple[np.ndarray, np.ndarray]:
    """``(p, q_wxyz)`` of the last frame of a URDF joint chain, in the first."""
    p, q = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    for xyz, rpy in chain:
        p = p + quat_rotate(q, np.asarray(xyz, dtype=float))
        q = quat_mul(q, rpy_quat(*rpy))
    return p, q / np.linalg.norm(q)


def wrist_camera_env_id(env_id: str, camera: str = DEFAULT_WRIST_CAMERA,
                        link: str = WRIST_MOUNT_LINK) -> str:
    """Register ``env_id`` again with ``camera`` added when its robot lacks one.

    The derived id keeps the original's horizon, default kwargs and wrappers;
    only the sensor list can differ, and only when the robot has no ``camera``.
    """
    from mani_skill.sensors.camera import CameraConfig, parse_camera_configs
    from mani_skill.utils.registration import REGISTERED_ENVS, register_env

    if env_id not in REGISTERED_ENVS:
        raise SystemExit(f"{env_id} is not a registered ManiSkill env")
    name, sep, version = env_id.rpartition("-v")
    derived = f"{name}WristCam-v{version}" if sep else f"{env_id}WristCam"
    if derived in REGISTERED_ENVS:
        return derived
    spec = REGISTERED_ENVS[env_id]
    mount_p, mount_q = chain_pose(WRIST_MOUNT_CHAIN)

    class WithWristCamera(spec.cls):
        @property
        def _default_sensor_configs(self):
            configs = parse_camera_configs(super()._default_sensor_configs)
            robot = ({} if self.agent is None
                     else parse_camera_configs(self.agent._sensor_configs))
            if camera in configs or camera in robot:
                return list(configs.values())
            import sapien

            return list(configs.values()) + [CameraConfig(
                camera, pose=sapien.Pose(p=mount_p.tolist(), q=mount_q.tolist()),
                width=128, height=128, fov=WRIST_FOV, near=0.01, far=100,
                mount=self.agent.robot.links_map[link],
            )]

    WithWristCamera.__name__ = f"{spec.cls.__name__}WithWristCamera"
    WithWristCamera.__qualname__ = WithWristCamera.__name__
    register_env(derived, max_episode_steps=spec.max_episode_steps,
                 asset_download_ids=spec.asset_download_ids,
                 **dict(spec.default_kwargs))(WithWristCamera)
    return derived


def make_figure_env(env_id: str, *, sensor_size: Sequence[int],
                    human_size: Sequence[int], sensor_shader: str,
                    human_shader: str, wrist_camera: str, control_mode: str,
                    reward_mode: str, reward_fallback: Sequence[str],
                    sim_backend: str, max_episode_steps: int = 0):
    """Returns ``(env, gym id, reward mode actually used)``."""
    import mani_skill.envs  # noqa: F401
    from envs.maniskill import _make_with_supported_reward

    sensors: Dict[str, Any] = dict(width=int(sensor_size[1]), height=int(sensor_size[0]))
    if sensor_shader:
        sensors["shader_pack"] = sensor_shader
    human: Dict[str, Any] = dict(width=int(human_size[1]), height=int(human_size[0]))
    if human_shader:
        human["shader_pack"] = human_shader
    gym_id = wrist_camera_env_id(env_id, wrist_camera)
    kwargs: Dict[str, Any] = dict(
        id=gym_id, obs_mode="rgb", control_mode=control_mode,
        render_mode="rgb_array", sim_backend=sim_backend,
        reward_mode=str(reward_mode), sensor_configs=sensors,
        human_render_camera_configs=human,
    )
    if int(max_episode_steps) > 0:
        kwargs["max_episode_steps"] = int(max_episode_steps)
    env = _make_with_supported_reward(kwargs, list(reward_fallback or []))
    return env, gym_id, str(kwargs["reward_mode"])


def sensor_frames(obs: dict, env_idx: int = ENV_IDX) -> Dict[str, np.ndarray]:
    frames: Dict[str, np.ndarray] = {}
    for camera, data in ((obs or {}).get("sensor_data") or {}).items():
        if "rgb" not in data:
            continue
        rgb = data["rgb"]
        arr = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
        arr = arr[env_idx][..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        frames[str(camera)] = np.ascontiguousarray(arr)
    return frames


def wrist_camera_source(env, camera: str) -> str:
    base = getattr(env, "unwrapped", env)
    if camera in (getattr(base, "_agent_sensor_configs", None) or {}):
        return "robot"
    return f"mounted on {WRIST_MOUNT_LINK} at panda_wristcam's camera_link offset"


class DiffCapture:
    """One attempt's frames, written as they arrive and kept only on success."""

    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.root = Path(args.out)
        self.roles = {"head": args.head_camera, "wrist": args.wrist_camera}
        self.seed = int(args.seed)
        self.writer: Optional[DiffEpisodeWriter] = None
        self.steps = 0
        self.first_success: Optional[int] = None

    @property
    def name(self) -> str:
        return f"{self.args.env_id}_seed{self.seed:04d}"

    def prepare(self, seed: int) -> None:
        """Refuse a taken name here: inside a hook it would read as a failed plan."""
        self.seed = int(seed)
        path = episode_path(self.root, self.name)
        if path.exists() and not self.args.overwrite:
            raise SystemExit(f"{path} already exists; pass --overwrite to replace "
                             "it, or write to a different --out")

    def _sensors(self, obs: dict) -> Dict[str, np.ndarray]:
        frames = sensor_frames(obs)
        return {role: frames[camera] for role, camera in self.roles.items()}

    def on_reset(self, obs: dict) -> None:
        self.close(commit=False)
        self.steps, self.first_success = 0, None
        self.writer = DiffEpisodeWriter(
            self.root, self.name, roles=list(self.roles),
            human_size=self.args.human_size, sensor_size=self.args.sensor_size,
            save_human=not self.args.no_human,
            save_frames=not self.args.diffs_only,
            max_gain=self.args.diff_max_gain, overwrite=self.args.overwrite,
        )
        self.writer.open()
        self.writer.write_step(
            step=0, sensors=self._sensors(obs),
            human=None if self.args.no_human else render_human(self.env),
            extra={"reward": None, "success": False},
        )

    def on_transition(self, obs, reward, _terminated, _truncated, info) -> None:
        self.steps += 1
        if self.writer is None or self.writer.count >= self.args.max_frames:
            return
        success = success_flag(info, ENV_IDX)
        if success and self.first_success is None:
            self.first_success = self.steps
        self.writer.write_step(
            step=self.steps, sensors=self._sensors(obs),
            human=None if self.args.no_human else render_human(self.env),
            extra={"reward": float(scalar(reward, ENV_IDX)), "success": success},
        )

    def close(self, *, commit: bool, metadata: Optional[Dict[str, Any]] = None
              ) -> Tuple[Optional[Path], Dict[str, float]]:
        writer, self.writer = self.writer, None
        if writer is None:
            return None, {}
        if not commit:
            writer.discard()
            return None, {}
        try:
            gains = writer.write_amplified(percentile=self.args.diff_percentile,
                                           gain=self.args.diff_gain,
                                           invert=self.args.diff_invert)
            payload = dict(metadata or {})
            payload.setdefault("diff", {})["gains"] = gains
            return writer.commit(payload), gains
        except BaseException:
            writer.discard()
            raise


def episode_metadata(args, capture: DiffCapture, attempt, *, gym_id: str,
                     reward_mode: str, horizon: Optional[int], robot_uids: str,
                     wrist_source: str) -> Dict[str, Any]:
    exported = max(0, (capture.writer.count if capture.writer else 0) - 1)
    mount_p, mount_q = chain_pose(WRIST_MOUNT_CHAIN)
    return {
        "env_id": args.env_id,
        "gym_id": gym_id,
        "title": args.title or figure_title(args.env_id),
        "policy": "motion_planning",
        "planner": "mani_skill.examples.motionplanning.panda.run.MP_SOLUTIONS",
        "seed": int(attempt.seed),
        "horizon": horizon,
        "reward_mode": reward_mode,
        "control_mode": args.control_mode,
        "sim_backend": args.sim_backend,
        "robot_uids": robot_uids,
        "frame_cameras": dict(capture.roles, human="human render camera"),
        "wrist_camera": {
            "source": wrist_source,
            "link": WRIST_MOUNT_LINK if wrist_source != "robot" else None,
            "pose_p": mount_p.tolist() if wrist_source != "robot" else None,
            "pose_q_wxyz": mount_q.tolist() if wrist_source != "robot" else None,
        },
        "human_size": list(args.human_size),
        "sensor_size": list(args.sensor_size),
        "shaders": {"sensor": args.sensor_shader or "env default",
                    "human": args.human_shader or "env default"},
        "frame_annotations": False,
        "graph": None,
        "max_frames": int(args.max_frames),
        "diff": {
            "reference": "previous exported frame",
            "percentile": args.diff_percentile,
            "max_gain": args.diff_max_gain,
            "gain_override": args.diff_gain,
            "inverted": bool(args.diff_invert),
        },
        "attempt": {
            "seed": int(attempt.seed),
            "success": capture.first_success is not None,
            "first_success_step": capture.first_success,
            "steps": exported,
            "planned_steps": int(capture.steps),
            "error": attempt.error,
        },
    }


def run(args) -> int:
    env, gym_id, reward_mode = make_figure_env(
        args.env_id, sensor_size=args.sensor_size, human_size=args.human_size,
        sensor_shader=args.sensor_shader, human_shader=args.human_shader,
        wrist_camera=args.wrist_camera, control_mode=args.control_mode,
        reward_mode=args.reward_mode, reward_fallback=args.reward_fallback,
        sim_backend=args.sim_backend, max_episode_steps=args.max_episode_steps,
    )
    capture = DiffCapture(env, args)
    runner = MotionPlanRunner(env, args.env_id, on_reset=capture.on_reset,
                              on_transition=capture.on_transition)
    obs, _ = env.reset(seed=int(args.seed))
    preflight(sensor_frames(obs), render_human(env), roles=capture.roles,
              sensor_size=args.sensor_size, human_size=args.human_size)
    horizon = episode_horizon(env)
    base = getattr(env, "unwrapped", env)
    robot_uids = str(getattr(base, "robot_uids", ""))
    wrist_source = wrist_camera_source(env, args.wrist_camera)
    print(f"env={args.env_id} gym_id={gym_id} robot={robot_uids} "
          f"reward_mode={reward_mode} horizon={horizon} "
          f"max_frames={args.max_frames}", flush=True)
    print(f"wrist camera {args.wrist_camera!r}: {wrist_source}", flush=True)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    kept: List[Path] = []
    seed, attempts = int(args.seed), 0
    try:
        while len(kept) < args.episodes and attempts < args.max_attempts:
            attempts += 1
            capture.prepare(seed)
            attempt = runner.attempt(seed)
            seed += 1
            ok = capture.first_success is not None
            path = None
            if ok:
                path, gains = capture.close(commit=True, metadata=episode_metadata(
                    args, capture, attempt, gym_id=gym_id,
                    reward_mode=reward_mode, horizon=horizon,
                    robot_uids=robot_uids, wrist_source=wrist_source))
            else:
                capture.close(commit=False)
            if attempt.success and not ok:
                status = f"dropped (success came after frame {args.max_frames - 1})"
            else:
                status = "kept" if path else "dropped"
            note = f" ({attempt.error})" if attempt.error else ""
            print(f"[{attempts}] seed={attempt.seed} success={attempt.success} "
                  f"planned_steps={capture.steps} first_success="
                  f"{capture.first_success} -> {status}{note}", flush=True)
            if path is not None:
                print("  difference gains: " + ", ".join(
                    f"{role}={value:.2f}x" for role, value in gains.items()),
                    flush=True)
                kept.append(path)
    except KeyboardInterrupt:
        print("\ninterrupted; staged episode discarded", flush=True)
    finally:
        capture.close(commit=False)
        env.close()

    print(f"\n{len(kept)}/{args.episodes} episodes written to {args.out}")
    for path in kept:
        print(f"  {path}")
    if not kept:
        print("no episode succeeded within the exported frames; raise "
              "--max-attempts or --max-frames")
        return 1
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Export the human view and the head/wrist frame "
                    "differences of successful motion-planning episodes",
    )
    p.add_argument("--env-id", required=True)
    p.add_argument("--title", default="",
                   help="figure title recorded in the manifest; defaults to the "
                        "env id")
    p.add_argument("--out", default="data/paper_figures")
    p.add_argument("--episodes", type=int, default=1,
                   help="successful episodes to keep")
    p.add_argument("--seed", type=int, default=0, help="first seed to try")
    p.add_argument("--max-attempts", type=int, default=25)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                   help="frames to export, counting the reset frame; an episode "
                        "is kept only if it succeeds within them")

    p.add_argument("--sensor-size", type=int, nargs=2,
                   default=list(PAPER_SENSOR_SIZE), metavar=("H", "W"))
    p.add_argument("--human-size", type=int, nargs=2,
                   default=list(PAPER_HUMAN_SIZE), metavar=("H", "W"))
    p.add_argument("--sensor-shader", default="default")
    p.add_argument("--human-shader", default="default")
    p.add_argument("--head-camera", default=DEFAULT_HEAD_CAMERA)
    p.add_argument("--wrist-camera", default=DEFAULT_WRIST_CAMERA)
    p.add_argument("--no-human", action="store_true")
    p.add_argument("--diffs-only", action="store_true")

    p.add_argument("--diff-percentile", type=float, default=99.5)
    p.add_argument("--diff-gain", type=float, default=0.0)
    p.add_argument("--diff-max-gain", type=float, default=16.0)
    p.add_argument("--diff-invert", action="store_true")

    p.add_argument("--control-mode", default="pd_joint_pos")
    p.add_argument("--sim-backend", default="cpu")
    p.add_argument("--reward-mode", default="normalized_dense")
    p.add_argument("--reward-fallback", nargs="*", default=["sparse"])
    p.add_argument("--max-episode-steps", type=int, default=0,
                   help="0 keeps the task's registered horizon")
    args = p.parse_args(argv)
    if args.max_frames < 2:
        p.error("--max-frames must be at least 2: a difference needs two frames")
    return args


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
