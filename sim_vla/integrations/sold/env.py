"""The target ManiSkill task, in the interface SOLD's online loop expects.

``sold/envs/__init__.py`` knows three suites -- ``mof``, ``gym`` and
``dmcontrol`` -- and ManiSkill is not one of them. Rather than adding a fourth
suite to the vendored tree, the environment is built here from
:class:`sim_vla.envs.maniskill.SimVlaEnv`, which is already constructed from the
*dataset's own metadata* and validates the live environment against it: the
robot, the controller, the camera set and resolution, the control frequency and
the reward mode all come from the recording rather than from a second set of
constants.

That matters more here than convenience. Stage 1A fits SAVi to one camera at
one resolution; an online environment that differs in either hands the slot
encoder pictures it has never decomposed, and the only symptom is that nothing
works.

What SOLD actually touches on an environment
--------------------------------------------

``reset() -> obs``, ``step(action) -> (obs, reward, done, info)``,
``action_space.{sample, low, high, shape}`` and ``max_episode_steps``. That is
the whole surface, so this provides exactly it -- and a minimal action space
rather than a gym dependency, because the offline stages use this object too
and they do not otherwise need gym.

Observations come back as ``uint8 (3, H, W)`` tensors, which is what
``train_sold`` divides by 255. ``info`` carries ``success`` when the task
reports it, because the evaluation loop reads it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class BoxSpace:
    """The four attributes SOLD reads off an action space."""

    low: np.ndarray
    high: np.ndarray
    seed_value: int = 0

    def __post_init__(self) -> None:
        self.low = np.asarray(self.low, dtype=np.float32)
        self.high = np.asarray(self.high, dtype=np.float32)
        self._rng = np.random.default_rng(int(self.seed_value))

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.low.shape)

    def sample(self) -> np.ndarray:
        return self._rng.uniform(self.low, self.high).astype(np.float32)


class SoldManiSkillEnv:
    """One ManiSkill episode source, matched to the demonstrations."""

    def __init__(self, metadata, *, image_size: Sequence[int],
                 camera: str, max_episode_steps: int = 150,
                 action_repeat: int = 1, seed: int = 0):
        from ...envs.maniskill import SimVlaEnv

        self.metadata = dict(metadata)
        self.camera = str(camera)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.action_repeat = max(int(action_repeat), 1)
        self.max_episode_steps = int(max_episode_steps)
        self._env = SimVlaEnv(self.metadata, graph_enabled=False,
                              max_steps=int(max_episode_steps) * self.action_repeat,
                              seed=int(seed))
        controller = dict(self.metadata.get("controller") or {})
        width = int(controller.get("action_dim") or 0)
        if width <= 0:
            raise SystemExit(
                "the dataset records no controller.action_dim; the action "
                "space cannot be built from it")
        low = controller.get("action_low")
        high = controller.get("action_high")
        self.action_space = BoxSpace(
            low=(np.full(width, -1.0, np.float32) if low is None
                 else np.broadcast_to(np.asarray(low, np.float32), (width,))),
            high=(np.full(width, 1.0, np.float32) if high is None
                  else np.broadcast_to(np.asarray(high, np.float32), (width,))),
            seed_value=int(seed))
        self._steps = 0

    # ------------------------------------------------------------ observation
    def _observation(self, obs) -> Any:
        import torch

        from ..observations import resize_uint8, to_chw

        if self.camera not in obs:
            raise KeyError(
                f"the environment produced {sorted(k for k in obs if k.startswith('image_'))} "
                f"and this run was configured for {self.camera!r}")
        frame = torch.as_tensor(np.asarray(obs[self.camera], dtype=np.uint8))
        return resize_uint8(to_chw(frame), self.image_size)

    # ---------------------------------------------------------------- driving
    def reset(self, seed: Optional[int] = None):
        self._steps = 0
        return self._observation(self._env.reset(seed))

    def step(self, action):
        import torch

        command = action.detach().cpu().numpy() if hasattr(action, "detach") \
            else np.asarray(action)
        command = np.asarray(command, dtype=np.float32).reshape(-1)
        reward = 0.0
        out = None
        for _ in range(self.action_repeat):
            out = self._env.step(command)
            reward += float(out["reward"])
            if out["is_last"]:
                break
        self._steps += 1
        done = bool(out["is_last"]) or self._steps >= self.max_episode_steps
        info: Dict[str, Any] = {"success": bool(out["success"])}
        if done and not out["is_last"]:
            info["TimeLimit.truncated"] = True
        return self._observation(out["obs"]), reward, done, info

    def close(self) -> None:
        self._env.close()


@dataclass
class SpecEnv:
    """An action space and an episode length, and nothing that can be stepped.

    ``SOLDModule.__init__`` reads only those two off its environment, so the
    offline stages construct the module without standing up a simulator. Any
    attempt to actually drive it raises rather than returning something
    plausible.
    """

    action_space: BoxSpace
    max_episode_steps: int

    def reset(self, *args, **kwargs):
        raise RuntimeError(
            "this is a specification, not an environment; the offline stages "
            "do not step one")

    def step(self, *args, **kwargs):
        return self.reset()


def spec_from(metadata, *, max_episode_steps: int, seed: int = 0) -> SpecEnv:
    controller = dict((metadata or {}).get("controller") or {})
    width = int(controller.get("action_dim") or 0)
    low = controller.get("action_low")
    high = controller.get("action_high")
    return SpecEnv(
        action_space=BoxSpace(
            low=(np.full(width, -1.0, np.float32) if low is None
                 else np.broadcast_to(np.asarray(low, np.float32), (width,))),
            high=(np.full(width, 1.0, np.float32) if high is None
                  else np.broadcast_to(np.asarray(high, np.float32), (width,))),
            seed_value=int(seed)),
        max_episode_steps=int(max_episode_steps))


def check_contract(env, source, *, log=print) -> Dict[str, Any]:
    """Refuse an environment that is not the one the demonstrations came from."""
    wanted = source.contract()
    problems = []
    if list(env.image_size) != list(wanted["image_size"]):
        problems.append(
            f"image size: demonstrations {wanted['image_size']} but the env "
            f"gives {list(env.image_size)}")
    if env.camera not in wanted["cameras"]:
        problems.append(
            f"camera: demonstrations used {wanted['cameras']} and the env is "
            f"configured for {env.camera!r}")
    if int(env.action_space.shape[0]) != int(wanted["action_dim"]):
        problems.append(
            f"action_dim: demonstrations {wanted['action_dim']} but the env "
            f"gives {int(env.action_space.shape[0])}")
    env_id = str(dict(env.metadata).get("env_id") or "")
    if env_id != wanted["env_id"]:
        problems.append(
            f"env_id: demonstrations {wanted['env_id']!r} but this env is "
            f"{env_id!r}")
    if problems:
        raise SystemExit(
            "the online environment is not the one these weights were trained "
            "against:\n  - " + "\n  - ".join(problems))
    if log:
        log(f"[sold] environment contract matches the demonstrations: {wanted}")
    return wanted
