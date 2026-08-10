"""Small ManiSkill observation wrappers required by the MS-HAB graph path."""

import gymnasium as gym

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common
from mani_skill.utils.wrappers import FlattenRGBDObservationWrapper


class NonPrivilegedObsWrapper(gym.ObservationWrapper):
    """Remove simulator-only task fields from the policy state."""

    PRIVILEGED_KEYS = {
        "is_grasped",
        "goal_pos",
        "obj_pose",
        "tcp_to_obj_pos",
        "obj_to_goal_pos",
        "obj_pose_wrt_base",
        "goal_pos_wrt_base",
    }

    def __init__(self, env) -> None:
        super().__init__(env)
        self._base_env: BaseEnv = env.unwrapped
        init_raw_obs = common.to_tensor(self._base_env._init_raw_obs)
        self._base_env.update_obs_space(self.observation(init_raw_obs))

    def observation(self, obs):
        if "extra" not in obs:
            return obs
        obs = dict(obs)
        obs["extra"] = {
            key: value
            for key, value in obs["extra"].items()
            if key not in self.PRIVILEGED_KEYS
        }
        return obs


class NamedCameraRGBWrapper(FlattenRGBDObservationWrapper):
    """Flatten state while retaining one RGB tensor per named camera.

    ``raw_obs`` is the graph producer's non-mutating access to segmentation.
    The graph builder must not call ``get_obs()`` because MS-HAB evaluation has
    stateful side effects.
    """

    def __init__(self, env, camera_keys) -> None:
        self._camera_keys = dict(camera_keys)
        self.raw_obs = None
        super().__init__(env, rgb=True, depth=False, state=True)

    def observation(self, obs):
        sensors = obs.get("sensor_data", {})
        missing = [cam for cam in self._camera_keys.values() if cam not in sensors]
        if missing:
            raise KeyError(
                f"cameras {missing} are not rendered; available: {sorted(sensors)}"
            )
        self.raw_obs = {
            **obs,
            "sensor_data": {cam: dict(fields) for cam, fields in sensors.items()},
        }
        frames = {
            key: sensors[cam]["rgb"].clone()
            for key, cam in self._camera_keys.items()
        }
        out = dict(super().observation(obs))
        out.pop("rgb", None)
        out.update(frames)
        return out
