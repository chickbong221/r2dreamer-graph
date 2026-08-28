"""The human-view render camera at figure resolution, plus its projection.

The eval video's camera and this one are the same third-person ``render_camera``
the policy never sees, so its resolution is free to be far larger than the
112px sensors the encoder reads. What a video cannot give a paper is a
*labelled* still: for that the camera also has to answer where a world point
lands in the frame, which is the second half of this module.

Two caches with different lifetimes, matching ``adapters.camera_projection``:
intrinsics are a property of the sensor config and are read once; extrinsics are
pose and are re-read every capture, because a task is free to move its render
camera between episodes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

# Private only by module convention -- these are the per-env row unwrap and the
# image-size probe the sensor-coverage path already uses, and re-deriving them
# here would let a figure disagree with the relation eligibility it illustrates.
from ..adapters.camera_projection import _dim, _row
from ..adapters.camera_projection import corners_world, local_aabb_corners

# 1000x1000 prints at ~3.3in / 300dpi, which is a full single-column figure.
DEFAULT_RENDER_SIZE: Tuple[int, int] = (1000, 1000)
# The sensors feed segmentation to the node builder and nothing else here, so
# they stay at the size the graph's masks are mined at rather than the figure's.
DEFAULT_SENSOR_SIZE: Tuple[int, int] = (128, 128)

_PREFERRED_CAMERA = "render_camera"


def make_figure_env(
    env_id: str,
    *,
    render_size: Sequence[int] = DEFAULT_RENDER_SIZE,
    sensor_size: Sequence[int] = DEFAULT_SENSOR_SIZE,
    control_mode: str = "pd_joint_pos",
    obs_mode: str = "rgb+segmentation",
    shader: str = "",
    sim_backend: str = "cpu",
):
    """A single-env ManiSkill task wired for figure capture.

    Batch size one on CPU is not a tunable: the scripted motion-planning
    solutions read ``pose.sp``, which exists only for an unbatched pose. That is
    the same constraint ``collect_maniskill_interactions`` runs under.

    ``obs_mode`` carries segmentation because the graph is built from masks; its
    RGB half is the policy's camera, not the figure's.
    """
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 - registers the tasks

    human = dict(width=int(render_size[1]), height=int(render_size[0]))
    kwargs: Dict[str, Any] = dict(
        id=env_id,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode="rgb_array",
        sim_backend=sim_backend,
        sensor_configs=dict(
            width=int(sensor_size[1]), height=int(sensor_size[0])
        ),
        human_render_camera_configs=human,
    )
    if not shader:
        return gym.make(**kwargs)

    # A ray-traced shader is selected per camera on purpose: setting it globally
    # would also pay for it on the sensors, which are 128px inputs to a mask
    # lookup. ManiSkill has spelled the per-camera selector both ways across
    # versions, so try the current name and then the older one.
    errors = []
    for field in ("shader_pack", "shader_dir"):
        kwargs["human_render_camera_configs"] = dict(human, **{field: shader})
        try:
            return gym.make(**kwargs)
        except Exception as exc:                          # noqa: BLE001
            errors.append(f"{field}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        f"the render camera rejected shader {shader!r} under both spellings "
        f"({'; '.join(errors)})"
    )


class FigureCamera:
    """Pixels and geometry for one human-render camera.

    ``capture`` is the image the paper prints; ``project`` is how a label finds
    the object it names. Both read the same camera at the same instant, so a
    label cannot drift from the pixels it annotates.
    """

    def __init__(self, env, camera_name: str = "", env_idx: int = 0):
        self.env = env
        self.env_idx = int(env_idx)
        self._requested = str(camera_name or "")
        self._intrinsic: Optional[np.ndarray] = None
        self._size: Optional[Tuple[int, int]] = None
        self._aabb: Dict[int, Optional[np.ndarray]] = {}

    def invalidate(self) -> None:
        """Drop the geometry cache. A reconfiguring reset destroys the actors it
        describes; the intrinsics survive because the sensor config does."""
        self._aabb.clear()

    # ------------------------------------------------------------- the camera
    @property
    def cameras(self) -> Dict[str, Any]:
        base = getattr(self.env, "unwrapped", self.env)
        scene = getattr(base, "scene", None)
        return dict(getattr(scene, "human_render_cameras", None) or {})

    @property
    def name(self) -> str:
        cams = self.cameras
        if not cams:
            raise RuntimeError(
                "env exposes no human_render_cameras; it has to be built with "
                "render_mode='rgb_array' and stepped past its first reset"
            )
        if self._requested:
            if self._requested not in cams:
                raise KeyError(
                    f"no human-render camera {self._requested!r}; "
                    f"have {sorted(cams)}"
                )
            return self._requested
        return _PREFERRED_CAMERA if _PREFERRED_CAMERA in cams else sorted(cams)[0]

    def capture(self) -> np.ndarray:
        """One RGB frame from the render camera, ``[H, W, 3]`` uint8."""
        frames = self.env.render()
        arr = frames.cpu().numpy() if hasattr(frames, "cpu") else np.asarray(frames)
        if arr.ndim == 4:
            arr = arr[min(self.env_idx, arr.shape[0] - 1)]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr[..., :3])

    def matrices(self) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """``(intrinsic, extrinsic, (width, height))`` for the current pose."""
        sensor = self.cameras[self.name]
        cam = getattr(sensor, "camera", sensor)
        if self._intrinsic is None:
            k = _row(cam.get_intrinsic_matrix(), self.env_idx).reshape(3, 3)
            self._intrinsic = k
            self._size = (_dim(cam, "width", k[0, 2]),
                          _dim(cam, "height", k[1, 2]))
        ext = _row(cam.get_extrinsic_matrix(), self.env_idx).reshape(3, 4)
        return self._intrinsic, ext, self._size        # type: ignore[return-value]

    # -------------------------------------------------------- world -> pixels
    def project(
        self, points_world, near: float = 1e-3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """``(uv, in_front)`` for ``[N, 3]`` world points.

        Rows behind the camera are flagged rather than dropped: a negative depth
        flips the sign of the projection, so its ``uv`` is a mirrored ghost, and
        a caller reducing several points to one box has to know which rows to
        ignore before it takes a min and a max.
        """
        pts = np.asarray(points_world, dtype=float).reshape(-1, 3)
        k, ext, _ = self.matrices()
        cam = pts @ ext[:, :3].T + ext[:, 3].reshape(1, 3)
        in_front = cam[:, 2] > near
        uvw = cam @ np.asarray(k, dtype=float).T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = uvw[:, :2] / uvw[:, 2:3]
        return uv, in_front

    def point_pixel(self, point_world) -> Optional[Tuple[float, float]]:
        """Pixel for one world point, or None when it is behind the camera."""
        uv, in_front = self.project(
            np.asarray(point_world, dtype=float).reshape(-1)[:3][None, :]
        )
        if not bool(in_front[0]) or not bool(np.isfinite(uv[0]).all()):
            return None
        return float(uv[0, 0]), float(uv[0, 1])

    def entity_corners(self, entity, pose_world) -> Optional[np.ndarray]:
        """World-frame AABB corners for one actor, cached per scene."""
        key = id(entity)
        if key not in self._aabb:
            self._aabb[key] = local_aabb_corners(entity, self.env_idx)
        local = self._aabb[key]
        if local is None or pose_world is None:
            return None
        return corners_world(
            local, np.asarray(pose_world, dtype=float).reshape(-1)
        )

    def entity_box(
        self, entity, pose_world, clip: bool = True
    ) -> Optional[Tuple[float, float, float, float]]:
        """``(x0, y0, x1, y1)`` pixels covering an actor's projected AABB.

        Clipped to the frame by default, because the caller uses the box to
        place a label: a table extending past both edges should be labelled at
        the centre of the part the reader can see, not at the centre of a box
        that is mostly outside the picture.
        """
        corners = self.entity_corners(entity, pose_world)
        if corners is None:
            return None
        uv, in_front = self.project(corners)
        keep = in_front & np.isfinite(uv).all(axis=1)
        if not keep.any():
            return None
        uv = uv[keep]
        x0, y0 = uv.min(axis=0)
        x1, y1 = uv.max(axis=0)
        if clip:
            _, _, (width, height) = self.matrices()
            x0, y0 = max(float(x0), 0.0), max(float(y0), 0.0)
            x1, y1 = min(float(x1), float(width)), min(float(y1), float(height))
            if x1 <= x0 or y1 <= y0:
                return None
        return float(x0), float(y0), float(x1), float(y1)
