"""Episode access at both ends of preprocessing, and the transforms between them.

* :class:`RawEpisodeSource` reads what preprocessing produces step by step --
  the LeRobot snapshot, annotations, tracks, geometry, the audit's action
  specification. It needs pandas and pyarrow and lives in the preprocessing
  environment.
* :class:`BuiltEpisodeStore` reads the packed dataset. Plain numpy, so the
  training environment needs nothing beyond what it already pins.
* :func:`state_features` and :class:`ActionTransform` are the two transforms
  that must be identical offline and on the robot, so both ends import them
  from here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..common import (
    episode_name,
    parse_episodes,
    read_json,
    read_jsonl,
    repo_path,
    stable_hash,
)
from .manifest import DatasetManifest
from .selection import load_selection, pilot_episodes

STATE_BLOCKS = {
    "joints": 6,
    "gripper": 1,
    "eef_xyz": 3,
    "eef_rot_sincos": 6,
    "velocity": 7,
    "effort": 7,
}


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def state_features(state_raw: np.ndarray, velocity: Optional[np.ndarray], effort: Optional[np.ndarray],
                   features: Sequence[str], effort_scale: float = 1000.0) -> np.ndarray:
    """The proprioceptive vector the encoder reads, from the 13-D recorded state.

    End-effector roll and yaw wrap between +pi and -pi several times per
    episode. Fed raw, a quarter-turn of the wrist would look like a jump of
    six radians, so the three angles enter as sine and cosine.
    """
    state_raw = np.asarray(state_raw, dtype=np.float64)
    parts = []
    for name in features:
        if name == "joints":
            parts.append(state_raw[..., 0:6])
        elif name == "gripper":
            parts.append(state_raw[..., 6:7])
        elif name == "eef_xyz":
            parts.append(state_raw[..., 7:10])
        elif name == "eef_rot_sincos":
            angles = state_raw[..., 10:13]
            parts.append(np.concatenate([np.sin(angles), np.cos(angles)], axis=-1))
        elif name == "velocity":
            if velocity is None:
                raise ValueError("state feature 'velocity' requested but no velocity given")
            parts.append(np.asarray(velocity, dtype=np.float64))
        elif name == "effort":
            if effort is None:
                raise ValueError("state feature 'effort' requested but no effort given")
            parts.append(np.asarray(effort, dtype=np.float64) / float(effort_scale))
        else:
            raise ValueError(f"unknown state feature {name!r}; known: {sorted(STATE_BLOCKS)}")
    return np.concatenate(parts, axis=-1).astype(np.float32)


def state_dim(features: Sequence[str]) -> int:
    return int(sum(STATE_BLOCKS[name] for name in features))


@dataclass
class ActionTransform:
    """Declared command dimensions, scaled into [-1, 1] and back.

    ``low``/``high`` are the range over every training episode, widened by
    ``margin`` so recorded commands sit strictly inside the actor's bounded
    support. Which dimensions are commands, in what representation and units,
    comes from the confirmed action mapping, never from the numbers alone.
    """

    indices: List[int]
    names: List[str]
    representation: str
    low: np.ndarray
    high: np.ndarray
    margin: float
    units: Dict[str, str] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "ActionTransform":
        norm = spec["normalization"]
        return cls(
            indices=[int(i) for i in spec["command_indices"]],
            names=list(spec["command_names"]),
            representation=str(spec["representation"]),
            low=np.asarray(norm["low"], dtype=np.float64),
            high=np.asarray(norm["high"], dtype=np.float64),
            margin=float(norm["margin"]),
            units={str(k): str(v) for k, v in (spec.get("units") or {}).items()},
            extra={"gripper": dict(spec.get("gripper", {}))},
        )

    @property
    def dim(self) -> int:
        return len(self.indices)

    def _center_half(self):
        center = (self.low + self.high) / 2.0
        half = np.maximum((self.high - self.low) / 2.0, 1e-6) * (1.0 + self.margin)
        return center, half

    def normalize(self, raw_actions: np.ndarray) -> np.ndarray:
        """Recorded 13-D actions -> executable commands in [-1, 1]."""
        center, half = self._center_half()
        commands = np.asarray(raw_actions, dtype=np.float64)[..., self.indices]
        return ((commands - center) / half).astype(np.float32)

    def denormalize(self, actions: np.ndarray) -> np.ndarray:
        """Policy output -> command values in the recorded units, clipped to the support."""
        center, half = self._center_half()
        return (np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0) * half + center)

    def identity(self) -> Dict[str, Any]:
        return {
            "indices": self.indices,
            "representation": self.representation,
            "units": dict(sorted(self.units.items())),
            "low": [round(float(v), 6) for v in self.low],
            "high": [round(float(v), 6) for v in self.high],
            "margin": self.margin,
        }

    def digest(self) -> str:
        return stable_hash(self.identity())


def resize_image(image: np.ndarray, size: Sequence[int], mode: str) -> np.ndarray:
    """``(H, W, 3)`` uint8 -> ``size`` = (height, width), stretched or letterboxed."""
    from PIL import Image

    height, width = int(size[0]), int(size[1])
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if mode == "stretch":
        return np.asarray(pil.resize((width, height), Image.BILINEAR), dtype=np.uint8)
    if mode != "letterbox":
        raise ValueError(f"resize_mode must be stretch or letterbox, got {mode!r}")
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    new_w, new_h = max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top, left = (height - new_h) // 2, (width - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = np.asarray(pil.resize((new_w, new_h), Image.BILINEAR))
    return canvas


def transform_boxes(boxes: np.ndarray, source_hw: Sequence[int], size: Sequence[int], mode: str) -> np.ndarray:
    """Normalised ``[x0, x1, y0, y1]`` boxes through the same resize as the image.

    Stretching scales each axis independently, which leaves normalised
    coordinates unchanged. Letterboxing shrinks the content and offsets it
    inside the padded canvas, so the coordinates move with it.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    if mode == "stretch":
        return boxes.copy()
    src_h, src_w = float(source_hw[0]), float(source_hw[1])
    height, width = float(size[0]), float(size[1])
    scale = min(width / src_w, height / src_h)
    new_w, new_h = round(src_w * scale), round(src_h * scale)
    left, top = (width - new_w) // 2, (height - new_h) // 2
    out = boxes.copy()
    out[..., 0:2] = (boxes[..., 0:2] * new_w + left) / width
    out[..., 2:4] = (boxes[..., 2:4] * new_h + top) / height
    empty = ~((boxes[..., 1] > boxes[..., 0]) & (boxes[..., 3] > boxes[..., 2]))
    out[empty] = 0.0
    return out


# --------------------------------------------------------------------------- #
# Raw artifacts
# --------------------------------------------------------------------------- #
class RawEpisodeSource:
    """Everything preprocessing has written so far, addressed by episode."""

    def __init__(self, configs: Mapping[str, Mapping[str, Any]], mode: Optional[str] = None):
        self.configs = configs
        self.dataset_cfg = configs["dataset"]
        self.paths = {key: repo_path(value) for key, value in self.dataset_cfg["paths"].items()}
        annotation_cfg = configs.get("annotation") or {}
        self.mode = mode or (annotation_cfg.get("annotation") or {}).get("mode", "full_episode")
        self._info = None
        self._lengths = None
        self._spec = None
        self._table_cache: Dict[int, Dict[str, np.ndarray]] = {}

    # ----------------------------------------------------------- source
    @property
    def source_root(self) -> str:
        return self.paths["source"]

    def info(self) -> Dict[str, Any]:
        if self._info is None:
            path = os.path.join(self.source_root, "meta", "info.json")
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"no LeRobot metadata at {path}; run `python -m real_robot.preprocessing.download`"
                )
            self._info = read_json(path)
        return self._info

    def source_record(self) -> Dict[str, Any]:
        return read_json(os.path.join(self.source_root, "source.json"))

    def lengths(self) -> Dict[int, int]:
        if self._lengths is None:
            rows = read_jsonl(os.path.join(self.source_root, "meta", "episodes.jsonl"))
            self._lengths = {int(r["episode_index"]): int(r["length"]) for r in rows}
        return self._lengths

    def available(self) -> List[int]:
        return sorted(self.lengths())

    def selection(self) -> Optional[Dict[str, Any]]:
        """The saved episode selection, or ``None`` before one is created."""
        try:
            return load_selection(self.dataset_cfg, self.available())
        except FileNotFoundError:
            return None

    def selections(self) -> Dict[str, List[int]]:
        """Named episode lists. ``training`` is every episode, whether or not a selection is saved yet."""
        out = {"training": self.available(), "pilot": pilot_episodes(self.dataset_cfg, self.lengths())}
        saved = self.selection()
        if saved is not None:
            out["training"] = [int(e) for e in saved["training"]]
            out["diagnostic"] = [int(e) for e in saved["diagnostic"]]
        return out

    def select(self, text: str) -> List[int]:
        return parse_episodes(text, self.selections(), self.available())

    def fps(self) -> float:
        return float(self.info()["fps"])

    def _chunk(self, episode: int) -> int:
        return int(episode) // int(self.info()["chunks_size"])

    def data_path(self, episode: int) -> str:
        rel = self.info()["data_path"].format(episode_chunk=self._chunk(episode), episode_index=int(episode))
        return os.path.join(self.source_root, rel)

    def video_key(self, camera: str) -> str:
        return self.dataset_cfg["source"]["cameras"][camera]

    def video_path(self, episode: int, camera: str) -> str:
        rel = self.info()["video_path"].format(episode_chunk=self._chunk(episode),
                                                video_key=self.video_key(camera),
                                                episode_index=int(episode))
        return os.path.join(self.source_root, rel)

    def table(self, episode: int) -> Dict[str, np.ndarray]:
        """The recorded rows of one episode as arrays, ordered by frame index."""
        episode = int(episode)
        if episode not in self._table_cache:
            import pandas as pd

            frame = pd.read_parquet(self.data_path(episode))
            frame = frame.sort_values("frame_index").reset_index(drop=True)
            fields = self.dataset_cfg["source"]["fields"]
            out = {
                "state": np.stack(frame[fields["state"]].to_numpy()).astype(np.float64),
                "action": np.stack(frame[fields["action"]].to_numpy()).astype(np.float64),
                "timestamp": frame["timestamp"].to_numpy().astype(np.float64),
                "frame_index": frame["frame_index"].to_numpy().astype(np.int64),
                "episode_index": frame["episode_index"].to_numpy().astype(np.int64),
            }
            for name in ("velocity", "effort"):
                column = fields.get(name)
                if column and column in frame.columns:
                    out[name] = np.stack(frame[column].to_numpy()).astype(np.float64)
            self._table_cache = {episode: out}
        return self._table_cache[episode]

    # ---------------------------------------------------- perception
    @property
    def spec(self):
        if self._spec is None:
            from ..graphs.schema import GraphSpec
            self._spec = GraphSpec.from_config(self.configs["graph"])
        return self._spec

    def _mode_path(self, kind: str, episode: int, suffix: str) -> str:
        return os.path.join(self.paths[kind], self.mode, episode_name(episode) + suffix)

    def annotation_path(self, episode: int) -> str:
        return self._mode_path("annotations", episode, ".json")

    def annotation(self, episode: int, require_valid: bool = True):
        from ..graphs.validate import EpisodeAnnotation

        path = self.annotation_path(episode)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no annotation at {path}; run annotate_episode for episode {episode}")
        annotation = EpisodeAnnotation.from_json(self.spec, read_json(path))
        if require_valid and not annotation.valid:
            raise ValueError(
                f"episode {episode}: annotation is invalid ({len(annotation.issues)} issue(s)); "
                f"see {path}"
            )
        return annotation

    def tracks_path(self, episode: int) -> str:
        return self._mode_path("tracks", episode, ".npz")

    def tracks(self, episode: int) -> Dict[str, np.ndarray]:
        path = self.tracks_path(episode)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no tracks at {path}; run track_objects for episode {episode}")
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    def geometry_path(self, episode: int) -> str:
        return self._mode_path("geometry", episode, ".npz")

    def geometry(self, episode: int) -> Dict[str, Any]:
        path = self.geometry_path(episode)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no geometry at {path}; run estimate_geometry for episode {episode}")
        with np.load(path, allow_pickle=False) as data:
            out: Dict[str, Any] = {key: data[key] for key in data.files}
        out["meta"] = read_json(path[:-4] + ".json")
        return out

    def action_spec(self) -> Dict[str, Any]:
        path = os.path.join(self.paths["audit"], "action_spec.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no action specification at {path}; run audit_dataset first")
        return read_json(path)

    def reward_inputs(self, episode: int):
        from ..rewards.kitchen import inputs_from_artifacts

        spec = self.action_spec()
        gripper = spec["gripper"]
        table = self.table(episode)
        return inputs_from_artifacts(self.spec, self.annotation(episode), self.geometry(episode),
                                     table["state"][:, int(gripper["state_index"])], gripper)


# --------------------------------------------------------------------------- #
# Packed dataset
# --------------------------------------------------------------------------- #
class BuiltEpisodeStore:
    """Packed episodes. Images are memory-mapped; everything else loads once."""

    def __init__(self, root: str, cache_episodes: int = 256):
        self.manifest = DatasetManifest.load(root)
        self._cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._cache_limit = int(cache_episodes)

    @property
    def root(self) -> str:
        return self.manifest.root

    def episodes(self, name: str) -> List[int]:
        """``training`` or ``diagnostic``: the built episodes of that selection."""
        return self.manifest.episodes(name)

    def load(self, episode: int, images: bool = True) -> Dict[str, np.ndarray]:
        episode = int(episode)
        if episode in self._cache:
            data = self._cache[episode]
        else:
            directory = self.manifest.episode_dir(episode)
            with np.load(os.path.join(directory, "arrays.npz"), allow_pickle=False) as arrays:
                data = {key: arrays[key] for key in arrays.files}
            if len(self._cache) >= self._cache_limit:
                self._cache.pop(next(iter(self._cache)))
            self._cache[episode] = data
        if not images:
            return data
        out = dict(data)
        directory = self.manifest.episode_dir(episode)
        for key in self.manifest.image_keys:
            out[key] = np.load(os.path.join(directory, f"{key}.npy"), mmap_mode="r")
        return out

    def meta(self, episode: int) -> Dict[str, Any]:
        return read_json(os.path.join(self.manifest.episode_dir(episode), "meta.json"))

    def length(self, episode: int) -> int:
        return int(self.load(episode, images=False)["obs_valid"].shape[0])
