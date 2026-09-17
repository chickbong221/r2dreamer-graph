"""The packed dataset's manifest and identity.

A packed dataset is only meaningful together with everything that produced it:
the pinned source revision, the episode selection, the Gemini model and
prompts, the frozen bin specification, ``K``, the geometry settings, the reward
version and fitted scales, the graph vocabulary, the declared action mapping and
its normaliser, and the annotation mode. All of it is written into
``manifest.json`` as ``identity``, and every consumer -- world-model training,
latent encoding, rollouts, policy training, the robot wrapper -- compares the
part it depends on before reading a single array.

The manifest also records the selection: every episode trains, and the
diagnostic episodes are a subset of them, stored by id rather than copied.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..common import (
    REMOVED_SPLITS,
    episode_name,
    read_json,
    repo_path,
    require_identity,
    stable_hash,
    utc_now,
    write_json,
)

MANIFEST_NAME = "manifest.json"
MANIFEST_FORMAT = "real_robot/kitchen-dataset-v2"
EARLIER_FORMATS = ("real_robot/kitchen-dataset-v1",)

# Frame arrays every packed episode carries, beside the graph keys and images.
FRAME_KEYS = (
    "state", "state_raw", "action", "action_raw", "velocity", "effort",
    "timestamp", "frame_index",
    "episode_begin", "recording_end", "task_terminal", "obs_valid", "graph_valid",
    "reward", "done", "transition_valid",
    "stage", "stage_q", "staged_score", "stage_S",
    "centroid_known",
)


class DatasetManifest:
    """``manifest.json`` of one packed dataset directory."""

    def __init__(self, root: str, data: Mapping[str, Any]):
        self.root = repo_path(root)
        self.data = dict(data)
        found = self.data.get("format")
        if found in EARLIER_FORMATS:
            raise ValueError(f"{self.root}: built with train/val/test splits by an earlier version of this "
                             "package; rebuild it under a new --name")
        if found != MANIFEST_FORMAT:
            raise ValueError(f"{self.root}: not a kitchen dataset manifest")

    @classmethod
    def load(cls, root: str) -> "DatasetManifest":
        path = os.path.join(repo_path(root), MANIFEST_NAME)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no dataset manifest at {path}; run build_dataset first")
        return cls(root, read_json(path))

    @classmethod
    def create(cls, root: str, *, identity: Mapping[str, Any], selection: Mapping[str, Any],
               shapes: Mapping[str, Any], model_inputs: Mapping[str, Any],
               graph: Mapping[str, Any], action: Mapping[str, Any]) -> "DatasetManifest":
        training = [int(i) for i in selection["training"]]
        diagnostic = [int(i) for i in selection["diagnostic"]]
        outside = sorted(set(diagnostic) - set(training))
        if outside:
            raise ValueError(f"diagnostic episodes {outside} are not training episodes")
        data = {
            "format": MANIFEST_FORMAT,
            "created": utc_now(),
            "identity": dict(identity),
            "identity_hash": stable_hash(identity),
            "selection": {
                "version": str(selection["version"]),
                "training": training,
                "diagnostic": diagnostic,
                "diagnostic_in_training": True,
            },
            "shapes": dict(shapes),
            "model_inputs": dict(model_inputs),
            "graph": dict(graph),
            "action": dict(action),
            "built": {},
        }
        return cls(root, data)

    def save(self) -> str:
        return write_json(os.path.join(self.root, MANIFEST_NAME), self.data)

    # ------------------------------------------------------------ queries
    @property
    def identity(self) -> Dict[str, Any]:
        return dict(self.data["identity"])

    @property
    def identity_hash(self) -> str:
        return str(self.data["identity_hash"])

    @property
    def content_hash(self) -> str:
        """Digest of what every packed episode was built from. A rebuilt episode changes it."""
        built = self.data.get("built", {})
        return stable_hash({key: (value or {}).get("inputs") for key, value in sorted(built.items())})

    def dataset_key(self) -> Dict[str, str]:
        """What a model trained on this dataset records about it: its contract and its contents."""
        return {"identity": self.identity_hash, "content": self.content_hash}

    @property
    def selection_version(self) -> str:
        return str(self.data["selection"]["version"])

    def _built(self) -> set:
        return set(int(k) for k in self.data.get("built", {}))

    def training_episodes(self) -> List[int]:
        """Every built episode of the training collection, in selection order."""
        built = self._built()
        return [i for i in self.data["selection"]["training"] if i in built]

    def diagnostic_episodes(self) -> List[int]:
        """The built diagnostic episodes. Each is also a training episode."""
        built = self._built()
        return [i for i in self.data["selection"]["diagnostic"] if i in built]

    def episodes(self, name: str) -> List[int]:
        if name == "training":
            return self.training_episodes()
        if name == "diagnostic":
            return self.diagnostic_episodes()
        if name in REMOVED_SPLITS:
            raise KeyError(f"{name!r}: there are no train/val/test splits; use training or diagnostic")
        raise KeyError(f"unknown episode selection {name!r}; use training or diagnostic")

    def all_episodes(self) -> List[int]:
        return sorted(self._built())

    def is_diagnostic(self, episode: int) -> bool:
        return int(episode) in self.data["selection"]["diagnostic"]

    def coverage(self) -> Dict[str, Any]:
        """Whether every training episode is packed exactly once, and nothing else is."""
        training = [int(i) for i in self.data["selection"]["training"]]
        built = self._built()
        missing = [i for i in training if i not in built]
        unexpected = sorted(built - set(training))
        repeated = sorted(i for i in set(training) if training.count(i) > 1)
        diagnostic_missing = [i for i in self.data["selection"]["diagnostic"] if i not in built]
        return {
            "training": len(training), "built": len(built), "missing": missing, "unexpected": unexpected,
            "repeated": repeated, "diagnostic_missing": diagnostic_missing,
            "complete": not missing and not unexpected and not repeated,
        }

    def require_complete(self, allow_partial: bool, what: str) -> Dict[str, Any]:
        coverage = self.coverage()
        if coverage["unexpected"] or coverage["repeated"]:
            raise ValueError(f"{self.root}: packed episodes outside the training selection "
                             f"{coverage['unexpected']} or listed twice {coverage['repeated']}")
        if not coverage["complete"] and not allow_partial:
            raise SystemExit(
                f"{what}: {self.root} packs {coverage['built']} of {coverage['training']} training episodes "
                f"(missing {coverage['missing'][:20]}{' ...' if len(coverage['missing']) > 20 else ''}). "
                "Build every episode, or pass --allow-partial-dataset for a smoke test."
            )
        return coverage

    def episode_dir(self, episode: int) -> str:
        return os.path.join(self.root, "episodes", episode_name(episode))

    def record_episode(self, episode: int, summary: Mapping[str, Any]) -> None:
        self.data.setdefault("built", {})[str(int(episode))] = dict(summary)

    @property
    def cameras(self) -> List[str]:
        return list(self.data["model_inputs"]["cameras"])

    @property
    def image_keys(self) -> List[str]:
        return [f"image_{camera}" for camera in self.cameras]

    @property
    def state_dim(self) -> int:
        return int(self.data["shapes"]["state"][-1])

    @property
    def action_dim(self) -> int:
        return int(self.data["shapes"]["action"][-1])

    @property
    def vocab_sizes(self) -> Dict[str, int]:
        return dict(self.data["graph"]["vocab_sizes"])

    def require(self, expected: Mapping[str, Any], what: str, fields: Optional[Sequence[str]] = None) -> None:
        require_identity(expected, self.identity, what, fields)
