"""A packed graph set, read beside the LeRobot data. Plain numpy."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np

from ..common import read_json, repo_path


class GraphStore:
    def __init__(self, root: str):
        self.root = repo_path(root)
        self.manifest: Dict[str, Any] = read_json(os.path.join(self.root, "manifest.json"))

    def episodes(self) -> List[int]:
        return sorted(int(key) for key in self.manifest["episodes"])

    def task(self, episode: int) -> str:
        return str(self.manifest["episodes"][str(int(episode))]["task"])

    def load(self, episode: int) -> Dict[str, np.ndarray]:
        """Every packed array for the episode's frames, plus ``graph_valid`` and the LeRobot indices."""
        entry = self.manifest["episodes"][str(int(episode))]
        with np.load(os.path.join(self.root, entry["file"]), allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    @property
    def vocab_sizes(self) -> Dict[str, int]:
        return dict(self.manifest["vocab_sizes"])
