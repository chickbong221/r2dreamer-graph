"""The pinned LeRobot v3.0 snapshot, addressed by episode.

v3.0 packs many episodes into each parquet and MP4 file; ``meta/episodes``
says which file holds an episode and where: its row range and, per camera, its
``[from_timestamp, to_timestamp)`` span in the concatenated video.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, Iterator, List, Mapping, Tuple

import numpy as np

from ..common import parse_episodes, read_json, repo_path

EPISODE_COLUMNS = ("episode_index", "tasks", "length", "data/chunk_index", "data/file_index",
                   "dataset_from_index", "dataset_to_index")
VIDEO_COLUMNS = ("chunk_index", "file_index", "from_timestamp", "to_timestamp")


class LeRobotSource:
    def __init__(self, configs: Mapping[str, Mapping[str, Any]]):
        self.configs = configs
        self.dataset_cfg = configs["dataset"]
        self.paths = {key: repo_path(value) for key, value in self.dataset_cfg["paths"].items()}
        self._info = None
        self._episodes = None
        self._graph = None

    @property
    def root(self) -> str:
        return self.paths["source"]

    def info(self) -> Dict[str, Any]:
        if self._info is None:
            path = os.path.join(self.root, "meta", "info.json")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"no LeRobot metadata at {path}; run "
                                        "`python -m real_robot.preprocessing.download`")
            info = read_json(path)
            if not str(info.get("codebase_version", "")).startswith("v3"):
                raise ValueError(f"{self.root} is LeRobot {info.get('codebase_version')}; this reader needs v3.x")
            self._info = info
        return self._info

    def source_record(self) -> Dict[str, Any]:
        return read_json(os.path.join(self.root, "source.json"))

    def fps(self) -> float:
        return float(self.info()["fps"])

    def cameras(self) -> Dict[str, str]:
        return {str(k): str(v) for k, v in self.dataset_cfg["source"]["cameras"].items()}

    # ---------------------------------------------------------------- index
    def episode_rows(self) -> Dict[int, Dict[str, Any]]:
        if self._episodes is None:
            import pandas as pd

            files = sorted(glob.glob(os.path.join(self.root, "meta", "episodes", "chunk-*", "file-*.parquet")))
            if not files:
                raise FileNotFoundError(f"no episode index under {self.root}/meta/episodes")
            columns = list(EPISODE_COLUMNS) + [f"videos/{key}/{name}" for key in self.cameras().values()
                                               for name in VIDEO_COLUMNS]
            frame = pd.concat([pd.read_parquet(path, columns=columns) for path in files], ignore_index=True)
            rows: Dict[int, Dict[str, Any]] = {}
            for record in frame.to_dict("records"):
                tasks = record["tasks"]
                record["tasks"] = [str(t) for t in (tasks.tolist() if hasattr(tasks, "tolist") else tasks)]
                rows[int(record["episode_index"])] = record
            self._episodes = rows
        return self._episodes

    def lengths(self) -> Dict[int, int]:
        return {episode: int(row["length"]) for episode, row in self.episode_rows().items()}

    def available(self) -> List[int]:
        return sorted(self.episode_rows())

    def task_text(self, episode: int) -> str:
        tasks = self.episode_rows()[int(episode)]["tasks"]
        if len(tasks) != 1:
            raise ValueError(f"episode {episode} has {len(tasks)} task strings: {tasks}")
        return tasks[0]

    @property
    def graph(self):
        if self._graph is None:
            from ..graphs.schema import GraphConfig
            self._graph = GraphConfig.from_config(self.configs["graph"])
        return self._graph

    def task_key(self, episode: int) -> str:
        return self.graph.task_for(self.task_text(episode))

    def spec(self, episode: int):
        return self.graph.spec(self.task_key(episode))

    def selections(self) -> Dict[str, List[int]]:
        """``pilot`` and one selection per task, named by its key in graph.yaml."""
        by_task: Dict[str, List[int]] = {key: [] for key in self.graph.tasks}
        for episode in self.available():
            by_task[self.task_key(episode)].append(episode)
        setting = self.dataset_cfg.get("pilot_episodes", "auto")
        if setting == "auto":
            lengths = self.lengths()
            pilot = []
            for episodes in by_task.values():
                ordered = sorted(episodes, key=lambda e: (lengths[e], e))
                if ordered:
                    pilot.append(ordered[len(ordered) // 2])
        else:
            pilot = [int(e) for e in setting]
        return {**by_task, "pilot": pilot}

    def select(self, text: str) -> List[int]:
        return parse_episodes(text, self.selections(), self.available())

    # ---------------------------------------------------------------- files
    def data_path(self, episode: int) -> str:
        row = self.episode_rows()[int(episode)]
        rel = self.info()["data_path"].format(chunk_index=int(row["data/chunk_index"]),
                                              file_index=int(row["data/file_index"]))
        return os.path.join(self.root, rel)

    def video_path(self, episode: int, camera: str) -> str:
        row = self.episode_rows()[int(episode)]
        key = self.cameras()[camera]
        rel = self.info()["video_path"].format(video_key=key, chunk_index=int(row[f"videos/{key}/chunk_index"]),
                                               file_index=int(row[f"videos/{key}/file_index"]))
        return os.path.join(self.root, rel)

    def video_span(self, episode: int, camera: str) -> Tuple[float, float]:
        row = self.episode_rows()[int(episode)]
        key = self.cameras()[camera]
        return float(row[f"videos/{key}/from_timestamp"]), float(row[f"videos/{key}/to_timestamp"])

    def global_index(self, episode: int) -> np.ndarray:
        """The dataset-wide ``index`` of each of the episode's rows."""
        row = self.episode_rows()[int(episode)]
        start, stop = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        if stop - start != int(row["length"]):
            raise ValueError(f"episode {episode}: rows {start}-{stop} do not match its length {row['length']}")
        return np.arange(start, stop, dtype=np.int64)

    def frames(self, episode: int, camera: str) -> Iterator[Tuple[int, np.ndarray]]:
        """``(frame_index, rgb)`` for every frame of the episode in one camera."""
        from ..preprocessing.prepare_videos import iter_segment

        start, end = self.video_span(episode, camera)
        yield from iter_segment(self.video_path(episode, camera), start, end,
                                self.lengths()[int(episode)], self.fps())
