"""SO-101 LeRobot episodes and their annotated scene graphs, as a sim_vla dataset.

    python -m sim_vla.data.convert_real --task stackcube
    python -m sim_vla.data.convert_real --task cubes_in_cup \
        --lerobot real_robot/outputs/source \
        --graphs real_robot/outputs/graphs/cubes_in_cup_graph_progress

Reads the LeRobot v3.0 snapshot the graphs were annotated on and one directory
written by ``real_robot.preprocessing.pack_graphs``, and writes the file that
``task.dataset`` names under ``--data real``, in the layout of
:mod:`sim_vla.data.writer`. An episode of ``N`` recorded frames becomes ``N``
observations and its first ``N - 1`` actions. Nothing records a reward or a
success, so rewards are zero and ``terminated``/``success`` are false.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from scenegraph.adapters.graph_pack import GRAPH_KEYS
from scenegraph.adapters.graph_vocab import (
    PAD_TOKEN,
    build_absolute_vocab,
    build_relation_vocab,
    build_temporal_vocab,
)
from scenegraph.core.relation_rules import EE_KEY

from .writer import DatasetWriter, field_kinds

GRAPHS_FORMAT = "real_robot/so101-graphs-v1"
ENV_PREFIX = "so101"
IMAGE_SIZE = (112, 112)
STATE_KEY = "observation.state"
ACTION_KEY = "action"
BOOKKEEPING = ("graph_valid", "frame_index", "episode_index", "index")


class ConversionError(ValueError):
    """The graphs and the recording disagree, or one of them is malformed."""


def image_key(camera: str) -> str:
    return f"image_{camera}"


def video_key(camera: str) -> str:
    return f"observation.images.{camera}"


def parse_episodes(text: str) -> Optional[List[int]]:
    """``"50,52-54"`` -> ``[50, 52, 53, 54]``; empty means every episode."""
    if not text.strip():
        return None
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            first, last = (int(v) for v in part.split("-", 1))
            out.extend(range(first, last + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


class Snapshot:
    """A LeRobot v3.0 snapshot, read one episode at a time."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        path = self.root / "meta" / "info.json"
        if not path.is_file():
            raise ConversionError(f"no LeRobot metadata at {path}")
        self.info = json.loads(path.read_text(encoding="utf-8"))
        version = str(self.info.get("codebase_version", ""))
        if not version.startswith("v3"):
            raise ConversionError(f"{self.root} is LeRobot {version or '?'}; this reads v3.x")
        self.fps = float(self.info["fps"])
        self.rows = self._episode_rows()
        self._table: Tuple[Optional[Path], Any] = (None, None)

    def _episode_rows(self) -> Dict[int, Dict[str, Any]]:
        import pandas as pd

        files = sorted(glob.glob(str(self.root / "meta" / "episodes" / "chunk-*" / "file-*.parquet")))
        if not files:
            raise ConversionError(f"no episode index under {self.root / 'meta' / 'episodes'}")
        frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
        return {int(row["episode_index"]): row for row in frame.to_dict("records")}

    def source(self) -> Dict[str, Any]:
        """``source.json`` from ``real_robot.preprocessing.download``, or ``{}``."""
        path = self.root / "source.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def feature(self, key: str) -> Dict[str, Any]:
        features = self.info.get("features") or {}
        if key not in features:
            raise ConversionError(f"the snapshot has no {key!r}; it has {sorted(features)}")
        return dict(features[key])

    def names(self, key: str) -> List[str]:
        feature = self.feature(key)
        width = int(np.prod(feature.get("shape") or [0]))
        names = [str(n) for n in (feature.get("names") or [])]
        return names if len(names) == width else [f"{key}[{i}]" for i in range(width)]

    def length(self, episode: int) -> int:
        if episode not in self.rows:
            raise ConversionError(f"episode {episode} is not in {self.root}")
        return int(self.rows[episode]["length"])

    def _data(self, episode: int):
        row = self.rows[episode]
        path = self.root / self.info["data_path"].format(
            chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"]))
        if self._table[0] != path:
            import pandas as pd

            columns = ["episode_index", "frame_index", "index", STATE_KEY, ACTION_KEY]
            self._table = (path, pd.read_parquet(path, columns=columns))
        table = self._table[1]
        return table[table["episode_index"] == episode].sort_values("frame_index")

    def episode(self, episode: int) -> Dict[str, np.ndarray]:
        """``state``, ``action`` and the dataset-wide ``index``, one row per frame."""
        length = self.length(episode)
        rows = self._data(episode)
        if not np.array_equal(rows["frame_index"].to_numpy(), np.arange(length)):
            raise ConversionError(f"episode {episode}: its rows are not frames 0..{length - 1}")
        return {
            "state": np.stack(rows[STATE_KEY].to_numpy()).astype(np.float32),
            "action": np.stack(rows[ACTION_KEY].to_numpy()).astype(np.float32),
            "index": rows["index"].to_numpy().astype(np.int64),
        }

    def video(self, episode: int, camera: str, size: Sequence[int]) -> np.ndarray:
        """``(length, H, W, 3)`` uint8 for one camera.

        The whole frame is resized and the aspect ratio is not kept: the graph
        boxes are normalised to the whole frame, and a crop would move them.
        """
        import cv2

        from real_robot.preprocessing.prepare_videos import iter_segment

        row = self.rows[episode]
        key = video_key(camera)
        path = self.root / self.info["video_path"].format(
            video_key=key, chunk_index=int(row[f"videos/{key}/chunk_index"]),
            file_index=int(row[f"videos/{key}/file_index"]))
        height, width = int(size[0]), int(size[1])
        frames = [cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
                  for _, rgb in iter_segment(str(path), float(row[f"videos/{key}/from_timestamp"]),
                                             float(row[f"videos/{key}/to_timestamp"]),
                                             self.length(episode), self.fps)]
        return np.stack(frames).astype(np.uint8)


def check_vocab(vocab: Mapping[str, Mapping[str, int]]) -> None:
    """Relation, label and change ids must be the repository's own."""
    shared = {"relation": build_relation_vocab(), "absolute": build_absolute_vocab(),
              "temporal": build_temporal_vocab()}
    for name, table in shared.items():
        if dict(vocab.get(name) or {}) != {PAD_TOKEN: 0, **table.token_to_id}:
            raise ConversionError(f"the graphs' {name} ids are not the repository's; repack them")
    if dict(vocab.get("entity") or {}).get(PAD_TOKEN) != 0:
        raise ConversionError("the graphs' entity vocabulary does not put padding at 0")


class Graphs:
    """One directory written by ``real_robot.preprocessing.pack_graphs``."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        path = self.root / "manifest.json"
        if not path.is_file():
            raise ConversionError(f"no manifest at {path}")
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != GRAPHS_FORMAT:
            raise ConversionError(f"{path} is {self.manifest.get('format')!r}, not {GRAPHS_FORMAT!r}")
        self.episodes = {int(k): dict(v) for k, v in (self.manifest.get("episodes") or {}).items()}
        if not self.episodes:
            raise ConversionError(f"{path} lists no episodes")
        tasks = sorted({str(entry["task"]) for entry in self.episodes.values()})
        if len(tasks) != 1:
            raise ConversionError(f"{path} mixes the tasks {tasks}; convert one task at a time")
        self.task = tasks[0]
        self.spec = dict(self.manifest["graph"]["tasks"][self.task])
        check_vocab(self.manifest["vocab"])

    def arrays(self, episode: int) -> Dict[str, np.ndarray]:
        entry = self.episodes.get(episode)
        if entry is None:
            raise ConversionError(f"episode {episode} has no graphs in {self.root}")
        with np.load(self.root / entry["file"]) as data:
            return {key: data[key] for key in data.files}


def graphs_digest(root: str | Path) -> str:
    """One hash over every file of a graphs directory, names included."""
    digest = hashlib.sha1()
    for path in sorted(p for p in Path(root).iterdir() if p.is_file()):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def stale_reason(out: str | Path, graphs_dir: str | Path,
                 size: Optional[Sequence[int]] = None) -> Optional[str]:
    """Why ``out`` does not hold these graphs' conversion, or None when it does."""
    out = Path(out)
    sidecar = out.with_suffix(".json")
    if not out.is_file() or not sidecar.is_file():
        return f"{out} does not exist"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    meta = payload.get("metadata") or {}
    source = meta.get("source") or {}
    if source.get("graphs_digest") != graphs_digest(graphs_dir):
        return f"the graphs in {graphs_dir} changed since {out} was written"
    if int(payload.get("count") or 0) != len(source.get("episodes") or []):
        return f"{out} is incomplete"
    if size is not None and list(meta.get("image_size") or []) != [int(v) for v in size]:
        return f"{out} was written at image size {meta.get('image_size')}"
    return None


def graph_metadata(manifest: Mapping[str, Any], task: str) -> Dict[str, Any]:
    """The ``graph`` block of a sim_vla dataset, from a graphs manifest.

    ``facts`` lists every fact the task's graphs carry, in stored orientation
    and by entity key, which is what the progress schedules compile against.
    """
    graph = manifest["graph"]
    spec = graph["tasks"][task]
    vocab = manifest["vocab"]
    keys = {ident: (EE_KEY if kind == "ee" else key) for ident, key, kind in spec["entities"]}
    without_pad = lambda table: {str(t): int(i) for t, i in table.items() if t != PAD_TOKEN}  # noqa: E731
    return {
        "config": dict(graph),
        "task": str(task),
        "temporal_k": int(spec["temporal_window"]),
        "n_max": int(spec["n_max"]),
        "e_max": int(spec["e_max"]),
        "n_cams": len(spec["cameras"]),
        "cameras": [str(c) for c in spec["cameras"]],
        "visibility_policy": "annotated",
        "use_target_flag": True,
        "object_object_spatial": True,
        "whitelist_dir": "",
        "whitelist_digest": "none",
        "thresholds_path": "",
        "thresholds_digest": "none",
        "vocab_sizes": {name: len(vocab[name]) for name in ("entity", "relation", "absolute", "temporal")},
        "entity_tokens": {str(t): int(i) for t, i in vocab["entity"].items()},
        "relation_tokens": without_pad(vocab["relation"]),
        "absolute_tokens": without_pad(vocab["absolute"]),
        "temporal_tokens": without_pad(vocab["temporal"]),
        "entities": [[keys[ident], str(kind)] for ident, _, kind in spec["entities"]],
        "facts": [[keys[src], keys[dst], str(relation)] for src, dst, relation in spec["facts"]],
        "annotation": {key: manifest.get(key) for key in (
            "name", "updated", "labels", "graph_config", "annotation_config", "prompt", "source_config")},
    }


def check_episode(snapshot: Snapshot, graphs: Graphs, episode: int) -> Dict[str, Any]:
    """One episode's recording and graphs, refused unless they line up frame for frame."""
    arrays = graphs.arrays(episode)
    missing = [key for key in GRAPH_KEYS + BOOKKEEPING if key not in arrays]
    if missing:
        raise ConversionError(f"episode {episode}: the graphs have no {missing}")
    n = int(graphs.episodes[episode]["n_frames"])
    length = snapshot.length(episode)
    if n != length:
        raise ConversionError(f"episode {episode}: {n} graph frames but {length} recorded frames")
    if n < 2:
        raise ConversionError(f"episode {episode}: {n} frames make no transition")
    wrong = [key for key in GRAPH_KEYS + BOOKKEEPING if arrays[key].shape[0] != n]
    if wrong:
        raise ConversionError(f"episode {episode}: {wrong} do not have {n} rows")
    incomplete = int(np.count_nonzero(~arrays["graph_valid"].astype(bool)))
    if incomplete:
        raise ConversionError(f"episode {episode}: {incomplete} frames have no complete graph")
    if not np.array_equal(arrays["frame_index"], np.arange(n)) or not np.all(arrays["episode_index"] == episode):
        raise ConversionError(f"episode {episode}: the graphs are not frames 0..{n - 1} of this episode")
    table = snapshot.episode(episode)
    if not np.array_equal(table["index"], arrays["index"]):
        raise ConversionError(f"episode {episode}: the graphs and the recording name different dataset rows")
    return table | {"graphs": {key: arrays[key] for key in GRAPH_KEYS}}


def build_metadata(*, snapshot: Snapshot, graphs: Graphs, env_id: str, cameras: Sequence[str],
                   size: Sequence[int], actions: np.ndarray, episodes: Sequence[int]) -> Dict[str, Any]:
    from graph_encoder_probe.dataset import git_revision

    source = snapshot.source()
    image_keys = [image_key(camera) for camera in cameras]
    return {
        "schema_version": 1,
        "env_id": env_id,
        "source": {
            "kind": "lerobot",
            "root": str(snapshot.root),
            "repo_id": source.get("repo_id") or (graphs.manifest.get("source") or {}).get("repo_id"),
            "revision": source.get("resolved_revision"),
            "codebase_version": snapshot.info.get("codebase_version"),
            "robot_type": snapshot.info.get("robot_type"),
            "fps": snapshot.fps,
            "episodes": [int(e) for e in episodes],
            "graphs": str(graphs.root),
            "graphs_digest": graphs_digest(graphs.root),
        },
        "reward_mode": "none",
        "cameras": list(cameras),
        "camera_keys": {camera: image_key(camera) for camera in cameras},
        "camera_sources": {camera: video_key(camera) for camera in cameras},
        "image_size": [int(size[0]), int(size[1])],
        "source_image_size": [int(v) for v in snapshot.feature(video_key(cameras[0]))["shape"][:2]],
        "proprio_names": snapshot.names(STATE_KEY),
        "proprio_fields": [["observation", "state"]],
        "privileged_fields": [],
        "controller": {
            "control_mode": "joint_position_target",
            "action_dim": int(actions.shape[-1]),
            "action_names": snapshot.names(ACTION_KEY),
            "action_low": actions.min(axis=0).astype(float).tolist(),
            "action_high": actions.max(axis=0).astype(float).tolist(),
            "control_freq": int(round(snapshot.fps)),
            "sim_freq": 0,
            "action_repeat": 1,
        },
        "graph": graph_metadata(graphs.manifest, graphs.task),
        "budget": {},
        "field_kinds": field_kinds(image_keys, GRAPH_KEYS),
        "notes": {
            "lengths": "fields of kind 'obs' have T+1 rows, 'step' have T",
            "frames": "one observation per recorded frame; the last recorded action has no "
                      "observation after it and is dropped",
            "reward": "nothing recorded: rewards are zero, terminated and success are false",
            "images": "the whole frame resized to image_size, aspect ratio not kept",
            "action_bounds": "the demonstrated minimum and maximum, not the joint limits",
            "centroids": "not annotated; graph_node_centroid is zero",
        },
        "versions": {"repo_revision": git_revision()},
    }


def convert(*, env_id: str, lerobot: str | Path, graphs_dir: str | Path, out: str | Path,
            size: Sequence[int] = IMAGE_SIZE, episodes: Optional[Sequence[int]] = None,
            overwrite: bool = False, log: Callable[[str], None] = print) -> Path:
    """Write one task's episodes. Every episode is checked before any is decoded."""
    out = Path(out)
    if out.exists() and not overwrite:
        raise ConversionError(f"{out} already exists; pass --overwrite to replace it")
    graphs = Graphs(graphs_dir)
    found = f"{ENV_PREFIX}/{graphs.task}"
    if found != env_id:
        raise ConversionError(f"{graphs.root} holds {found!r} graphs, the task is {env_id!r}")
    snapshot = Snapshot(lerobot)
    annotated = (graphs.manifest.get("source") or {}).get("revision")
    recorded = snapshot.source().get("resolved_revision")
    if annotated and recorded and annotated != recorded:
        raise ConversionError(f"the graphs were annotated on revision {annotated}, "
                              f"the snapshot is {recorded}")
    if not recorded:
        log(f"[convert_real] {snapshot.root} has no source.json; its revision is not checked")
    cameras = [str(c) for c in graphs.spec["cameras"]]
    for camera in cameras:
        snapshot.feature(video_key(camera))

    chosen = sorted(graphs.episodes) if episodes is None else sorted(int(e) for e in episodes)
    checked = {episode: check_episode(snapshot, graphs, episode) for episode in chosen}
    actions = np.concatenate([item["action"][:-1] for item in checked.values()])
    metadata = build_metadata(snapshot=snapshot, graphs=graphs, env_id=env_id, cameras=cameras,
                              size=size, actions=actions, episodes=chosen)

    with DatasetWriter(out, metadata, overwrite=overwrite) as writer:
        for episode, item in checked.items():
            steps = len(item["action"]) - 1
            entry = graphs.episodes[episode]
            written = writer.add(
                images={image_key(camera): snapshot.video(episode, camera, size) for camera in cameras},
                proprio=item["state"],
                graphs=item["graphs"],
                actions=item["action"][:-1],
                rewards=np.zeros(steps, dtype=np.float32),
                terminated=np.zeros(steps, dtype=bool),
                truncated=np.zeros(steps, dtype=bool),
                success=np.zeros(steps, dtype=bool),
                env_states={},
                privileged={},
                info={"seed": None, "end_reason": "recording_end", "settled_steps": None,
                      "source_episode": int(episode),
                      "graph_annotation": entry.get("annotation"),
                      "answer_hash": entry.get("answer_hash")})
            log(f"[convert_real] episode {episode}: {steps + 1} frames -> traj_{written}")
    log(f"[convert_real] wrote {len(checked)} episodes, {len(actions)} transitions -> {out}")
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert SO-101 episodes and their scene graphs into a sim_vla dataset")
    parser.add_argument("--task", required=True, help="a --data real task, e.g. stackcube or cubes_in_cup")
    parser.add_argument("--lerobot", default=None, help="the LeRobot v3.0 snapshot; default task.source.lerobot")
    parser.add_argument("--graphs", default=None, help="the packed graphs directory; default task.source.graphs")
    parser.add_argument("--out", default=None, help="the .h5 to write; default task.dataset")
    parser.add_argument("--image-size", type=int, nargs=2, default=list(IMAGE_SIZE), metavar=("H", "W"))
    parser.add_argument("--episodes", default="", help="a subset such as 50,52-54; default every episode")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    from ..config import load_config

    args = parse_args(argv)
    task = load_config(args.task, "dreamer", data="real")["task"]
    source = dict(task.get("source") or {})
    lerobot, graphs = args.lerobot or source.get("lerobot"), args.graphs or source.get("graphs")
    if not lerobot or not graphs:
        raise SystemExit("pass --lerobot and --graphs; the task config names no source")
    try:
        convert(env_id=str(task["env_id"]), lerobot=lerobot, graphs_dir=graphs,
                out=args.out or task["dataset"], size=args.image_size,
                episodes=parse_episodes(args.episodes), overwrite=args.overwrite)
    except ConversionError as exc:
        raise SystemExit(f"[convert_real] {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
