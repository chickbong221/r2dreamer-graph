"""The packed-graph cache: shards on disk, one dense frame table in memory.

Only the encoder's own inputs are stored. RGB and simulator state are what make
a rollout expensive to keep, and the probe reads neither, so a run that has
collected once never starts the simulator again.

Field names and dtypes are the repository's, unchanged -- ``pack_graph``
produces them and ``compact_graph`` reads them back, and a cache that widened a
uint8 or renamed a key would be measuring a different tensor than training
does. ``ShardWriter.add`` asserts that on the way in rather than trusting it.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from scenegraph.adapters.graph_pack import GRAPH_KEYS

# Mirrors ``scenegraph.adapters.graph_obs._DTYPES``; the tests assert the two
# agree. Duplicated rather than imported so loading a cache needs neither torch
# nor a simulator install.
FIELD_DTYPES: dict[str, np.dtype] = {
    "graph_node_ent": np.dtype(np.uint8),
    "graph_node_bbox": np.dtype(np.float16),
    "graph_node_centroid": np.dtype(np.float32),
    "graph_node_target": np.dtype(np.uint8),
    "graph_edge_src": np.dtype(np.uint8),
    "graph_edge_dst": np.dtype(np.uint8),
    "graph_edge_rel": np.dtype(np.uint8),
    "graph_edge_abs": np.dtype(np.uint8),
    "graph_edge_temp": np.dtype(np.uint8),
}

# Per-frame provenance. Not model input: the report names the episode and step a
# pair came from, and a seed nobody recorded makes a surprising pair impossible
# to go back and look at.
INDEX_DTYPES: dict[str, np.dtype] = {
    "episode": np.dtype(np.int32),
    "seed": np.dtype(np.int64),
    "frame": np.dtype(np.int32),
    "success": np.dtype(bool),
}

SHARD_PREFIX = "shard_"
META_NAME = "meta.json"
# A handful of frames as the packer emitted them, kept beside the shards so the
# round-trip check can compare the cache against its source rather than against
# a second copy of itself.
SAMPLE_NAME = "packed_sample.npz"


def save_packed_sample(out_dir: str, rows: Sequence[tuple[int, Mapping[str, np.ndarray]]]) -> Optional[str]:
    """Write ``(cache row, packed arrays)`` pairs straight from the packer."""
    if not rows:
        return None
    payload: dict[str, np.ndarray] = {
        key: np.stack([packed[key] for _, packed in rows]).astype(FIELD_DTYPES[key], copy=False)
        for key in GRAPH_KEYS
    }
    payload["row"] = np.asarray([row for row, _ in rows], dtype=np.int64)
    path = os.path.join(out_dir, SAMPLE_NAME)
    np.savez_compressed(path, **payload)
    return path


def load_packed_sample(path: str) -> Optional[tuple[np.ndarray, dict[str, np.ndarray]]]:
    """``(rows, fields)`` if a sample was written, else ``None``."""
    full = os.path.join(path, SAMPLE_NAME)
    if not os.path.isfile(full):
        return None
    with np.load(full) as data:
        return np.asarray(data["row"]), {key: data[key] for key in GRAPH_KEYS}


def git_revision() -> str:
    """The revision the cache was produced at, or ``unknown`` outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except Exception:                                      # noqa: BLE001
        return "unknown"
    return out.stdout.strip() or "unknown"


def check_fields(fields: Mapping[str, np.ndarray], *, frame_axis: bool) -> None:
    """Every packed key present, in the repository's dtype.

    ``frame_axis`` distinguishes one frame's arrays from a stacked table; the
    dtype contract is the same either way and is the half that actually rots.
    """
    missing = [key for key in GRAPH_KEYS if key not in fields]
    if missing:
        raise KeyError(f"packed graph is missing {missing}")
    extra = [key for key in fields if key not in GRAPH_KEYS]
    if extra:
        raise KeyError(f"packed graph carries unknown keys {extra}")
    for key in GRAPH_KEYS:
        want = FIELD_DTYPES[key]
        got = np.asarray(fields[key]).dtype
        if got != want:
            raise TypeError(f"{key} is {got}, the packer emits {want}")
    if frame_axis:
        counts = {int(np.asarray(fields[key]).shape[0]) for key in GRAPH_KEYS}
        if len(counts) != 1:
            raise ValueError(f"fields disagree on frame count: {sorted(counts)}")


@dataclass
class GraphFrames:
    """A dense table of packed graphs, one row per frame."""

    fields: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        check_fields(self.fields, frame_axis=True)
        # A cache, not state: ``select`` and ``concat`` build a different table
        # and must not inherit it. Set here rather than declared as a field so
        # it never reaches a comparison, a copy or a save.
        self._resident: Optional[dict] = None
        self._resident_device = None

    def __len__(self) -> int:
        return int(self.fields["graph_node_ent"].shape[0])

    @property
    def n_max(self) -> int:
        return int(self.fields["graph_node_ent"].shape[1])

    @property
    def e_max(self) -> int:
        return int(self.fields["graph_edge_rel"].shape[1])

    @property
    def n_cams(self) -> int:
        return int(self.fields["graph_node_bbox"].shape[2])

    def frame(self, i: int) -> dict[str, np.ndarray]:
        """One frame's arrays, copied. Pair construction edits these."""
        return {key: np.array(arr[i], copy=True) for key, arr in self.fields.items()}

    def select(self, indices: Sequence[int]) -> "GraphFrames":
        idx = np.asarray(indices, dtype=np.int64)
        return GraphFrames({key: arr[idx] for key, arr in self.fields.items()})

    def concat(self, other: "GraphFrames") -> "GraphFrames":
        return GraphFrames(
            {
                key: np.concatenate([self.fields[key], other.fields[key]], 0)
                for key in GRAPH_KEYS
            }
        )

    def device_bytes(self) -> int:
        """What holding the whole table on a device would cost."""
        return sum(
            arr.nbytes * (2 if key == "graph_node_bbox" else 1)   # float16 -> float32
            for key, arr in self.fields.items()
        )

    def to_device(self, device) -> "GraphFrames":
        """Hold the whole table on ``device``, so a batch is an on-device gather.

        These graphs are small -- a batch of 128 is about 150 KB -- but building
        one on the host means nine separate host-to-device copies out of
        pageable numpy memory, and a pageable copy is synchronous. At a few
        milliseconds per update that is most of the update. The pool is tens of
        megabytes, so it simply lives on the device instead.

        The tensors are exactly the ones ``torch_batch`` built before, widened
        once instead of per batch, so no number changes.
        """
        import torch

        device = torch.device(device)
        resident = {}
        for key in GRAPH_KEYS:
            tensor = torch.from_numpy(np.ascontiguousarray(self.fields[key]))
            if key == "graph_node_bbox":
                tensor = tensor.to(torch.float32)
            resident[key] = tensor.to(device)
        self._resident, self._resident_device = resident, device
        return self

    @property
    def resident_device(self):
        return self._resident_device

    def torch_batch(self, indices, device=None) -> dict:
        """Packed arrays as tensors, ready for ``GraphEncoder``.

        Only the box table is retyped: it is stored at the replay buffer's
        float16 and the run is float32 throughout. The widening is exact, and
        the encoder would do it internally anyway -- doing it here keeps one
        precision visible at the call site.

        ``indices`` may be a device tensor, which is how the training loop keeps
        an epoch's shuffled order on the device and slices it there.
        """
        import torch

        if self._resident is not None and (
            device is None or torch.device(device) == self._resident_device
        ):
            if torch.is_tensor(indices):
                idx = indices.to(self._resident_device, torch.int64)
            else:
                idx = torch.as_tensor(
                    np.asarray(indices, dtype=np.int64), device=self._resident_device
                )
            return {key: value.index_select(0, idx) for key, value in self._resident.items()}

        if torch.is_tensor(indices):
            indices = indices.detach().cpu().numpy()
        idx = np.asarray(indices, dtype=np.int64)
        out = {}
        for key in GRAPH_KEYS:
            sel = np.ascontiguousarray(self.fields[key][idx])
            tensor = torch.from_numpy(sel)
            if key == "graph_node_bbox":
                tensor = tensor.to(torch.float32)
            out[key] = tensor if device is None else tensor.to(device)
        return out

    def fingerprint(self, indices: Optional[Sequence[int]] = None) -> str:
        """Content hash over the packed bytes, so pairs cannot be replayed
        against a cache they were not built from."""
        digest = hashlib.sha256()
        rows = None if indices is None else np.asarray(indices, dtype=np.int64)
        for key in GRAPH_KEYS:
            arr = self.fields[key] if rows is None else self.fields[key][rows]
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(arr).tobytes())
        return digest.hexdigest()


@dataclass
class GraphDataset:
    """Collected frames plus their provenance and the collection metadata."""

    frames: GraphFrames
    index: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def n_cams(self) -> int:
        return self.frames.n_cams

    @property
    def entity_vocab(self) -> int:
        """Rows the entity embedding table needs.

        Read from the recorded vocabulary, not from the ids that happen to
        appear: an entity the collection never saw still has to have a row, or
        a later run on the same cache would index past the table.
        """
        sizes = self.meta.get("vocab_sizes") or {}
        if "entity" not in sizes:
            raise KeyError("dataset meta carries no vocab_sizes.entity")
        return int(sizes["entity"])

    def episode_bounds(self) -> list[tuple[int, int]]:
        """``[start, stop)`` per episode, in collection order."""
        episodes = np.asarray(self.index["episode"])
        if episodes.size == 0:
            return []
        edges = np.flatnonzero(np.diff(episodes)) + 1
        starts = np.concatenate([[0], edges])
        stops = np.concatenate([edges, [episodes.size]])
        return [(int(a), int(b)) for a, b in zip(starts, stops)]

    @classmethod
    def load(cls, path: str) -> "GraphDataset":
        meta_path = os.path.join(path, META_NAME)
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"no packed-graph cache at {path!r} (expected {META_NAME}). "
                "Run the collect stage first."
            )
        with open(meta_path) as handle:
            meta = json.load(handle)
        shards = sorted(
            name
            for name in os.listdir(path)
            if name.startswith(SHARD_PREFIX) and name.endswith(".npz")
        )
        if not shards:
            raise FileNotFoundError(f"cache at {path!r} has no shards")
        parts: list[dict[str, np.ndarray]] = []
        index_parts: list[dict[str, np.ndarray]] = []
        for name in shards:
            with np.load(os.path.join(path, name)) as data:
                parts.append({key: data[key] for key in GRAPH_KEYS})
                index_parts.append({key: data[key] for key in INDEX_DTYPES})
        fields = {key: np.concatenate([p[key] for p in parts], 0) for key in GRAPH_KEYS}
        index = {
            key: np.concatenate([p[key] for p in index_parts], 0) for key in INDEX_DTYPES
        }
        dataset = cls(GraphFrames(fields), index, meta)
        recorded = int(meta.get("frames", len(dataset)))
        if recorded != len(dataset):
            raise ValueError(
                f"cache at {path!r} records {recorded} frames but its shards hold "
                f"{len(dataset)}; the cache is incomplete"
            )
        return dataset

    def compatible_with(self, collect_cfg: Mapping) -> tuple[bool, str]:
        """Whether this cache was produced by the collection now configured."""
        settings = self.meta.get("collect") or {}
        for key in (
            "env_id", "n_max", "e_max", "use_target_flag",
            "object_object_spatial", "visibility_policy", "control_mode",
        ):
            want, got = collect_cfg.get(key), settings.get(key)
            if want != got:
                return False, f"{key}: cache has {got!r}, config asks {want!r}"
        if len(self) == 0:
            return False, "cache is empty"
        return True, "compatible"


class ShardWriter:
    """Buffers frames and flushes compressed shards, so a long collection never
    holds the whole run in memory at once."""

    def __init__(self, out_dir: str, shard_size: int = 4096):
        self.out_dir = str(out_dir)
        self.shard_size = int(shard_size)
        if self.shard_size <= 0:
            raise ValueError(f"shard_size must be positive, got {shard_size}")
        os.makedirs(self.out_dir, exist_ok=True)
        # A half-finished previous collection under the same path would be
        # concatenated into this one on load, silently.
        for name in os.listdir(self.out_dir):
            if name.startswith(SHARD_PREFIX) and name.endswith(".npz"):
                os.remove(os.path.join(self.out_dir, name))
        self._buffer: list[tuple[dict[str, np.ndarray], dict[str, object]]] = []
        self._shards = 0
        self.frames = 0

    def add(self, packed: Mapping[str, np.ndarray], index: Mapping[str, object]) -> None:
        check_fields(packed, frame_axis=False)
        missing = [key for key in INDEX_DTYPES if key not in index]
        if missing:
            raise KeyError(f"frame index is missing {missing}")
        self._buffer.append(
            ({key: np.asarray(packed[key]) for key in GRAPH_KEYS}, dict(index))
        )
        self.frames += 1
        if len(self._buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        payload: dict[str, np.ndarray] = {
            key: np.stack([frame[key] for frame, _ in self._buffer]).astype(
                FIELD_DTYPES[key], copy=False
            )
            for key in GRAPH_KEYS
        }
        for key, dtype in INDEX_DTYPES.items():
            payload[key] = np.asarray(
                [row[key] for _, row in self._buffer], dtype=dtype
            )
        path = os.path.join(self.out_dir, f"{SHARD_PREFIX}{self._shards:05d}.npz")
        np.savez_compressed(path, **payload)
        self._shards += 1
        self._buffer.clear()

    def close(self, meta: Mapping) -> str:
        self.flush()
        payload = dict(meta)
        payload["frames"] = self.frames
        payload["shards"] = self._shards
        payload["revision"] = payload.get("revision") or git_revision()
        path = os.path.join(self.out_dir, META_NAME)
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        return path


def write_frames(
    out_dir: str,
    frames: Iterable[Mapping[str, np.ndarray]],
    index: Iterable[Mapping[str, object]],
    meta: Mapping,
    *,
    shard_size: int = 4096,
) -> str:
    """Whole-table convenience path, used by the tests and the round-trip check."""
    writer = ShardWriter(out_dir, shard_size)
    for packed, row in zip(frames, index):
        writer.add(packed, row)
    return writer.close(meta)
