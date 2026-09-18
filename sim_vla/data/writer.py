"""One HDF5 file per task, one group per episode, plus a metadata sidecar.

Layout, for an episode with ``T`` executed actions::

    traj_<id>/
        obs/image_<cam>     (T+1, H, W, 3) uint8
        obs/proprio         (T+1, D)       float32
        obs/graph_<key>     (T+1, ...)     the nine packed arrays
        actions             (T, A)         float32
        rewards             (T,)           float32
        terminated          (T,)           bool
        truncated           (T,)           bool
        success             (T,)           bool
        env_states/...      (T+1, ...)     nested, as the simulator gave it
        privileged/<key>    (T+1, ...)     never a model input

The observation arrays are one longer than the transition arrays, and which is
which is declared in ``schema.field_kinds`` rather than left to be inferred --
see :mod:`sim_vla.data.schema`.

The sidecar is written after every episode rather than at the end, so an
interrupted run leaves a file whose finished episodes are all described. The
order is deliberate: the HDF5 is flushed first and the sidecar is then replaced
atomically, so a sidecar that lists an episode is never newer than the data it
lists. Each group also carries a ``complete`` attribute written last, which is
what :func:`incomplete_groups` looks for.

That is crash *recovery*, not crash *proof*. A process killed mid-write can
still leave a group without its attribute, and nothing here protects against a
storage failure -- it protects against the run stopping.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from .schema import KIND_OBS, KIND_STEP


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(str(v) for v in value)
    return str(value)


def atomic_replace(tmp: Path, target: Path, attempts: int = 5,
                   delay: float = 0.05) -> None:
    """Move ``tmp`` over ``target``, retrying a transient lock.

    On Windows a freshly written file can be held briefly by the indexer or a
    virus scanner, and ``os.replace`` then fails with a permission error even
    though nothing in this process has it open. It is transient -- observed
    about once in six runs of the test suite -- but this runs after every
    episode of a collection that lasts hours, so once is enough to lose a
    worker. POSIX never takes this path.

    The last resort is a direct write, which gives up atomicity rather than the
    run: a sidecar that might be torn is worth more than a dead worker with
    hundreds of episodes already on disk.
    """
    for attempt in range(max(int(attempts), 1)):
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                break
            time.sleep(delay * (attempt + 1))
    target.write_text(tmp.read_text(encoding="utf-8"), encoding="utf-8")
    tmp.unlink(missing_ok=True)


def _leaves(prefix: str, payload: Mapping[str, Any]) -> List[tuple]:
    """Every array in a nested tree, with a dotted name for the error message."""
    out: List[tuple] = []
    for key, value in (payload or {}).items():
        name = f"{prefix}.{key}"
        if isinstance(value, Mapping):
            out.extend(_leaves(name, value))
        else:
            out.append((name, np.asarray(value)))
    return out


def incomplete_groups(path: str | Path) -> List[str]:
    """Episode groups written without their completion marker.

    A group gets ``complete`` only after every dataset in it exists, so one
    without the attribute is an episode a killed process left half-written.
    Reported rather than deleted: which of them to drop is a question about the
    dataset, not about the file.
    """
    import h5py

    with h5py.File(path, "r") as handle:
        return [key for key in handle
                if key.startswith("traj_")
                and not bool(handle[key].attrs.get("complete", False))]


def write_nested(group, name: str, payload: Mapping[str, Any]) -> None:
    """Write a nested dict of arrays, preserving its shape.

    The simulator state is a tree of per-actor and per-articulation entries and
    it is stored as one: flattening it here would mean inventing a key
    convention that the restore path does not share.
    """
    sub = group.create_group(name)

    def walk(node, data: Mapping[str, Any]) -> None:
        for key, value in data.items():
            if isinstance(value, Mapping):
                walk(node.create_group(str(key)), value)
            else:
                node.create_dataset(str(key), data=np.asarray(value))

    walk(sub, payload)


class DatasetWriter:
    """Accumulates accepted episodes into one task's file."""

    # RGB compresses well and is most of the file; gzip 5 is what ManiSkill's
    # own recorder uses for image data, and matching it keeps read speed
    # comparable to the trajectories this sits beside.
    _IMAGE_COMPRESSION = dict(compression="gzip", compression_opts=5)

    def __init__(self, path: str | Path, metadata: Mapping[str, Any],
                 *, overwrite: bool = False):
        import h5py

        self.path = Path(path)
        # Refused rather than truncated. Re-running a collection command with
        # the same name is the normal way to add to a dataset, and opening "w"
        # turns that into eight hours of deletion.
        if self.path.exists() and not overwrite:
            raise SystemExit(
                f"{self.path} already exists. Pass --overwrite to replace it, "
                f"or collect under a different --name and merge.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._h5 = h5py.File(self.path, "w")
        self.metadata = dict(metadata)
        self.episodes: List[Dict[str, Any]] = []
        self.count = 0

    # ------------------------------------------------------------------ write
    def add(
        self,
        *,
        images: Mapping[str, np.ndarray],
        proprio: np.ndarray,
        graphs: Mapping[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        success: np.ndarray,
        env_states: Mapping[str, Any],
        privileged: Mapping[str, np.ndarray],
        info: Mapping[str, Any],
    ) -> int:
        """Commit one episode. Returns the id it was written under."""
        steps = int(len(actions))
        expected = steps + 1
        obs_arrays = [("proprio", proprio), *images.items(), *graphs.items()]
        # Nested trees are checked leaf by leaf. Checking only the flat arrays
        # let an episode through with three observations, two actions and a
        # single row of simulator state -- every per-step array agreeing with
        # every other one except the two nobody looked at.
        obs_arrays += _leaves("env_states", env_states)
        obs_arrays += _leaves("privileged", privileged)
        for name, arr in obs_arrays:
            if len(arr) != expected:
                raise ValueError(
                    f"{name} has {len(arr)} rows, expected {expected} for an "
                    f"episode of {steps} actions -- observation arrays carry "
                    f"the reset frame as well")
        for name, arr in (("rewards", rewards), ("terminated", terminated),
                          ("truncated", truncated), ("success", success)):
            if len(arr) != steps:
                raise ValueError(
                    f"{name} has {len(arr)} rows, expected {steps}")

        episode_id = self.count
        group = self._h5.create_group(f"traj_{episode_id}")
        obs = group.create_group("obs")
        for key, arr in images.items():
            obs.create_dataset(key, data=np.asarray(arr, dtype=np.uint8),
                               **self._IMAGE_COMPRESSION)
        obs.create_dataset("proprio", data=np.asarray(proprio, dtype=np.float32))
        for key, arr in graphs.items():
            obs.create_dataset(key, data=np.asarray(arr))

        group.create_dataset("actions", data=np.asarray(actions, dtype=np.float32))
        group.create_dataset("rewards", data=np.asarray(rewards, dtype=np.float32))
        group.create_dataset("terminated", data=np.asarray(terminated, dtype=bool))
        group.create_dataset("truncated", data=np.asarray(truncated, dtype=bool))
        group.create_dataset("success", data=np.asarray(success, dtype=bool))
        write_nested(group, "env_states", env_states)
        if privileged:
            write_nested(group, "privileged", privileged)
        # Written last, and it is what tells a recovery pass this group has
        # every dataset it claims to.
        group.attrs["complete"] = True

        self.episodes.append(dict(info) | {
            "episode_id": episode_id,
            "steps": steps,
            "observations": expected,
        })
        self.count += 1
        self._flush_sidecar()
        return episode_id

    # ----------------------------------------------------------------- finish
    def _flush_sidecar(self) -> None:
        """Publish the episode list, never ahead of the data it describes.

        The HDF5 is flushed first so that an entry in the sidecar always has
        its group on disk, and the JSON is replaced atomically so an interrupt
        mid-write leaves the previous complete sidecar rather than a truncated
        one.
        """
        self._h5.flush()
        payload = {
            "metadata": self.metadata,
            "episodes": self.episodes,
            "count": self.count,
        }
        tmp = self.sidecar.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=_json_default),
                       encoding="utf-8")
        atomic_replace(tmp, self.sidecar)

    @property
    def sidecar(self) -> Path:
        return self.path.with_suffix(".json")

    def close(self, summary: Optional[Mapping[str, Any]] = None) -> None:
        if summary:
            self.metadata = dict(self.metadata) | {"collection": dict(summary)}
        self._flush_sidecar()
        self._h5.close()

    def __enter__(self) -> "DatasetWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def field_kinds(image_keys, graph_keys) -> Dict[str, str]:
    """Which arrays are indexed by observation and which by transition."""
    kinds = {key: KIND_OBS for key in image_keys}
    kinds |= {key: KIND_OBS for key in graph_keys}
    kinds |= {
        "proprio": KIND_OBS,
        "env_states": KIND_OBS,
        "privileged": KIND_OBS,
        "actions": KIND_STEP,
        "rewards": KIND_STEP,
        "terminated": KIND_STEP,
        "truncated": KIND_STEP,
        "success": KIND_STEP,
    }
    return kinds
