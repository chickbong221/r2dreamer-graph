"""Reading collected demonstrations as learning batches, under a graph switch.

Two things live here and they are deliberately separate. An *episode* is
everything the collector stored -- images, proprioception, graphs, rewards,
simulator state, privileged fields. A *batch* is the subset a given arm is
allowed to learn from, and which subset that is depends on one flag.

The allowlist is the whole point. ``model.graph.enabled=false`` must mean the
graph never reaches the model, and the way to guarantee that is not to filter
tensors late but to never read the arrays at all: :func:`batch_fields` decides
what the loader opens, so a baseline batch has no graph key to leak through an
encoder that happened to iterate its inputs. The graph-isolation test asserts
this by corrupting the stored graphs and checking a baseline batch is
byte-identical.

Episodes stop at their first terminal step. ManiSkill sets ``terminated`` from
the task's own success flag, and the collector keeps a few steps past the point
success settles, so the recorded tail sits *after* a genuine terminal
transition. Learning from it would teach a continuation head that episodes go
on after they end. The steps are still on disk -- this drops them from batches,
not from the dataset.

Arrays come out as numpy. Turning them into tensors is the trainer's job: it
knows the device and the dtype policy, and keeping torch out of the loader is
what lets the alignment be tested without a GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from scenegraph.adapters.graph_pack import GRAPH_KEYS

# Recorded per observation, and inputs for both arms.
SENSOR_FIELDS: Tuple[str, ...] = ("proprio",)
# Recorded per transition. Supervision and masking, never encoder inputs.
SUPERVISION_FIELDS: Tuple[str, ...] = (
    "actions", "rewards", "terminated", "truncated", "success",
)
# Recorded, never learned from. Kept for diagnostics and for re-rendering.
DIAGNOSTIC_FIELDS: Tuple[str, ...] = ("env_states", "privileged")


@dataclass(frozen=True)
class BatchFields:
    """Exactly which stored arrays one arm's batches may contain."""

    images: Tuple[str, ...]
    observations: Tuple[str, ...]
    supervision: Tuple[str, ...]
    graph: Tuple[str, ...]

    @property
    def all(self) -> Tuple[str, ...]:
        return self.images + self.observations + self.graph + self.supervision


def batch_fields(metadata: Mapping[str, Any], *, graph_enabled: bool) -> BatchFields:
    """The allowlist for an arm.

    Cameras come from the dataset's own metadata rather than a constant: both
    arms get every camera the episodes were recorded with, because turning the
    graph off is not a reason to take a wrist view away.
    """
    images = tuple(sorted(str(v) for v in
                          (metadata.get("camera_keys") or {}).values()))
    return BatchFields(
        images=images,
        observations=SENSOR_FIELDS,
        supervision=SUPERVISION_FIELDS,
        graph=tuple(GRAPH_KEYS) if graph_enabled else (),
    )


@dataclass
class EpisodeRef:
    """One episode's identity and shape, without its arrays."""

    dataset: Path
    group: str
    episode_id: int
    steps: int                 # actions actually usable, after the terminal cut
    recorded_steps: int        # actions as stored
    seed: Optional[int]
    end_reason: str
    settled_steps: Optional[int]
    terminal: bool             # the usable window ends on a real terminal step

    @property
    def observations(self) -> int:
        return self.steps + 1


def _first_terminal(terminated: np.ndarray) -> Optional[int]:
    """Index of the first terminal transition, if there is one."""
    hits = np.flatnonzero(np.asarray(terminated, dtype=bool))
    return int(hits[0]) if hits.size else None


class DemoDataset:
    """One task's collected demonstrations, read under an arm's allowlist.

    Opened lazily and kept open: a sampler asks for many small windows and
    reopening the file per window is most of the cost of reading it.
    """

    def __init__(self, path: str | Path, *, graph_enabled: bool,
                 stop_at_terminal: bool = True):
        self.path = Path(path)
        self.sidecar = self.path.with_suffix(".json")
        if not self.path.exists() or not self.sidecar.exists():
            raise FileNotFoundError(
                f"expected a dataset and its sidecar at {self.path}")
        payload = json.loads(self.sidecar.read_text(encoding="utf-8"))
        self.metadata: Dict[str, Any] = dict(payload.get("metadata") or {})
        self.graph_enabled = bool(graph_enabled)
        self.stop_at_terminal = bool(stop_at_terminal)
        self.fields = batch_fields(self.metadata, graph_enabled=graph_enabled)
        self._handle = None
        self.episodes: List[EpisodeRef] = self._index(payload.get("episodes") or [])

    # ------------------------------------------------------------------ index
    def _index(self, entries: Sequence[Mapping[str, Any]]) -> List[EpisodeRef]:
        refs: List[EpisodeRef] = []
        handle = self._open()
        for entry in entries:
            group = f"traj_{int(entry['episode_id'])}"
            if group not in handle:
                continue
            terminated = np.asarray(handle[group]["terminated"][()], dtype=bool)
            recorded = int(terminated.shape[0])
            cut = _first_terminal(terminated) if self.stop_at_terminal else None
            # The terminal transition itself is kept; what is dropped is
            # everything after it.
            steps = recorded if cut is None else cut + 1
            refs.append(EpisodeRef(
                dataset=self.path, group=group,
                episode_id=int(entry["episode_id"]), steps=steps,
                recorded_steps=recorded,
                seed=entry.get("seed"),
                end_reason=str(entry.get("end_reason") or ""),
                settled_steps=entry.get("settled_steps"),
                terminal=cut is not None,
            ))
        return refs

    def _open(self):
        if self._handle is None:
            import h5py

            self._handle = h5py.File(self.path, "r")
        return self._handle

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "DemoDataset":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.episodes)

    @property
    def total_steps(self) -> int:
        return sum(ref.steps for ref in self.episodes)

    # ------------------------------------------------------------------- read
    def read(self, ref: EpisodeRef, start: int, stop: int) -> Dict[str, np.ndarray]:
        """Arrays for transitions ``[start, stop)`` of one episode.

        Observation-indexed fields come back with one extra row -- the
        observation the last action led to -- so a caller always has ``o_t``
        and ``o_{t+1}`` for every action in the window.
        """
        if not 0 <= start < stop <= ref.steps:
            raise IndexError(
                f"[{start}, {stop}) is not inside episode {ref.episode_id} "
                f"of {ref.steps} usable steps")
        group = self._open()[ref.group]
        obs = group["obs"]
        out: Dict[str, np.ndarray] = {}
        for key in self.fields.images + self.fields.graph:
            out[key] = np.asarray(obs[key][start:stop + 1])
        for key in self.fields.observations:
            out[key] = np.asarray(obs[key][start:stop + 1])
        for key in self.fields.supervision:
            out[key] = np.asarray(group[key][start:stop])
        return out

    def diagnostics(self, ref: EpisodeRef, start: int, stop: int) -> Dict[str, Any]:
        """Simulator and privileged arrays. Never part of a batch.

        Separate method, not a flag on :meth:`read`: a field reachable through
        the same call an encoder makes is a field that eventually reaches the
        encoder.
        """
        group = self._open()[ref.group]
        out: Dict[str, Any] = {}
        for name in DIAGNOSTIC_FIELDS:
            if name in group:
                out[name] = _read_tree(group[name], start, stop + 1)
        return out


def _read_tree(node, start: int, stop: int) -> Any:
    import h5py

    if isinstance(node, h5py.Group):
        return {key: _read_tree(node[key], start, stop) for key in node}
    return np.asarray(node[start:stop])
