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

**Terminations follow the online environment, not the recording.** The trainer
builds its ManiSkill env with ``ignore_terminations=True``
(``envs/maniskill.py:528``), so online an episode never terminates: it runs to
the horizon and the value function bootstraps there. The demonstrations were
recorded without that wrapper, and ManiSkill sets ``terminated`` from the
task's own success flag, so every demo carries terminal transitions the online
env would never produce.

Honouring the recording would train a continuation head on a signal that does
not exist at rollout time. So under the default policy the recorded
``terminated`` is ignored, ``is_terminal`` is false everywhere, and the
collector's post-success steps are ordinary steps rather than a tail after an
ending. ``ignore_terminations=False`` restores the recording's own semantics
and cuts each episode at its first terminal step; it is the right setting only
for a trainer configured the same way.

**First success is not settled success.** The flag flickers -- PickCube's
success asks for a static robot as well as a placed cube -- so the step an
episode *reaches* success and the step it *keeps* success are different
numbers, and only the second is where a demonstration could be cut. Both are
computed per episode and neither is inferred from the other.

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
    settled_steps: Optional[int]   # as the collector recorded it
    terminal: bool                 # the usable window ends on a real terminal step
    first_success: Optional[int]   # actions to the first success, flicker included
    settled_success: Optional[int] # actions to the success that held to the end

    @property
    def observations(self) -> int:
        return self.steps + 1


def _first_terminal(terminated: np.ndarray) -> Optional[int]:
    """Index of the first terminal transition, if there is one."""
    hits = np.flatnonzero(np.asarray(terminated, dtype=bool))
    return int(hits[0]) if hits.size else None


def success_milestones(success: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
    """``(first, settled)`` in actions, from a per-step success array.

    They differ whenever the flag drops again, which it does: the two are kept
    apart because a demonstration can only be cut at the second, while the
    first is what a naive reading of the array reports.
    """
    flags = np.asarray(success, dtype=bool).reshape(-1)
    if flags.size == 0:
        return None, None
    hits = np.flatnonzero(flags)
    first = int(hits[0]) + 1 if hits.size else None
    if not bool(flags[-1]):
        return first, None
    misses = np.flatnonzero(~flags)
    return first, (int(misses[-1]) + 2 if misses.size else 1)


class DemoDataset:
    """One task's collected demonstrations, read under an arm's allowlist.

    Opened lazily and kept open: a sampler asks for many small windows and
    reopening the file per window is most of the cost of reading it.
    """

    def __init__(self, path: str | Path, *, graph_enabled: bool,
                 ignore_terminations: bool = True):
        self.path = Path(path)
        self.sidecar = self.path.with_suffix(".json")
        if not self.path.exists() or not self.sidecar.exists():
            raise FileNotFoundError(
                f"expected a dataset and its sidecar at {self.path}")
        payload = json.loads(self.sidecar.read_text(encoding="utf-8"))
        self.metadata: Dict[str, Any] = dict(payload.get("metadata") or {})
        self.graph_enabled = bool(graph_enabled)
        # Matches the online env. See the module docstring: honouring the
        # recording's terminations would train against a signal the rollout
        # never produces.
        self.ignore_terminations = bool(ignore_terminations)
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
            node = handle[group]
            terminated = np.asarray(node["terminated"][()], dtype=bool)
            recorded = int(terminated.shape[0])
            first, settled = success_milestones(
                np.asarray(node["success"][()], dtype=bool))
            # Under the online policy there is no terminal to cut at, so every
            # recorded step is usable -- including the collector's pad, which
            # is only a tail if something ended before it.
            cut = None if self.ignore_terminations else _first_terminal(terminated)
            steps = recorded if cut is None else cut + 1
            refs.append(EpisodeRef(
                dataset=self.path, group=group,
                episode_id=int(entry["episode_id"]), steps=steps,
                recorded_steps=recorded,
                seed=entry.get("seed"),
                end_reason=str(entry.get("end_reason") or ""),
                settled_steps=entry.get("settled_steps"),
                terminal=cut is not None,
                first_success=first, settled_success=settled,
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
