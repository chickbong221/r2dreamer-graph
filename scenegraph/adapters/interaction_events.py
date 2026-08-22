"""Success-gated interaction evidence for normal ManiSkill.

A bucket is ``relation / src_key / dst_key``. Buckets are discovered from what
actually happened in episodes that later succeeded -- nothing is declared in
advance. Directed relations keep their semantic orientation; symmetric ones are
stored once in canonical entity-key order, matching runtime pair ordering.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

EE_KEY = "ee"

# Symmetric: no semantic direction, so the pair is stored in key order.
SYMMETRIC_RELATIONS = frozenset({"contact"})
# Directed: the caller supplies the orientation and it is preserved.
DIRECTED_RELATIONS = frozenset({"grasp", "support", "contain"})


@dataclass(frozen=True, order=True)
class BucketKey:
    relation: str
    src: str
    dst: str

    def __str__(self) -> str:
        return f"{self.relation} / {self.src} / {self.dst}"


def make_bucket(relation: str, src: str, dst: str) -> BucketKey:
    """Canonicalize a bucket. Symmetric relations sort their endpoints.

    ``ee`` is never sorted into second place: an ee-object relation is directed
    by construction even when the relation itself is symmetric.
    """
    if relation in SYMMETRIC_RELATIONS and EE_KEY not in (src, dst):
        if dst < src:
            src, dst = dst, src
    return BucketKey(relation, src, dst)


@dataclass
class InteractionEvent:
    """One observation. ``payload`` is relation-specific and opaque here."""
    bucket: BucketKey
    frame: int
    payload: Dict[str, Any] = field(default_factory=dict)


class EpisodeEvidence:
    """One episode's events behind a success gate.

    Events accumulate unconditionally; nothing leaves unless the episode
    succeeded. Pre-success approach frames are kept, because the episode that
    contains them succeeded.
    """

    def __init__(self) -> None:
        self.events: List[InteractionEvent] = []
        self.success_once: bool = False

    def observe_success(self, flag: Any) -> None:
        self.success_once = self.success_once or bool(flag)

    def add(self, event: InteractionEvent) -> None:
        self.events.append(event)

    def __len__(self) -> int:
        return len(self.events)

    def reset(self) -> None:
        self.events = []
        self.success_once = False

    def commit(self, sink: "BucketStore") -> int:
        """Move events into ``sink`` if the episode succeeded. Returns the
        number committed; a failed episode contributes zero."""
        if not self.success_once:
            self.reset()
            return 0
        n = 0
        for event in self.events:
            if sink.add(event):
                n += 1
        sink.end_episode({e.bucket for e in self.events})
        self.reset()
        return n


class BucketStore:
    """Discovered buckets and their samples, capped per bucket."""

    def __init__(self, target: int = 300) -> None:
        self.target = int(target)
        self.samples: Dict[BucketKey, List[InteractionEvent]] = defaultdict(list)
        self.seen_counts: Dict[BucketKey, int] = defaultdict(int)
        self.frozen: Optional[set] = None
        self.episodes: int = 0
        # Successful episodes each bucket appeared in at all. Separates a real
        # interaction from an incidental brush far better than a raw count: a
        # brush is frequent within the rare episode that has it.
        self.episode_presence: Dict[BucketKey, int] = defaultdict(int)
        # Buckets rejected because discovery had already frozen.
        self.late: Dict[BucketKey, int] = defaultdict(int)
        self._min_presence_seen: float = 0.0
        # Buckets frozen out as incidental. Reported, never chased: an object
        # the gripper brushes in one episode of twenty cannot reach target, and
        # waiting for it would stall the whole run.
        self.excluded: Dict[BucketKey, float] = {}
        # Episodes a late bucket appeared in, so the report can say whether it
        # would have survived the presence gate had discovery seen it.
        self.late_presence: Dict[BucketKey, int] = defaultdict(int)

    def add(self, event: InteractionEvent) -> bool:
        bucket = event.bucket
        if bucket in self.excluded:
            return False
        if self.frozen is not None and bucket not in self.frozen:
            self.late[bucket] += 1
            return False
        self.seen_counts[bucket] += 1
        if len(self.samples[bucket]) >= self.target:
            return False
        self.samples[bucket].append(event)
        return True

    def end_episode(self, buckets: Optional[Iterable[BucketKey]] = None) -> None:
        self.episodes += 1
        for bucket in buckets or ():
            if self.frozen is None or bucket in self.frozen:
                self.episode_presence[bucket] += 1
            elif bucket not in self.excluded:
                self.late_presence[bucket] += 1

    def presence(self, bucket: BucketKey) -> float:
        """Fraction of committed episodes this bucket appeared in.

        An excluded bucket reports the rate measured when it was frozen out.
        It stops accumulating presence at that point while episodes keep
        counting, so a live ratio would decay toward zero and misreport why
        the bucket was dropped.
        """
        if bucket in self.excluded:
            return self.excluded[bucket]
        if not self.episodes:
            return 0.0
        return self.episode_presence[bucket] / self.episodes

    def incidental(self, min_presence: float) -> List[BucketKey]:
        """Buckets too rare to be a task interaction and not already frozen
        out. Freeze reports its own; this catches the rest, which is what a
        pilot run (no freeze) needs."""
        return [b for b in self.buckets()
                if b not in self.excluded
                and self.presence(b) < min_presence]

    def freeze(self, min_presence: float = 0.0) -> None:
        """Stop admitting new buckets. Existing ones keep filling.

        Buckets below ``min_presence`` are frozen out here rather than at the
        end: discovery has settled by now, so presence is measured over enough
        episodes to trust, and excluding them is what lets the run finish.
        """
        known = set(self.samples) | set(self.seen_counts)
        if min_presence > 0.0:
            for bucket in sorted(known):
                rate = self.presence(bucket)
                if rate < min_presence:
                    self.excluded[bucket] = rate
        self.frozen = known - set(self.excluded)
        self._min_presence_seen = min_presence

    def buckets(self) -> List[BucketKey]:
        return sorted(set(self.samples) | set(self.seen_counts))

    def complete(self) -> List[BucketKey]:
        return [b for b in self.buckets()
                if len(self.samples[b]) >= self.target]

    def incomplete(self) -> List[BucketKey]:
        """Buckets still short of target, excluding incidental ones."""
        return [b for b in self.buckets()
                if b not in self.excluded
                and len(self.samples[b]) < self.target]

    def is_done(self) -> bool:
        wanted = [b for b in self.buckets() if b not in self.excluded]
        return bool(wanted) and not self.incomplete()

    def complete_buckets(self) -> List[BucketKey]:
        """The only buckets a whitelist may be built from."""
        return [b for b in self.complete() if b not in self.excluded]

    def report(self) -> str:
        lines = [f"episodes committed: {self.episodes}",
                 f"target per bucket: {self.target}"]
        for b in self.buckets():
            if b in self.excluded:
                continue          # reported once, on its own SKIP line
            have, seen = len(self.samples[b]), self.seen_counts[b]
            mark = "ok " if have >= self.target else "SHORT"
            lines.append(
                f"  {mark} {have:4d}/{self.target}  (seen {seen}, "
                f"in {self.presence(b):.0%} of episodes)  {b}")
        for b, rate in sorted(self.excluded.items()):
            lines.append(f"  SKIP  incidental at {rate:.0%} of episodes: {b}")
        for b, n in sorted(self.late.items()):
            rate = self.late_presence[b] / self.episodes if self.episodes else 0.0
            note = ("below the presence gate, would have been dropped anyway"
                    if rate < self._min_presence_seen
                    else "ABOVE the presence gate -- real, and it was missed")
            lines.append(
                f"  LATE  {n} events in {rate:.1%} of episodes; {note}: {b}")
        return "\n".join(lines)


class DiscoveryWindow:
    """Ends discovery once no new bucket has appeared for ``patience``
    successful episodes. Only stabilizes the bucket set; it is not the
    collection stopping criterion."""

    def __init__(self, patience: int = 25) -> None:
        self.patience = int(patience)
        self.since_new = 0
        self.known: set = set()

    def observe(self, buckets: Iterable[BucketKey]) -> None:
        new = {b for b in buckets} - self.known
        if new:
            self.known |= new
            self.since_new = 0
        else:
            self.since_new += 1

    @property
    def settled(self) -> bool:
        return self.since_new >= self.patience


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #
# Payload fields averaged across a group's selected frames.
AVERAGED_VECTORS = (
    "contact_position", "force_vector", "anchor_a_local", "anchor_b_local",
)
# Averaged and then renormalized to unit length.
NORMALIZED_VECTORS = ("contact_normal", "normal_a_local", "normal_b_local")
# Everything else rides through from the anchor frame rather than an allowlist.
# An allowlist silently drops any field added upstream, and the drop surfaces
# only as an empty component list several stages later.
_REDUCED_KEYS = frozenset(AVERAGED_VECTORS) | frozenset(NORMALIZED_VECTORS) | {
    "force",
}

GROUP_GAP = 5      # observed steps without the predicate before a group ends
GROUP_WINDOW = 5   # positive frames averaged, centred on the peak


def _mean(vectors):
    arr = [v for v in vectors if v is not None]
    if not arr:
        return None
    return [float(x) for x in (sum(map(_np_array, arr)) / len(arr))]


def _np_array(v):
    import numpy as np
    return np.asarray(v, dtype=float)


def _unit(vector):
    import numpy as np
    if vector is None:
        return None
    arr = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(arr))
    return None if norm <= 0.0 else (arr / norm).tolist()


@dataclass
class _OpenGroup:
    first_frame: int
    last_frame: int
    frames: List[int] = field(default_factory=list)
    payloads: List[Dict[str, Any]] = field(default_factory=list)


def reduce_group(group: "_OpenGroup", mode: str,
                 window: int = GROUP_WINDOW) -> Dict[str, Any]:
    """One group -> one sample.

    ``peak`` averages up to ``window`` frames around the strongest force and
    takes orientations from the peak frame. ``last`` keeps the final positive
    frame, which is what a grasp interval means: the pose at release.
    """
    forces = [float(p.get("force", 0.0) or 0.0) for p in group.payloads]
    peak_i = max(range(len(forces)), key=forces.__getitem__) if forces else 0

    if mode == "last":
        chosen = [len(group.payloads) - 1]
        anchor_i = chosen[0]
    else:
        half = max(window // 2, 0)
        lo = max(0, peak_i - half)
        hi = min(len(group.payloads), lo + window)
        lo = max(0, hi - window)
        chosen = list(range(lo, hi))
        anchor_i = peak_i

    picked = [group.payloads[i] for i in chosen]
    anchor = group.payloads[anchor_i] if group.payloads else {}

    out: Dict[str, Any] = {
        "peak_force": max(forces) if forces else 0.0,
        "mean_force": (sum(forces[i] for i in chosen) / len(chosen)
                       if chosen else 0.0),
        "duration": group.last_frame - group.first_frame + 1,
        "n_frames": len(group.payloads),
        "n_averaged": len(chosen),
        "first_frame": group.first_frame,
        "peak_frame": group.frames[anchor_i] if group.frames else group.first_frame,
        "last_frame": group.last_frame,
    }
    for key in AVERAGED_VECTORS:
        value = _mean([p.get(key) for p in picked])
        if value is not None:
            out[key] = value
    for key in NORMALIZED_VECTORS:
        value = _unit(_mean([p.get(key) for p in picked]))
        if value is not None:
            out[key] = value
    for key, value in anchor.items():
        if key not in _REDUCED_KEYS and key not in out:
            out[key] = value
    return out


class GroupAccumulator:
    """Turns per-step positives into one sample per physical interaction.

    A group survives gaps shorter than ``gap`` observed steps and ends after
    ``gap`` consecutive steps without the predicate. A single-step touch is a
    valid group. Two groups are never averaged together.
    """

    def __init__(self, mode: str = "peak", gap: int = GROUP_GAP,
                 window: int = GROUP_WINDOW):
        if mode not in ("peak", "last"):
            raise ValueError(f"unknown group mode {mode!r}")
        self.mode = mode
        self.gap = int(gap)
        self.window = int(window)
        self.open: Dict[BucketKey, _OpenGroup] = {}

    def observe(self, bucket: BucketKey, frame: int,
                payload: Dict[str, Any]) -> None:
        group = self.open.get(bucket)
        if group is None:
            group = _OpenGroup(first_frame=frame, last_frame=frame)
            self.open[bucket] = group
        group.last_frame = frame
        group.frames.append(frame)
        group.payloads.append(payload)

    def tick(self, frame: int) -> List[InteractionEvent]:
        """Close groups idle for ``gap`` steps. Call once per control step."""
        done = []
        for bucket in list(self.open):
            group = self.open[bucket]
            if frame - group.last_frame >= self.gap:
                done.append(self._close(bucket))
        return done

    def close_all(self) -> List[InteractionEvent]:
        """Finalize every open group, e.g. when an episode ends mid-grasp."""
        return [self._close(b) for b in list(self.open)]

    def _close(self, bucket: BucketKey) -> InteractionEvent:
        group = self.open.pop(bucket)
        return InteractionEvent(
            bucket, group.first_frame, reduce_group(group, self.mode,
                                                    self.window))


# --------------------------------------------------------------------------- #
# Spatial statistics for relation bin derivation
# --------------------------------------------------------------------------- #
class BinStats:
    """Running maxima over every pair, every step.

    Relation bins must describe the range a run actually spans. Sampling only
    when a predicate fires would calibrate every bin against contact distance
    alone, so "far" would never be reachable and the token would mean nothing.
    """

    def __init__(self, reservoir: int = 20000, seed: int = 0) -> None:
        self.maxes: Dict[str, float] = defaultdict(float)
        # A max alone cannot bin a bimodal distribution: tabletop scenes put
        # object pairs near zero and table pairs near the table origin, 0.9m
        # below its own surface, so equal-width bins over the max collapse
        # both modes into one label. Quantiles adapt to where the data is.
        self.samples: Dict[str, List[float]] = defaultdict(list)
        self.seen: Dict[str, int] = defaultdict(int)
        self.capacity = int(reservoir)
        self._rng = __import__("random").Random(seed)
        self._prev: Dict[Tuple[str, str], Tuple[float, float]] = {}

    def _record(self, key: str, value: float) -> None:
        self.maxes[key] = max(self.maxes[key], abs(value))
        self.seen[key] += 1
        bucket = self.samples[key]
        if len(bucket) < self.capacity:
            bucket.append(float(value))
            return
        j = self._rng.randrange(self.seen[key])
        if j < self.capacity:
            bucket[j] = float(value)

    def new_episode(self) -> None:
        self._prev.clear()

    def observe(self, poses: Dict[str, Any], frame: int) -> None:
        import itertools as _it

        import numpy as np

        keys = sorted(poses)
        for a, b in _it.combinations(keys, 2):
            pa = np.asarray(poses[a], dtype=float)
            pb = np.asarray(poses[b], dtype=float)
            planar = float(np.linalg.norm(pa[:2] - pb[:2]))
            height = float(pa[2] - pb[2])
            self._record("planar_distance", planar)
            self._record("height_offset", height)
            prev = self._prev.get((a, b))
            if prev is not None:
                self._record("planar_distance_change", planar - prev[0])
                self._record("height_offset_change", height - prev[1])
            self._prev[(a, b)] = (planar, height)

    def merge(self, other: Dict[str, float]) -> None:
        for key, value in (other or {}).items():
            self.maxes[key] = max(self.maxes[key], float(value))

    def as_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.maxes.items()}

    def reservoir(self) -> Dict[str, Any]:
        """The raw sample per statistic, as float32.

        The samples travel rather than precomputed quantiles: quantiles cannot
        be merged across shards -- averaging or overwriting them both give a
        distribution no worker observed -- while reservoirs concatenate.
        """
        import numpy as np

        return {k: np.asarray(v, dtype=np.float32)
                for k, v in self.samples.items() if v}
