"""Success-gated interaction evidence for normal ManiSkill.

A bucket is ``relation / src_key / dst_key``. Buckets are discovered from what
actually happened in episodes that later succeeded -- nothing is declared in
advance. Directed relations keep their semantic orientation; symmetric ones are
stored once in canonical entity-key order, matching runtime pair ordering.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..core.spatial_metrics import EE_OBJECT_SCOPE, stat_key

EE_KEY = "ee"

# Tags on the keyed calibration reservoir, so the miner routes a sample by what
# it is rather than by re-deriving it from the key strings.
KIND_EE_OBJECT = "ee-object"
KIND_OBJECT_SITE = "object-site"
KIND_OBJECT_REGION = "object-region"

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


def _scalar_at(value: Any, env_idx: int) -> Optional[float]:
    """One float from a possibly-batched info value, or None if it is not one.

    Nested structures and multi-element-per-env arrays are not predicates and
    are reported rather than coerced.
    """
    if value is None or isinstance(value, (str, bytes, dict, list, tuple, set)):
        return None
    if isinstance(value, (bool, int, float)):
        return float(value)
    arr = getattr(value, "cpu", None)
    arr = arr() if callable(arr) else value
    try:
        arr = np.asarray(arr)
    except Exception:
        return None
    if arr.dtype.kind not in "biuf" or arr.size == 0:
        return None
    if arr.ndim == 0:
        return float(arr)
    flat = arr.reshape(-1)
    if arr.ndim == 1:
        return float(flat[min(env_idx, flat.size - 1)])
    return None


class InfoTrace:
    """Per-frame scalar and boolean values from the environment's own ``info``.

    The task's decomposition of its own success predicate is the one account of
    phase structure that does not come from our detectors, and it exists only
    while the episode runs. Recorded per frame, reduced on commit.
    """

    def __init__(self, env_idx: int = 0) -> None:
        self.env_idx = int(env_idx)
        self.values: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self.skipped: Dict[str, str] = {}

    def reset(self) -> None:
        self.values = defaultdict(list)
        self.skipped = {}

    def observe(self, info: Any, frame: int) -> None:
        if not isinstance(info, dict):
            return
        for key, value in info.items():
            scalar = _scalar_at(value, self.env_idx)
            if scalar is None:
                self.skipped.setdefault(key, type(value).__name__)
                continue
            self.values[str(key)].append((int(frame), scalar))

    def reduce(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
        """``(predicate spans, scalar series, key kinds)``.

        A key whose every observed value is 0 or 1 is a predicate and reduces to
        onset/release spans -- the same shape interaction milestones use, so the
        two orderings are directly comparable. Anything else stays frame-aligned
        because its shape is the signal.
        """
        spans: Dict[str, Any] = {}
        series: Dict[str, Any] = {}
        kinds: Dict[str, str] = {}
        for key, samples in self.values.items():
            if not samples:
                continue
            vals = [v for _, v in samples]
            if all(v in (0.0, 1.0) for v in vals):
                kinds[key] = "predicate"
                spans[key] = _true_spans(samples)
            else:
                kinds[key] = "scalar"
                series[key] = (
                    np.asarray([f for f, _ in samples], dtype=np.int32),
                    np.asarray(vals, dtype=np.float32),
                )
        for key, why in self.skipped.items():
            kinds.setdefault(str(key), f"ignored:{why}")
        return spans, series, kinds


def _true_spans(samples: List[Tuple[int, float]]) -> Tuple[Tuple[int, int], ...]:
    """Maximal runs where the value is 1, as ``(onset, release)`` frames.

    Runs are kept separate here, unlike interaction milestones: a predicate that
    turns off and on again is exactly the evidence that a phase was undone.
    """
    out: List[Tuple[int, int]] = []
    start = prev = None
    for frame, value in samples:
        if value == 1.0:
            if start is None:
                start = frame
            prev = frame
        elif start is not None:
            out.append((start, prev))
            start = prev = None
    if start is not None:
        out.append((start, prev))
    return tuple(out)


@dataclass
class EpisodeRecord:
    """One successful episode, every stream sharing its index."""
    interactions: Tuple[Tuple[str, str, str, int, int], ...]
    predicates: Dict[str, Any] = field(default_factory=dict)
    scalars: Dict[str, Any] = field(default_factory=dict)
    kinds: Dict[str, str] = field(default_factory=dict)
    frames: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Plain types only. A shard outlives the class that wrote it."""
        return {
            "interactions": self.interactions,
            "predicates": {k: v for k, v in self.predicates.items()},
            "scalars": {k: (f, v) for k, (f, v) in self.scalars.items()},
            "kinds": dict(self.kinds),
            "frames": int(self.frames),
        }


def interaction_spans(
    events: Iterable[InteractionEvent],
) -> Tuple[Tuple[str, str, str, int, int], ...]:
    """One ``(relation, src, dst, onset, release)`` per bucket, in onset order.

    One span per bucket, not one per group: a grasp that momentarily breaks
    opens a second group, and two entries would read as two milestones. The
    span runs from the earliest onset to the latest release, so ``release`` is
    when the robot finally let go rather than when it first slipped.
    """
    span: Dict[BucketKey, Tuple[int, int]] = {}
    for event in events:
        on = int(event.frame)
        off = int(event.payload.get("last_frame", on))
        prev = span.get(event.bucket)
        span[event.bucket] = (
            (on, off) if prev is None
            else (min(prev[0], on), max(prev[1], off))
        )
    return tuple(
        (b.relation, b.src, b.dst, on, off)
        for b, (on, off) in sorted(
            span.items(), key=lambda kv: (kv[1][0], kv[1][1], kv[0]))
    )


class EpisodeEvidence:
    """One episode's events behind a success gate.

    Events accumulate unconditionally; nothing leaves unless the episode
    succeeded. Pre-success approach frames are kept, because the episode that
    contains them succeeded.
    """

    def __init__(self, env_idx: int = 0) -> None:
        self.events: List[InteractionEvent] = []
        self.info = InfoTrace(env_idx)
        self.frames: int = 0
        self.success_once: bool = False

    def observe_success(self, flag: Any) -> None:
        self.success_once = self.success_once or bool(flag)

    def observe_info(self, info: Any, frame: int) -> None:
        self.info.observe(info, frame)
        self.frames = max(self.frames, int(frame) + 1)

    def add(self, event: InteractionEvent) -> None:
        self.events.append(event)

    def __len__(self) -> int:
        return len(self.events)

    def reset(self) -> None:
        self.events = []
        self.info.reset()
        self.frames = 0
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
        spans, series, kinds = self.info.reduce()
        sink.add_episode(EpisodeRecord(
            interactions=interaction_spans(self.events),
            predicates=spans, scalars=series, kinds=kinds,
            frames=self.frames,
        ))
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
        # One entry per successful episode: which interactions happened, and
        # when. Buckets say what a task involves; this says in what order, and
        # that is the only thing a phase schedule can be mined from.
        self.traces: List["EpisodeRecord"] = []

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

    def add_episode(self, record: "EpisodeRecord") -> None:
        """Store one successful episode's evidence.

        Independent of ``add``: a bucket at its sample cap stops storing
        payloads but its ordering still counts, and a bucket frozen out as
        incidental is still evidence about this episode's sequence.
        """
        self.traces.append(record)

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
    """Spatial statistics for bin derivation, split by calibration scope.

    Bins must describe the range a run actually spans, so this samples every
    pair every step rather than only when a predicate fires.

    EE-object scales are recorded here directly. Object-object scales are not:
    they are measured at runtime through mined surface anchors, and an object
    origin is not that point -- a table's sits ~0.9m below its own top. What
    travels instead is a reservoir of raw pose pairs at ``t`` and ``t - K``,
    which the miner reprojects once the anchors exist. Changes use the same
    ``K`` as the runtime temporal buffer.
    """

    def __init__(self, reservoir: int = 20000, seed: int = 0,
                 horizon: int = 5, pose_reservoir: int = 20000,
                 keyed_reservoir: int = 20000) -> None:
        self.maxes: Dict[str, float] = defaultdict(float)
        self.samples: Dict[str, List[float]] = defaultdict(list)
        self.seen: Dict[str, int] = defaultdict(int)
        self.capacity = int(reservoir)
        self._rng = __import__("random").Random(seed)
        self.horizon = max(1, int(horizon))
        self.pose_capacity = int(pose_reservoir)
        self.pose_pairs: List[Dict[str, Any]] = []
        self.pose_seen = 0
        # Its own reservoir, deliberately. The object-pair one runs at capacity
        # for a 300-episode task -- PlaceSphere's shard holds exactly 20000 --
        # so routing keyed samples into it would evict the object-object
        # calibration one-for-one with every sample added.
        self.keyed_capacity = int(keyed_reservoir)
        self.keyed_pairs: List[Dict[str, Any]] = []
        self.keyed_seen = 0
        self._history: Dict[
            Tuple[str, str], deque[Tuple[float, float]]
        ] = defaultdict(lambda: deque(maxlen=self.horizon + 1))
        self._pose_history: Dict[
            Tuple[str, str], deque[Tuple[List[float], List[float]]]
        ] = defaultdict(lambda: deque(maxlen=self.horizon + 1))
        self._keyed_history: Dict[
            Tuple[str, str, str], deque[Tuple[List[float], List[float]]]
        ] = defaultdict(lambda: deque(maxlen=self.horizon + 1))

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

    def _record_pose_pair(self, sample: Dict[str, Any]) -> None:
        self.pose_seen += 1
        if len(self.pose_pairs) < self.pose_capacity:
            self.pose_pairs.append(sample)
            return
        j = self._rng.randrange(self.pose_seen)
        if j < self.pose_capacity:
            self.pose_pairs[j] = sample

    def _record_keyed_pair(self, sample: Dict[str, Any]) -> None:
        self.keyed_seen += 1
        if len(self.keyed_pairs) < self.keyed_capacity:
            self.keyed_pairs.append(sample)
            return
        j = self._rng.randrange(self.keyed_seen)
        if j < self.keyed_capacity:
            self.keyed_pairs[j] = sample

    def observe_keyed_pair(self, kind: str, src_key: str, dst_key: str,
                           src_pose, dst_pose) -> None:
        """Record one calibration pair that keeps both endpoint identities.

        The unkeyed end-effector streams cannot be split after the fact: they
        record a height and discard which object produced it, so a shard has no
        way to tell an end-effector-to-sphere sample from an
        end-effector-to-table one. Everything that needs a per-family or
        per-pair scale travels through here instead, tagged with ``kind`` so
        the miner can route it without re-deriving what it is.
        """
        import numpy as np

        src = [float(v) for v in np.asarray(src_pose, dtype=float).reshape(-1)[:7]]
        dst = [float(v) for v in np.asarray(dst_pose, dtype=float).reshape(-1)[:7]]
        if len(src) < 3 or len(dst) < 3:
            return
        history = self._keyed_history[(kind, src_key, dst_key)]
        history.append((src, dst))
        prev_src = prev_dst = None
        if len(history) == self.horizon + 1:
            prev_src, prev_dst = history[0]
        self._record_keyed_pair({
            "kind": kind,
            "src_key": src_key, "dst_key": dst_key,
            "src_pose": src, "dst_pose": dst,
            "prev_src_pose": prev_src, "prev_dst_pose": prev_dst,
        })

    def new_episode(self) -> None:
        self._history.clear()
        self._pose_history.clear()
        self._keyed_history.clear()

    def observe(self, poses: Dict[str, Any], frame: int,
                dynamic: Optional[Iterable[str]] = None) -> None:
        """``dynamic`` names the keys physics can move.

        A pair with no dynamic endpoint is dropped: its relative
        geometry is fixed for the run, so it would pin the scale at a
        constant no pair that actually moves ever reaches. Omitting the
        argument keeps every pair, which is what the tests want.
        """
        import itertools as _it

        import numpy as np

        movable = None if dynamic is None else set(dynamic)
        keys = sorted(poses)
        for a, b in _it.combinations(keys, 2):
            pa = np.asarray(poses[a], dtype=float)
            pb = np.asarray(poses[b], dtype=float)
            if a == EE_KEY or b == EE_KEY:
                self._observe_ee(a, b, pa, pb)
                continue
            if movable is not None and a not in movable and b not in movable:
                continue
            self._observe_object_pair(a, b, pa, pb)

    def _observe_ee(self, a, b, pa, pb) -> None:
        import numpy as np

        # Keyed first, and kept whatever the shared streams do with it: the
        # per-family scales are derived from these and from nothing else.
        ee_key, obj_key = (a, b) if a == EE_KEY else (b, a)
        self.observe_keyed_pair(KIND_EE_OBJECT, ee_key, obj_key,
                                pa if a == EE_KEY else pb,
                                pb if a == EE_KEY else pa)
        scope = EE_OBJECT_SCOPE
        planar = float(np.linalg.norm(pa[:2] - pb[:2]))
        height = float(pa[2] - pb[2])
        self._record(stat_key(scope, "planar-distance"), planar)
        self._record(stat_key(scope, "height-offset"), height)
        history = self._history[(a, b)]
        history.append((planar, height))
        if len(history) == self.horizon + 1:
            old_planar, old_height = history[0]
            self._record(
                f"{stat_key(scope, 'planar-distance')}_change",
                planar - old_planar)
            self._record(
                f"{stat_key(scope, 'height-offset')}_change",
                height - old_height)

    def _observe_object_pair(self, a, b, pa, pb) -> None:
        history = self._pose_history[(a, b)]
        pose_a = [float(v) for v in pa[:7]]
        pose_b = [float(v) for v in pb[:7]]
        history.append((pose_a, pose_b))
        prev_a = prev_b = None
        if len(history) == self.horizon + 1:
            prev_a, prev_b = history[0]
        self._record_pose_pair({
            "key_a": a, "key_b": b,
            "pose_a": pose_a, "pose_b": pose_b,
            "prev_pose_a": prev_a, "prev_pose_b": prev_b,
        })

    def merge(self, other: Dict[str, float]) -> None:
        for key, value in (other or {}).items():
            self.maxes[key] = max(self.maxes[key], float(value))

    def as_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.maxes.items()}

    def pose_samples(self) -> List[Dict[str, Any]]:
        """Raw object-pair poses the miner reprojects through mined anchors."""
        return list(self.pose_pairs)

    def reservoir(self) -> Dict[str, Any]:
        """The raw sample per statistic, as float32.

        The samples travel rather than precomputed quantiles: quantiles cannot
        be merged across shards -- averaging or overwriting them both give a
        distribution no worker observed -- while reservoirs concatenate.
        """
        import numpy as np

        return {k: np.asarray(v, dtype=np.float32)
                for k, v in self.samples.items() if v}
