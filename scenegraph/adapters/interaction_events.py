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
        sink.end_episode()
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
        # Buckets rejected because discovery had already frozen.
        self.late: Dict[BucketKey, int] = defaultdict(int)

    def add(self, event: InteractionEvent) -> bool:
        bucket = event.bucket
        if self.frozen is not None and bucket not in self.frozen:
            self.late[bucket] += 1
            return False
        self.seen_counts[bucket] += 1
        if len(self.samples[bucket]) >= self.target:
            return False
        self.samples[bucket].append(event)
        return True

    def end_episode(self) -> None:
        self.episodes += 1

    def freeze(self) -> None:
        """Stop admitting new buckets. Existing ones keep filling."""
        self.frozen = set(self.samples) | set(self.seen_counts)

    def buckets(self) -> List[BucketKey]:
        return sorted(set(self.samples) | set(self.seen_counts))

    def complete(self) -> List[BucketKey]:
        return [b for b in self.buckets()
                if len(self.samples[b]) >= self.target]

    def incomplete(self) -> List[BucketKey]:
        return [b for b in self.buckets()
                if len(self.samples[b]) < self.target]

    def is_done(self) -> bool:
        buckets = self.buckets()
        return bool(buckets) and not self.incomplete()

    def report(self) -> str:
        lines = [f"episodes committed: {self.episodes}",
                 f"target per bucket: {self.target}"]
        for b in self.buckets():
            have, seen = len(self.samples[b]), self.seen_counts[b]
            mark = "ok " if have >= self.target else "SHORT"
            lines.append(f"  {mark} {have:4d}/{self.target}  (seen {seen})  {b}")
        for b, n in sorted(self.late.items()):
            lines.append(f"  LATE  discovered after freeze, {n} events: {b}")
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
