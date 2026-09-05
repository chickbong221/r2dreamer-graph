"""Whitelist admission, appearance retention, and the episode vertex index.

Pipeline per frame:

    apply_whitelist(candidates)   # hard eligibility gate
    -> merge_retained(...)        # re-inject every node seen this episode
    -> EntityRegistry.assign(...) # stable index, explicit overflow policy

No scoring or secondary contact-based admission path exists. Ordinary tasks
raise on overflow. MS-HAB Pick reserves its task nodes and retains a bounded
FIFO of context nodes.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

from .persistence import _snapshot
from .schema import Node
from .whitelist import Whitelist, match_key

class EntityRegistry:
    """Bounded vertex index. One instance per episode.

    ``ee`` always holds index 0. The default retains indices until reset and
    raises on overflow. Explicit FIFO mode can replace unprotected residents,
    reusing their rows without disturbing the surviving nodes.
    """

    def __init__(self, n_max: int):
        self.n_max = int(n_max)
        self._index: Dict[str, int] = {}
        self._first_seen: Dict[str, int] = {}
        self._ever_seen: Set[str] = set()
        self._seen_clock = 0
        self._next = 1  # 0 is reserved for the end effector
        self.evicted_ids: List[str] = []
        self.rejected_ids: List[str] = []

    def reset_episode(self) -> None:
        self._index.clear()
        self._first_seen.clear()
        self._ever_seen.clear()
        self._seen_clock = 0
        self._next = 1
        self.evicted_ids.clear()
        self.rejected_ids.clear()

    def __len__(self) -> int:
        return len(self._index)

    def index_of(self, entity_id: str) -> Optional[int]:
        return self._index.get(entity_id)

    @property
    def episode_entities(self) -> int:
        """Distinct object instances presented to ``assign`` this episode.

        With FIFO this can exceed current occupancy; it is not a row count.
        """
        return len(self._ever_seen)

    def assign(self, nodes: Dict[str, Node], *, overflow: str = "error",
               protected: Iterable[str] = ()) -> Dict[str, Node]:
        """Index objects with strict retention or protected FIFO."""
        self.evicted_ids.clear()
        self.rejected_ids.clear()
        if overflow == "fifo":
            return self._assign_fifo(nodes, protected)
        if overflow != "error":
            raise ValueError(f"unknown node overflow policy {overflow!r}")
        admitted: Dict[str, Node] = {}
        pending: List[str] = []
        for ent_id, node in nodes.items():
            if node.node_type == "ee":
                node.index = 0
                admitted[ent_id] = node
                continue
            self._ever_seen.add(ent_id)
            idx = self._index.get(ent_id)
            if idx is None:
                pending.append(ent_id)
                continue
            node.index = idx
            admitted[ent_id] = node

        for ent_id in sorted(pending):
            if ent_id not in self._first_seen:
                self._first_seen[ent_id] = self._seen_clock
                self._seen_clock += 1

        capacity = max(0, self.n_max - 1)  # excluding the end effector
        for ent_id in sorted(pending, key=lambda k: self._first_seen[k]):
            node = nodes[ent_id]
            if len(self._index) >= capacity:
                raise RuntimeError(
                    f"entity registry full: {ent_id!r} needs a row but all "
                    f"{capacity} object slots are held "
                    f"({sorted(self._index)}). Retention never evicts."
                )
            idx = self._next
            self._next += 1
            self._index[ent_id] = idx
            node.index = idx
            admitted[ent_id] = node
        return admitted

    def _assign_fifo(self, nodes: Dict[str, Node],
                     protected: Iterable[str]) -> Dict[str, Node]:
        """Protect task instances and keep the newest context arrivals.

        First-seen order survives eviction. A discarded instance still visible
        on the next frame must not masquerade as a new arrival and rotate all
        the slots. A new episode clears that order along with the registry.
        """
        keep = {key for key in protected if key and key != "ee"}
        objects = {key for key, node in nodes.items() if node.node_type != "ee"}
        missing = keep - objects
        if missing:
            raise RuntimeError(f"protected graph nodes are missing: {sorted(missing)}")
        capacity = max(0, self.n_max - 1)
        if len(keep) > capacity:
            raise RuntimeError(
                f"n_max={self.n_max} cannot hold the end effector and "
                f"protected nodes {sorted(keep)}")
        self._ever_seen.update(objects)
        for key in sorted(objects):
            if key not in self._first_seen:
                self._first_seen[key] = self._seen_clock
                self._seen_clock += 1
        context = sorted(objects - keep, key=self._first_seen.get, reverse=True)
        chosen = keep | set(context[:capacity - len(keep)])
        self.evicted_ids = sorted(set(self._index) - chosen)
        self.rejected_ids = sorted(objects - chosen)
        for key in self.evicted_ids:
            del self._index[key]
        free = [row for row in range(1, self.n_max)
                if row not in self._index.values()]
        pending = sorted(chosen - set(self._index),
                         key=lambda key: (key not in keep, self._first_seen[key]))
        for key, row in zip(pending, free):
            self._index[key] = row
        admitted = {}
        for key, node in nodes.items():
            if node.node_type == "ee":
                node.index = 0
            elif key in chosen:
                node.index = self._index[key]
            else:
                continue
            admitted[key] = node
        return admitted

class NodeSelector:
    """Stateful selector. One instance per episode.

    Holds the active subtask's ``Whitelist``. GraphBuilder may update it when
    MS-HAB advances to another subtask.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        sel = cfg["selection"]
        self.n_max = int(sel["n_max"])

        # Active whitelist; set by GraphBuilder before selection. None
        # means "no asset loaded yet" and triggers a fail-loud error in the
        # selection path -- the hard gate explicitly forbids a silent
        # "admit everything" fallback.
        self._whitelist: Optional[Whitelist] = None

        self._history: Dict[str, Node] = {}
        self._last_seen: Dict[str, int] = {}

    # ---------------------------------------------------------------- reset
    def reset_episode(self) -> None:
        self._history.clear()
        self._last_seen.clear()

    # ---------------------------------------------------------------- persistence
    def merge_retained(
        self, fresh: Dict[str, Node], frame: int
    ) -> Dict[str, Node]:
        """Re-inject every node admitted earlier this episode and absent now.

        Retention is unconditional: once a camera has seen a whitelisted
        object, it stays a vertex until reset. A re-injected node carries no
        pixels, so its box is zero; its pose is refreshed from the simulator
        before relations are built, not taken from this snapshot.
        """
        merged = dict(fresh)
        for ent_id, snap in self._history.items():
            if ent_id in merged:
                continue
            last = self._last_seen.get(ent_id)
            if last is None:
                continue
            merged[ent_id] = Node(
                node_id=snap.node_id,
                node_type=snap.node_type,
                name=snap.name,
                visible=False,
                segmentation_ids=[],
                pixel_area=0,
                pose_world=list(snap.pose_world) if snap.pose_world else None,
                index=snap.index,
                source=snap.source,
                attributes=dict(snap.attributes),
            )
        return merged

    # ---------------------------------------------------------------- whitelist
    def set_whitelist(self, whitelist: Whitelist) -> None:
        """Bind the active subtask's whitelist."""
        self._whitelist = whitelist

    @property
    def whitelist(self) -> Optional[Whitelist]:
        return self._whitelist

    def apply_whitelist(self, nodes: Dict[str, Node]) -> Dict[str, Node]:
        """Hard gate: keep ``ee`` plus every node whose ``match_key`` is in the
        active whitelist. Everything else is dropped before indexing.

        Instances are not filtered. A per-target whitelist marks every member
        ``interacted``, not just the target, so an instance filter would delete
        the supporters and the scene background along with the sibling copies.
        Same-category siblings stay in the graph and ``graph_node_target``
        names the goal among them.

        Raises if no whitelist is bound.
        """
        if self._whitelist is None:
            raise RuntimeError(
                "NodeSelector.apply_whitelist called with no whitelist bound. "
                "GraphBuilder must call set_whitelist() during episode reset."
            )
        wl = self._whitelist
        kept: Dict[str, Node] = {}
        for nid, n in nodes.items():
            if n.node_type == "ee":
                kept[nid] = n
                continue
            key = match_key(n)
            if not wl.contains(key):
                continue
            roles = wl.roles(key)
            n.attributes["whitelist_key"] = key
            n.attributes["whitelist_roles"] = sorted(roles)
            n.attributes["interaction_types"] = sorted(wl.types(key))
            kept[nid] = n
        return kept

    # ---------------------------------------------------------------- commit
    def commit(self, nodes: Dict[str, Node], frame: int) -> None:
        """Snapshot every visible whitelisted object for history bookkeeping."""
        for ent_id, n in nodes.items():
            if n is None or n.node_type != "object":
                continue
            if n.visible:
                self._last_seen[ent_id] = frame
                self._history[ent_id] = _snapshot(n)

    def evict(self, node_ids: Iterable[str]) -> None:
        """Stop history from re-injecting FIFO victims on the next frame."""
        for node_id in node_ids:
            self._history.pop(node_id, None)
            self._last_seen.pop(node_id, None)


