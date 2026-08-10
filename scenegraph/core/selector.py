"""Whitelist admission, appearance retention, and the append-only vertex index.

Pipeline per frame:

    apply_whitelist(candidates)   # hard eligibility gate
    -> merge_persistent(...)      # re-inject nodes that left the view
    -> EntityRegistry.assign(...) # append-only vertex index

No scoring or secondary contact-based admission path exists. Whitelist role
priority is consulted only when the registry is at capacity.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .persistence import _snapshot
from .schema import Node
from .whitelist import Whitelist, match_key


def _role_rank(node: Node) -> int:
    roles = set(node.attributes.get("whitelist_roles") or [])
    if "interacted" in roles:
        return 0
    if "support" in roles:
        return 1
    return 2


class EntityRegistry:
    """Append-only vertex index. One instance per episode.

    ``ee`` always holds index 0. Object indices are handed out on first sight
    and never reused or reordered while the episode runs, so a node that leaves
    and later re-enters the view keeps its position. Appearance retention is
    separate and lives in the adapter-level cache.
    """

    def __init__(self, n_max: int):
        self.n_max = int(n_max)
        self._index: Dict[str, int] = {}
        self._free: List[int] = []
        self._next = 1  # 0 is reserved for the end effector
        self.overflow_drops = 0

    def reset_episode(self) -> None:
        self._index.clear()
        self._free.clear()
        self._next = 1
        self.overflow_drops = 0

    def __len__(self) -> int:
        return len(self._index)

    def index_of(self, entity_id: str) -> Optional[int]:
        return self._index.get(entity_id)

    def assign(self, nodes: Dict[str, Node]) -> Dict[str, Node]:
        """Index every object node, dropping the lowest-priority overflow.

        Returns the admitted subset. Capacity is only reachable when the active
        whitelist admits more instances than ``n_max``; the candidate replaces a
        resident only when it outranks the worst one, and the displaced entry
        releases its index.
        """
        admitted: Dict[str, Node] = {}
        pending: List[str] = []
        for ent_id, node in nodes.items():
            if node.node_type == "ee":
                node.index = 0
                admitted[ent_id] = node
                continue
            idx = self._index.get(ent_id)
            if idx is None:
                pending.append(ent_id)
                continue
            node.index = idx
            admitted[ent_id] = node

        capacity = self.n_max - 1  # excluding the end effector
        for ent_id in sorted(pending, key=lambda k: (_role_rank(nodes[k]), k)):
            node = nodes[ent_id]
            if len(self._index) >= capacity:
                worst = self._worst_resident(admitted)
                if worst is None or _role_rank(nodes[ent_id]) >= _role_rank(
                    admitted[worst]
                ):
                    self.overflow_drops += 1
                    continue
                self.release(worst)
                admitted.pop(worst, None)
            idx = self._free.pop(0) if self._free else self._next
            if idx == self._next:
                self._next += 1
            self._index[ent_id] = idx
            node.index = idx
            admitted[ent_id] = node
        return admitted

    def _worst_resident(self, admitted: Dict[str, Node]) -> Optional[str]:
        candidates = [k for k, n in admitted.items() if n.node_type != "ee"]
        if not candidates:
            return None
        return max(candidates, key=lambda k: (_role_rank(admitted[k]), k))

    def release(self, entity_id: str) -> None:
        idx = self._index.pop(entity_id, None)
        if idx is not None:
            self._free.append(idx)
            self._free.sort()


class NodeSelector:
    """Stateful selector. One instance per episode.

    Holds the active subtask's ``Whitelist``. GraphBuilder may update it when
    MS-HAB advances to another subtask.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        sel = cfg["selection"]
        self.n_max = int(sel["n_max"])
        self.k_persist = int(sel["k_persist"])

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
    def merge_persistent(
        self, fresh: Dict[str, Node], frame: int
    ) -> Dict[str, Node]:
        """Re-inject nodes seen within ``k_persist`` frames that are missing
        from the current frame's visible set.

        ``k_persist == 0`` disables persistence entirely; ``k_persist < 0``
        means "never evict" -- the node stays for the whole episode once seen,
        which is the configuration the scene graph is defined against.
        """
        if self.k_persist == 0:
            return fresh
        merged = dict(fresh)
        for ent_id, snap in self._history.items():
            if ent_id in merged:
                continue
            last = self._last_seen.get(ent_id)
            if last is None:
                continue
            if self.k_persist >= 0 and (frame - last) > self.k_persist:
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
                steps_since_seen=frame - last,
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

    def evict(self, evicted_ids: Iterable[str]) -> None:
        for ent_id in evicted_ids:
            self._history.pop(ent_id, None)
            self._last_seen.pop(ent_id, None)

    def evict_expired(self, frame: int) -> List[str]:
        """Drop history entries whose age exceeds ``k_persist`` frames.

        ``k_persist < 0`` means "never evict within an episode"; ``k_persist
        == 0`` disables persistence and nothing is retained to evict.
        """
        if self.k_persist <= 0:
            return []
        expired: List[str] = []
        for ent_id, last in list(self._last_seen.items()):
            if (frame - last) > self.k_persist:
                expired.append(ent_id)
        self.evict(expired)
        return expired
