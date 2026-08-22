"""Whitelist admission, appearance retention, and the episode vertex index.

Pipeline per frame:

    apply_whitelist(candidates)   # hard eligibility gate
    -> merge_retained(...)        # re-inject every node seen this episode
    -> EntityRegistry.assign(...) # stable index, diversity-aware overflow

No scoring or secondary contact-based admission path exists. Capacity is a hard
ceiling: a scene that needs more rows than ``n_max`` raises rather than choosing
a vertex to lose.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

from .persistence import _snapshot
from .schema import Node
from .whitelist import Whitelist, match_key

class EntityRegistry:
    """Bounded vertex index. One instance per episode.

    ``ee`` always holds index 0. Object indices are handed out on first sight
    and held until reset -- nothing displaces a resident, and no slot is
    reclaimed inside an episode. A scene needing more rows than ``n_max`` is a
    configuration error and raises.
    """

    def __init__(self, n_max: int):
        self.n_max = int(n_max)
        self._index: Dict[str, int] = {}
        self._first_seen: Dict[str, int] = {}
        self._ever_seen: Set[str] = set()
        self._seen_clock = 0
        self._next = 1  # 0 is reserved for the end effector

    def reset_episode(self) -> None:
        self._index.clear()
        self._first_seen.clear()
        self._ever_seen.clear()
        self._seen_clock = 0
        self._next = 1

    def __len__(self) -> int:
        return len(self._index)

    def index_of(self, entity_id: str) -> Optional[int]:
        return self._index.get(entity_id)

    @property
    def episode_entities(self) -> int:
        """Distinct object instances presented to ``assign`` this episode.

        Under retention this equals live occupancy: every instance admitted
        this episode still holds its row.
        """
        return len(self._ever_seen)

    def assign(self, nodes: Dict[str, Node]) -> Dict[str, Node]:
        """Index every object. Overflow raises; nothing is ever displaced.

        Under unconditional retention a slot is held for the whole episode, so
        there is no resident whose eviction would be correct -- dropping one
        deletes facts the progress target reads. The graph builder checks
        capacity first and raises with scene detail; this is the backstop for
        callers that do not.
        """
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


