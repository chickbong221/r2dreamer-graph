"""Which vertices and facts a kitchen frame carries, and in which orientation.

Relations, labels and temporal applicability are the repository's
(:mod:`scenegraph.core.relation_rules`); this module only says which of them
exist between which kitchen entities. Orientation follows the same rules the
simulator builder and the schedule compiler use:

* end-effector facts are stored ``ee -> object`` and never sorted;
* object-object facts are stored in ``pair_sort_key`` order -- whitelist key,
  then node id -- so a pair means the same thing in every frame and every
  episode.

When an annotation names a pair the other way round, the label that carries
direction is mirrored with the swap: ``height-offset`` exchanges ``above`` and
``below`` (and its temporal change flips sign), ``support`` and ``contain``
exchange ``src-holds`` and ``dst-holds``. Every other label is symmetric.

All object-object facts use the sorted order, compatibility families included.
The simulator builder emits ``support-``/``contain-compatibility`` in role order
(supporter or container first); for the tasks it was mined on that coincides
with key order, and the schedule compiler assumes key order. Here the pot sorts
after the banana and the lid, so role order would disagree with the compiler;
key order keeps every consumer of these graphs reading the same row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from scenegraph.adapters.graph_vocab import EE_TOKEN, PAD_TOKEN
from scenegraph.core.relation_rules import (
    ABS_LABELS,
    CHANGE_LABELS,
    DST_HOLDS,
    RELATION_TYPES,
    SRC_HOLDS,
    TEMPORAL_RELATIONS,
)
from scenegraph.core.schedule import ANTISYMMETRIC, DIRECTIONAL

EE_ID = "ee"
EE_NODE_ID = "ee"
ENTITY_TYPES = ("ee", "object")

EE_OBJECT = "ee-object"
OBJECT_OBJECT = "object-object"

# Which endpoint kinds each relation joins, as relation_rules emits them.
# ``reached`` needs a declared site; the kitchen declares none, so it stays in
# the vocabulary and out of the facts.
RELATION_SCOPES: Dict[str, frozenset] = {
    "contact": frozenset({EE_OBJECT, OBJECT_OBJECT}),
    "grasp": frozenset({EE_OBJECT}),
    "support": frozenset({OBJECT_OBJECT}),
    "contain": frozenset({OBJECT_OBJECT}),
    "planar-distance": frozenset({EE_OBJECT, OBJECT_OBJECT}),
    "height-offset": frozenset({EE_OBJECT, OBJECT_OBJECT}),
    "grasp-compatibility": frozenset({EE_OBJECT}),
    "contact-compatibility": frozenset({EE_OBJECT, OBJECT_OBJECT}),
    "support-compatibility": frozenset({OBJECT_OBJECT}),
    "contain-compatibility": frozenset({OBJECT_OBJECT}),
    "reached": frozenset(),
}

HEIGHT_MIRROR: Dict[str, str] = {
    "far-below": "far-above", "below": "above", "level": "level",
    "above": "below", "far-above": "far-below",
}
DIRECTION_MIRROR: Dict[str, str] = {SRC_HOLDS: DST_HOLDS, DST_HOLDS: SRC_HOLDS}
# v_t - v_{t-K} changes sign with the endpoints of an antisymmetric value.
CHANGE_MIRROR: Dict[str, str] = {
    "decrease-fast": "increase-fast", "decrease-slow": "increase-slow",
    "stable": "stable",
    "increase-slow": "decrease-slow", "increase-fast": "decrease-fast",
}


class SpecError(ValueError):
    """A graph configuration that cannot be packed or read consistently."""


@dataclass(frozen=True)
class Entity:
    id: str
    key: str
    name: str
    type: str

    @property
    def node_id(self) -> str:
        # The end effector's node id is the one the viewer palette pins and
        # the schedule compiler resolves by type; objects use their key.
        return EE_NODE_ID if self.type == "ee" else self.key


@dataclass(frozen=True)
class Fact:
    """One stored fact slot, already in canonical orientation."""

    src: str
    dst: str
    relation: str

    @property
    def temporal(self) -> bool:
        return self.relation in TEMPORAL_RELATIONS

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.src, self.dst, self.relation)

    def label(self) -> str:
        return f"{self.relation}({self.src}, {self.dst})"


def mirror_absolute(relation: str, label: str) -> str:
    if relation in ANTISYMMETRIC:
        if label not in HEIGHT_MIRROR:
            raise SpecError(f"{relation} has no mirror for label {label!r}")
        return HEIGHT_MIRROR[label]
    if relation in DIRECTIONAL:
        return DIRECTION_MIRROR.get(label, label)
    return label


def mirror_temporal(relation: str, label: Optional[str]) -> Optional[str]:
    if label is None or relation not in ANTISYMMETRIC:
        return label
    if label not in CHANGE_MIRROR:
        raise SpecError(f"{relation} has no mirror for change label {label!r}")
    return CHANGE_MIRROR[label]


class GraphSpec:
    """The validated kitchen graph configuration."""

    def __init__(
        self,
        *,
        version: str,
        entities: Sequence[Entity],
        facts: Sequence[Fact],
        targets: Sequence[str],
        points: Mapping[str, Sequence[str]],
        temporal_window: int,
        cameras: Sequence[str],
        n_max: int,
        e_max: int,
        centroid_origin: Sequence[float] = (0.0, 0.0, 0.0),
        centroid_scale: Any = 1.0,
    ):
        self.version = str(version)
        self.entities: Tuple[Entity, ...] = tuple(entities)
        self.facts: Tuple[Fact, ...] = tuple(facts)
        self.targets: Tuple[str, ...] = tuple(targets)
        self.points: Dict[str, Tuple[str, ...]] = {k: tuple(v) for k, v in points.items()}
        self.temporal_window = int(temporal_window)
        self.cameras: Tuple[str, ...] = tuple(cameras)
        self.n_max = int(n_max)
        self.e_max = int(e_max)
        self.centroid_origin = [float(v) for v in centroid_origin]
        self.centroid_scale = centroid_scale
        self._by_id = {e.id: e for e in self.entities}
        self._fact_index = {f.key: i for i, f in enumerate(self.facts)}
        self._validate()

    # ------------------------------------------------------------------ build
    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "GraphSpec":
        entities = [
            Entity(id=str(e["id"]), key=str(e["key"]), name=str(e.get("name", e["id"])),
                   type=str(e.get("type", "object")))
            for e in cfg["entities"]
        ]
        by_id = {e.id: e for e in entities}
        facts: List[Fact] = []
        for entry in cfg["facts"]:
            pair = list(entry["pair"])
            if len(pair) != 2:
                raise SpecError(f"fact pair {pair} must name two entities")
            for relation in entry["relations"]:
                a, b = str(pair[0]), str(pair[1])
                if a not in by_id or b not in by_id:
                    raise SpecError(f"fact pair {pair} names an unknown entity")
                src, dst, _ = _canonical_order(by_id, a, b)
                facts.append(Fact(src=src, dst=dst, relation=str(relation)))
        return cls(
            version=cfg.get("version", "unversioned"),
            entities=entities,
            facts=facts,
            targets=[str(t) for t in cfg.get("targets", ())],
            points={str(k): [str(p) for p in v] for k, v in (cfg.get("points") or {}).items()},
            temporal_window=int(cfg["temporal_window_frames"]),
            cameras=[str(c) for c in cfg["cameras"]],
            n_max=int(cfg["n_max"]),
            e_max=int(cfg["e_max"]),
            centroid_origin=cfg.get("centroid_origin", (0.0, 0.0, 0.0)),
            centroid_scale=cfg.get("centroid_scale", 1.0),
        )

    def _validate(self) -> None:
        problems: List[str] = []
        ids = [e.id for e in self.entities]
        keys = [e.key for e in self.entities]
        if len(set(ids)) != len(ids):
            problems.append(f"duplicate entity ids in {ids}")
        if len(set(keys)) != len(keys):
            problems.append(f"duplicate entity keys in {keys}")
        ees = [e for e in self.entities if e.type == "ee"]
        if len(ees) != 1 or ees[0].id != EE_ID or ees[0].key != EE_TOKEN:
            problems.append(
                f"exactly one end effector with id {EE_ID!r} and key {EE_TOKEN!r} is "
                f"required; got {[(e.id, e.key) for e in ees]}"
            )
        for entity in self.entities:
            if entity.type not in ENTITY_TYPES:
                problems.append(f"entity {entity.id!r} has type {entity.type!r}")
            if entity.type == "object" and entity.key in (EE_TOKEN, PAD_TOKEN):
                problems.append(f"object {entity.id!r} uses the reserved key {entity.key!r}")
        if len(self.entities) > self.n_max:
            problems.append(f"{len(self.entities)} entities exceed n_max={self.n_max}")
        if len(self.facts) > self.e_max:
            problems.append(f"{len(self.facts)} facts exceed e_max={self.e_max}")
        seen = set()
        for fact in self.facts:
            if fact.relation not in RELATION_TYPES:
                problems.append(f"unknown relation {fact.relation!r}")
                continue
            if fact.src == fact.dst:
                problems.append(f"{fact.label()} relates an entity to itself")
                continue
            scope = self.scope(fact.src, fact.dst)
            if scope not in RELATION_SCOPES[fact.relation]:
                problems.append(f"{fact.label()} is not a {scope} relation")
            if fact.key in seen:
                problems.append(f"{fact.label()} is listed twice")
            seen.add(fact.key)
        for target in self.targets:
            entity = self._by_id.get(target)
            if entity is None or entity.type != "object":
                problems.append(f"target {target!r} is not an object entity")
        for owner in self.points:
            if owner not in self._by_id:
                problems.append(f"points declared for unknown entity {owner!r}")
        if not self.cameras:
            problems.append("no cameras")
        if self.temporal_window < 1:
            problems.append("temporal_window_frames must be at least 1")
        if problems:
            raise SpecError("invalid graph configuration:\n  " + "\n  ".join(problems))

    # --------------------------------------------------------------- queries
    def entity(self, entity_id: str) -> Entity:
        return self._by_id[entity_id]

    def has_entity(self, entity_id: str) -> bool:
        return entity_id in self._by_id

    @property
    def entity_ids(self) -> Tuple[str, ...]:
        return tuple(e.id for e in self.entities)

    @property
    def object_ids(self) -> Tuple[str, ...]:
        return tuple(e.id for e in self.entities if e.type == "object")

    def entity_row_index(self, entity_id: str) -> int:
        """Position in :attr:`entities` (not the packed row)."""
        return self.entity_ids.index(entity_id)

    def scope(self, a: str, b: str) -> str:
        kinds = {self._by_id[a].type, self._by_id[b].type}
        return EE_OBJECT if "ee" in kinds else OBJECT_OBJECT

    def canonical_order(self, a: str, b: str) -> Tuple[str, str, bool]:
        return _canonical_order(self._by_id, a, b)

    def canonicalize(self, relation: str, a: str, b: str, label: Optional[str] = None,
                     temp_label: Optional[str] = None
                     ) -> Tuple[str, str, Optional[str], Optional[str]]:
        """Stored orientation for a fact, with direction-carrying labels mirrored."""
        src, dst, swapped = self.canonical_order(a, b)
        if swapped:
            if label is not None:
                label = mirror_absolute(relation, label)
            temp_label = mirror_temporal(relation, temp_label)
        return src, dst, label, temp_label

    def fact_index(self, src: str, dst: str, relation: str) -> Optional[int]:
        return self._fact_index.get((src, dst, relation))

    def facts_for_pair(self, a: str, b: str) -> List[Fact]:
        src, dst, _ = self.canonical_order(a, b)
        return [f for f in self.facts if f.src == src and f.dst == dst]

    @staticmethod
    def legal_labels(relation: str) -> List[str]:
        return list(ABS_LABELS[relation])

    @staticmethod
    def change_labels() -> List[str]:
        return list(CHANGE_LABELS)

    @property
    def relations_in_use(self) -> Tuple[str, ...]:
        seen: List[str] = []
        for fact in self.facts:
            if fact.relation not in seen:
                seen.append(fact.relation)
        return tuple(seen)

    def identity(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "entities": [[e.id, e.key, e.type] for e in self.entities],
            "facts": [list(f.key) for f in self.facts],
            "targets": list(self.targets),
            "points": {k: list(v) for k, v in sorted(self.points.items())},
            "temporal_window": self.temporal_window,
            "cameras": list(self.cameras),
            "n_max": self.n_max,
            "e_max": self.e_max,
            "centroid_origin": self.centroid_origin,
            "centroid_scale": self.centroid_scale,
        }


def _sort_key(entity: Entity) -> Tuple[str, str]:
    # relation_rules.pair_sort_key: (whitelist key, node id).
    return (entity.key, entity.node_id)


def _canonical_order(by_id: Mapping[str, Entity], a: str, b: str) -> Tuple[str, str, bool]:
    ea, eb = by_id[a], by_id[b]
    if ea.type == "ee":
        return a, b, False
    if eb.type == "ee":
        return b, a, True
    if _sort_key(eb) < _sort_key(ea):
        return b, a, True
    return a, b, False


def load_spec(cfg: Optional[Mapping[str, Any]] = None) -> GraphSpec:
    if cfg is None:
        from ..common import load_config
        cfg = load_config("graph")
    return GraphSpec.from_config(cfg)


def iter_fact_labels(spec: GraphSpec, labels: Iterable[Optional[str]]) -> Iterable[Tuple[Fact, str]]:
    for fact, label in zip(spec.facts, labels):
        if label is not None:
            yield fact, label
