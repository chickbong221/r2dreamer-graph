"""The entity vocabulary and, per task, which entities and facts a frame carries.

Relations, labels and temporal applicability are the repository's
(:mod:`scenegraph.core.relation_rules`). Orientation follows the simulator
builder: end-effector facts are stored ``ee -> object``; object-object facts in
``(whitelist key, node id)`` order. A pair named the other way round has its
direction-carrying label mirrored: ``height-offset`` swaps ``above`` and
``below`` (and its temporal change flips sign), ``support`` and ``contain``
swap ``src-holds`` and ``dst-holds``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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

# ``reached`` needs a declared site; none is declared, so it stays out of the facts.
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
    description: str = ""
    reference: str = ""

    @property
    def node_id(self) -> str:
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


def _sort_key(entity: Entity) -> Tuple[str, str]:
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


class GraphSpec:
    """One task: its entities (in vocabulary order), facts and target objects."""

    def __init__(
        self,
        *,
        version: str,
        task: str,
        instruction: str,
        target_rule: str,
        entities: Sequence[Entity],
        facts: Sequence[Fact],
        targets: Sequence[str],
        temporal_window: int,
        cameras: Sequence[str],
        n_max: int,
        e_max: int,
    ):
        self.version = str(version)
        self.task = str(task)
        self.instruction = str(instruction)
        self.target_rule = str(target_rule)
        self.entities: Tuple[Entity, ...] = tuple(entities)
        self.facts: Tuple[Fact, ...] = tuple(facts)
        self.targets: Tuple[str, ...] = tuple(targets)
        self.temporal_window = int(temporal_window)
        self.cameras: Tuple[str, ...] = tuple(cameras)
        self.n_max = int(n_max)
        self.e_max = int(e_max)
        self._by_id = {e.id: e for e in self.entities}
        self._fact_index = {f.key: i for i, f in enumerate(self.facts)}
        self._validate()

    def _validate(self) -> None:
        problems: List[str] = []
        ees = [e for e in self.entities if e.type == "ee"]
        if len(ees) != 1 or ees[0].id != EE_ID or ees[0].key != EE_TOKEN:
            problems.append(
                f"exactly one end effector with id {EE_ID!r} and key {EE_TOKEN!r} is "
                f"required; got {[(e.id, e.key) for e in ees]}"
            )
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
        if not self.targets:
            problems.append("no targets")
        for target in self.targets:
            entity = self._by_id.get(target)
            if entity is None or entity.type != "object":
                problems.append(f"target {target!r} is not an object entity of this task")
        if not self.cameras:
            problems.append("no cameras")
        if self.temporal_window < 1:
            problems.append("temporal_window_frames must be at least 1")
        if problems:
            raise SpecError(f"invalid graph configuration for task {self.task!r}:\n  " + "\n  ".join(problems))

    def entity(self, entity_id: str) -> Entity:
        return self._by_id[entity_id]

    def has_entity(self, entity_id: str) -> bool:
        return entity_id in self._by_id

    @property
    def entity_ids(self) -> Tuple[str, ...]:
        return tuple(e.id for e in self.entities)

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
            "task": self.task,
            "entities": [[e.id, e.key, e.type] for e in self.entities],
            "facts": [list(f.key) for f in self.facts],
            "targets": list(self.targets),
            "temporal_window": self.temporal_window,
            "cameras": list(self.cameras),
            "n_max": self.n_max,
            "e_max": self.e_max,
        }


class GraphConfig:
    """The whole dataset: one entity vocabulary, one :class:`GraphSpec` per task."""

    def __init__(self, cfg: Mapping[str, Any]):
        self.version = str(cfg.get("version", "unversioned"))
        self.temporal_window = int(cfg["temporal_window_frames"])
        self.n_max = int(cfg["n_max"])
        self.e_max = int(cfg["e_max"])
        self.cameras: Tuple[str, ...] = tuple(str(c) for c in cfg["cameras"])
        self.entities: Tuple[Entity, ...] = tuple(
            Entity(id=str(e["id"]), key=str(e["key"]), name=str(e.get("name", e["id"])),
                   type=str(e.get("type", "object")), description=str(e.get("description", "")),
                   reference=str(e.get("reference", "")))
            for e in cfg["entities"]
        )
        self._validate_entities()
        by_id = {e.id: e for e in self.entities}
        self.tasks: Dict[str, GraphSpec] = {}
        self._task_of_text: Dict[str, str] = {}
        for key, task in (cfg.get("tasks") or {}).items():
            wanted = [str(i) for i in task["entities"]]
            unknown = [i for i in wanted if i not in by_id]
            if unknown:
                raise SpecError(f"task {key!r} names unknown entities {unknown}")
            entities = [e for e in self.entities if e.id in wanted]
            local = {e.id: e for e in entities}
            facts: List[Fact] = []
            for entry in task["facts"]:
                pair = [str(p) for p in entry["pair"]]
                if len(pair) != 2 or any(p not in local for p in pair):
                    raise SpecError(f"task {key!r}: fact pair {pair} must name two of its entities")
                src, dst, _ = _canonical_order(local, pair[0], pair[1])
                facts.extend(Fact(src=src, dst=dst, relation=str(r)) for r in entry["relations"])
            self.tasks[str(key)] = GraphSpec(
                version=self.version, task=str(key), instruction=str(task.get("instruction", "")),
                target_rule=str(task.get("target_rule", "")), entities=entities, facts=facts,
                targets=[str(t) for t in task["targets"]], temporal_window=self.temporal_window,
                cameras=self.cameras, n_max=self.n_max, e_max=self.e_max,
            )
            for text in task.get("dataset_tasks") or ():
                text = str(text).strip()
                if text in self._task_of_text:
                    raise SpecError(f"dataset task {text!r} is mapped to both "
                                    f"{self._task_of_text[text]!r} and {key!r}")
                self._task_of_text[text] = str(key)
        if not self.tasks:
            raise SpecError("graph configuration defines no tasks")

    def _validate_entities(self) -> None:
        problems: List[str] = []
        ids = [e.id for e in self.entities]
        keys = [e.key for e in self.entities]
        if len(set(ids)) != len(ids):
            problems.append(f"duplicate entity ids in {ids}")
        if len(set(keys)) != len(keys):
            problems.append(f"duplicate entity keys in {keys}")
        for entity in self.entities:
            if entity.type not in ENTITY_TYPES:
                problems.append(f"entity {entity.id!r} has type {entity.type!r}")
            if entity.type == "object" and entity.key in (EE_TOKEN, PAD_TOKEN):
                problems.append(f"object {entity.id!r} uses the reserved key {entity.key!r}")
        if problems:
            raise SpecError("invalid graph configuration:\n  " + "\n  ".join(problems))

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "GraphConfig":
        return cls(cfg)

    def task_for(self, text: str) -> str:
        key = self._task_of_text.get(str(text).strip())
        if key is None:
            raise KeyError(f"dataset task {text!r} is not mapped to a task in graph.yaml "
                           f"(known: {sorted(self._task_of_text)})")
        return key

    def spec(self, task: str) -> GraphSpec:
        return self.tasks[task]

    def identity(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "entities": [[e.id, e.key, e.type] for e in self.entities],
            "cameras": list(self.cameras),
            "n_max": self.n_max,
            "e_max": self.e_max,
            "temporal_window": self.temporal_window,
            "tasks": {key: spec.identity() for key, spec in sorted(self.tasks.items())},
        }


def load_graph_config(cfg: Optional[Mapping[str, Any]] = None) -> GraphConfig:
    if cfg is None:
        from ..common import load_config
        cfg = load_config("graph")
    return GraphConfig.from_config(cfg)
