"""Compile a task phase schedule against the mined assets.

A schedule names semantic roles and relation clauses. The runtime speaks
relation ids, absolute-label ids, entity-vocabulary ids and canonical pair
orientation. Everything that translates between the two happens here, once,
before training -- and every way a schedule can be unsatisfiable is an error
here rather than a silent zero later.

Three translations are easy to get wrong and are the reason this is not done
inline:

* **Pair orientation.** Object-object edges are stored in ``pair_sort_key``
  order, so a clause written ``tool -> cube`` is looked up as ``cube -> tool``.
  End-effector edges are always ``ee -> object`` and are not sorted.
* **Antisymmetric labels.** Swapping the endpoints of ``height-offset`` turns
  ``above`` into ``below``. The mirror is applied with the swap.
* **Directional physical labels.** ``support`` and ``contain`` name the holder
  by role; whether that is ``src-holds`` or ``dst-holds`` depends on which key
  sorts first, which is a property of the strings and not of the physics.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..adapters.graph_vocab import build_absolute_vocab, build_relation_vocab
from .relation_rules import SPATIAL_RELATIONS
from .sites import (
    SITE_PREFIX,
    SITE_REGION,
    SiteDeclaration,
    goal_pairs,
    parse_site_declarations,
)
from .spatial_metrics import (
    EE_OBJECT_SCOPE,
    OBJECT_OBJECT_SCOPE,
    OBJECT_REGION_PLANAR_KEY,
    OBJECT_SITE_HEIGHT_KEY,
    OBJECT_SITE_PLANAR_KEY,
    ee_family_bin_key,
    spatial_bin_key,
)

SCHEDULE_SCHEMA_VERSION = 1
EE_ROLE = "ee"
EE_KEY = "ee"

# Relations whose label flips when the endpoints are swapped.
_MIRROR: Dict[str, str] = {
    "far-below": "far-above", "below": "above", "level": "level",
    "above": "below", "far-above": "far-below",
}
ANTISYMMETRIC = frozenset({"height-offset"})
# Relations whose label names which endpoint holds the other.
DIRECTIONAL = frozenset({"support", "contain"})
SRC_HOLDS, DST_HOLDS = "src-holds", "dst-holds"


class ScheduleError(ValueError):
    """A schedule that cannot be scored. Always names task, phase and clause."""


@dataclass(frozen=True)
class Clause:
    """One rung. ``src_key``/``dst_key`` are already in stored edge order."""
    relation: str
    relation_id: int
    src_key: str
    dst_key: str
    src_entity_id: int
    dst_entity_id: int
    labels: Tuple[str, ...]
    label_ids: Tuple[int, ...]
    weight: float

    @property
    def slot(self) -> Tuple[int, int, int]:
        """What the runtime looks up. Rungs on one fact share a slot, so the
        edge is found once however many rungs read it."""
        return (self.relation_id, self.src_entity_id, self.dst_entity_id)


@dataclass(frozen=True)
class Phase:
    name: str
    weight: float
    clauses: Tuple[Clause, ...]
    completions: Tuple[Clause, ...]
    # Current-frame gate. The phase earns its clause quality only on frames
    # where every one of these holds; a completed later phase still grants the
    # full weight through cumulative credit, so a gate can never cost terminal
    # potential. Weightless by construction -- a gate that paid would be a
    # rung, and the ordering it expresses would become a reward of its own.
    requires: Tuple[Clause, ...] = ()

    @property
    def completion(self) -> Clause:
        """First completion clause, for legacy callers and diagnostics.

        Runtime completion is the conjunction of :attr:`completions`. Existing
        schedules have exactly one, so their compiled API and behaviour stay
        unchanged.
        """
        return self.completions[0]


@dataclass(frozen=True)
class CompiledSchedule:
    env_id: str
    roles: Dict[str, str]
    role_entity_ids: Dict[str, int]
    phases: Tuple[Phase, ...]

    @property
    def slots(self) -> Tuple[Tuple[int, int, int], ...]:
        """Distinct ``(relation, src entity, dst entity)`` facts the schedule
        reads, in stable order. Several rungs usually share one."""
        seen: List[Tuple[int, int, int]] = []
        for phase in self.phases:
            for clause in (*phase.clauses, *phase.completions,
                           *phase.requires):
                if clause.slot not in seen:
                    seen.append(clause.slot)
        return tuple(seen)

    @property
    def entity_ids(self) -> Tuple[int, ...]:
        """Every entity id a frame must resolve, in stable order."""
        seen: List[int] = []
        for slot in self.slots:
            for ent in slot[1:]:
                if ent not in seen:
                    seen.append(ent)
        return tuple(seen)


# --------------------------------------------------------------------------- #
# what the mined assets can actually score
# --------------------------------------------------------------------------- #
def _types(members: Dict[str, Any], key: str) -> set:
    return set((members.get(key) or {}).get("interaction_types") or ())


def _has(objects: Dict[str, Any], key: str, field: str) -> bool:
    return bool((objects.get(key) or {}).get(field))


def scorable_relations(
    objects: Dict[str, Any], members: Dict[str, Any],
    bin_edges: Dict[str, Any],
    sites: Optional[Dict[str, SiteDeclaration]] = None,
    structural: Optional[set] = None,
) -> Dict[str, Dict[str, bool]]:
    """Per pair, which relations the runtime can emit with meaning.

    ``contact`` needs both endpoints to carry the token; the compatibility
    families need the mined components behind them. The distinction matters
    because a compatibility relation with no components is still *emitted* --
    labelled ``unobserved``, exactly as a pair too far apart to judge -- so a
    clause naming it would score zero forever with nothing to show for it.
    """
    keys = sorted(members)
    # ``reached`` is scorable only where a validated site declaration names
    # that exact pair. Without this it would degrade into a generic proximity
    # alias: any two objects have a distance, but only a declared pair has an
    # environment tolerance that makes crossing it mean success.
    declared = goal_pairs(sites or {})
    reached_on = lambda a, b: tuple(sorted((a, b))) in declared
    # An extended support plane emits no public planar-distance, so a clause
    # naming one would score a fact the runtime never produces. Caught here
    # rather than at the first frame: a schedule that reaches for the table's
    # horizontal position is describing a goal the scene does not contain.
    surfaces = set(structural or ())
    # A region pair is scored on its own planar scale and has no height at all.
    regions = {key for key, decl in (sites or {}).items()
               if decl.site_type == SITE_REGION}
    region_spatial = {
        "planar-distance": bool(bin_edges.get(OBJECT_REGION_PLANAR_KEY)),
        "height-offset": False,
    }
    # Virtual, non-region sites carry a ladder on the object-site scale.
    ladders = {key for key, decl in (sites or {}).items()
               if key.startswith(SITE_PREFIX) and decl.site_type != SITE_REGION}
    ladder_spatial = {
        "planar-distance": bool(bin_edges.get(OBJECT_SITE_PLANAR_KEY)),
        "height-offset": bool(bin_edges.get(OBJECT_SITE_HEIGHT_KEY)),
    }
    # Scope-aware: the ee-object and object-object scales are mined
    # separately, so a pair can be scorable in one and not the other.
    # Height is calibrated per family, so what makes an ee pair scorable is
    # that member's own scale -- resolved below, per key.
    families = {
        key: (entry or {}).get("family")
        for key, entry in members.items() if isinstance(entry, dict)
    }
    any_family = any(families.values())

    def _ee_height(key: str) -> bool:
        if not any_family:
            return bool(bin_edges.get(
                spatial_bin_key(EE_OBJECT_SCOPE, "height-offset")))
        family = families.get(key)
        return bool(family) and bool(bin_edges.get(ee_family_bin_key(family)))

    ee_planar = bool(bin_edges.get(
        spatial_bin_key(EE_OBJECT_SCOPE, "planar-distance")))
    obj_spatial = {r: bool(bin_edges.get(
        spatial_bin_key(OBJECT_OBJECT_SCOPE, r))) for r in SPATIAL_RELATIONS}
    out: Dict[str, Dict[str, bool]] = {}
    for key in keys:
        out[f"{EE_KEY} / {key}"] = {
            "contact": "contact" in _types(members, key),
            "grasp": "grasp" in _types(members, key),
            "grasp-compatibility": _has(objects, key, "grasp_components"),
            "contact-compatibility": _has(objects, key, "contact_components"),
            "reached": reached_on(EE_KEY, key),
            "planar-distance": ee_planar and key not in surfaces,
            "height-offset": _ee_height(key),
        }
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            both = _types(members, a) & _types(members, b)
            out[f"{a} / {b}"] = {
                "contact": "contact" in both,
                "support": "support" in both,
                "contain": "contain" in both,
                "contact-compatibility": _has(objects, a, "contact_components")
                                         and _has(objects, b, "contact_components"),
                "support-compatibility": _pairing(objects, a, b,
                                                  "support_components",
                                                  "bottom_components"),
                "contain-compatibility": _pairing(objects, a, b,
                                                  "contain_components",
                                                  "key_components"),
                "reached": reached_on(a, b),
                **obj_spatial,
                **(region_spatial if (a in regions or b in regions) else {}),
                **(ladder_spatial if (a in ladders or b in ladders) else {}),
                **({"planar-distance": False}
                   if (a in surfaces or b in surfaces) else {}),
            }
    return out


def _pairing(objects, a: str, b: str, host: str, guest: str) -> bool:
    return ((_has(objects, a, host) and _has(objects, b, guest))
            or (_has(objects, b, host) and _has(objects, a, guest)))


# --------------------------------------------------------------------------- #
# compilation
# --------------------------------------------------------------------------- #
def load_assets(
    env_id: str, configs: str,
) -> Tuple[Dict, Dict, Dict, Dict[str, SiteDeclaration], set]:
    """``(objects, members, bin_edges, sites, structural_surfaces)``."""
    aff = os.path.join(configs, "affordances", f"{env_id}.json")
    wl = os.path.join(configs, "subtask_whitelists", env_id, "task_all.json")
    for path in (aff, wl):
        if not os.path.isfile(path):
            raise ScheduleError(
                f"{env_id}: {path} is missing. Mine the task's assets first."
            )
    with open(aff) as handle:
        objects = json.load(handle).get("objects", {})
    with open(wl) as handle:
        whitelist = json.load(handle)
    sites = parse_site_declarations(whitelist.get("sites"), where=env_id)
    members = whitelist.get("members", {})
    structural = {
        key for key, entry in members.items()
        if isinstance(entry, dict) and entry.get("structural_surface")
    }
    for key, decl in sites.items():
        for named in (key, decl.subject_key):
            if named != EE_KEY and named not in members:
                raise ScheduleError(
                    f"{env_id}: site {key!r} names {named!r}, which is not a "
                    "whitelist member. A site and its subject both need "
                    "vocabulary rows before a frame can resolve them."
                )
    return (objects, members, whitelist.get("bin_edges", {}), sites,
            structural)


def _order(a_key: str, b_key: str) -> Tuple[str, str, bool]:
    """Stored edge order for a pair, and whether that swapped the arguments.

    End-effector edges are emitted ``ee -> object`` and never sorted; object
    pairs follow ``pair_sort_key``, whose first component is the whitelist key.
    """
    if a_key == EE_KEY:
        return a_key, b_key, False
    if b_key == EE_KEY:
        return b_key, a_key, True
    if b_key < a_key:
        return b_key, a_key, True
    return a_key, b_key, False


def _resolve_labels(relation: str, labels: Sequence[str], swapped: bool,
                    where: str) -> Tuple[str, ...]:
    if relation in ANTISYMMETRIC and swapped:
        try:
            return tuple(_MIRROR[label] for label in labels)
        except KeyError as exc:
            raise ScheduleError(
                f"{where}: {relation} has no mirror for label {exc.args[0]!r}"
            ) from None
    return tuple(labels)


def compile_schedule(raw: Dict[str, Any], objects: Dict[str, Any],
                     members: Dict[str, Any], bin_edges: Dict[str, Any],
                     entity_vocab,
                     sites: Optional[Dict[str, SiteDeclaration]] = None,
                     structural: Optional[set] = None,
                     ) -> CompiledSchedule:
    """Validate and translate one schedule. Raises on anything unscorable."""
    env_id = str(raw.get("env_id") or "?")
    version = int(raw.get("_schema_version", 0))
    if version != SCHEDULE_SCHEMA_VERSION:
        raise ScheduleError(
            f"{env_id}: schedule schema v{version}, expected "
            f"v{SCHEDULE_SCHEMA_VERSION}"
        )
    roles = dict(raw.get("roles") or {})
    if not roles:
        raise ScheduleError(f"{env_id}: schedule names no roles")

    duplicates = [k for k in set(roles.values())
                  if list(roles.values()).count(k) > 1]
    if duplicates:
        raise ScheduleError(
            f"{env_id}: roles {sorted(duplicates)} share one entity key, so a "
            "frame cannot tell them apart. Give the task an explicit role "
            "token rather than letting one row serve two roles."
        )
    role_ids: Dict[str, int] = {}
    for role, key in roles.items():
        if key not in members:
            raise ScheduleError(
                f"{env_id}: role {role!r} names {key!r}, which is not a "
                f"whitelist member ({sorted(members)})"
            )
        try:
            role_ids[role] = entity_vocab.encode(key)
        except KeyError:
            raise ScheduleError(
                f"{env_id}: role {role!r} names {key!r}, absent from the "
                "entity vocabulary. Re-mine the task."
            ) from None

    scorable = scorable_relations(
        objects, members, bin_edges, sites, structural)
    relations, absolute = build_relation_vocab(), build_absolute_vocab()

    def entity_id(key: str) -> int:
        return entity_vocab.ee_id if key == EE_KEY else entity_vocab.encode(key)

    def key_of(token: str, where: str) -> str:
        if token == EE_ROLE:
            return EE_KEY
        if token not in roles:
            raise ScheduleError(f"{where}: unknown role {token!r}")
        return roles[token]

    phases: List[Phase] = []
    for index, rawphase in enumerate(raw.get("phases") or ()):
        name = str(rawphase.get("name") or index)
        where = f"{env_id}/{name}"
        weight = float(rawphase.get("weight", 0.0))
        clauses = tuple(
            _clause(c, key_of, scorable, relations, absolute, entity_id, where)
            for c in rawphase.get("clauses") or ()
        )
        if not clauses:
            raise ScheduleError(f"{where}: phase has no clauses")
        inner = sum(c.weight for c in clauses)
        if abs(inner - weight) > 1e-6:
            raise ScheduleError(
                f"{where}: clause weights sum to {inner:.6f}, phase weight is "
                f"{weight:.6f}. A phase that cannot reach its own weight can "
                "never be completed."
            )
        requires = _phase_requires(
            rawphase, key_of, scorable, relations, absolute, entity_id, where)
        completion = dict(rawphase.get("completion") or {})
        if "all_of" in completion:
            if set(completion) != {"all_of"}:
                raise ScheduleError(
                    f"{where}/completion: 'all_of' cannot be mixed with a "
                    "single completion clause"
                )
            raw_completions = completion["all_of"]
            if not isinstance(raw_completions, list) or not raw_completions:
                raise ScheduleError(
                    f"{where}/completion: 'all_of' must be a non-empty list"
                )
        else:
            raw_completions = [completion]
        completions = []
        for raw_completion in raw_completions:
            if not isinstance(raw_completion, dict):
                raise ScheduleError(
                    f"{where}/completion: every 'all_of' item must be a clause"
                )
            item = dict(raw_completion)
            item.setdefault("src", rawphase.get("src", EE_ROLE))
            item.setdefault("dst", rawphase.get("dst", EE_ROLE))
            item.setdefault("weight", 0.0)
            completions.append(_clause(
                item, key_of, scorable, relations, absolute, entity_id,
                f"{where}/completion",
            ))
        phases.append(Phase(
            name=name, weight=weight, clauses=clauses,
            completions=tuple(completions), requires=requires,
        ))

    if not phases:
        raise ScheduleError(f"{env_id}: schedule has no phases")
    total = sum(p.weight for p in phases)
    if abs(total - 1.0) > 1e-6:
        raise ScheduleError(
            f"{env_id}: phase weights sum to {total:.6f}, not 1. The potential "
            "is bounded by construction only when they do."
        )
    return CompiledSchedule(env_id=env_id, roles=roles,
                            role_entity_ids=role_ids, phases=tuple(phases))


def _phase_requires(rawphase: Dict[str, Any], key_of, scorable, relations,
                    absolute, entity_id, where: str) -> Tuple[Clause, ...]:
    """Compile a phase's current-frame gate.

    ``requires`` says *when* a phase's rungs pay, not *what* they pay for. Peg
    needs its containment ladder inert until the peg head has reached the hole
    mouth and touched the box, because otherwise the alignment reward is
    available from across the table and the schedule stops describing an order
    at all.

    Deliberately not a latch. The gate reads only this frame, so the potential
    stays a function of the observed state and ``gamma * Phi' - Phi`` still
    telescopes exactly. The price is a transient dip when a gate flickers,
    which is the trade the schedule owner accepted.
    """
    raw = rawphase.get("requires")
    if raw is None:
        return ()
    if not isinstance(raw, dict) or set(raw) != {"all_of"}:
        raise ScheduleError(
            f"{where}/requires: expected an object with exactly one key "
            "'all_of'"
        )
    items = raw["all_of"]
    if not isinstance(items, list) or not items:
        raise ScheduleError(
            f"{where}/requires: 'all_of' must be a non-empty list"
        )
    out: List[Clause] = []
    for item in items:
        if not isinstance(item, dict):
            raise ScheduleError(
                f"{where}/requires: every 'all_of' item must be a clause"
            )
        if float(item.get("weight", 0.0)) != 0.0:
            raise ScheduleError(
                f"{where}/requires: a gate cannot carry weight. It decides "
                "whether the phase's own clauses pay; paying for it as well "
                "would double-count the milestone and break the phase's "
                "weight sum."
            )
        entry = dict(item)
        entry.setdefault("src", rawphase.get("src", EE_ROLE))
        entry.setdefault("dst", rawphase.get("dst", EE_ROLE))
        entry["weight"] = 0.0
        out.append(_clause(entry, key_of, scorable, relations, absolute,
                           entity_id, f"{where}/requires"))
    return tuple(out)


def _clause(raw: Dict[str, Any], key_of, scorable, relations, absolute,
            entity_id, where: str) -> Clause:
    relation = str(raw.get("relation") or "")
    if relation not in relations.token_to_id:
        raise ScheduleError(f"{where}: unknown relation {relation!r}")
    src_key = key_of(str(raw.get("src", EE_ROLE)), where)
    dst_key = key_of(str(raw.get("dst", EE_ROLE)), where)
    if src_key == dst_key:
        raise ScheduleError(f"{where}: clause relates {src_key!r} to itself")

    a_key, b_key, swapped = _order(src_key, dst_key)
    pair = f"{a_key} / {b_key}"
    if pair not in scorable:
        raise ScheduleError(
            f"{where}: no pair {pair!r} exists in this task's assets"
        )
    if not scorable[pair].get(relation):
        raise ScheduleError(
            f"{where}: {relation!r} is not scorable on {pair!r}. The mined "
            "assets carry no components or bins behind it, so the runtime "
            "would emit 'unobserved' every frame and this clause would score "
            "zero for the whole episode."
        )

    holder = raw.get("holder")
    if relation in DIRECTIONAL:
        if not holder:
            raise ScheduleError(
                f"{where}: {relation!r} needs a 'holder' role -- the label "
                "names which endpoint holds the other, and which of "
                "src-holds/dst-holds that is depends on key order."
            )
        holder_key = key_of(str(holder), where)
        if holder_key not in (a_key, b_key):
            raise ScheduleError(
                f"{where}: holder {holder_key!r} is not an endpoint of {pair!r}"
            )
        labels: Tuple[str, ...] = (
            SRC_HOLDS if holder_key == a_key else DST_HOLDS,
        )
    else:
        if holder:
            raise ScheduleError(
                f"{where}: {relation!r} is not directional; 'holder' would be "
                "silently ignored"
            )
        labels = _resolve_labels(
            relation, [str(x) for x in raw.get("labels") or ()], swapped, where)
    if not labels:
        raise ScheduleError(f"{where}: clause names no labels")

    try:
        label_ids = tuple(absolute.encode(label) for label in labels)
    except KeyError as exc:
        raise ScheduleError(f"{where}: unknown label {exc.args[0]!r}") from None
    return Clause(
        relation=relation, relation_id=relations.encode(relation),
        src_key=a_key, dst_key=b_key,
        src_entity_id=entity_id(a_key), dst_entity_id=entity_id(b_key),
        labels=labels, label_ids=label_ids,
        weight=float(raw.get("weight", 0.0)),
    )


def compile_from_files(env_id: str, schedule_dir: str, configs: str,
                       entity_vocab) -> CompiledSchedule:
    path = os.path.join(schedule_dir, f"{env_id}.json")
    if not os.path.isfile(path):
        raise ScheduleError(
            f"{env_id}: no schedule at {path}. progress.mode=task_schedule "
            "requires one per task."
        )
    with open(path) as handle:
        raw = json.load(handle)
    objects, members, bin_edges, sites, structural = load_assets(
        env_id, configs)
    return compile_schedule(
        raw, objects, members, bin_edges, entity_vocab, sites=sites,
        structural=structural)
