"""Per-subtask whitelist used as the runtime's sole relevance gate.

Asset shape (``_schema_version: 4``)::

    {
      "_schema_version": 4,
      "subtask": "pick",
      "target": "actor:024_bowl",
      "members": {
        "actor:024_bowl": {
            "roles": ["interacted"],
            "interaction_types": ["contact", "grasp"],
            "kind": "actor"
        },
        "link:cabinet/drawer3": {
            "roles": ["support"],
            "interaction_types": ["support"],
            "kind": "link"
        }
      },
      "bin_edges": { "<relation>": [edges...], ... }
    }

Per-member ``interaction_types`` controls which compatibility edges runtime
emits for the object:

  * ``contact`` -- ee touched (or for obj-obj, both endpoints touched in demos)
  * ``grasp``   -- grasp predicate fired in demos
  * ``support`` -- participated in an obj-obj support pair in demos
  * ``contain`` -- participated in an obj-obj contain pair in demos

Match-key conventions:

  * Free actors: ``actor:<canonical object id>``.
  * Articulation links: ``link:<articulation instance>/<link name>``.

v3 assets without the new ``support`` / ``contain`` tokens still load (those
edges simply never emit).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .affordance import canonical_affordance_key
from .entity_identity import entity_kind, normalize_asset_key, stable_entity_key
from .schema import Node


WHITELIST_SCHEMA_VERSION = 4


# --------------------------------------------------------------------------- #
# Match key (used by both runtime and miner)
# --------------------------------------------------------------------------- #
def match_key(node: Node) -> Optional[str]:
    """Resolve the cross-frame whitelist key for one node."""
    name = node.name
    if not name:
        return None
    attrs = node.attributes or {}
    stable = attrs.get("entity_key")
    if stable:
        return normalize_asset_key(str(stable), attrs.get("entity_kind"))
    if attrs.get("is_actor"):
        return normalize_asset_key(canonical_affordance_key(name) or name, "actor")
    if attrs.get("is_link") or attrs.get("is_articulation_link"):
        return normalize_asset_key(name, "link")
    return normalize_asset_key(name)


def entity_match_key(entity) -> Optional[str]:
    """Entity-level twin of ``match_key``: identical resolution, no Node.

    Lets the runtime test whitelist admission before constructing a node
    (and paying its pose read) for entities that can never pass the gate.
    """
    if entity is None:
        return None
    name = getattr(entity, "name", str(entity))
    if not name:
        return None
    kind = entity_kind(entity)
    stable = stable_entity_key(entity)
    if stable:
        return normalize_asset_key(str(stable), kind)
    if kind == "actor":
        return normalize_asset_key(canonical_affordance_key(name) or name, "actor")
    if kind == "link":
        return normalize_asset_key(name, "link")
    return normalize_asset_key(name)


# --------------------------------------------------------------------------- #
# Interaction-type vocabulary
# --------------------------------------------------------------------------- #
INTERACTION_CONTACT = "contact"
INTERACTION_GRASP = "grasp"
INTERACTION_SUPPORT = "support"
INTERACTION_CONTAIN = "contain"

_VALID_INTERACTION_TYPES = frozenset({
    INTERACTION_CONTACT, INTERACTION_GRASP,
    INTERACTION_SUPPORT, INTERACTION_CONTAIN,
})


# --------------------------------------------------------------------------- #
# Asset
# --------------------------------------------------------------------------- #
@dataclass
class Whitelist:
    subtask: str = ""
    target: str = ""
    by_key: Dict[str, Set[str]] = field(default_factory=dict)
    interaction_types: Dict[str, Set[str]] = field(default_factory=dict)
    bin_edges: Dict[str, List[float]] = field(default_factory=dict)
    source_path: Optional[str] = None

    @property
    def empty(self) -> bool:
        return not self.by_key

    def contains(self, key: Optional[str]) -> bool:
        if key is None:
            return False
        return key in self.by_key

    def roles(self, key: Optional[str]) -> Set[str]:
        if key is None:
            return set()
        return set(self.by_key.get(key, set()))

    def types(self, key: Optional[str]) -> Set[str]:
        if key is None:
            return set()
        return set(self.interaction_types.get(key, set()))


# Parsed assets keyed by absolute path with an (mtime_ns, size) signature so
# the per-episode whitelist re-bind does not re-parse JSON from disk. Cached
# instances are shared across builders and must stay immutable.
_WHITELIST_CACHE: Dict[str, Tuple[Tuple[int, int], Whitelist]] = {}


def load_whitelist(path: str) -> Whitelist:
    """Load a per-subtask whitelist. Raises FileNotFoundError if missing."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"per-subtask whitelist not found at {path!r}; mine it with "
            "tools/build_subtask_whitelists.py before running the probe"
        )
    abspath = os.path.abspath(path)
    st = os.stat(abspath)
    sig = (st.st_mtime_ns, st.st_size)
    hit = _WHITELIST_CACHE.get(abspath)
    if hit is not None and hit[0] == sig:
        return hit[1]
    with open(path, "r") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"whitelist {path!r}: expected JSON object at root")

    by_key: Dict[str, Set[str]] = {}
    interaction_types: Dict[str, Set[str]] = {}
    members = raw.get("members", {})
    if not isinstance(members, dict):
        raise ValueError(f"whitelist {path!r}: 'members' must be an object")
    for k, entry in members.items():
        if not isinstance(k, str) or k.startswith("_"):
            continue
        roles_set: Set[str] = set()
        itypes_set: Set[str] = set()
        kind = None
        if isinstance(entry, dict):
            roles = entry.get("roles")
            if isinstance(roles, (list, tuple)):
                for r in roles:
                    if isinstance(r, str):
                        roles_set.add(r)
            itypes = entry.get("interaction_types")
            if isinstance(itypes, (list, tuple)):
                for t in itypes:
                    if isinstance(t, str) and t in _VALID_INTERACTION_TYPES:
                        itypes_set.add(t)
            kind = entry.get("kind")
        normalized = normalize_asset_key(k, kind)
        if normalized:
            by_key[normalized] = roles_set
            interaction_types[normalized] = itypes_set

    if not by_key:
        raise ValueError(f"whitelist {path!r}: 'members' is empty")

    bin_edges: Dict[str, List[float]] = {}
    raw_edges = raw.get("bin_edges", {})
    if isinstance(raw_edges, dict):
        for rel, edges in raw_edges.items():
            if not isinstance(rel, str) or not isinstance(edges, (list, tuple)):
                continue
            parsed: List[float] = []
            ok = True
            for x in edges:
                try:
                    v = float(x)
                except (TypeError, ValueError):
                    ok = False
                    break
                if not math.isfinite(v):
                    ok = False
                    break
                parsed.append(v)
            if ok and parsed:
                bin_edges[rel] = parsed

    wl = Whitelist(
        subtask=str(raw.get("subtask", "") or ""),
        target=str(raw.get("target", "") or ""),
        by_key=by_key,
        interaction_types=interaction_types,
        bin_edges=bin_edges,
        source_path=path,
    )
    _WHITELIST_CACHE[abspath] = (sig, wl)
    return wl


# --------------------------------------------------------------------------- #
# Bin-edge derivation
# --------------------------------------------------------------------------- #
# Equal-width splits of [0, max] (unsigned) and [-max, max] (signed).
# 5-label unsigned: 4 edges at max/5, 2*max/5, 3*max/5, 4*max/5.
# 5-label signed:   4 edges at -3*max/5, -max/5, max/5, 3*max/5.
# Sensitive 5-label signed changes keep the same fast outer thresholds but
# shrink the stable center band to +/-max/12.
# Compatibility absolute edges are fixed because the score is already in [0,1].


def _equal_width_5_unsigned(max_v: float) -> Optional[List[float]]:
    if max_v <= 0 or not math.isfinite(max_v):
        return None
    return [max_v / 5.0, 2.0 * max_v / 5.0,
            3.0 * max_v / 5.0, 4.0 * max_v / 5.0]


def _equal_width_5_signed(max_v: float) -> Optional[List[float]]:
    if max_v <= 0 or not math.isfinite(max_v):
        return None
    return [-3.0 * max_v / 5.0, -max_v / 5.0,
            max_v / 5.0, 3.0 * max_v / 5.0]


def _sensitive_width_5_signed(max_v: float) -> Optional[List[float]]:
    if max_v <= 0 or not math.isfinite(max_v):
        return None
    return [-3.0 * max_v / 5.0, -max_v / 12.0,
            max_v / 12.0, 3.0 * max_v / 5.0]


# Relations and the kind of bin-derivation they use.
_BIN_DERIVATION = {
    "planar-distance":              ("unsigned5",        "planar_distance"),
    "height-offset":                ("signed5",          "height_offset"),
    "planar-distance-change":       ("signed5-sensitive", "planar_distance_change"),
    "height-offset-change":         ("signed5-sensitive", "height_offset_change"),
    "grasp-compatibility-change":   ("signed5-sensitive", "grasp_compatibility_change"),
    "contact-compatibility-change": ("signed5-sensitive", "contact_compatibility_change"),
    "support-compatibility-change": ("signed5-sensitive", "support_compatibility_change"),
    "contain-compatibility-change": ("signed5-sensitive", "contain_compatibility_change"),
}


def derive_bin_edges(max_values: Dict[str, float]) -> Dict[str, List[float]]:
    """Derive runtime bin edges from per-relation demo maxes.

    Compatibility absolute edges are always ``[1/3, 2/3]`` because the score
    is normalized to ``[0, 1]`` before binning.
    """
    out: Dict[str, List[float]] = {
        "grasp-compatibility":   [1.0 / 3.0, 2.0 / 3.0],
        "contact-compatibility": [1.0 / 3.0, 2.0 / 3.0],
        "support-compatibility": [1.0 / 3.0, 2.0 / 3.0],
        "contain-compatibility": [1.0 / 3.0, 2.0 / 3.0],
    }
    for relation, (kind, src) in _BIN_DERIVATION.items():
        try:
            v = float(max_values.get(src, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        edges: Optional[List[float]] = None
        if kind == "unsigned5":
            edges = _equal_width_5_unsigned(v)
        elif kind == "signed5":
            edges = _equal_width_5_signed(v)
        elif kind == "signed5-sensitive":
            edges = _sensitive_width_5_signed(v)
        if edges is not None:
            out[relation] = edges
    return out


# --------------------------------------------------------------------------- #
# Filename resolver
# --------------------------------------------------------------------------- #
def resolve_whitelist_path(
    whitelist_dir: Optional[str],
    subtask_type: Optional[str],
    target_canonical: Optional[str],
) -> Optional[str]:
    """Return the on-disk path for ``<subtask>_<target>.json`` if it exists."""
    if not whitelist_dir or not subtask_type or not target_canonical:
        return None
    target_slug = whitelist_target_slug(target_canonical)
    fname = f"{subtask_type}_{target_slug}.json"
    path = os.path.join(whitelist_dir, fname)
    return path if os.path.isfile(path) else None


def whitelist_target_slug(target_key: str) -> str:
    """Filesystem-safe target portion shared by runtime and offline miner."""
    value = str(target_key).split(":", 1)[-1]
    return value.replace("/", "__").replace("\\", "__").replace(":", "_")
