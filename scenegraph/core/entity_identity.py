"""Stable identities shared by offline collection and runtime graph building.

The graph has only two node types (``ee`` and ``object``).  Free actors and
articulation links are both ordinary object nodes; their stable keys differ so
that a support link and a handle link never collapse into one entity.
"""

from __future__ import annotations

import re
from typing import Optional

from .affordance import canonical_affordance_key


_ENV_PREFIX_RE = re.compile(r"^env-\d+_")
# ReplicaCAD articulation names carry a scene-config-set tag like
# ``scs-[0]_`` or ``scs-[2,3]_`` that varies per build config. The same
# logical articulation (e.g. ``kitchen_counter-0``) shows up under different
# prefixes across scene sets, so the offline whitelist would be unmatched at
# runtime if we kept it. Strip it for cross-scene key portability.
_SCS_PREFIX_RE = re.compile(r"^scs-\[[0-9,]+\]_")
# Leading YCB-style numeric asset id (``024_bowl-0`` -> ``bowl-0``). Only
# applied to display names, never to whitelist keys.
_ASSET_NUM_PREFIX_RE = re.compile(r"^\d+_")


def entity_name(entity) -> str:
    return str(getattr(entity, "name", None) or entity)


def entity_kind(entity) -> str:
    name = type(entity).__name__
    if name == "Actor":
        return "actor"
    if name == "Link":
        return "link"
    return "other"


def _articulation(entity):
    for attr in ("articulation", "parent_articulation"):
        value = getattr(entity, attr, None)
        if value is not None:
            return value
    for method in ("get_articulation", "get_parent_articulation"):
        fn = getattr(entity, method, None)
        if callable(fn):
            try:
                value = fn()
            except Exception:
                continue
            if value is not None:
                return value
    return None


def canonical_scene_name(name: Optional[str]) -> Optional[str]:
    """Strip per-environment + scene-config-set prefixes, preserving instance
    suffixes. ``env-0_scs-[2,3]_fridge-0`` -> ``fridge-0``."""
    if not name:
        return None
    stripped = _ENV_PREFIX_RE.sub("", str(name))
    stripped = _SCS_PREFIX_RE.sub("", stripped)
    return stripped or None


def display_name(name: Optional[str]) -> str:
    """Short label for overlays and node-graph rendering.

    Strips the same env/scs prefixes as ``canonical_scene_name`` and, in
    addition, one leading YCB-style ``<digits>_`` asset id so labels stay
    readable in a small figure (``env-0_024_bowl-0`` -> ``bowl-0``).
    """
    if not name:
        return ""
    s = canonical_scene_name(name) or str(name)
    return _ASSET_NUM_PREFIX_RE.sub("", s) or s


def canonical_actor_key(name: Optional[str]) -> Optional[str]:
    """Cross-scene actor key: no ``env-N_``, no ``scs-[i,j]_``, no ``-N``.

    Merged ReplicaCAD scene actors carry the same scene-config-set tag the
    articulations do, so one logical asset appears as
    ``scs-[2,3]_frl_apartment_chair_01`` in one build config and
    ``scs-[6,7]_frl_apartment_chair_01`` in another. Keeping the tag made a
    whitelist mined in one set of build configs unmatchable in any other, and
    split one chair into four vocabulary entries. The link path has always
    stripped it; this is the actor twin of that.

    Only the whitelist key collapses. ``stable_node_id`` still carries the
    instance, so two chairs standing in one scene remain two nodes.
    """
    if not name:
        return None
    scene = canonical_scene_name(name) or str(name)
    return canonical_affordance_key(scene) or scene


def stable_entity_key(entity) -> Optional[str]:
    """Return ``actor:<id>`` or ``link:<articulation>/<link>``.

    Link qualification is best-effort because SAPIEN versions expose the
    parent articulation through different attributes.  The bare link name is
    retained as a deterministic fallback.
    """
    if entity is None:
        return None
    name = entity_name(entity)
    kind = entity_kind(entity)
    if kind == "actor":
        return f"actor:{canonical_actor_key(name) or name}"
    if kind == "link":
        link_name = canonical_scene_name(name) or name
        art = _articulation(entity)
        art_name = canonical_scene_name(entity_name(art)) if art is not None else None
        qualified = f"{art_name}/{link_name}" if art_name else link_name
        return f"link:{qualified}"
    return f"object:{canonical_scene_name(name) or name}"


def stable_node_id(entity) -> str:
    if entity_kind(entity) == "actor":
        # Node identity preserves the simulator instance suffix while the
        # whitelist key intentionally canonicalizes actors by asset type.
        name = canonical_scene_name(entity_name(entity)) or entity_name(entity)
        return f"actor:{name}"
    return stable_entity_key(entity) or f"object:{entity_name(entity)}"


def normalize_asset_key(key: Optional[str], kind: Optional[str] = None) -> Optional[str]:
    """Normalize new and legacy whitelist keys to the stable-key namespace.

    An already-namespaced key has its payload canonicalized rather than passed
    through. That is what lets evidence persisted before the actor path stripped
    ``scs-[i,j]_`` be re-mined without recollecting: rollout pickles carry the
    prefixed key verbatim, and every reader of them funnels through here.
    """
    if not key:
        return None
    value = str(key)
    for prefix in ("actor:", "link:", "object:"):
        if value.startswith(prefix):
            payload = value[len(prefix):]
            if not payload:
                return None
            if prefix == "actor:":
                return f"actor:{canonical_actor_key(payload) or payload}"
            # Link and object payloads keep their instance suffix -- a bare
            # ``fridge-0`` fallback must not become ``fridge`` -- so only the
            # scene-set tag comes off.
            return f"{prefix}{canonical_scene_name(payload) or payload}"
    if kind == "actor":
        return f"actor:{canonical_actor_key(value) or value}"
    if kind == "link":
        return f"link:{canonical_scene_name(value) or value}"
    return value
