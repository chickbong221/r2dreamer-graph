"""Scene entity extraction for normal ManiSkill tabletop tasks.

MS-HAB picks objects from a task plan. Normal ManiSkill has none, so the scene
itself is authoritative: take every segmentation entity, drop the robot and the
ground plane, keep the rest. Kinematic actors are kept -- bins, receptacles and
the box-with-hole are kinematic but are real graph objects.

The table is kept too. It is the surface everything rests on, so
``support / table-workspace / cubeA`` is the primary support bucket of every
tabletop task; dropping it would leave those objects supported by nothing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from ..core.entity_identity import entity_name, stable_entity_key
from .privileged_state import per_env_segmentation_id_map

# Fallback only; identity via the scene builder is preferred, since a task
# may rename its ground.
BACKGROUND_NAMES = frozenset({"ground"})

# Attributes ManiSkill scene builders are reachable through.
_SCENE_BUILDER_ATTRS = ("table_scene", "scene_builder", "_scene_builder")


def _unwrap(env):
    return env.unwrapped if hasattr(env, "unwrapped") else env


def background_ids(env) -> Set[int]:
    """``id()`` of the ground plane a scene builder owns.

    Only the ground. ``scene_objects`` is not used because it bundles the table
    in with it, and the table is a task object here.
    """
    base = _unwrap(env)
    found: Set[int] = set()
    for attr in _SCENE_BUILDER_ATTRS:
        builder = getattr(base, attr, None)
        if builder is None:
            continue
        ent = getattr(builder, "ground", None)
        if ent is not None:
            found.add(id(ent))
    return found


def robot_link_names(env) -> Set[str]:
    agent = getattr(_unwrap(env), "agent", None)
    robot = getattr(agent, "robot", None) if agent is not None else None
    if robot is None:
        return set()
    try:
        return {entity_name(link) for link in robot.get_links()}
    except Exception:
        return set()


def is_background(entity, bg_ids: Set[int]) -> bool:
    if id(entity) in bg_ids:
        return True
    return entity_name(entity) in BACKGROUND_NAMES


def scene_entities(
    env, env_idx: int = 0, seg_id_map: Optional[Dict[int, Any]] = None,
) -> List[Any]:
    """Task objects for one env, deduped by stable key.

    Merged-view aliasing must be on, or PegInsertionSide yields ``peg_<i>``.
    """
    if seg_id_map is None:
        seg_id_map = per_env_segmentation_id_map(env, env_idx)
    bg_ids = background_ids(env)
    robot_names = robot_link_names(env)

    out: List[Any] = []
    seen: Set[str] = set()
    for seg_id, ent in sorted(seg_id_map.items()):
        if not seg_id or ent is None:
            continue
        if entity_name(ent) in robot_names or is_background(ent, bg_ids):
            continue
        key = stable_entity_key(ent)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(ent)
    return out


def scene_entity_keys(env, env_idx: int = 0) -> List[str]:
    return [stable_entity_key(e) for e in scene_entities(env, env_idx)]
