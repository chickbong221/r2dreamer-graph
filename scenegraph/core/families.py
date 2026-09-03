"""Which height scale a member is measured on, and what counts as a surface.

Two classifications the runtime cannot do without, and cannot recover from the
graph itself:

* **Structural surface.** A tabletop is measured from its top face, not from
  its actor origin a metre below. Getting this wrong set a deadband from a
  metre of table clearance and made every end-effector height read ``level``
  in seven tasks at once.
* **End-effector height family.** ``level`` has to mean one thing per scale.
  A two-centimetre lift above a manipuland and a hand hovering over a counter
  are not the same measurement, so each family carries its own quantiles.

Shared by both miners on purpose. ManiSkill mines one task at a time from an
interaction shard; MS-HAB mines per (subtask, target) from rollout pickles.
They arrive at the same question with differently shaped evidence, and two
copies of these rules would drift into two meanings for one token.

Pure dict logic -- no numpy, no simulator, no I/O.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from . import spatial_metrics

# Smaller horizontal half-extent at or above which a supporter is an extended
# plane rather than a localized receptacle. Size is the only discriminator
# available: in PlaceSphere the bin and the table carry byte-identical roles
# and interaction types, because both are kinematic and both support the
# sphere. Their sizes differ by an order of magnitude.
STRUCTURAL_SURFACE_MIN_HALF_EXTENT = 0.30


def structural_surfaces(
    extents: Dict[str, Any], members: Dict[str, Any],
) -> Dict[str, str]:
    """``{key: reason}`` for every member that is an extended support plane.

    Two conditions, both necessary. It has to actually support something -- a
    large object nothing ever rests on is scenery, not a surface -- and its
    smaller horizontal half-extent has to reach
    :data:`STRUCTURAL_SURFACE_MIN_HALF_EXTENT`.

    A member whose collision geometry could not be read is left unclassified
    rather than assumed small: a missing measurement is not evidence of
    absence, and a table quietly demoted to an ordinary object reinstates the
    metre of origin error this exists to remove. Report those through
    :func:`unclassified_supporters` instead.
    """
    out: Dict[str, str] = {}
    for key, entry in sorted(members.items()):
        if "support" not in ((entry or {}).get("interaction_types") or ()):
            continue
        half = _half_extents(extents.get(key))
        if half is None:
            continue
        horizontal = min(float(half[0]), float(half[1]))
        if horizontal >= STRUCTURAL_SURFACE_MIN_HALF_EXTENT:
            out[key] = (
                f"collision half-extent {horizontal:.3f}m horizontal >= "
                f"{STRUCTURAL_SURFACE_MIN_HALF_EXTENT}m"
            )
    return out


def unclassified_supporters(
    extents: Dict[str, Any], members: Dict[str, Any],
) -> List[str]:
    """Supporters whose collision geometry the evidence never recorded.

    Reported rather than defaulted. Every one of them is a member the runtime
    will measure from its actor origin, and if one is a tabletop that is
    exactly the failure the classification exists to remove.
    """
    return sorted(
        key for key, entry in members.items()
        if "support" in ((entry or {}).get("interaction_types") or ())
        and _half_extents(extents.get(key)) is None
    )


def object_families(
    members: Dict[str, Any],
    holders: Set[str],
    supported: Set[str],
    structural: Iterable[str],
    declared_sites: Optional[Iterable[str]] = None,
) -> Dict[str, Optional[str]]:
    """The end-effector height family for every member, in strict precedence.

    1. A structural surface is one, whatever else it does. A tabletop is also
       grasped by nothing and supports everything, so the later rules would
       reach it too -- and give it the wrong answer.
    2. Anything the demos grasped is a manipuland. This is what the gripper
       approaches and lifts, and its heights are the ones the deadband has to
       resolve.
    3. Anything that held something else -- the supporter of a support pair or
       the container of a contain pair -- is a receptacle. Read from the
       directed pairs, not from ``interaction_types``, which is a flat set per
       member and says only that the object took part.
    4. Anything that rests on something else and holds nothing is also a
       manipuland. PullCubeTool grasps the *tool*, never the cube -- the cube
       is dragged by it -- so rule 2 misses the one object the task is about.
       Being supported while supporting nothing is what a scene's movable
       objects have in common, however they are moved.
    5. A member with no interactions that a reviewed declaration names as a
       site is a goal marker: PickCube's ``actor:goal_site`` has no collision
       geometry, so it appears in no pair and exists purely to be measured
       against.

    Anything else is left ``None``, **including a member with no interactions
    that no declaration names**. That case used to fall into rule 5, which is
    how a behaviour-free kitchen counter would have been labelled a goal
    marker and scored on that family's deadband -- silently, because rule 5
    returns a family rather than None. It is now ambiguous, and ambiguity
    stops the mine. See :func:`ambiguous_families`.

    Virtual ``spatial:`` sites are not passed here at all: they carry no body
    for the gripper to be near or above, and the miners exclude them before
    calling.
    """
    structural = set(structural)
    declared = set(declared_sites or ())
    out: Dict[str, Optional[str]] = {}
    for key, entry in members.items():
        types = set((entry or {}).get("interaction_types") or ())
        if key in structural:
            out[key] = spatial_metrics.FAMILY_STRUCTURAL
        elif "grasp" in types:
            out[key] = spatial_metrics.FAMILY_MANIPULAND
        elif key in holders:
            out[key] = spatial_metrics.FAMILY_RECEPTACLE
        elif key in supported:
            out[key] = spatial_metrics.FAMILY_MANIPULAND
        elif not types and key in declared:
            out[key] = spatial_metrics.FAMILY_GOAL_MARKER
        else:
            out[key] = None
    return out


def ambiguous_families(families: Dict[str, Optional[str]]) -> List[str]:
    """Members no rule classified.

    A member that took part in interactions but is neither grasped, nor a
    holder, nor structural. Falling back to a family would give it another
    family's deadband, which is how one token comes to mean two heights; the
    miner refuses to write the asset instead.
    """
    return sorted(key for key, family in families.items() if not family)


def directed_pairs(buckets: Iterable[str]) -> Tuple[Set[str], Set[str]]:
    """``(holders, supported)`` from ``<kind>/<holder>/<held>`` bucket names.

    The ManiSkill miner keys its evidence by that string; MS-HAB keeps the two
    sides as sets already. Both reach :func:`object_families` the same way.
    """
    holders: Set[str] = set()
    supported: Set[str] = set()
    for bucket in buckets:
        parts = [part.strip() for part in str(bucket).split("/")]
        if len(parts) == 3 and parts[0] in ("support", "contain"):
            holders.add(parts[1])
            supported.add(parts[2])
    return holders, supported


def reference_surface_from_supports(entry: Dict[str, Any]) -> Optional[Dict]:
    """The top face of a supporter, in its own object frame, or None.

    Derived from the support anchors already mined against it -- those points
    lie on the face things rest on, which is the plane by definition. The
    normal is forced outward here rather than only at read time, so the stored
    asset means what it says.

    Emitted for any object with support components, not only for classified
    surfaces: the runtime reads it only when the whitelist marks that member
    structural, and the classification is not known until the extents have
    been read. Without it, marking a counter structural makes
    ``reference_plane_world`` raise and the height edge vanish.
    """
    import numpy as np

    supports = (entry or {}).get("support_components") or []
    anchors = [np.asarray(c["surface_anchor"], dtype=float)
               for c in supports if c.get("surface_anchor") is not None]
    if not anchors:
        return None
    normals = [np.asarray(c["surface_normal"], dtype=float)
               for c in supports if c.get("surface_normal") is not None]
    normal = np.array([0.0, 0.0, 1.0])
    if normals:
        oriented = spatial_metrics.oriented_normal(
            np.mean(np.vstack(normals), axis=0))
        if oriented is not None:
            normal = oriented
    round6 = lambda v: [round(float(x), 6)
                        for x in np.asarray(v, dtype=float).reshape(-1)]
    return {
        "anchor": round6(np.mean(np.vstack(anchors), axis=0)),
        "outward_normal": round6(normal),
        "n_samples": len(anchors),
        "provenance": "mean of mined support-surface anchors, normal forced "
                      "outward (support normals are mined from contact "
                      "forces and point into the supporter)",
    }


def _half_extents(entry) -> Optional[List[float]]:
    """Half-extents out of either evidence shape, or None.

    ManiSkill stores the bare list under a key; the MS-HAB collector stores a
    record carrying the read status beside it, because "no extent" and "small"
    have to stay distinguishable.
    """
    if isinstance(entry, dict):
        entry = entry.get("half_extents")
    if not entry or len(entry) < 3:
        return None
    return list(entry)
