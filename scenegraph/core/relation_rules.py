"""Hyper-relational facts from privileged state.

One fact per admissible ``(src, relation, dst)``: absolute state ``sigma`` plus,
for spatial and affordance families, a change ``delta`` filled in by the
temporal buffer.

* physical state: ``contact`` (ee--obj, obj--obj), ``grasp`` (ee--obj),
  ``support`` / ``contain`` (obj--obj, directed). Binary and mutually
  independent -- a grasped object is also in contact.
* spatial (ee--obj): ``planar-distance``, ``height-offset``.
* goal-spatial: ``reached`` (obj--site). Binary, terminal, and measured
  against the environment's own tolerance rather than a mined bin, so it is
  emitted only for pairs a validated :class:`SiteSpec` declares.
* affordance: ``grasp-`` / ``contact-`` / ``support-`` / ``contain-
  compatibility``. Scored for every admissible instance; the near gate only
  picks the label, emitting ``unobserved`` when far. Scoring outside the gate
  is what keeps the temporal change continuous across it.

Admissible = both endpoints in frame (or one is the protected
target) and both carrying the whitelist
``interaction_types`` token; affordance also needs the mined components. The
one exception is the active subtask target, which the builder replays while it
is occluded and which therefore keeps all six of its end-effector facts. Other
facts touching a node that left the view are the graph builder's business.

Bin edges come from the mined whitelist alone (``cfg["bin_edges"]``). There is
no rule-based fallback: a relation the asset does not calibrate is not
emitted, so a token never means a hand-picked distance.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from . import spatial_metrics
from .affordance import (
    CompatibilityMeasurement,
    components_for_partner,
    compatibility_components,
    lookup_components,
    lookup_contact_components,
    lookup_contain_components,
    lookup_bottom_components,
    lookup_key_components,
    lookup_reference_surface,
    lookup_support_components,
    select_active_component,
    transform_anchors,
)
from .containment import (
    contain_compatibility,
    contain_holds,
    obj_contact_compatibility,
    support_compatibility,
)
from .sites import SiteSpec, reached_holds, site_distance
from .schema import Edge, Graph, Node
from ..adapters.privileged_state import PrivilegedState


# --------------------------------------------------------------------------- #
# Relation vocabulary
# --------------------------------------------------------------------------- #
PHYSICAL_RELATIONS: Tuple[str, ...] = ("contact", "grasp", "support", "contain")
SPATIAL_RELATIONS: Tuple[str, ...] = ("planar-distance", "height-offset")
AFFORDANCE_RELATIONS: Tuple[str, ...] = (
    "grasp-compatibility", "contact-compatibility",
    "support-compatibility", "contain-compatibility",
)
# Goal-spatial. Binary and terminal: the environment's own success geometry,
# not a mined scale, so it carries no bin key and no temporal change. Appended
# last on purpose -- inserting it beside the other spatial relations would
# shift every affordance relation id and silently invalidate trained heads.
GOAL_RELATIONS: Tuple[str, ...] = ("reached",)
RELATION_TYPES: Tuple[str, ...] = (
    PHYSICAL_RELATIONS + SPATIAL_RELATIONS + AFFORDANCE_RELATIONS
    + GOAL_RELATIONS
)

# Relations that carry a temporal-change label (mu^rho == 1).
TEMPORAL_RELATIONS = frozenset(SPATIAL_RELATIONS + AFFORDANCE_RELATIONS)

NOT_HOLDS = "not-holds"
HOLDS = "holds"
UNOBSERVED = "unobserved"
SRC_HOLDS = "src-holds"
DST_HOLDS = "dst-holds"

# Physical relations whose direction carries meaning.
DIRECTED_RELATIONS: Tuple[str, ...] = ("support", "contain")
# Their affordance twins. Role orientation comes from mined components, which
# are fixed for the episode, so these stay in COMPAT_LABELS and only collapse
# to a single emission.
DIRECTED_COMPAT_RELATIONS: Tuple[str, ...] = (
    "support-compatibility", "contain-compatibility",
)

PHYSICAL_LABELS: List[str] = [NOT_HOLDS, HOLDS]
DIRECTIONAL_PHYSICAL_LABELS: List[str] = [NOT_HOLDS, SRC_HOLDS, DST_HOLDS]
COMPAT_BIN_LABELS: List[str] = ["match", "partial-match", "poor-match"]
COMPAT_LABELS: List[str] = COMPAT_BIN_LABELS + [UNOBSERVED]
SPATIAL_LABELS: Dict[str, List[str]] = {
    "planar-distance": ["very-near", "near", "medium", "far", "very-far"],
    "height-offset": ["far-below", "below", "level", "above", "far-above"],
}
# Shared signed change vocabulary. Index 0 is the most negative change, so for
# distances it reads as approaching and for mismatch scores as fitting better.
CHANGE_LABELS: List[str] = [
    "decrease-fast", "decrease-slow", "stable", "increase-slow", "increase-fast",
]

# --------------------------------------------------------------------------- #
# Scoped calibration keys
# --------------------------------------------------------------------------- #
# The vocabulary stays two relations. The scale does not: an end-effector reach
# and a table-to-bin offset are different distributions, so one shared set of
# edges makes a token mean two distances.
EE_OBJECT_SCOPE = spatial_metrics.EE_OBJECT_SCOPE
OBJECT_OBJECT_SCOPE = spatial_metrics.OBJECT_OBJECT_SCOPE
SPATIAL_SCOPES: Tuple[str, ...] = spatial_metrics.SPATIAL_SCOPES
OBJECT_REGION_PLANAR_KEY = spatial_metrics.OBJECT_REGION_PLANAR_KEY
EE_HEIGHT_FAMILIES = spatial_metrics.EE_HEIGHT_FAMILIES
ee_family_bin_key = spatial_metrics.ee_family_bin_key
OBJECT_SITE_PLANAR_KEY = spatial_metrics.OBJECT_SITE_PLANAR_KEY
OBJECT_SITE_HEIGHT_KEY = spatial_metrics.OBJECT_SITE_HEIGHT_KEY
spatial_bin_key = spatial_metrics.spatial_bin_key
change_bin_key = spatial_metrics.change_bin_key

SCOPED_SPATIAL_KEYS: Tuple[str, ...] = tuple(
    spatial_bin_key(scope, relation)
    for scope in SPATIAL_SCOPES for relation in SPATIAL_RELATIONS
)

# Absolute-state vocabulary per relation, used by the encoder and the decoder.
# One edge per unordered pair in stable key order, with the role in the label,
# so a pair's direction never depends on whether the predicate holds.
def abs_labels_for() -> Dict[str, List[str]]:
    """Legal absolute labels per relation."""
    labels: Dict[str, List[str]] = {
        **{r: PHYSICAL_LABELS for r in PHYSICAL_RELATIONS},
        **SPATIAL_LABELS,
        **{r: COMPAT_LABELS for r in AFFORDANCE_RELATIONS},
        # Reuses the physical pair, so sigma does not grow.
        **{r: PHYSICAL_LABELS for r in GOAL_RELATIONS},
    }
    for relation in DIRECTED_RELATIONS:
        labels[relation] = DIRECTIONAL_PHYSICAL_LABELS
    return labels


ABS_LABELS: Dict[str, List[str]] = abs_labels_for()

# Label sets that pair with binned edges. Compatibility bins only cover the
# three scored labels; ``unobserved`` is assigned outside the binning path.
# Keyed by calibration key, not relation: the unscoped spatial names are gone
# so a pre-split asset cannot resolve and quietly label with the wrong scale.
_BIN_LABELS: Dict[str, List[str]] = {
    **{spatial_bin_key(scope, relation): labels
       for scope in SPATIAL_SCOPES
       for relation, labels in SPATIAL_LABELS.items()},
    **{change_bin_key(k): CHANGE_LABELS for k in SCOPED_SPATIAL_KEYS},
    **{r: COMPAT_BIN_LABELS for r in AFFORDANCE_RELATIONS},
    **{f"{r}-change": CHANGE_LABELS for r in AFFORDANCE_RELATIONS},
    # Planar only, registered by hand. See spatial_metrics: the scope tuple is
    # a cross-product generator and a region has no height target.
    OBJECT_REGION_PLANAR_KEY: SPATIAL_LABELS["planar-distance"],
    change_bin_key(OBJECT_REGION_PLANAR_KEY): CHANGE_LABELS,
    OBJECT_SITE_PLANAR_KEY: SPATIAL_LABELS["planar-distance"],
    change_bin_key(OBJECT_SITE_PLANAR_KEY): CHANGE_LABELS,
    OBJECT_SITE_HEIGHT_KEY: SPATIAL_LABELS["height-offset"],
    change_bin_key(OBJECT_SITE_HEIGHT_KEY): CHANGE_LABELS,
    **{ee_family_bin_key(f): SPATIAL_LABELS["height-offset"]
       for f in EE_HEIGHT_FAMILIES},
    **{change_bin_key(ee_family_bin_key(f)): CHANGE_LABELS
       for f in EE_HEIGHT_FAMILIES},
}


def required_bin_keys(cfg: dict) -> Tuple[str, ...]:
    """Calibration keys this configuration cannot label without.

    Scope-aware because MS-HAB emits no object-object spatial edges: requiring
    a height scale it never mines would reject every one of its whitelists.
    Object-object planar is still needed there, since the obj-obj
    compatibility near gate reads it. Change relations stay out -- an asset
    legitimately omits one the demos never moved.
    """
    keys = [spatial_bin_key(EE_OBJECT_SCOPE, "planar-distance")]
    families = sorted(set((cfg.get("families") or {}).values()))
    if families:
        keys.extend(ee_family_bin_key(f) for f in families)
    else:
        # Legacy asset: one shared end-effector height scale.
        keys.append(spatial_bin_key(EE_OBJECT_SCOPE, "height-offset"))
    keys.extend(AFFORDANCE_RELATIONS)
    # Required only where a region site exists to measure against. A task
    # without one must not be made to carry a scale nothing in it produces.
    if region_site_keys(cfg):
        keys.append(OBJECT_REGION_PLANAR_KEY)
    if ladder_site_keys(cfg):
        keys.extend((OBJECT_SITE_PLANAR_KEY, OBJECT_SITE_HEIGHT_KEY))
    oo_spatial = bool(cfg.get("object_object_spatial", False))
    oo_compat = bool(
        (cfg.get("affordances") or {}).get("object_object_compatibility", True)
    )
    if oo_spatial or oo_compat:
        keys.append(spatial_bin_key(OBJECT_OBJECT_SCOPE, "planar-distance"))
    if oo_spatial:
        keys.append(spatial_bin_key(OBJECT_OBJECT_SCOPE, "height-offset"))
    return tuple(keys)


def temporal_bin_key(relation: str) -> str:
    return f"{relation}-change"


def _planar_near_labels() -> Set[str]:
    labels = SPATIAL_LABELS["planar-distance"]
    if len(labels) >= 5:
        return set(labels[:2])
    return {labels[0]}


def bin_label(value: float, edges: List[float], labels: List[str]) -> str:
    """Upper-exclusive ascending bins. ``len(labels) == len(edges) + 1``."""
    idx = int(np.searchsorted(edges, value, side="right"))
    idx = min(idx, len(labels) - 1)
    return labels[idx]


def _get_bin_spec(cfg: dict, relation: str) -> Optional[Tuple[List[float], List[str]]]:
    """Resolve ``(edges, labels)`` for ``relation`` from the mined bins.

    ``cfg["bin_edges"]`` is the only source. The graph builder has already
    refused to run without a mined union asset and checked that it calibrates
    every absolute relation, so a None here means a change relation the demos
    never moved -- that relation carries no label rather than borrowing one
    from a hand-written scale.
    """
    edges = (cfg.get("bin_edges") or {}).get(relation)
    if not edges:
        return None
    labels = _BIN_LABELS.get(relation)
    if labels is None or len(labels) != len(edges) + 1:
        return None
    return list(edges), list(labels)


def _compat_norm(cfg: dict) -> Dict[str, float]:
    """Per-component normalizers (metres / radians) shared by all compat scorers.

    Defaults:
      pos      = 0.10 m   (close-approach distance)
      orient   = pi / 2   (90deg half-cone)
      width    = 0.04 m   (gripper spread)
      xy       = 0.05 m   (in-plane support offset)
      vertical = 0.03 m   (support gap/interpenetration)
      radial   = 0.02 m   (contain radial slack)
      axial    = 0.03 m   (contain axial slack)
    """
    norm = cfg.get("compat_norm") or {}
    out = {
        "pos":      float(norm.get("pos",      0.10)),
        "orient":   float(norm.get("orient",   np.pi / 2.0)),
        "width":    float(norm.get("width",    0.04)),
        "xy":       float(norm.get("xy",       0.05)),
        "vertical": float(norm.get("vertical", 0.03)),
        "radial":   float(norm.get("radial",   0.02)),
        "axial":    float(norm.get("axial",    0.03)),
    }
    for k, v in out.items():
        if not np.isfinite(v) or v <= 0:
            out[k] = 1.0
    return out


def interaction_types(node: Node) -> Set[str]:
    """Mined interaction tokens for one node, written by the whitelist gate."""
    return set(node.attributes.get("interaction_types") or ())


# Pose arrays are ``[x, y, z, qw, qx, qy, qz]`` (SAPIEN).
def _xyz(node: Node) -> Optional[np.ndarray]:
    if node.pose_world is None:
        return None
    return np.asarray(node.pose_world[:3], dtype=float)


def planar_distance_xyz(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def height_offset_xyz(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[2] - b[2])


def _whitelist_key(node: Node) -> Optional[str]:
    key = node.attributes.get("whitelist_key")
    return str(key) if key else None


def support_anchor_spec(a: Node, b: Node, aff_set):
    """``(anchor_a, radial_a, anchor_b, radial_b)`` for the pair, or None.

    Link origins are not on the contact surface -- a table's sits ~0.9m below
    its own top -- so origin geometry reports a bin resting on a table as far
    above it. The anchors keep describing the pair after physical support
    ends, so a lift changes height without changing what height means.
    """
    if aff_set is None or a.pose_world is None or b.pose_world is None:
        return None
    for supporter, supported in ((a, b), (b, a)):
        sup = components_for_partner(
            lookup_support_components(aff_set, supporter),
            _whitelist_key(supported),
        )
        bot = components_for_partner(
            lookup_bottom_components(aff_set, supported),
            _whitelist_key(supporter),
        )
        if len(sup) != 1 or len(bot) != 1:
            continue
        s_anchor = getattr(sup[0], "surface_anchor_obj_frame", None)
        b_anchor = getattr(bot[0], "bottom_anchor_obj_frame", None)
        radial = getattr(bot[0], "radial_offset", None)
        if s_anchor is None or (b_anchor is None and radial is None):
            continue
        if supporter is a:
            return s_anchor, None, b_anchor, radial
        return b_anchor, radial, s_anchor, None
    return None


def object_pair_measures(a: Node, b: Node, aff_set) -> Tuple[float, float]:
    """``(planar, signed height)`` for an object pair, in ``a - b`` order."""
    spec = support_anchor_spec(a, b, aff_set) or (None, None, None, None)
    points = spatial_metrics.pair_points(
        a.pose_world, b.pose_world, spec[0], spec[2], spec[1], spec[3])
    return spatial_metrics.measures(*points)


def planar_distance(a: Node, b: Node) -> Optional[float]:
    pa, pb = _xyz(a), _xyz(b)
    if pa is None or pb is None:
        return None
    return planar_distance_xyz(pa, pb)


def height_offset(a: Node, b: Node) -> Optional[float]:
    pa, pb = _xyz(a), _xyz(b)
    if pa is None or pb is None:
        return None
    return height_offset_xyz(pa, pb)


# --------------------------------------------------------------------------- #
# Structural surfaces
# --------------------------------------------------------------------------- #
def structural_surface_keys(cfg: dict) -> Set[str]:
    """Whitelist keys the asset marked as extended support planes."""
    return set(cfg.get("structural_surfaces") or ())


def node_family(node: Node, cfg: dict) -> Optional[str]:
    """The mined end-effector height family for one node, or None.

    None means the asset never classified it. Callers that need a scale raise
    rather than picking one: an unclassified member silently borrowing another
    family's deadband is how a token comes to mean two different heights.
    """
    families = cfg.get("families") or {}
    key = _whitelist_key(node)
    return families.get(key) if key else None


def ee_height_bin_key(node: Node, cfg: dict) -> str:
    """Which height scale labels this end-effector pair.

    Falls back to the single shared scale only when the asset declares no
    families at all -- a legacy MS-HAB whitelist, which was mined before the
    split and whose members were never classified. A families-aware asset that
    omits one member is an error, not a reason to reach for the old scale.
    """
    families = cfg.get("families") or {}
    if not families:
        return spatial_bin_key(EE_OBJECT_SCOPE, "height-offset")
    family = node_family(node, cfg)
    if not family:
        raise ValueError(
            f"{_whitelist_key(node)!r} has no mined end-effector height "
            "family, but this asset classifies its other members. Labelling "
            "it on another family's scale would make 'level' mean two "
            "different heights in one graph. Re-mine the task."
        )
    return ee_family_bin_key(family)


def region_site_keys(cfg: dict) -> Set[str]:
    """Declared sites whose geometry is a region rather than a point."""
    from .sites import SITE_REGION
    return {
        key for key, decl in (cfg.get("site_declarations") or {}).items()
        if getattr(decl, "site_type", None) == SITE_REGION
    }


def is_region_site(node: Node, cfg: dict) -> bool:
    key = _whitelist_key(node)
    return bool(key) and key in region_site_keys(cfg)


def is_virtual_site(node: Node) -> bool:
    """Whether this node stands for goal geometry with no simulator body.

    Keyed on the ``spatial:`` namespace rather than on a runtime flag, because
    the distinction that matters is whether an actor backs it. PickCube's
    ``actor:goal_site`` is a real kinematic sphere with pixels and a mined
    family; a hole mouth and a goal region are neither.
    """
    from .sites import SITE_PREFIX
    key = _whitelist_key(node)
    return bool(key) and key.startswith(SITE_PREFIX)


def ladder_site_keys(cfg: dict) -> Set[str]:
    """Virtual sites that carry a distance ladder.

    Virtual only: PickCube's goal marker is a real actor with real pixels and
    an object-object ladder that already works, and re-scoping it would mean
    re-mining a task nothing is wrong with. Regions are excluded because they
    have their own planar-only scale and no height target.
    """
    from .sites import SITE_PREFIX, SITE_REGION
    return {
        key for key, decl in (cfg.get("site_declarations") or {}).items()
        if str(key).startswith(SITE_PREFIX)
        and getattr(decl, "site_type", None) != SITE_REGION
    }


def is_ladder_site(node: Node, cfg: dict) -> bool:
    key = _whitelist_key(node)
    return bool(key) and key in ladder_site_keys(cfg)


def is_declared_site_pair(cfg: dict, site: Node, other: Node) -> bool:
    """Whether this pair is the one the site's declaration names.

    A site node pairs with every other object in the scene, but only one of
    those pairs was declared, calibrated and scheduled. PullCubeTool's goal
    region emitted a second edge for the *tool*, labelled on a scale mined
    from the cube -- an extra fact in the world-model target that no schedule
    reads and no sample calibrated. The miner already filters to the declared
    pair; this is the same rule on the emitting side.
    """
    decl = (cfg.get("site_declarations") or {}).get(_whitelist_key(site))
    if decl is None:
        return False
    return _whitelist_key(other) == getattr(decl, "subject_key", None)


def _spec_for(cfg: dict, key: str):
    for spec in cfg.get("site_specs") or ():
        if spec.key == key:
            return spec
    return None


def is_structural_surface(node: Node, cfg: dict) -> bool:
    key = _whitelist_key(node)
    return bool(key) and key in structural_surface_keys(cfg)


def reference_plane_world(node: Node, aff_set):
    """``(anchor world, outward normal world)`` for a structural surface.

    Raises rather than falling back. A member the asset called structural but
    never gave a plane cannot be measured against its surface, and the only
    available fallback -- the actor origin -- is the ~0.9m error this whole
    change exists to remove. Failing closed here is what makes the mined
    property load-bearing instead of advisory.
    """
    surface = lookup_reference_surface(aff_set, node) if aff_set else None
    if surface is None:
        raise ValueError(
            f"{_whitelist_key(node)!r} is marked structural_surface but its "
            "affordance asset carries no 'reference_surface'. Height would "
            "fall back to the actor origin, which for a table sits ~0.9m "
            "below its own top. Re-mine the task's assets."
        )
    if node.pose_world is None:
        return None
    anchor = spatial_metrics.anchor_world(
        node.pose_world, surface.anchor_obj_frame)
    rot = _quat_rot(node.pose_world)
    if anchor is None or rot is None:
        return None
    normal = rot @ np.asarray(
        surface.outward_normal_obj_frame, dtype=float).reshape(3)
    return anchor, normal


def _quat_rot(pose7) -> Optional[np.ndarray]:
    pose = np.asarray(pose7, dtype=float).reshape(-1)
    if pose.size < 7 or not np.all(np.isfinite(pose[:7])):
        return None
    w, x, y, z = pose[3:7]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def _bottom_point(node: Node, partner_key: Optional[str], aff_set):
    """The node's lower contact point in world, or its origin.

    A sphere's bottom is ``centre - r * up`` whatever its quaternion says, so a
    ball rolling in place keeps a constant height above the table.
    """
    comps = components_for_partner(
        lookup_bottom_components(aff_set, node), partner_key) if aff_set else []
    if len(comps) == 1:
        point = spatial_metrics.anchor_world(
            node.pose_world,
            getattr(comps[0], "bottom_anchor_obj_frame", None),
            getattr(comps[0], "radial_offset", None),
        )
        if point is not None:
            return point
    return _xyz(node)


def surface_relative_height(
    surface_node: Node, point: np.ndarray, aff_set,
) -> Optional[float]:
    """Height of a world point above a structural surface's plane."""
    plane = reference_plane_world(surface_node, aff_set)
    if plane is None or point is None:
        return None
    return spatial_metrics.surface_height(point, plane[0], plane[1])


def _resolve_entity(
    node: Node, state: PrivilegedState, graph: Optional[Graph] = None,
    cfg: Optional[dict] = None,
):
    """Node -> live simulator entity (for force queries).

    A retained node carries no segmentation ids -- nothing saw it this frame --
    so the seg-id lookup finds nothing exactly when it matters most, with the
    object in the gripper hiding itself. The builder's cache holds the
    association made when the node was first seen and is consulted first;
    ``state.active_obj`` remains the fallback for the active target alone.

    Returning None here is not benign: a force query on None reads zero, which
    would be emitted as a confident ``not-holds``. Callers must not reach this
    for a node the builder never resolved.
    """
    if cfg is not None:
        cached = (cfg.get("_entity_cache") or {}).get(node.node_id)
        if cached is not None:
            return cached
    name = node.name
    for seg_id in node.segmentation_ids:
        ent = state.seg_id_map.get(seg_id)
        if ent is not None and getattr(ent, "name", None) == name:
            return ent
    ents = [state.seg_id_map.get(s) for s in node.segmentation_ids]
    ents = [e for e in ents if e is not None]
    if not ents:
        return _active_target_entity(node, state, graph)
    best_ent, best_count = None, 0
    for ent in ents:
        count = sum(1 for e in ents if e is ent)
        if count > best_count:
            best_ent, best_count = ent, count
    return best_ent


def _active_target_entity(
    node: Node, state: PrivilegedState, graph: Optional[Graph]
):
    """``state.active_obj``, but only for the exact node the builder flagged."""
    if graph is None or state.active_obj is None:
        return None
    target_id = graph.meta.get("active_target_node_id")
    if not target_id or node.node_id != target_id:
        return None
    return state.active_obj


def _resolve_active_anchor(node: Node, state: PrivilegedState, cfg: dict):
    """``(anchor_world (3,), component, a_star)`` or ``(None, None, None)``.

    Component index is cached per node and reused across frames so the
    compatibility reference does not jump mid-rollout. The world anchor is
    re-derived each frame from the node's current pose.
    """
    aff_set = cfg.get("affordance_set")
    if aff_set is None or getattr(aff_set, "is_empty", lambda: True)():
        return None, None, None
    if state.tcp_pose_world is None or node.pose_world is None:
        return None, None, None
    tcp_world = np.asarray(state.tcp_pose_world[:3], dtype=float)
    if tcp_world.shape[0] != 3 or not np.all(np.isfinite(tcp_world)):
        return None, None, None

    comps = lookup_components(aff_set, node)
    if not comps:
        return None, None, None
    anchors_world = transform_anchors(node.pose_world, comps)
    if anchors_world is None:
        return None, None, None
    cache = cfg.setdefault("_affordance_selection_cache", {})
    cached = cache.get(node.node_id)
    if isinstance(cached, int) and 0 <= cached < len(comps):
        a_star = cached
    else:
        tcp_axis_local = cfg["grasp"].get(
            "tcp_approach_axis_local", [0.0, 0.0, 1.0]
        )
        orientation_weight = float(
            cfg.get("affordances", {}).get("orientation_selection_weight", 0.10)
        )
        a_star = select_active_component(
            tcp_world,
            anchors_world,
            components=comps,
            obj_pose_world=node.pose_world,
            tcp_pose_world=state.tcp_pose_world,
            tcp_axis_local=tcp_axis_local,
            orientation_weight=orientation_weight,
        )
        if a_star is not None:
            cache[node.node_id] = int(a_star)
    if a_star is None:
        return None, None, None

    node.attributes["affordance_a_star"] = int(a_star)
    return anchors_world[a_star], comps[a_star], a_star


def _compatibility_score(
    meas: CompatibilityMeasurement, norm: Dict[str, float], *, include_width: bool,
) -> float:
    """Unweighted mean of per-component [0,1] mismatches.

    Missing components (orientation without ``approach_dir``, width without
    gripper qpos) are skipped from the average rather than treated as 1.
    """
    parts: List[float] = []
    parts.append(min(meas.pos_mismatch / norm["pos"], 1.0))
    if meas.orient_mismatch is not None:
        parts.append(min(meas.orient_mismatch / norm["orient"], 1.0))
    if include_width and meas.width_mismatch is not None:
        parts.append(min(meas.width_mismatch / norm["width"], 1.0))
    if not parts:
        return 1.0
    return float(np.mean(parts))


def _mean_normalized(parts: List[float]) -> float:
    if not parts:
        return 1.0
    return float(np.mean([min(max(p, 0.0), 1.0) for p in parts]))


def _compat_edge(
    src: str, dst: str, relation: str, score: Optional[float],
    near: bool, spec, attributes: Optional[dict] = None,
) -> Edge:
    """One affordance fact. The score is kept as ``raw_value`` even when the
    near gate labels it ``unobserved``, so the temporal change stays continuous
    across the gate."""
    if score is None or not near:
        label = UNOBSERVED
    else:
        label = bin_label(score, spec[0], spec[1])
    return Edge(
        src, dst, relation, label, raw_value=score,
        attributes=dict(attributes or {}),
    )


def _eligible_objects(graph: Graph) -> List[Node]:
    """Objects a camera covers this frame, whether or not pixels survived.

    ``in_frame``, not ``visible``: an object the robot arm hides is still one
    the cameras cover, and its facts are still derivable from the view. What is
    excluded is an object outside every frustum, where emitting a fact would
    supervise something no observation contains.
    """
    return [
        n for n in graph.nodes
        if n.node_type == "object" and n.in_frame and n.pose_world is not None
    ]


def _protected_target(graph: Graph) -> Optional[Node]:
    """The active target when it is out of frame, or None.

    The one node whose facts continue regardless: the progress ladder reads
    them every step, and the subtask does not pause because a camera turned
    away.
    """
    target_id = graph.meta.get("active_target_node_id")
    if not target_id:
        return None
    node = graph.get_node(target_id)
    if node is None or node.node_type != "object" or node.pose_world is None:
        return None
    return None if node.in_frame else node


def _ee_object_nodes(graph: Graph) -> List[Node]:
    """Objects the end-effector relations are computed against.

    Every eligible object, plus the protected target whether or not a camera
    covers it. The facts the progress ladder reads are all defined from the
    current end-effector pose against the target's current pose, and both are
    known while it is hidden.
    """
    nodes = _eligible_objects(graph)
    retained = _protected_target(graph)
    return nodes if retained is None else nodes + [retained]


# --------------------------------------------------------------------------- #
# ee -> object
# --------------------------------------------------------------------------- #
def ee_object_spatial_edges(
    graph: Graph, state: PrivilegedState, cfg: dict
) -> List[Edge]:
    """Object-center spatial facts for every visible object, plus the target.

    Both facts are recomputed from the *current* end-effector position each
    frame, so an occluded target's distance and height keep changing as the
    robot moves even though its own centroid does not.
    """
    ee = graph.get_node("ee")
    if ee is None or ee.pose_world is None:
        return []
    ee_xyz = np.asarray(ee.pose_world[:3], dtype=float)
    pd_key = spatial_bin_key(EE_OBJECT_SCOPE, "planar-distance")
    pd_spec = _get_bin_spec(cfg, pd_key)

    aff_set = cfg.get("affordance_set")
    edges: List[Edge] = []
    for node in _ee_object_nodes(graph):
        # A virtual site has no body for the gripper to be near or above. It
        # is also, deliberately, given no height family by the miner, so
        # asking for one raises -- correctly, since inventing a family would
        # label a distance to nothing on a real object's scale.
        if is_virtual_site(node):
            continue
        obj_xyz = _xyz(node)
        structural = is_structural_surface(node, cfg)
        # Per family, so a metre of table clearance cannot set the deadband
        # that a two-centimetre lift has to register against.
        ho_key = ee_height_bin_key(node, cfg)
        ho_spec = _get_bin_spec(cfg, ho_key)
        # A structural surface's origin names no place to approach: it is the
        # centre of a metre-wide plane, so the planar distance to it says
        # nothing the policy can act on. The height above it says a great deal.
        if pd_spec is not None and not structural:
            d = planar_distance_xyz(ee_xyz, obj_xyz)
            edges.append(Edge(
                "ee", node.node_id, "planar-distance",
                bin_label(d, pd_spec[0], pd_spec[1]), raw_value=d,
                bin_key=pd_key,
            ))
        if ho_spec is not None:
            if structural:
                dz = surface_relative_height(node, ee_xyz, aff_set)
                if dz is None:
                    continue
            else:
                dz = height_offset_xyz(ee_xyz, obj_xyz)
            edges.append(Edge(
                "ee", node.node_id, "height-offset",
                bin_label(dz, ho_spec[0], ho_spec[1]), raw_value=dz,
                bin_key=ho_key,
            ))
    return edges


def ee_object_physical_edges(
    graph: Graph, state: PrivilegedState, cfg: dict
) -> List[Edge]:
    """Binary ``contact`` and ``grasp`` facts, gated on the mined tokens.

    The two are independent: a grasped object reports ``grasp: holds`` and
    ``contact: holds`` at the same time.

    Both are live simulator queries, so the retained target answers them while
    invisible -- which is the case that matters, since an object in the gripper
    is usually an object the cameras have lost.
    """
    ee = graph.get_node("ee")
    if ee is None or ee.pose_world is None:
        return []
    eps_contact = cfg["contact"]["eps_force"]
    grasp_angle = cfg["grasp"]["max_angle"]

    edges: List[Edge] = []
    for node in _ee_object_nodes(graph):
        types = interaction_types(node)
        emit_contact = "contact" in types
        emit_grasp = "grasp" in types
        if not emit_contact and not emit_grasp:
            continue
        ent = _resolve_entity(node, state, graph, cfg)
        if emit_contact:
            force = state.ee_object_contact_force(ent)
            edges.append(Edge(
                "ee", node.node_id, "contact",
                HOLDS if force > eps_contact else NOT_HOLDS, raw_value=force,
            ))
        if emit_grasp:
            grasped = state.is_grasping(ent, max_angle=grasp_angle)
            edges.append(Edge(
                "ee", node.node_id, "grasp",
                HOLDS if grasped else NOT_HOLDS, raw_value=float(bool(grasped)),
            ))
    return edges


def ee_object_affordance_edges(
    graph: Graph, state: PrivilegedState, cfg: dict
) -> List[Edge]:
    """ee--object affordance facts.

    Admissible when the object carries the mined token and a grasp component
    resolves. The compatibility score is computed for every admissible object
    regardless of distance; the near gate only decides whether the label is a
    score bin or ``unobserved``.

    The retained target is admissible too: the anchor is re-derived from its
    retained pose and scored against the current TCP, so both compatibility
    rungs stay live through occlusion.
    """
    ee = graph.get_node("ee")
    if ee is None or ee.pose_world is None or state.tcp_pose_world is None:
        return []
    ee_xyz = np.asarray(ee.pose_world[:3], dtype=float)
    pd_spec = _get_bin_spec(
        cfg, spatial_bin_key(EE_OBJECT_SCOPE, "planar-distance"))
    if pd_spec is None:
        return []
    grasp_spec = _get_bin_spec(cfg, "grasp-compatibility")
    contact_spec = _get_bin_spec(cfg, "contact-compatibility")
    if grasp_spec is None and contact_spec is None:
        return []
    near_labels = _planar_near_labels()
    norm = _compat_norm(cfg)
    tcp_axis_local = cfg["grasp"].get("tcp_approach_axis_local", [0.0, 0.0, 1.0])
    gripper_width = getattr(state, "gripper_width", None)

    edges: List[Edge] = []
    for node in _ee_object_nodes(graph):
        types = interaction_types(node)
        emit_grasp = grasp_spec is not None and "grasp" in types
        emit_contact = contact_spec is not None and "contact" in types
        if not emit_grasp and not emit_contact:
            continue

        anchor_world, comp, a_star = _resolve_active_anchor(node, state, cfg)
        if anchor_world is None or comp is None or a_star is None:
            continue

        obj_xyz = _xyz(node)
        d = planar_distance_xyz(ee_xyz, obj_xyz)
        near = bin_label(d, pd_spec[0], pd_spec[1]) in near_labels

        meas = compatibility_components(
            comp, a_star, anchor_world,
            obj_pose_world=node.pose_world,
            tcp_pose_world=state.tcp_pose_world,
            tcp_axis_local=tcp_axis_local,
            gripper_width=gripper_width,
        )

        if emit_grasp:
            score = _compatibility_score(meas, norm, include_width=True)
            edges.append(_compat_edge(
                "ee", node.node_id, "grasp-compatibility",
                score, near, grasp_spec,
            ))
        if emit_contact:
            score = _compatibility_score(meas, norm, include_width=False)
            edges.append(_compat_edge(
                "ee", node.node_id, "contact-compatibility",
                score, near, contact_spec,
            ))
    return edges


# --------------------------------------------------------------------------- #
# object -> object
# --------------------------------------------------------------------------- #
def pair_sort_key(node: Node) -> Tuple[str, str]:
    """Cross-episode ordering key. ``whitelist_key`` is semantic and stable;
    ``node_id`` separates instances that share one key."""
    attrs = node.attributes or {}
    return (str(attrs.get("whitelist_key") or ""), str(node.node_id))


def is_dynamic(node: Node) -> bool:
    """Whether physics can move this body.

    Unknown reads as dynamic: a missing body type must not silently
    delete facts.
    """
    body_type = (node.attributes or {}).get("body_type")
    if not body_type:
        return True
    return str(body_type) == "dynamic"


def _immobile_pair(a: Node, b: Node) -> bool:
    """Neither endpoint can move, so the pair is scene layout.

    PlaceSphere's bin and table are both kinematic. PhysX solves no
    contact between them, so no anchor is ever mined and the pair falls
    back to link origins -- and a table origin sits ~0.9m below its own
    top. Emitting that costs an edge slot to say the same wrong thing
    every frame, and mining it sets the height scale for every pair
    that does move.
    """
    return not is_dynamic(a) and not is_dynamic(b)


def _object_pairs(graph: Graph) -> List[Tuple[Node, Node]]:
    """Unordered object pairs in stable key order.

    Sorted by ``pair_sort_key`` rather than by registry position, which shifts
    as nodes arrive. A pair's ``(a, b)`` orientation has to mean the same thing
    every frame for a single stored edge to be readable. Pairs neither
    endpoint can move are dropped: nothing the policy does changes them.
    """
    objs = sorted(_eligible_objects(graph), key=pair_sort_key)
    out: List[Tuple[Node, Node]] = []
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            if _immobile_pair(objs[i], objs[j]):
                continue
            out.append((objs[i], objs[j]))
    # The protected target pairs with anything in frame even when it is not.
    # A place subtask's defining fact is target-to-receptacle, and losing it
    # whenever the camera turns away would delete the evidence mid-subtask.
    # Two out-of-frame endpoints stay unpaired: nothing observed either.
    target = _protected_target(graph)
    if target is not None:
        for other in objs:
            out.append((other, target)
                       if pair_sort_key(other) < pair_sort_key(target)
                       else (target, other))
    return out


def _pair_planar_distance(a: Node, b: Node) -> float:
    """Center-to-center planar distance in metres. Uses the object-frame origin;
    conservative for the physics short-circuit because true contact distance is
    always <= center distance + summed extents."""
    return float(np.linalg.norm(
        np.asarray(a.pose_world[:2], dtype=float)
        - np.asarray(b.pose_world[:2], dtype=float)
    ))


def _both(types_a: Set[str], types_b: Set[str], token: str) -> bool:
    return token in types_a and token in types_b


def object_object_physical_edges(
    graph: Graph, state: PrivilegedState, cfg: dict
) -> List[Edge]:
    """Binary ``contact`` / ``support`` / ``contain`` facts for visible pairs.

    All three are evaluated independently -- a supported object in contact with
    its supporter reports both. Each emits once per pair in stable key order;
    ``support`` and ``contain`` name the role in the label, so a pair's
    direction never depends on whether the predicate holds.

    Pairs whose centers exceed the maximum plausible contact distance skip the
    SAPIEN force query and report ``not-holds`` directly: two rigid bodies
    cannot exert a contact force at that separation.
    """
    eps_contact = cfg["contact"]["eps_force"]
    min_vertical_ratio = cfg["support"].get("min_vertical_force_ratio", 0.5)
    pair_force_max_distance = float(cfg.get("pair_force_max_distance", 2.0))
    aff_set = cfg.get("affordance_set")
    edges: List[Edge] = []
    for a, b in _object_pairs(graph):
        ta, tb = interaction_types(a), interaction_types(b)
        want_contact = _both(ta, tb, "contact")
        want_support = _both(ta, tb, "support")
        want_contain = _both(ta, tb, "contain")
        if not (want_contact or want_support or want_contain):
            continue

        if want_contain and aff_set is not None:
            oriented = []
            for container, containee in ((a, b), (b, a)):
                container_comps = lookup_contain_components(aff_set, container)
                key_comps = lookup_key_components(aff_set, containee)
                if not container_comps or not key_comps:
                    continue
                held = any(
                    contain_holds(
                        container.pose_world, cc, containee.pose_world, kc,
                    )
                    for cc in container_comps for kc in key_comps
                )
                oriented.append((container, containee, held))
            if len(oriented) > 1:
                raise ValueError(
                    f"contain role is ambiguous for "
                    f"{a.node_id!r}/{b.node_id!r}: both orientations have "
                    "mined components. Extend the schema rather than "
                    "emitting two edges."
                )
            if oriented:
                container, _, held = oriented[0]
                label = NOT_HOLDS
                if held:
                    label = SRC_HOLDS if container is a else DST_HOLDS
                edges.append(Edge(
                    a.node_id, b.node_id, "contain", label,
                    raw_value=float(held),
                    attributes={"contain_role": "container"},
                ))

        if not (want_contact or want_support):
            continue

        far = (
            pair_force_max_distance > 0.0
            and _pair_planar_distance(a, b) > pair_force_max_distance
        )
        if far:
            force = 0.0
            force_vector = np.zeros(3, dtype=float)
        else:
            force_vector = np.asarray(
                state.pairwise_force_vector(
                    _resolve_entity(a, state, graph, cfg),
                    _resolve_entity(b, state, graph, cfg),
                ), dtype=float,
            )
            force = float(np.linalg.norm(force_vector))
        in_contact = force > eps_contact

        if want_contact:
            edges.append(Edge(
                a.node_id, b.node_id, "contact",
                HOLDS if in_contact else NOT_HOLDS, raw_value=force,
            ))

        if want_support:
            # Direction from the contact force sign, not pose-center dz.
            # Link-frame origins are usually not at the contact surface (a
            # drawer's origin sits at the drawer front, not on its top face), so
            # "supporter = lower-z endpoint" flips whenever a tall/thin
            # supporter carries a short/wide supported object. ManiSkill's
            # ``get_pairwise_contact_forces`` returns "force on ``a`` due to
            # ``b``" (see mani_skill/envs/scene.py:789), so:
            #   fz < 0  -> b's weight pushes a down -> a is supporter
            #   fz > 0  -> reaction pushes a up     -> b is supporter
            supporter = None
            if in_contact and force > 0.0:
                if abs(float(force_vector[2])) / force >= min_vertical_ratio:
                    supporter = a if float(force_vector[2]) < 0.0 else b
            label = NOT_HOLDS
            if supporter is a:
                label = SRC_HOLDS
            elif supporter is b:
                label = DST_HOLDS
            edges.append(Edge(
                a.node_id, b.node_id, "support", label,
                raw_value=force if supporter is not None else 0.0,
                attributes={"support_role": "supporter"},
            ))
    return edges


def object_object_affordance_edges(
    graph: Graph, state: PrivilegedState, cfg: dict,
    skip_pairs: Optional[Set[Tuple[str, str]]] = None,
) -> List[Edge]:
    """obj--obj affordance facts: contact / support / contain compatibility.

    Admissible when both endpoints carry the mined token and the matching
    components exist. Pairs beyond ``object_object_compat_max_distance`` skip
    scoring and report ``unobserved`` with no value, so no temporal change is
    accumulated for a pair that is nowhere near interacting. ``skip_pairs``
    contains unordered node-id pairs whose physical relation was already true
    at episode start; those are static scene layout, not affordances to pursue.
    """
    pd_spec = _get_bin_spec(
        cfg, spatial_bin_key(OBJECT_OBJECT_SCOPE, "planar-distance"))
    if pd_spec is None:
        return []
    near_labels = _planar_near_labels()
    aff_set = cfg.get("affordance_set")
    if aff_set is None:
        return []

    aff_cfg = cfg.get("affordances", {})
    contact_spec = (
        _get_bin_spec(cfg, "contact-compatibility")
        if bool(aff_cfg.get("object_object_contact_compatibility", True))
        else None
    )
    support_spec = (
        _get_bin_spec(cfg, "support-compatibility")
        if bool(aff_cfg.get("object_object_support_compatibility", False))
        else None
    )
    contain_spec = (
        _get_bin_spec(cfg, "contain-compatibility")
        if bool(aff_cfg.get("object_object_contain_compatibility", True))
        else None
    )
    if contact_spec is None and support_spec is None and contain_spec is None:
        return []

    max_distance = float(aff_cfg.get("object_object_compat_max_distance", 2.0))
    norm = _compat_norm(cfg)

    edges: List[Edge] = []
    for a, b in _object_pairs(graph):
        pair = tuple(sorted((a.node_id, b.node_id)))
        if skip_pairs is not None and pair in skip_pairs:
            continue
        ta, tb = interaction_types(a), interaction_types(b)
        # Same metric the spatial edges report, so "near" means the distance
        # the label names rather than an origin-to-origin one.
        d, _ = object_pair_measures(a, b, aff_set)
        scored = max_distance <= 0.0 or d <= max_distance
        near = scored and bin_label(d, pd_spec[0], pd_spec[1]) in near_labels

        if contact_spec is not None and _both(ta, tb, "contact"):
            a_comps = lookup_contact_components(aff_set, a)
            b_comps = lookup_contact_components(aff_set, b)
            if a_comps and b_comps:
                score = None
                if scored:
                    meas = obj_contact_compatibility(
                        a.pose_world, a_comps, b.pose_world, b_comps,
                        a.attributes.get("whitelist_key"),
                        b.attributes.get("whitelist_key"),
                    )
                    if meas is not None:
                        parts = [meas.pos_mismatch / norm["pos"]]
                        if meas.orient_mismatch is not None:
                            parts.append(meas.orient_mismatch / norm["orient"])
                        score = _mean_normalized(parts)
                edges.append(_compat_edge(
                    a.node_id, b.node_id, "contact-compatibility",
                    score, near, contact_spec,
                ))

        if support_spec is not None and _both(ta, tb, "support"):
            oriented = []
            for supporter, supported in ((a, b), (b, a)):
                sup_comps = components_for_partner(
                    lookup_support_components(aff_set, supporter),
                    _whitelist_key(supported))
                bot_comps = components_for_partner(
                    lookup_bottom_components(aff_set, supported),
                    _whitelist_key(supporter))
                if sup_comps and bot_comps:
                    oriented.append((supporter, supported, sup_comps, bot_comps))
            if len(oriented) > 1:
                raise ValueError(
                    f"support-compatibility role is ambiguous for "
                    f"{a.node_id!r}/{b.node_id!r}: both orientations have "
                    "mined components. Extend the schema rather than emitting "
                    "two edges."
                )
            for supporter, supported, sup_comps, bot_comps in oriented:
                score = None
                if scored:
                    meas = support_compatibility(
                        supporter.pose_world, sup_comps,
                        supported.pose_world, bot_comps,
                    )
                    if meas is not None:
                        parts = [
                            meas.xy_mismatch / norm["xy"],
                            meas.vertical_mismatch / norm["vertical"],
                        ]
                        if meas.orient_mismatch is not None:
                            parts.append(meas.orient_mismatch / norm["orient"])
                        score = _mean_normalized(parts)
                edges.append(_compat_edge(
                    supporter.node_id, supported.node_id, "support-compatibility",
                    score, near, support_spec,
                    attributes={"support_role": "supporter"},
                ))

        if contain_spec is not None and _both(ta, tb, "contain"):
            oriented = []
            for container, containee in ((a, b), (b, a)):
                con_comps = lookup_contain_components(aff_set, container)
                key_comps = lookup_key_components(aff_set, containee)
                if con_comps and key_comps:
                    oriented.append((container, containee, con_comps, key_comps))
            if len(oriented) > 1:
                raise ValueError(
                    f"contain-compatibility role is ambiguous for "
                    f"{a.node_id!r}/{b.node_id!r}: both orientations have "
                    "mined components. Extend the schema rather than emitting "
                    "two edges."
                )
            for container, containee, con_comps, key_comps in oriented:
                score = None
                if scored:
                    meas = contain_compatibility(
                        container.pose_world, con_comps,
                        containee.pose_world, key_comps,
                    )
                    if meas is not None:
                        parts = [
                            meas.radial_mismatch / norm["radial"],
                            meas.axial_mismatch / norm["axial"],
                        ]
                        if meas.orient_mismatch is not None:
                            parts.append(meas.orient_mismatch / norm["orient"])
                        score = _mean_normalized(parts)
                edges.append(_compat_edge(
                    container.node_id, containee.node_id, "contain-compatibility",
                    score, near, contain_spec,
                    attributes={"contain_role": "container"},
                ))
    return edges


def _site_ladder_edges(a: Node, b: Node, site: Node, cfg: dict,
                       pd_spec, ho_spec) -> List[Edge]:
    """Planar and height rungs for one object-to-site pair.

    Emitted in ``a - b`` order like every other pair, so a site sorting first
    inverts the height the same way a structural surface does.
    """
    from .sites import site_pair_points

    spec = _spec_for(cfg, _whitelist_key(site))
    other = b if site is a else a
    if spec is None:
        raise ValueError(
            f"{_whitelist_key(site)!r} is a declared ladder site but no live "
            f"spec was resolved for frame; the provider must run before edges "
            "are emitted."
        )
    points = site_pair_points(spec, other.pose_world)
    if points is None:
        return []
    source, target = points
    d = float(np.linalg.norm(source[:2] - target[:2]))
    dz = float(source[2] - target[2])
    if site is a:
        dz = -dz

    out: List[Edge] = []
    if pd_spec is not None:
        out.append(Edge(
            a.node_id, b.node_id, "planar-distance",
            bin_label(d, pd_spec[0], pd_spec[1]), raw_value=d,
            bin_key=OBJECT_SITE_PLANAR_KEY,
        ))
    if ho_spec is not None:
        out.append(Edge(
            a.node_id, b.node_id, "height-offset",
            bin_label(dz, ho_spec[0], ho_spec[1]), raw_value=dz,
            bin_key=OBJECT_SITE_HEIGHT_KEY,
        ))
    return out


def object_object_spatial_edges(
    graph: Graph, state: PrivilegedState, cfg: dict
) -> List[Edge]:
    """Object-center spatial facts for every eligible object pair.

    The carry phase of a place task is ``cubeA`` approaching ``cubeB``, and
    without these the only distance in the graph is end-effector-to-something.
    A schedule would have to substitute the gripper's distance for the object's,
    which stops being the same quantity the moment the object is released.

    Direction is carried by the label, not the endpoints: ``height-offset`` is
    antisymmetric and its ``above``/``below`` bins already say which way round
    the pair is, so one edge per pair says everything two would.
    """
    pd_key = spatial_bin_key(OBJECT_OBJECT_SCOPE, "planar-distance")
    ho_key = spatial_bin_key(OBJECT_OBJECT_SCOPE, "height-offset")
    pd_spec = _get_bin_spec(cfg, pd_key)
    ho_spec = _get_bin_spec(cfg, ho_key)
    if pd_spec is None and ho_spec is None:
        return []

    region_spec = _get_bin_spec(cfg, OBJECT_REGION_PLANAR_KEY)
    site_pd_spec = _get_bin_spec(cfg, OBJECT_SITE_PLANAR_KEY)
    site_ho_spec = _get_bin_spec(cfg, OBJECT_SITE_HEIGHT_KEY)
    aff_set = cfg.get("affordance_set")
    edges: List[Edge] = []
    for a, b in _object_pairs(graph):
        # A ladder site is measured through the site's own source point -- for
        # PegInsertionSide the peg *head*, not the peg origin -- by the one
        # function the miner also calls. Calibrating on the origin and reading
        # the head is the drift this module exists to prevent.
        site = (a if is_ladder_site(a, cfg)
                else b if is_ladder_site(b, cfg) else None)
        if site is not None:
            # Same discipline as the region above: one declared pair per site.
            if is_declared_site_pair(cfg, site, b if site is a else a):
                edges.extend(_site_ladder_edges(
                    a, b, site, cfg, site_pd_spec, site_ho_spec))
            continue
        d, dz = object_pair_measures(a, b, aff_set)
        # A region is a disc, not a body: planar distance to its centre is the
        # whole of its geometry, on its own scale, and there is no height to
        # report. Handled before the structural branch because the two are
        # mutually exclusive and this one owns both of the pair's outputs.
        region = (a if is_region_site(a, cfg)
                  else b if is_region_site(b, cfg) else None)
        if region is not None:
            other = b if region is a else a
            if (region_spec is not None
                    and is_declared_site_pair(cfg, region, other)):
                edges.append(Edge(
                    a.node_id, b.node_id, "planar-distance",
                    bin_label(d, region_spec[0], region_spec[1]), raw_value=d,
                    bin_key=OBJECT_REGION_PLANAR_KEY,
                ))
            continue
        surface = (a if is_structural_surface(a, cfg)
                   else b if is_structural_surface(b, cfg) else None)
        if surface is not None:
            other = b if surface is a else a
            height = surface_relative_height(
                surface, _bottom_point(other, _whitelist_key(surface), aff_set),
                aff_set)
            if height is None:
                continue
            # The label is read in ``a - b`` order, so a surface in the first
            # position inverts: "table above cube" is the cube's height below
            # the tabletop.
            dz = height if surface is b else -height
        if pd_spec is not None and surface is None:
            edges.append(Edge(
                a.node_id, b.node_id, "planar-distance",
                bin_label(d, pd_spec[0], pd_spec[1]), raw_value=d,
                bin_key=pd_key,
            ))
        if ho_spec is not None:
            edges.append(Edge(
                a.node_id, b.node_id, "height-offset",
                bin_label(dz, ho_spec[0], ho_spec[1]), raw_value=dz,
                bin_key=ho_key,
            ))
    return edges


# --------------------------------------------------------------------------- #
# goal sites
# --------------------------------------------------------------------------- #
def _node_for_key(graph: Graph, key: str) -> Optional[Node]:
    """The single node carrying ``key``, or None. Several is not a match.

    Two nodes under one entity key would make the pair ambiguous, and the
    replay potential already refuses to guess between them; refusing here keeps
    the two refusals consistent instead of emitting an edge the scorer will
    then discard.
    """
    hits = [n for n in graph.nodes
            if (n.attributes or {}).get("whitelist_key") == key]
    return hits[0] if len(hits) == 1 else None


def goal_edges(graph: Graph, state: PrivilegedState, cfg: dict) -> List[Edge]:
    """One ``reached`` fact per declared site pair, every frame.

    Unconditional by contract. The replay potential requires exactly one edge
    per scored relation and masks the entire frame when one is missing, so a
    subject the cameras lost still emits -- its pose and the site's are both
    known -- and a provider that failed raises here rather than letting a stale
    pose be read as a confident ``not-holds``.

    Endpoints are stored in ``pair_sort_key`` order, the same order the
    schedule compiler resolves a clause to, so a clause written subject-first
    finds the fact whichever way the keys happen to sort.
    """
    specs: Sequence[SiteSpec] = cfg.get("site_specs") or ()
    if not specs:
        return []
    edges: List[Edge] = []
    for spec in specs:
        spec.validate(f"{graph.env_id}/frame {graph.frame}")
        site_node = _node_for_key(graph, spec.key)
        subject_node = _node_for_key(graph, spec.subject_key)
        if site_node is None or subject_node is None:
            missing = spec.key if site_node is None else spec.subject_key
            raise ValueError(
                f"reached({spec.subject_key!r}, {spec.key!r}) cannot be "
                f"emitted: no unique node for {missing!r} at frame "
                f"{graph.frame}. A scheduled site pair has to resolve every "
                "frame -- a missing fact masks the whole frame's potential "
                "rather than scoring zero."
            )
        distance = site_distance(spec, subject_node.pose_world)
        held = reached_holds(spec, subject_node.pose_world)
        if distance is None or held is None:
            raise ValueError(
                f"reached({spec.subject_key!r}, {spec.key!r}) is unresolvable "
                f"at frame {graph.frame}: the subject or the site has no "
                "finite pose."
            )
        a, b = sorted((subject_node, site_node), key=pair_sort_key)
        edges.append(Edge(
            a.node_id, b.node_id, "reached",
            HOLDS if held else NOT_HOLDS,
            raw_value=distance,
            attributes={
                "site_key": spec.key,
                "subject_key": spec.subject_key,
                "metric": spec.metric,
                "tolerance": float(spec.tolerance),
            },
        ))
    return edges


def build_absolute_edges(
    graph: Graph, state: PrivilegedState, cfg: dict,
    initial_physical_pairs: Optional[Set[Tuple[str, str]]] = None,
    capture_initial: bool = False,
) -> None:
    """Append every admissible fact to ``graph.edges`` in place."""
    graph.edges.extend(goal_edges(graph, state, cfg))
    graph.edges.extend(ee_object_spatial_edges(graph, state, cfg))
    graph.edges.extend(ee_object_physical_edges(graph, state, cfg))
    graph.edges.extend(ee_object_affordance_edges(graph, state, cfg))
    # Off for MS-HAB: its progress ladder is end-effector-to-target and these
    # would be new facts in an environment nothing asked to change. Named
    # explicitly rather than inferred from the edge contract, which is a
    # separate axis.
    if bool(cfg.get("object_object_spatial", False)):
        graph.edges.extend(object_object_spatial_edges(graph, state, cfg))
    physical_edges = object_object_physical_edges(graph, state, cfg)
    graph.edges.extend(physical_edges)

    aff_cfg = cfg.get("affordances", {})
    suppress_initial = bool(
        aff_cfg.get("suppress_initial_physical_pair_compatibility", True)
    )
    # Capture is driven by the caller's once-per-episode flag, not by frame
    # numbering: a builder stepped without an episode boundary would otherwise
    # union a second scene's pairs onto the first.
    if (
        suppress_initial
        and capture_initial
        and initial_physical_pairs is not None
    ):
        held_labels = {HOLDS, SRC_HOLDS, DST_HOLDS}
        for edge in physical_edges:
            if edge.label in held_labels:
                initial_physical_pairs.add(tuple(sorted((edge.src, edge.dst))))
    if bool(cfg.get("affordances", {}).get("object_object_compatibility", True)):
        skip_pairs = initial_physical_pairs if suppress_initial else None
        graph.edges.extend(object_object_affordance_edges(
            graph, state, cfg, skip_pairs=skip_pairs,
        ))
