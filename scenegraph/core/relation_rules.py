"""Hyper-relational facts from privileged state.

One fact per admissible ``(src, relation, dst)``: absolute state ``sigma`` plus,
for spatial and affordance families, a change ``delta`` filled in by the
temporal buffer.

* physical state: ``contact`` (ee--obj, obj--obj), ``grasp`` (ee--obj),
  ``support`` / ``contain`` (obj--obj, directed). Binary and mutually
  independent -- a grasped object is also in contact.
* spatial (ee--obj): ``planar-distance``, ``height-offset``.
* affordance: ``grasp-`` / ``contact-`` / ``support-`` / ``contain-
  compatibility``. Scored for every admissible instance; the near gate only
  picks the label, emitting ``unobserved`` when far. Scoring outside the gate
  is what keeps the temporal change continuous across it.

Admissible = both endpoints visible and both carrying the whitelist
``interaction_types`` token; affordance also needs the mined components. The
one exception is the active subtask target, which the builder replays while it
is occluded and which therefore keeps all six of its end-effector facts. Other
facts touching a node that left the view are the graph builder's business.

Bin edges come from the mined whitelist alone (``cfg["bin_edges"]``). There is
no rule-based fallback: a relation the asset does not calibrate is not
emitted, so a token never means a hand-picked distance.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .affordance import (
    CompatibilityMeasurement,
    compatibility_components,
    lookup_components,
    lookup_contact_components,
    lookup_contain_components,
    lookup_bottom_components,
    lookup_key_components,
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
RELATION_TYPES: Tuple[str, ...] = (
    PHYSICAL_RELATIONS + SPATIAL_RELATIONS + AFFORDANCE_RELATIONS
)

# Relations that carry a temporal-change label (mu^rho == 1).
TEMPORAL_RELATIONS = frozenset(SPATIAL_RELATIONS + AFFORDANCE_RELATIONS)

NOT_HOLDS = "not-holds"
HOLDS = "holds"
UNOBSERVED = "unobserved"

PHYSICAL_LABELS: List[str] = [NOT_HOLDS, HOLDS]
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

# Absolute-state vocabulary per relation, used by the encoder and the decoder.
ABS_LABELS: Dict[str, List[str]] = {
    **{r: PHYSICAL_LABELS for r in PHYSICAL_RELATIONS},
    **SPATIAL_LABELS,
    **{r: COMPAT_LABELS for r in AFFORDANCE_RELATIONS},
}

# Label sets that pair with binned edges. Compatibility bins only cover the
# three scored labels; ``unobserved`` is assigned outside the binning path.
_BIN_LABELS: Dict[str, List[str]] = {
    **SPATIAL_LABELS,
    **{r: COMPAT_BIN_LABELS for r in AFFORDANCE_RELATIONS},
    **{f"{r}-change": CHANGE_LABELS for r in TEMPORAL_RELATIONS},
}


# Absolute relations the runtime cannot label at all without mined edges.
# ``derive_bin_edges`` always emits the four compatibility scales (the score is
# already normalised to [0, 1]) and derives the two spatial ones from the demo
# statistics, so a v4 asset missing any of these was not mined against the task
# being run. Change relations stay out: an asset legitimately omits one when
# the demos never moved it.
REQUIRED_BIN_RELATIONS: Tuple[str, ...] = SPATIAL_RELATIONS + AFFORDANCE_RELATIONS


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


def _resolve_entity(node: Node, state: PrivilegedState, graph: Optional[Graph] = None):
    """Node -> live simulator entity (for force queries).

    A retained target carries no segmentation ids -- nothing saw it this frame
    -- so the usual seg-id lookup returns nothing and its contact and grasp
    facts would vanish exactly when they matter most, with the object in the
    gripper hiding itself. ``state.active_obj`` is the handle for that one
    node, so it is used as a fallback and only ever for the node whose id the
    builder named as the active target.
    """
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


def _visible_objects(graph: Graph) -> List[Node]:
    return [
        n for n in graph.nodes
        if n.node_type == "object" and n.visible and n.pose_world is not None
    ]


def _retained_target(graph: Graph) -> Optional[Node]:
    """The active target when the builder is replaying it through occlusion."""
    target_id = graph.meta.get("active_target_node_id")
    if not target_id:
        return None
    node = graph.get_node(target_id)
    if node is None or node.node_type != "object" or node.pose_world is None:
        return None
    return None if node.visible else node


def _ee_object_nodes(graph: Graph) -> List[Node]:
    """Objects the end-effector relations are computed against.

    Every visible object, plus the retained active target whether or not a
    camera saw it this frame. Occlusion must not punch a hole in the target's
    relation history: the six facts the progress ladder reads are all defined
    from the current end-effector pose against the target's current pose, and
    both are known while it is hidden. Object--object facts keep the ordinary
    visible-only rule -- there is no second retained endpoint to pair with.
    """
    nodes = _visible_objects(graph)
    retained = _retained_target(graph)
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
    pd_spec = _get_bin_spec(cfg, "planar-distance")
    ho_spec = _get_bin_spec(cfg, "height-offset")

    edges: List[Edge] = []
    for node in _ee_object_nodes(graph):
        obj_xyz = _xyz(node)
        if pd_spec is not None:
            d = planar_distance_xyz(ee_xyz, obj_xyz)
            edges.append(Edge(
                "ee", node.node_id, "planar-distance",
                bin_label(d, pd_spec[0], pd_spec[1]), raw_value=d,
            ))
        if ho_spec is not None:
            dz = height_offset_xyz(ee_xyz, obj_xyz)
            edges.append(Edge(
                "ee", node.node_id, "height-offset",
                bin_label(dz, ho_spec[0], ho_spec[1]), raw_value=dz,
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
        ent = _resolve_entity(node, state, graph)
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
    pd_spec = _get_bin_spec(cfg, "planar-distance")
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
def _object_pairs(graph: Graph) -> List[Tuple[Node, Node]]:
    objs = [n for n in _visible_objects(graph) if n.segmentation_ids]
    out: List[Tuple[Node, Node]] = []
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            out.append((objs[i], objs[j]))
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
    its supporter reports both. ``contact`` is undirected and emitted once per
    pair; ``support`` and ``contain`` are directed and emitted for both
    orderings, with at most one of the two labelled ``holds``.

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
                edges.append(Edge(
                    container.node_id, containee.node_id, "contain",
                    HOLDS if held else NOT_HOLDS, raw_value=float(held),
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
                    _resolve_entity(a, state), _resolve_entity(b, state)
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
            for src, dst in ((a, b), (b, a)):
                holds = supporter is src
                edges.append(Edge(
                    src.node_id, dst.node_id, "support",
                    HOLDS if holds else NOT_HOLDS,
                    raw_value=force if holds else 0.0,
                    attributes={"support_role": "supporter"},
                ))
    return edges


def object_object_affordance_edges(
    graph: Graph, state: PrivilegedState, cfg: dict
) -> List[Edge]:
    """obj--obj affordance facts: contact / support / contain compatibility.

    Admissible when both endpoints carry the mined token and the matching
    components exist. Pairs beyond ``object_object_compat_max_distance`` skip
    scoring and report ``unobserved`` with no value, so no temporal change is
    accumulated for a pair that is nowhere near interacting.
    """
    pd_spec = _get_bin_spec(cfg, "planar-distance")
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
        a_xyz, b_xyz = _xyz(a), _xyz(b)
        ta, tb = interaction_types(a), interaction_types(b)
        d = planar_distance_xyz(a_xyz, b_xyz)
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
            for supporter, supported in ((a, b), (b, a)):
                sup_comps = lookup_support_components(aff_set, supporter)
                bot_comps = lookup_bottom_components(aff_set, supported)
                if not sup_comps or not bot_comps:
                    continue
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
            for container, containee in ((a, b), (b, a)):
                con_comps = lookup_contain_components(aff_set, container)
                key_comps = lookup_key_components(aff_set, containee)
                if not con_comps or not key_comps:
                    continue
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


def build_absolute_edges(
    graph: Graph, state: PrivilegedState, cfg: dict
) -> None:
    """Append every admissible fact to ``graph.edges`` in place."""
    graph.edges.extend(ee_object_spatial_edges(graph, state, cfg))
    graph.edges.extend(ee_object_physical_edges(graph, state, cfg))
    graph.edges.extend(ee_object_affordance_edges(graph, state, cfg))
    graph.edges.extend(object_object_physical_edges(graph, state, cfg))
    if bool(cfg.get("affordances", {}).get("object_object_compatibility", True)):
        graph.edges.extend(object_object_affordance_edges(graph, state, cfg))
