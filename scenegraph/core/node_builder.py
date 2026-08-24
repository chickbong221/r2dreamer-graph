"""Build the two-type node set (``ee`` + ``object``) for one frame.

The builder excludes background, folds gripper links into the single ``ee``
node, and creates object nodes for visible non-robot actors and links. It also
derives each node's per-camera appearance support: a normalised bounding box
and the patch-grid coverage the frozen encoder's features are pooled under.

Task relevance is decided later by the hard per-subtask whitelist.  This
module deliberately avoids name-based scene filtering so a visible supporter
or articulation link cannot be discarded before the whitelist sees it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .entity_identity import entity_kind, stable_entity_key, stable_node_id
from .schema import Node
from .mask_extractor import (
    MaskAccumulator,
    extract_camera_obs,
    mask_for_id,
    pick_camera,
)
from ..adapters.privileged_state import (
    PrivilegedState,
    entity_pose_world_array,
)


# --------------------------------------------------------------------------- #
# Entity classification
# --------------------------------------------------------------------------- #
def _entity_name(entity) -> str:
    return getattr(entity, "name", str(entity))


def _is_actor(entity) -> bool:
    return type(entity).__name__ == "Actor"


def _is_link(entity) -> bool:
    return type(entity).__name__ == "Link"


def _is_robot_link(entity, robot_links: set, link_names=None) -> bool:
    if not _is_link(entity):
        return False
    if entity in robot_links:
        return True
    # Fallback by name match (merged views can break identity equality).
    if link_names is None:
        link_names = {getattr(l, "name", None) for l in robot_links}
    return getattr(entity, "name", None) in link_names


def _is_ee_link(entity, ee_links: List[Any], link_names=None) -> bool:
    if entity in ee_links:
        return True
    if link_names is None:
        link_names = {getattr(l, "name", None) for l in ee_links}
    return getattr(entity, "name", None) in link_names


def canonical_object_key(entity) -> str:
    """Stable node id. ManiSkill already provides stable simulator objects."""
    return stable_node_id(entity)


# --------------------------------------------------------------------------- #
# Node factories
# --------------------------------------------------------------------------- #
def make_ee_node(state: PrivilegedState) -> Node:
    return Node(
        node_id="ee",
        node_type="ee",
        name="end_effector",
        visible=False,                      # set True once a mask is merged in
        pose_world=list(state.tcp_pose_world)
        if state.tcp_pose_world is not None
        else None,
        source="segmentation",
    )


def make_object_node(entity, state: PrivilegedState) -> Node:
    pose_world = None
    try:
        arr = entity_pose_world_array(entity, state.env_idx)
        pose_world = list(arr) if arr is not None else None
    except Exception:
        pose_world = None
    return Node(
        node_id=canonical_object_key(entity),
        node_type="object",
        name=_entity_name(entity),
        visible=True,
        pose_world=pose_world,
        source="segmentation",
        attributes=dict(
            is_actor=_is_actor(entity),
            is_link=_is_link(entity),
            is_articulation_link=_is_link(entity),
            entity_kind=entity_kind(entity),
            entity_key=stable_entity_key(entity),
            # 'dynamic' | 'kinematic' | 'static'. A pair with no
            # dynamic endpoint cannot change shape during an episode.
            body_type=str(getattr(entity, "px_body_type", "") or ""),
        ),
    )


# --------------------------------------------------------------------------- #
# Main builder
# --------------------------------------------------------------------------- #
def fill_appearance(
    nodes: Dict[str, Node], seg_by_cam: List[np.ndarray], grid: int,
) -> None:
    """Attach the per-camera bbox and patch-grid coverage to every node.

    ``grid`` is the frozen encoder's patch-grid resolution. Coverage stays
    fractional rather than binary so a patch a node only partly occupies
    contributes proportionally to its pooled embedding.

    A camera with no pixels for a node leaves that row zero, which is what the
    model reads back as invisible in that camera. Segmentation ids are
    scene-global, so a node seen by one camera only lands correctly in both.
    """
    for seg in seg_by_cam:
        H, W = seg.shape
        if H % grid or W % grid:
            raise ValueError(
                f"segmentation {H}x{W} is not divisible by patch grid {grid}"
            )
    shape = (len(seg_by_cam), grid * grid)
    for node in nodes.values():
        node.bbox = np.zeros((shape[0], 4), np.float32)
        node.patch_weights = np.zeros(shape, np.float32)
        if not node.segmentation_ids:
            continue
        for cam, seg in enumerate(seg_by_cam):
            H, W = seg.shape
            fy, fx = H // grid, W // grid
            m = np.isin(seg, node.segmentation_ids)
            ys, xs = np.nonzero(m)
            if ys.size == 0:
                continue
            # Exclusive maxima, so a one-pixel node still has nonzero extent.
            node.bbox[cam] = (
                xs.min() / W, (xs.max() + 1) / W,
                ys.min() / H, (ys.max() + 1) / H,
            )
            cov = m.reshape(grid, fy, grid, fx).sum((1, 3)) / (fy * fx)
            node.patch_weights[cam] = cov.reshape(-1)


def fill_bboxes(nodes: Dict[str, Node], seg_by_cam: List[np.ndarray]) -> None:
    """Attach the per-camera bbox to every node, and nothing else.

    The relation-only pooled contract needs boxes but no patch coverage, so this
    is deliberately not ``fill_appearance`` with a flag: that function's cost is
    the per-node ``isin`` sweep and the patch-grid reduction, both of which exist
    only to feed the frozen appearance encoder.

    One vectorised pass per camera. Segmentation ids are mapped to node rows
    through a lookup table, so a node owning several ids costs nothing extra and
    there is no Python loop over nodes doing full-frame work. Row 0 of the table
    is the sentinel that absorbs background, ids this builder created no node
    for, and anything out of range -- it is dropped afterwards, so no id can
    index row -1 or leak into a real node's extent.

    Boxes come out normalised to [0, 1] with exclusive maxima, byte-identical to
    what ``fill_appearance`` produces, so the two paths stay interchangeable.
    Node rows here are builder-local; whitelist admission and the registry's
    capacity rules run later and decide the packed position.
    """
    n_cams = len(seg_by_cam)
    keys = list(nodes)
    for node in nodes.values():
        node.bbox = np.zeros((n_cams, 4), np.float32)

    owner: Dict[int, int] = {}
    for row, key in enumerate(keys, start=1):
        for seg_id in nodes[key].segmentation_ids:
            seg_id = int(seg_id)
            if seg_id > 0:
                owner[seg_id] = row
    if not owner:
        return

    max_id = max(owner)
    lut = np.zeros(max_id + 1, np.int32)
    for seg_id, row in owner.items():
        lut[seg_id] = row
    n_rows = len(keys) + 1

    for cam, seg in enumerate(seg_by_cam):
        H, W = seg.shape
        # Ids above this table (another camera saw a larger one) and any
        # non-positive id fall to the sentinel row rather than wrapping.
        known = (seg > 0) & (seg <= max_id)
        rows = lut[np.where(known, seg, 0)]
        # Two boolean projections instead of one mask per node: each is a single
        # scatter over the frame, and duplicate writes are all ``True``.
        seen_x = np.zeros((n_rows, W), bool)
        seen_y = np.zeros((n_rows, H), bool)
        seen_x[rows, np.arange(W)[None, :]] = True
        seen_y[rows, np.arange(H)[:, None]] = True
        present = seen_x.any(1)
        # argmax on a boolean row is the first True; reversed gives the last.
        # Meaningless for an all-False row, which ``present`` filters out.
        x0 = seen_x.argmax(1)
        x1 = W - seen_x[:, ::-1].argmax(1)
        y0 = seen_y.argmax(1)
        y1 = H - seen_y[:, ::-1].argmax(1)
        for row, key in enumerate(keys, start=1):
            if present[row]:
                nodes[key].bbox[cam] = (
                    x0[row] / W, x1[row] / W, y0[row] / H, y1[row] / H,
                )


def _ingest_camera(
    seg: np.ndarray,
    state: PrivilegedState,
    nodes: Dict[str, Node],
    area_by_key: Dict[str, int],
    masks: MaskAccumulator,
    *,
    need_masks: bool,
    admit: Optional[Callable[[Any], bool]] = None,
) -> None:
    """Union one camera's segmentation into the shared node dict.

    Per-segment pixel counts come from one ``bincount`` pass, so visibility
    costs a single sweep of the frame and never materialises a per-node mask.

    ``admit`` is an optional early whitelist gate: entities it rejects are
    skipped before node construction. It must admit a superset of what the
    downstream ``apply_whitelist`` keeps so the final graph is unchanged.

    ``seed_scene``, when given, is the gate for :func:`seed_scene_nodes`: every
    actor it admits becomes a vertex whether or not it rendered.
    """
    flat = seg.reshape(-1)
    counts_by_id = np.bincount(flat)

    robot_link_names = getattr(state, "robot_link_names", None)
    ee_link_names = getattr(state, "ee_link_names", None)
    for seg_id in np.nonzero(counts_by_id)[0]:
        seg_id = int(seg_id)
        if seg_id == 0:
            continue
        count = int(counts_by_id[seg_id])
        entity = state.seg_id_map.get(seg_id)
        if entity is None:
            continue

        if _is_robot_link(entity, state.robot_links, robot_link_names):
            if _is_ee_link(entity, state.ee_links, ee_link_names):
                if need_masks:
                    masks.add("ee", mask_for_id(seg, seg_id))
                nodes["ee"].visible = True
                nodes["ee"].segmentation_ids.append(seg_id)
                area_by_key["ee"] += count
            continue

        if admit is not None and not admit(entity):
            continue

        key = canonical_object_key(entity)
        if key not in nodes:
            nodes[key] = make_object_node(entity, state)
            area_by_key[key] = 0
        nodes[key].segmentation_ids.append(seg_id)
        area_by_key[key] += count
        if need_masks:
            masks.add(key, mask_for_id(seg, seg_id))
        nodes[key].pixel_area = area_by_key[key]


def seed_scene_nodes(
    nodes: Dict[str, Node],
    state: PrivilegedState,
    *,
    admit: Optional[Callable[[Any], bool]] = None,
) -> None:
    """Add a pixel-less node for every admissible actor no camera rendered.

    Segmentation is the only way an entity becomes a vertex, so an actor that
    renders nothing -- a goal marker the task hides before sensor capture, an
    object fully inside a container -- is never one, and a schedule role bound
    to it can never resolve. Mining does not have that blind spot: it admits
    spatial-only members from poses, so the whitelist declares objects the
    runtime graph then cannot produce.

    ``seg_id_map`` covers ``scene.actors`` rather than the pixels of any frame,
    so it lists them all. The seeded node carries no segmentation ids, which
    leaves its box zero -- the agreed signal for "no pixels this frame" -- while
    the caller refreshes its centroid from the simulator like any other.

    Tabletop scenes only. Under ``projected_camera`` the scene is a whole
    apartment and admissibility is not the question; camera coverage is.
    """
    for entity in state.seg_id_map.values():
        if entity is None or _is_robot_link(
            entity, state.robot_links, state.robot_link_names
        ):
            continue
        key = canonical_object_key(entity)
        if key in nodes:
            continue
        if admit is not None and not admit(entity):
            continue
        node = make_object_node(entity, state)
        node.visible = False
        node.source = "scene"
        nodes[key] = node


def build_nodes(
    obs: dict,
    state: PrivilegedState,
    *,
    camera: Optional[str] = None,
    seg_override: Optional[np.ndarray] = None,
    seg_overrides: Optional[Dict[str, np.ndarray]] = None,
    rgb_override: Optional[np.ndarray] = None,
    camera_override: Optional[str] = None,
    record_camera: Optional[str] = None,
    camera_order: Optional[List[str]] = None,
    need_masks: bool = True,
    admit: Optional[Callable[[Any], bool]] = None,
    patch_grid: int = 8,
    appearance: bool = True,
    bbox: bool = True,
    seed_scene: Optional[Callable[[Any], bool]] = None,
) -> Tuple[Dict[str, Node], MaskAccumulator, str, np.ndarray]:
    """Return (nodes_by_id, masks, record_camera_name, rgb).

    ``seg_overrides`` (dict of ``cam -> [H, W]``) unions visibility across
    cameras and yields one bbox and coverage grid per camera, ordered by
    ``camera_order``. Overlay masks are collected only for ``record_camera``.
    ``seg_override`` (singular) is the single-camera path used by the offline
    probe.
    """
    if seg_overrides is not None:
        if not seg_overrides:
            raise ValueError("seg_overrides is empty")
    elif seg_override is not None:
        seg_overrides = {camera_override or camera or "fetch_head": seg_override}
    else:
        cam = pick_camera(obs, camera)
        rgb_from_obs, seg, _ = extract_camera_obs(obs, cam, state.env_idx)
        seg_overrides = {cam: seg}
        rgb_override = rgb_override if rgb_override is not None else rgb_from_obs

    order = list(camera_order) if camera_order else list(seg_overrides)
    missing = [c for c in order if c not in seg_overrides]
    if missing:
        raise KeyError(
            f"camera_order names {missing} with no segmentation; "
            f"got {sorted(seg_overrides)}"
        )
    cam = record_camera or camera_override or camera
    if cam not in order:
        cam = order[0]

    seg_by_cam = [seg_overrides[name] for name in order]
    H, W = seg_by_cam[0].shape
    rgb = rgb_override if rgb_override is not None else \
        np.zeros((H, W, 3), dtype=np.uint8)

    masks = MaskAccumulator(H, W)
    nodes: Dict[str, Node] = {"ee": make_ee_node(state)}
    area_by_key: Dict[str, int] = {"ee": 0}

    for name, cam_seg in zip(order, seg_by_cam):
        _ingest_camera(
            cam_seg, state, nodes, area_by_key, masks,
            need_masks=need_masks and name == cam,
            admit=admit,
        )

    nodes["ee"].pixel_area = area_by_key["ee"]
    if seed_scene is not None:
        # Its own gate, not ``admit``: that one is relaxed on recording paths
        # so overlays keep every visible entity, and seeding without a
        # whitelist would add the ground and the walls.
        seed_scene_nodes(nodes, state, admit=seed_scene)
    # Patch coverage exists only to feed the appearance encoder, and computing
    # it is the expensive half. Boxes are wanted on their own by the pooled
    # relation contract, which reads no RGB and builds no DINO, so the two are
    # separate switches rather than one.
    if appearance:
        fill_appearance(nodes, seg_by_cam, patch_grid)
    elif bbox:
        fill_bboxes(nodes, seg_by_cam)
    return nodes, masks, cam, rgb
