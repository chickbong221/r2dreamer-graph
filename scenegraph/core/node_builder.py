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
    fill_appearance(nodes, seg_by_cam, patch_grid)
    return nodes, masks, cam, rgb
