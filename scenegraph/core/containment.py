"""Geometric containment + obj-obj compatibility scorers.

The geometric ``contain`` detector follows ManiSkill's PegInsertionSide-v1
``has_peg_inserted``: transform the containee's key point into the container's
local frame, then check axial coordinate within ``[0, depth]`` and radial
distance to the entry axis within ``opening_radius``.

The four obj-obj compatibility scorers all return an unweighted mean of
per-component mismatches normalized to ``[0, 1]`` (same shape as the existing
ee-object grasp/contact scorer). Missing components are skipped from the
average rather than treated as a worst-case mismatch.

All inputs are SAPIEN-style poses (``[xyz, qw, qx, qy, qz]``) and frames are
metres / radians.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .affordance import (
    BottomComponent,
    ContactComponent,
    ContainComponent,
    KeyComponent,
    SupportComponent,
    inv_transform_point,
    transform_dir,
    transform_point,
)


# --------------------------------------------------------------------------- #
# Geometric contain (absolute physical-state edge)
# --------------------------------------------------------------------------- #
def contain_holds(
    container_pose_world: List[float],
    container: ContainComponent,
    containee_pose_world: List[float],
    key: KeyComponent,
) -> bool:
    """True iff containee's key point lies inside the container's entry volume.

    The check is the PegInsertionSide ``has_peg_inserted`` template:

        axial  = (key_in_container - entry_anchor) . entry_axis
        radial = || (key_in_container - entry_anchor) - axial * entry_axis ||
        contain = (0 <= axial <= depth) and (radial <= opening_radius)
    """
    key_world = transform_point(containee_pose_world, key.key_anchor_obj_frame)
    if key_world is None:
        return False
    key_in_container = inv_transform_point(container_pose_world, key_world)
    if key_in_container is None:
        return False
    delta = key_in_container - np.asarray(
        container.entry_anchor_obj_frame, dtype=float
    ).reshape(3)
    axis = np.asarray(container.entry_axis_obj_frame, dtype=float).reshape(3)
    axial = float(np.dot(delta, axis))
    radial_vec = delta - axial * axis
    radial = float(np.linalg.norm(radial_vec))
    if not (0.0 <= axial <= float(container.depth)):
        return False
    return radial <= float(container.opening_radius)


# --------------------------------------------------------------------------- #
# Obj-obj contact-compatibility
# --------------------------------------------------------------------------- #
@dataclass
class ObjContactMeasurement:
    """Raw per-component mismatches for obj-obj contact compatibility."""
    a_index: int
    b_index: int
    pos_mismatch: float                   # m
    orient_mismatch: Optional[float]      # rad (None if either normal absent)


def _pick_closest_pair(
    a_anchors_world: np.ndarray, b_anchors_world: np.ndarray,
) -> Tuple[int, int, float]:
    """Indices ``(i, j)`` whose world anchors are closest and the distance."""
    diffs = a_anchors_world[:, None, :] - b_anchors_world[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    i, j = np.unravel_index(int(np.argmin(dists)), dists.shape)
    return int(i), int(j), float(dists[i, j])


def obj_contact_compatibility(
    a_pose_world: List[float],
    a_components: List[ContactComponent],
    b_pose_world: List[float],
    b_components: List[ContactComponent],
) -> Optional[ObjContactMeasurement]:
    """Score the closest contact-anchor pair between two objects.

    ``pos_mismatch`` is the world distance between the matched anchors.
    ``orient_mismatch`` is the angle between A's outward normal and B's
    inverted outward normal (they should oppose at a real contact).
    Returns None if either side has no components.
    """
    if not a_components or not b_components:
        return None
    a_anchors = []
    for c in a_components:
        p = transform_point(a_pose_world, c.anchor_obj_frame)
        if p is None:
            return None
        a_anchors.append(p)
    b_anchors = []
    for c in b_components:
        p = transform_point(b_pose_world, c.anchor_obj_frame)
        if p is None:
            return None
        b_anchors.append(p)
    a_arr = np.asarray(a_anchors, dtype=float)
    b_arr = np.asarray(b_anchors, dtype=float)
    i, j, dist = _pick_closest_pair(a_arr, b_arr)

    orient: Optional[float] = None
    n_a = a_components[i].outward_normal_obj_frame
    n_b = b_components[j].outward_normal_obj_frame
    if n_a is not None and n_b is not None:
        a_dir = transform_dir(a_pose_world, n_a)
        b_dir = transform_dir(b_pose_world, n_b)
        if a_dir is not None and b_dir is not None:
            cos = float(np.clip(np.dot(a_dir, -b_dir), -1.0, 1.0))
            orient = float(np.arccos(cos))
    return ObjContactMeasurement(
        a_index=i, b_index=j, pos_mismatch=dist, orient_mismatch=orient,
    )


# --------------------------------------------------------------------------- #
# Support-compatibility
# --------------------------------------------------------------------------- #
@dataclass
class SupportMeasurement:
    supporter_index: int
    supported_index: int
    xy_mismatch: float                # m, in supporter surface plane
    vertical_mismatch: float          # m, along surface normal
    orient_mismatch: Optional[float]  # rad between supported.bottom_normal and -surface_normal


def support_compatibility(
    supporter_pose_world: List[float],
    supporter_components: List[SupportComponent],
    supported_pose_world: List[float],
    supported_components: List[BottomComponent],
) -> Optional[SupportMeasurement]:
    """Score how well ``supported`` is positioned to rest on ``supporter``.

    For each (surface, bottom) pair we compute the bottom anchor's offset from
    the surface anchor in the supporter's surface plane (``xy_mismatch``) and
    along the surface normal (``vertical_mismatch``); ``xy_mismatch`` is
    clipped to zero inside the supporter's footprint radius. The pair that
    minimizes ``xy_mismatch + |vertical_mismatch|`` is reported.
    """
    if not supporter_components or not supported_components:
        return None
    best: Optional[Tuple[float, int, int, float, float, Optional[float]]] = None
    for i, sc in enumerate(supporter_components):
        surface_anchor_world = transform_point(
            supporter_pose_world, sc.surface_anchor_obj_frame,
        )
        normal_world = transform_dir(
            supporter_pose_world, sc.surface_normal_obj_frame,
        )
        if surface_anchor_world is None or normal_world is None:
            continue
        for j, bc in enumerate(supported_components):
            bottom_anchor_world = transform_point(
                supported_pose_world, bc.bottom_anchor_obj_frame,
            )
            if bottom_anchor_world is None:
                continue
            delta = bottom_anchor_world - surface_anchor_world
            vertical = float(np.dot(delta, normal_world))
            in_plane = delta - vertical * normal_world
            raw_xy = float(np.linalg.norm(in_plane))
            xy_mismatch = max(0.0, raw_xy - float(sc.footprint_radius))

            orient: Optional[float] = None
            bottom_normal_world = transform_dir(
                supported_pose_world, bc.bottom_normal_obj_frame,
            )
            if bottom_normal_world is not None:
                cos = float(np.clip(np.dot(bottom_normal_world, -normal_world), -1.0, 1.0))
                orient = float(np.arccos(cos))

            score = xy_mismatch + abs(vertical)
            cand = (score, i, j, xy_mismatch, vertical, orient)
            if best is None or score < best[0]:
                best = cand
    if best is None:
        return None
    _score, i, j, xy_mismatch, vertical, orient = best
    return SupportMeasurement(
        supporter_index=i, supported_index=j,
        xy_mismatch=xy_mismatch,
        vertical_mismatch=abs(vertical),
        orient_mismatch=orient,
    )


# --------------------------------------------------------------------------- #
# Contain-compatibility
# --------------------------------------------------------------------------- #
@dataclass
class ContainMeasurement:
    container_index: int
    key_index: int
    radial_mismatch: float           # m past the opening_radius (clipped at 0)
    axial_mismatch: float            # m past the [0, depth] interval (clipped at 0)
    orient_mismatch: Optional[float] # rad between key_axis and entry_axis


def contain_compatibility(
    container_pose_world: List[float],
    container_components: List[ContainComponent],
    containee_pose_world: List[float],
    key_components: List[KeyComponent],
) -> Optional[ContainMeasurement]:
    """Score how aligned ``containee`` is with each entry of ``container``.

    For each (entry, key) pair we transform the key anchor into the
    container's local frame, decompose the offset along the entry axis vs.
    perpendicular, and report mismatches in the same units as ``contain_holds``
    expects. The pair minimizing ``radial_mismatch + axial_mismatch`` is
    chosen.
    """
    if not container_components or not key_components:
        return None
    best: Optional[Tuple[float, int, int, float, float, Optional[float]]] = None
    for i, cc in enumerate(container_components):
        axis_obj = np.asarray(cc.entry_axis_obj_frame, dtype=float).reshape(3)
        entry_anchor_obj = np.asarray(cc.entry_anchor_obj_frame, dtype=float).reshape(3)
        for j, kc in enumerate(key_components):
            key_world = transform_point(containee_pose_world, kc.key_anchor_obj_frame)
            if key_world is None:
                continue
            key_in_container = inv_transform_point(container_pose_world, key_world)
            if key_in_container is None:
                continue
            delta = key_in_container - entry_anchor_obj
            axial = float(np.dot(delta, axis_obj))
            radial_vec = delta - axial * axis_obj
            radial = float(np.linalg.norm(radial_vec))
            radial_mismatch = max(0.0, radial - float(cc.opening_radius))
            if axial < 0.0:
                axial_mismatch = -axial
            elif axial > float(cc.depth):
                axial_mismatch = axial - float(cc.depth)
            else:
                axial_mismatch = 0.0

            orient: Optional[float] = None
            entry_axis_world = transform_dir(container_pose_world, axis_obj)
            key_axis_world = transform_dir(containee_pose_world, kc.key_axis_obj_frame)
            if entry_axis_world is not None and key_axis_world is not None:
                cos = float(np.clip(
                    np.dot(entry_axis_world, key_axis_world), -1.0, 1.0,
                ))
                orient = float(np.arccos(cos))

            score = radial_mismatch + axial_mismatch
            cand = (score, i, j, radial_mismatch, axial_mismatch, orient)
            if best is None or score < best[0]:
                best = cand
    if best is None:
        return None
    _score, i, j, radial_mismatch, axial_mismatch, orient = best
    return ContainMeasurement(
        container_index=i, key_index=j,
        radial_mismatch=radial_mismatch,
        axial_mismatch=axial_mismatch,
        orient_mismatch=orient,
    )
