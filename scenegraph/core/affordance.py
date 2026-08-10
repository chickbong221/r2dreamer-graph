"""Affordance components mined from manipulation-success rollouts.

The asset is keyed per stable object identity. Each object can carry several
component lists, one per affordance relation:

* ``grasp`` (legacy ``components``): ``anchor``, ``approach_dir``, ``width`` -
  used by ee--object grasp-compatibility (the original mining target).
* ``contact``: ``anchor`` + ``outward_normal`` - used by obj--obj
  contact-compatibility (ee--object contact-compatibility reuses the grasp
  component without ``width``).
* ``support``: ``surface_anchor`` + ``surface_normal`` + ``footprint_radius``
  on the supporter side.
* ``bottom``: ``bottom_anchor`` + ``bottom_normal`` on the supported side.
* ``contain``: ``entry_anchor`` + ``entry_axis`` + ``opening_radius`` +
  ``depth`` on the container side, modeled on PegInsertionSide's
  ``box_hole_pose`` / ``box_hole_radii``.
* ``key``: ``key_anchor`` + ``key_axis`` on the containee side (peg head + peg
  long axis).

All frames are OBJECT-local, metres. Asset shape::

    {
      "_schema_version": 3,
      "objects": {
        "<canonical_key>": {
          "grasp_components":  [{"anchor": [..], "approach_dir": [..]?, "width": w}, ...],
          "contact_components": [{"anchor": [..], "outward_normal": [..]?}, ...],
          "support_components": [{"surface_anchor": [..], "surface_normal": [..], "footprint_radius": r}, ...],
          "bottom_components":  [{"bottom_anchor": [..], "bottom_normal": [..]}, ...],
          "contain_components": [{"entry_anchor": [..], "entry_axis": [..], "opening_radius": r, "depth": d}, ...],
          "key_components":     [{"key_anchor": [..], "key_axis": [..]}, ...]
        }
      }
    }

Legacy v2 assets used ``components`` for grasp components and no obj--obj
sections; the loader still accepts that shape (treats ``components`` as
``grasp_components``).
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TypeVar

import numpy as np

from .schema import Node


@dataclass
class AffordanceComponent:
    """Grasp component (legacy name kept for backwards compatibility)."""
    anchor_obj_frame: np.ndarray            # (3,) OBJECT frame, metres
    preferred_width: float                  # qpos[-2] + qpos[-1]
    approach_dir_obj_frame: Optional[np.ndarray] = None  # unit, OBJECT frame


# Alias used by new builders for readability.
GraspComponent = AffordanceComponent


@dataclass
class ContactComponent:
    """One contact point on an object, used for obj--obj contact-compatibility.

    ``outward_normal`` is the contact surface normal pointing OUT of the
    object. At a real contact between two objects A and B, A's outward normal
    should be anti-parallel to B's outward normal at the matching point.
    """
    anchor_obj_frame: np.ndarray
    outward_normal_obj_frame: Optional[np.ndarray] = None


@dataclass
class SupportComponent:
    """Surface descriptor on a supporter object."""
    surface_anchor_obj_frame: np.ndarray
    surface_normal_obj_frame: np.ndarray   # unit, points OUT of supporter's surface
    footprint_radius: float                # half-extent of the valid xy region


@dataclass
class BottomComponent:
    """Bottom-contact descriptor on a supported object."""
    bottom_anchor_obj_frame: np.ndarray
    bottom_normal_obj_frame: np.ndarray    # unit, points DOWN out of the object


@dataclass
class ContainComponent:
    """Entry descriptor on a container object (PegInsertionSide-style)."""
    entry_anchor_obj_frame: np.ndarray
    entry_axis_obj_frame: np.ndarray       # unit, points INTO the interior
    opening_radius: float
    depth: float


@dataclass
class KeyComponent:
    """Leading-point descriptor on a containee object."""
    key_anchor_obj_frame: np.ndarray
    key_axis_obj_frame: np.ndarray         # unit, long axis of the object


@dataclass
class AffordanceSet:
    """Per-relation component lists, keyed by canonical object id.

    ``by_object`` keeps the legacy name and stores grasp components so existing
    call sites continue to work; the other dicts are addressed via the
    relation-specific lookup helpers.
    """
    by_object: Dict[str, List[AffordanceComponent]] = field(default_factory=dict)
    contact_by_object: Dict[str, List[ContactComponent]] = field(default_factory=dict)
    support_by_object: Dict[str, List[SupportComponent]] = field(default_factory=dict)
    bottom_by_object: Dict[str, List[BottomComponent]] = field(default_factory=dict)
    contain_by_object: Dict[str, List[ContainComponent]] = field(default_factory=dict)
    key_by_object: Dict[str, List[KeyComponent]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (
            self.by_object or self.contact_by_object or self.support_by_object
            or self.bottom_by_object or self.contain_by_object or self.key_by_object
        )


def _normalize(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return v / n


def _quat_wxyz_to_rotmat(q: np.ndarray) -> Optional[np.ndarray]:
    """SAPIEN (qw, qx, qy, qz) -> 3x3 rotation matrix."""
    qn = _normalize(np.asarray(q, dtype=float))
    if qn is None:
        return None
    w, x, y, z = qn
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


# MS-HAB actor names look like ``env-0_024_bowl-3``; canonical form is ``024_bowl``.
_ENV_PREFIX_RE = re.compile(r"^env-\d+_")
_INSTANCE_SUFFIX_RE = re.compile(r"-\d+$")


def canonical_affordance_key(name: Optional[str]) -> Optional[str]:
    """Strip ``env-N_`` prefix and ``-N`` instance suffix (``env-0_024_bowl-3`` -> ``024_bowl``)."""
    if not name:
        return None
    key = _ENV_PREFIX_RE.sub("", str(name))
    key = _INSTANCE_SUFFIX_RE.sub("", key)
    return key or None


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
def _parse_vec3(v) -> Optional[np.ndarray]:
    if v is None:
        return None
    arr = np.asarray(v, dtype=float).reshape(-1)
    if arr.size != 3 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _parse_unit_vec3(v) -> Optional[np.ndarray]:
    arr = _parse_vec3(v)
    if arr is None:
        return None
    return _normalize(arr)


def _parse_scalar(v) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _parse_grasp_components(comps_raw) -> List[AffordanceComponent]:
    out: List[AffordanceComponent] = []
    if not isinstance(comps_raw, list):
        return out
    for c in comps_raw:
        if not isinstance(c, dict):
            continue
        anchor = _parse_vec3(c.get("anchor"))
        width = _parse_scalar(c.get("width"))
        if anchor is None or width is None:
            continue
        out.append(AffordanceComponent(
            anchor_obj_frame=anchor,
            preferred_width=width,
            approach_dir_obj_frame=_parse_unit_vec3(c.get("approach_dir")),
        ))
    return out


def _parse_contact_components(comps_raw) -> List[ContactComponent]:
    out: List[ContactComponent] = []
    if not isinstance(comps_raw, list):
        return out
    for c in comps_raw:
        if not isinstance(c, dict):
            continue
        anchor = _parse_vec3(c.get("anchor"))
        if anchor is None:
            continue
        out.append(ContactComponent(
            anchor_obj_frame=anchor,
            outward_normal_obj_frame=_parse_unit_vec3(c.get("outward_normal")),
        ))
    return out


def _parse_support_components(comps_raw) -> List[SupportComponent]:
    out: List[SupportComponent] = []
    if not isinstance(comps_raw, list):
        return out
    for c in comps_raw:
        if not isinstance(c, dict):
            continue
        anchor = _parse_vec3(c.get("surface_anchor"))
        normal = _parse_unit_vec3(c.get("surface_normal"))
        radius = _parse_scalar(c.get("footprint_radius"))
        if anchor is None or normal is None or radius is None or radius <= 0:
            continue
        out.append(SupportComponent(
            surface_anchor_obj_frame=anchor,
            surface_normal_obj_frame=normal,
            footprint_radius=radius,
        ))
    return out


def _parse_bottom_components(comps_raw) -> List[BottomComponent]:
    out: List[BottomComponent] = []
    if not isinstance(comps_raw, list):
        return out
    for c in comps_raw:
        if not isinstance(c, dict):
            continue
        anchor = _parse_vec3(c.get("bottom_anchor"))
        normal = _parse_unit_vec3(c.get("bottom_normal"))
        if anchor is None or normal is None:
            continue
        out.append(BottomComponent(
            bottom_anchor_obj_frame=anchor,
            bottom_normal_obj_frame=normal,
        ))
    return out


def _parse_contain_components(comps_raw) -> List[ContainComponent]:
    out: List[ContainComponent] = []
    if not isinstance(comps_raw, list):
        return out
    for c in comps_raw:
        if not isinstance(c, dict):
            continue
        anchor = _parse_vec3(c.get("entry_anchor"))
        axis = _parse_unit_vec3(c.get("entry_axis"))
        radius = _parse_scalar(c.get("opening_radius"))
        depth = _parse_scalar(c.get("depth"))
        if anchor is None or axis is None or radius is None or depth is None:
            continue
        if radius <= 0 or depth <= 0:
            continue
        out.append(ContainComponent(
            entry_anchor_obj_frame=anchor,
            entry_axis_obj_frame=axis,
            opening_radius=radius,
            depth=depth,
        ))
    return out


def _parse_key_components(comps_raw) -> List[KeyComponent]:
    out: List[KeyComponent] = []
    if not isinstance(comps_raw, list):
        return out
    for c in comps_raw:
        if not isinstance(c, dict):
            continue
        anchor = _parse_vec3(c.get("key_anchor"))
        axis = _parse_unit_vec3(c.get("key_axis"))
        if anchor is None or axis is None:
            continue
        out.append(KeyComponent(
            key_anchor_obj_frame=anchor,
            key_axis_obj_frame=axis,
        ))
    return out


def load_affordance_set(path: Optional[str]) -> AffordanceSet:
    """Load JSON asset; missing/unreadable file -> empty set (warn).

    Accepts both the new per-relation shape and the legacy v2 shape where the
    only key was ``components`` (which we treat as ``grasp_components``).
    """
    if not path or not os.path.isfile(path):
        warnings.warn(
            f"affordance asset not found at {path!r}; "
            "affordance relations will be silently skipped.",
            RuntimeWarning,
        )
        return AffordanceSet()

    with open(path, "r") as f:
        raw = json.load(f)

    out = AffordanceSet()
    objects = raw.get("objects", {}) if isinstance(raw, dict) else {}
    for key, entry in objects.items():
        if not isinstance(key, str) or key.startswith("_"):
            continue
        if not isinstance(entry, dict):
            continue
        # Grasp components: prefer ``grasp_components``; fall back to legacy
        # ``components`` for v2 assets.
        grasp_raw = entry.get("grasp_components")
        if grasp_raw is None:
            grasp_raw = entry.get("components")
        grasp = _parse_grasp_components(grasp_raw)
        if grasp:
            out.by_object[str(key)] = grasp

        contact = _parse_contact_components(entry.get("contact_components"))
        if contact:
            out.contact_by_object[str(key)] = contact

        support = _parse_support_components(entry.get("support_components"))
        if support:
            out.support_by_object[str(key)] = support

        bottom = _parse_bottom_components(entry.get("bottom_components"))
        if bottom:
            out.bottom_by_object[str(key)] = bottom

        contain = _parse_contain_components(entry.get("contain_components"))
        if contain:
            out.contain_by_object[str(key)] = contain

        keyc = _parse_key_components(entry.get("key_components"))
        if keyc:
            out.key_by_object[str(key)] = keyc
    return out


def _obj_rotmat(obj_pose_world: Optional[List[float]]) -> Optional[np.ndarray]:
    if obj_pose_world is None or len(obj_pose_world) < 7:
        return None
    q = np.asarray(obj_pose_world[3:7], dtype=float)
    return _quat_wxyz_to_rotmat(q)


def transform_anchors(
    obj_pose_world: Optional[List[float]],
    components,
) -> Optional[np.ndarray]:
    """(N,3) world anchors. ``obj_pose_world`` is ``[xyz, qw, qx, qy, qz]``.

    Accepts any component sequence whose elements expose an attribute named
    ``anchor_obj_frame`` (grasp / contact components do).
    """
    if obj_pose_world is None or len(obj_pose_world) < 7 or not components:
        return None
    p = np.asarray(obj_pose_world[:3], dtype=float)
    if not np.all(np.isfinite(p)):
        return None
    R = _obj_rotmat(obj_pose_world)
    if R is None:
        return None
    anchors_obj = np.stack(
        [np.asarray(c.anchor_obj_frame, dtype=float).reshape(3) for c in components],
        axis=0,
    )
    return p[None, :] + anchors_obj @ R.T


def transform_point(
    obj_pose_world: Optional[List[float]], point_obj: np.ndarray,
) -> Optional[np.ndarray]:
    """Transform a single 3-vector from object frame to world."""
    if obj_pose_world is None or len(obj_pose_world) < 7:
        return None
    p = np.asarray(obj_pose_world[:3], dtype=float)
    R = _obj_rotmat(obj_pose_world)
    if R is None or not np.all(np.isfinite(p)):
        return None
    return p + R @ np.asarray(point_obj, dtype=float).reshape(3)


def transform_dir(
    obj_pose_world: Optional[List[float]], dir_obj: np.ndarray,
) -> Optional[np.ndarray]:
    """Rotate a unit direction from object frame to world frame."""
    R = _obj_rotmat(obj_pose_world)
    if R is None:
        return None
    out = R @ np.asarray(dir_obj, dtype=float).reshape(3)
    return _normalize(out)


def inv_transform_point(
    obj_pose_world: Optional[List[float]], point_world: np.ndarray,
) -> Optional[np.ndarray]:
    """``inv(pose) * point`` for object pose ``[xyz, qw, qx, qy, qz]``."""
    if obj_pose_world is None or len(obj_pose_world) < 7:
        return None
    p = np.asarray(obj_pose_world[:3], dtype=float)
    R = _obj_rotmat(obj_pose_world)
    if R is None or not np.all(np.isfinite(p)):
        return None
    return R.T @ (np.asarray(point_world, dtype=float).reshape(3) - p)


def transform_approach_dir(
    obj_pose_world: Optional[List[float]],
    component: AffordanceComponent,
) -> Optional[np.ndarray]:
    """World-frame unit approach direction, or None if not mined / degenerate."""
    if component.approach_dir_obj_frame is None:
        return None
    return transform_dir(obj_pose_world, component.approach_dir_obj_frame)


def select_active_component(
    tcp_xyz_world: np.ndarray,
    anchors_world: Optional[np.ndarray],
    *,
    components: Optional[List[AffordanceComponent]] = None,
    obj_pose_world: Optional[List[float]] = None,
    tcp_pose_world: Optional[List[float]] = None,
    tcp_axis_local: Optional[List[float]] = None,
    orientation_weight: float = 0.10,
) -> Optional[int]:
    """Index of the candidate closest to the TCP. Adds ``orientation_weight * angle/pi``
    (metres) when TCP/component orientations are available."""
    if anchors_world is None or len(anchors_world) == 0:
        return None
    tcp = np.asarray(tcp_xyz_world, dtype=float).reshape(-1)
    if tcp.shape[0] != 3 or not np.all(np.isfinite(tcp)):
        return None
    diffs = anchors_world - tcp[None, :]
    scores = np.linalg.norm(diffs, axis=1)

    if (
        components is not None
        and obj_pose_world is not None
        and tcp_pose_world is not None
        and tcp_axis_local is not None
        and orientation_weight > 0.0
    ):
        tcp_R = _quat_wxyz_to_rotmat(np.asarray(tcp_pose_world[3:7], dtype=float))
        if tcp_R is not None:
            tcp_dir = _normalize(
                tcp_R @ np.asarray(tcp_axis_local, dtype=float).reshape(3)
            )
            if tcp_dir is not None:
                for idx, comp in enumerate(components[: len(scores)]):
                    aff_dir = transform_approach_dir(obj_pose_world, comp)
                    if aff_dir is None:
                        continue
                    cos = float(np.clip(np.dot(tcp_dir, aff_dir), -1.0, 1.0))
                    angle = float(np.arccos(cos))
                    scores[idx] += orientation_weight * (angle / np.pi)

    return int(np.argmin(scores))


# --------------------------------------------------------------------------- #
# Per-relation lookups
# --------------------------------------------------------------------------- #
_T = TypeVar("_T")


def _lookup_in(
    table: Dict[str, List[_T]], node: Node,
) -> Optional[List[_T]]:
    if not table:
        return None

    entity_key = node.attributes.get("entity_key") if node.attributes else None
    if entity_key:
        ek = str(entity_key)
        if ek in table:
            return table[ek]
        if ek.startswith("actor:"):
            legacy_actor = ek.split(":", 1)[1]
            if legacy_actor in table:
                return table[legacy_actor]

    mshab_id = node.attributes.get("mshab_obj_id") if node.attributes else None
    key = canonical_affordance_key(mshab_id) if mshab_id else None
    if key and key in table:
        return table[key]

    key = canonical_affordance_key(node.name)
    if key and key in table:
        return table[key]
    return None


def lookup_components(
    aff_set: AffordanceSet, node: Node,
) -> Optional[List[AffordanceComponent]]:
    """Grasp components for a node. Kept as the canonical name for legacy callers."""
    if aff_set is None or aff_set.is_empty():
        return None
    return _lookup_in(aff_set.by_object, node)


def lookup_grasp_components(
    aff_set: AffordanceSet, node: Node,
) -> Optional[List[AffordanceComponent]]:
    return lookup_components(aff_set, node)


def lookup_contact_components(
    aff_set: AffordanceSet, node: Node,
) -> Optional[List[ContactComponent]]:
    if aff_set is None:
        return None
    return _lookup_in(aff_set.contact_by_object, node)


def lookup_support_components(
    aff_set: AffordanceSet, node: Node,
) -> Optional[List[SupportComponent]]:
    if aff_set is None:
        return None
    return _lookup_in(aff_set.support_by_object, node)


def lookup_bottom_components(
    aff_set: AffordanceSet, node: Node,
) -> Optional[List[BottomComponent]]:
    if aff_set is None:
        return None
    return _lookup_in(aff_set.bottom_by_object, node)


def lookup_contain_components(
    aff_set: AffordanceSet, node: Node,
) -> Optional[List[ContainComponent]]:
    if aff_set is None:
        return None
    return _lookup_in(aff_set.contain_by_object, node)


def lookup_key_components(
    aff_set: AffordanceSet, node: Node,
) -> Optional[List[KeyComponent]]:
    if aff_set is None:
        return None
    return _lookup_in(aff_set.key_by_object, node)


def has_affordance(aff_set: AffordanceSet, node: Node) -> bool:
    """True if the node has any mined grasp components (legacy semantics).

    Used by node classification to decide ``interactive_object`` eligibility.
    """
    return bool(lookup_components(aff_set, node))


# --------------------------------------------------------------------------- #
# Compatibility components (ee--object, grasp + contact reuse same primitives)
# --------------------------------------------------------------------------- #
@dataclass
class CompatibilityMeasurement:
    """Raw per-component mismatches against the active affordance component.

    ``pos_mismatch`` is metres, ``orient_mismatch`` is radians in ``[0, pi]``,
    ``width_mismatch`` is metres (or None when the gripper width is unknown).
    ``a_star`` is the index of the component the TCP scored against.
    """

    a_star: int
    pos_mismatch: float
    orient_mismatch: Optional[float]
    width_mismatch: Optional[float]


def compatibility_components(
    component: AffordanceComponent,
    a_star: int,
    anchor_world: np.ndarray,
    obj_pose_world: List[float],
    tcp_pose_world: np.ndarray,
    tcp_axis_local: List[float],
    gripper_width: Optional[float],
) -> CompatibilityMeasurement:
    """Mismatches between current gripper config and ``component`` in world frame.

    Each mismatch is signed-magnitude: orientation mismatch is unsigned (an
    angle), gripper-width mismatch is the absolute delta. All three are fed
    through the demo-derived normalizers downstream.
    """
    tcp_xyz = np.asarray(tcp_pose_world[:3], dtype=float).reshape(3)
    pos = float(np.linalg.norm(tcp_xyz - np.asarray(anchor_world, dtype=float).reshape(3)))

    orient: Optional[float] = None
    aff_dir = transform_approach_dir(obj_pose_world, component)
    if aff_dir is not None and len(tcp_pose_world) >= 7:
        R = _quat_wxyz_to_rotmat(np.asarray(tcp_pose_world[3:7], dtype=float))
        if R is not None:
            tcp_dir = _normalize(
                R @ np.asarray(tcp_axis_local, dtype=float).reshape(3)
            )
            if tcp_dir is not None:
                cos = float(np.clip(np.dot(tcp_dir, aff_dir), -1.0, 1.0))
                orient = float(np.arccos(cos))

    width: Optional[float] = None
    if gripper_width is not None and np.isfinite(gripper_width):
        width = float(abs(float(gripper_width) - float(component.preferred_width)))

    return CompatibilityMeasurement(
        a_star=int(a_star),
        pos_mismatch=pos,
        orient_mismatch=orient,
        width_mismatch=width,
    )
