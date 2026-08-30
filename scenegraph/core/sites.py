"""Spatial sites: goal geometry the scene has no actor for.

A site is a pose the environment defines but no segmentation id names -- a
PickCube goal marker, the mouth of a PegInsertionSide hole, the region around
the robot base a PullCubeTool episode has to drag the cube into. They enter the
graph as ordinary vertices with a stable key and a live pose, and they carry
exactly one relation the scene cannot otherwise express: ``reached``.

Two objects, deliberately separate:

* :class:`SiteDeclaration` is what the mined asset stores. It names the pair,
  the metric and where the tolerance comes from, and it is what the schedule
  compiler validates a ``reached`` clause against. It holds no geometry, so it
  cannot go stale against a re-randomized scene.
* :class:`SiteSpec` is what a provider returns each frame: the declaration plus
  the live pose, the live tolerance and the source point the distance is
  measured from.

The source point matters more than it looks. PegInsertionSide's milestone is
the peg *head* reaching the hole mouth, not the peg's origin reaching it, and
the mined bins for that ladder have to be calibrated on the same point the
runtime reads -- otherwise a token means one distance offline and another
online. Both sides call :func:`site_pair_points`, which is the only place that
choice is made.

Pure numpy: providers live in ``adapters`` because they read a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Set, Tuple

import numpy as np

# Site keys live in their own namespace so ``normalize_asset_key`` passes them
# through untouched and no canonicalisation collapses two of them into one.
SITE_PREFIX = "spatial:"

SITE_POINT = "point"
SITE_SURFACE = "surface"
SITE_REGION = "region"
SITE_TYPES: Tuple[str, ...] = (SITE_POINT, SITE_SURFACE, SITE_REGION)

METRIC_EUCLIDEAN = "euclidean"
METRIC_PLANAR = "planar"
METRICS: Tuple[str, ...] = (METRIC_EUCLIDEAN, METRIC_PLANAR)

# How the subject's source point is derived. ``origin`` is the subject node's
# own pose; ``provider`` means the provider supplies a live point (the peg
# head), and a spec claiming it must actually carry one.
SOURCE_ORIGIN = "origin"
SOURCE_PROVIDER = "provider"
SOURCES: Tuple[str, ...] = (SOURCE_ORIGIN, SOURCE_PROVIDER)


class SiteError(ValueError):
    """A site declaration or observation that cannot be scored."""


# --------------------------------------------------------------------------- #
# Asset-level declaration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SiteDeclaration:
    """What the whitelist stores about one goal pair.

    Deliberately geometry-free. The tolerance and the pose are read live, so a
    task that re-randomizes its goal every episode cannot drift away from a
    number frozen at mining time.
    """

    key: str                 # site entity key, ``spatial:...`` or an actor key
    site_type: str
    subject_key: str
    metric: str
    source: str
    provenance: str

    def validate(self, where: str = "site") -> None:
        if not self.key:
            raise SiteError(f"{where}: site declaration has no key")
        if self.site_type not in SITE_TYPES:
            raise SiteError(
                f"{where}: site {self.key!r} has unknown site_type "
                f"{self.site_type!r}; expected one of {list(SITE_TYPES)}"
            )
        if self.metric not in METRICS:
            raise SiteError(
                f"{where}: site {self.key!r} has unknown metric "
                f"{self.metric!r}; expected one of {list(METRICS)}"
            )
        if self.source not in SOURCES:
            raise SiteError(
                f"{where}: site {self.key!r} has unknown source "
                f"{self.source!r}; expected one of {list(SOURCES)}"
            )
        if not self.subject_key:
            raise SiteError(
                f"{where}: site {self.key!r} names no subject. 'reached' is "
                "scorable for one declared pair, never as a generic proximity "
                "alias."
            )
        if self.subject_key == self.key:
            raise SiteError(
                f"{where}: site {self.key!r} names itself as its subject"
            )
        if not self.provenance:
            raise SiteError(
                f"{where}: site {self.key!r} records no provenance. The "
                "tolerance comes from the environment and the reader has to be "
                "able to tell which predicate it mirrors."
            )


def parse_site_declarations(
    raw: Any, where: str = "asset",
) -> Dict[str, SiteDeclaration]:
    """Parse the asset's ``sites`` section. A missing section means no sites."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SiteError(f"{where}: 'sites' must be an object")
    out: Dict[str, SiteDeclaration] = {}
    for key, entry in raw.items():
        if not isinstance(key, str) or key.startswith("_"):
            continue
        if not isinstance(entry, dict):
            raise SiteError(f"{where}: site {key!r} must be an object")
        decl = SiteDeclaration(
            key=key,
            site_type=str(entry.get("site_type", "") or ""),
            subject_key=str(entry.get("subject", "") or ""),
            metric=str(entry.get("metric", "") or ""),
            source=str(entry.get("source", SOURCE_ORIGIN) or SOURCE_ORIGIN),
            provenance=str(entry.get("provenance", "") or ""),
        )
        decl.validate(where)
        out[key] = decl
    return out


def declaration_to_dict(decl: SiteDeclaration) -> Dict[str, Any]:
    return {
        "site_type": decl.site_type,
        "subject": decl.subject_key,
        "metric": decl.metric,
        "source": decl.source,
        "provenance": decl.provenance,
    }


def goal_pairs(declarations: Dict[str, SiteDeclaration]) -> Set[Tuple[str, str]]:
    """Unordered ``(subject, site)`` key pairs ``reached`` may be scored on."""
    return {
        tuple(sorted((decl.subject_key, decl.key)))
        for decl in declarations.values()
    }


# --------------------------------------------------------------------------- #
# Per-frame observation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SiteSpec:
    """One site as it stands this frame.

    ``tolerance`` is the environment's own success radius, read live rather
    than mined: PickCube's ``goal_thresh`` and PullCubeTool's 0.6 m are the
    exact spatial half of each task's success predicate, and a mined
    approximation of them would be a different predicate wearing the same name.
    """

    declaration: SiteDeclaration
    pose_world: np.ndarray                            # [7] xyz + wxyz
    tolerance: float
    subject_point_world: Optional[np.ndarray] = None  # source override
    axis_world: Optional[np.ndarray] = None           # normal / entry axis

    @property
    def key(self) -> str:
        return self.declaration.key

    @property
    def subject_key(self) -> str:
        return self.declaration.subject_key

    @property
    def metric(self) -> str:
        return self.declaration.metric

    def validate(self, where: str = "site") -> None:
        self.declaration.validate(where)
        pose = np.asarray(self.pose_world, dtype=float).reshape(-1)
        if pose.size < 7 or not np.all(np.isfinite(pose[:7])):
            raise SiteError(
                f"{where}: site {self.key!r} has no finite live pose. A site "
                "whose provider failed must raise here rather than emit a "
                "confident 'not-holds' from a stale pose."
            )
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise SiteError(
                f"{where}: site {self.key!r} has non-positive tolerance "
                f"{self.tolerance!r}"
            )
        if (self.declaration.source == SOURCE_PROVIDER
                and self.subject_point_world is None):
            raise SiteError(
                f"{where}: site {self.key!r} declares source 'provider' but "
                "the provider supplied no subject point. Falling back to the "
                "subject's origin would calibrate one point and read another."
            )


def site_pair_points(
    spec: SiteSpec, subject_pose_world: Optional[Sequence[float]],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """``(source point, site point)`` in world, or None when unresolvable.

    The single definition of what a site pair measures. Runtime edge emission,
    the calibration collector and the bin re-projection all route through here,
    so the peg-head override cannot be applied in one and forgotten in another.
    """
    site = np.asarray(spec.pose_world, dtype=float).reshape(-1)
    if site.size < 3 or not np.all(np.isfinite(site[:3])):
        return None
    if spec.subject_point_world is not None:
        source = np.asarray(spec.subject_point_world, dtype=float).reshape(-1)
    elif subject_pose_world is not None:
        source = np.asarray(subject_pose_world, dtype=float).reshape(-1)
    else:
        return None
    if source.size < 3 or not np.all(np.isfinite(source[:3])):
        return None
    return source[:3], site[:3]


def site_distance(
    spec: SiteSpec, subject_pose_world: Optional[Sequence[float]],
) -> Optional[float]:
    """Distance under the site's own metric, or None when unresolvable."""
    points = site_pair_points(spec, subject_pose_world)
    if points is None:
        return None
    source, site = points
    if spec.metric == METRIC_PLANAR:
        return float(np.linalg.norm(source[:2] - site[:2]))
    return float(np.linalg.norm(source - site))


def reached_holds(
    spec: SiteSpec, subject_pose_world: Optional[Sequence[float]],
) -> Optional[bool]:
    """Whether the pair satisfies the environment's spatial success test.

    Strictly the spatial half. PickCube also requires the robot to be static
    and PullCubeTool evaluates nothing else, so this is an exact predicate for
    one and an exact component of the other -- never an approximation of
    either.

    A region is ``<`` because PullCubeTool's ``evaluate`` is; a point and a
    surface are ``<=`` because PickCube's ``is_obj_placed`` is. The asymmetry
    is the environments', not ours.
    """
    distance = site_distance(spec, subject_pose_world)
    if distance is None:
        return None
    if spec.declaration.site_type == SITE_REGION:
        return bool(distance < float(spec.tolerance))
    return bool(distance <= float(spec.tolerance))
