"""Mine normal-ManiSkill shards into the runtime affordance and whitelist assets.

Reads ``data/maniskill_evidence/<env-id>/shard_*.pkl`` and writes the same two
asset shapes MS-HAB uses, so the runtime loads them through the existing
loaders:

    scenegraph/configs/affordances/<env-id>.json
    scenegraph/configs/subtask_whitelists/<env-id>/task_all.json

Only complete, non-incidental buckets reach the assets. A bucket that never
filled, or that was frozen out as a brush, is reported and excluded -- mining
an affordance from a near-miss is worse than not having one.

    python -m scenegraph.tools.build_maniskill_assets --env-id StackCube-v1
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scenegraph.adapters.contact_geometry import directions_to_local, to_local
from scenegraph.core.affordance import transform_dir
from scenegraph.adapters.interaction_events import (
    EE_KEY, KIND_EE_OBJECT, KIND_OBJECT_REGION, KIND_OBJECT_SITE,
)
from scenegraph.core import spatial_metrics
from scenegraph.core.spatial_metrics import (
    EE_OBJECT_SCOPE,
    OBJECT_OBJECT_SCOPE,
    SPATIAL_SCOPES,
    spatial_bin_key,
    stat_key,
)
from scenegraph.core.relation_rules import required_bin_keys
from scenegraph.core.sites import parse_site_declarations
from scenegraph.core.whitelist import WHITELIST_SCHEMA_VERSION, derive_bin_edges

# Equal-population edges for unsigned distance only. Signed quantities keep
# the symmetric, zero-centred deadbands produced by ``derive_bin_edges``:
# otherwise a skewed scene distribution can classify exactly zero as
# ``above`` or ``increase-slow``.
_EDGE_PROBS = (0.2, 0.4, 0.6, 0.8)
_QUANTILE_KEYS = {
    stat_key(scope, "planar-distance"): spatial_bin_key(scope, "planar-distance")
    for scope in SPATIAL_SCOPES
}

# Signed scales come from one magnitude, and a raw maximum hands that
# magnitude to the worst frame in the run. PullCubeTool knocks its cube off
# the table in ~9% of frames; the 0.88m that produces is a true measurement
# and a useless scale, since it puts every on-table height in one bin.
# Equal-population planar edges never had this problem -- a tail cannot move
# a quantile -- so only the signed streams need the guard.
#
# Changes take a much higher quantile than absolutes. Most frames are
# stationary, so their distribution is a spike at zero with a thin tail of
# real motion; a low quantile would shrink the stable band to nothing and
# report every frame as moving. Only the discontinuity a fall produces
# (0.38 against a clean maximum of 0.08) needs dropping.
_ABS_QUANTILE = 0.9
_CHANGE_QUANTILE = 0.99
# Below this a quantile is noise; keep the maximum instead.
_MIN_ROBUST_SAMPLES = 100


def _robust_scale(values, is_change: bool) -> Optional[float]:
    """Quantile of the magnitudes, or None when there is too little data."""
    arr = np.abs(np.asarray(values, dtype=float).reshape(-1))
    arr = arr[np.isfinite(arr)]
    if arr.size < _MIN_ROBUST_SAMPLES:
        return None
    q = _CHANGE_QUANTILE if is_change else _ABS_QUANTILE
    value = float(np.quantile(arr, q))
    return value if value > 0.0 else None

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "scenegraph" / "configs"
# Reviewed site declarations are repository input, read regardless of where
# the mined assets are written. Conflating the two made a pilot that wrote to
# /tmp look for its declarations there too, and find none.
SITES_DIR = CONFIGS / "sites"
AFFORDANCE_SCHEMA_VERSION = 5
# Shards below this carry no interaction traces, no env predicate traces
# and no raw presence counts.
SHARD_SCHEMA_MIN = 4
# Below this a shard has no keyed calibration reservoir and no collision
# extents. Both are unrecoverable after the fact: the end-effector streams
# record a height and discard which object produced it, and nothing else in the
# shard measures how big an actor is. A v4 shard still mines the assets it
# always could; it just cannot produce per-family scales, site or region
# scales, or a structural-surface classification.
SHARD_SCHEMA_KEYED = 5
TASK_SUBTASK = "task"
TASK_TARGET = "all"

# A supporter whose smaller horizontal half-extent reaches this is an extended
# plane rather than a localized destination. Measured, not guessed: across the
# shipped tasks the tabletop reports 0.605 m against 0.025 m for PlaceSphere's
# bin, 0.123 m for the peg box and 0.022 m for the peg, so 0.30 m sits five
# times clear of both sides. The *smaller* horizontal extent is used because a
# long thin object -- a peg, a rail -- is not a surface you can place things
# anywhere on.
#
# Flatness deliberately plays no part: the tabletop's vertical half-extent is
# 0.46 m because the actor includes its legs, so an aspect-ratio test would
# reject the one object this exists to find.
STRUCTURAL_SURFACE_MIN_HALF_EXTENT = 0.30

# A pair whose two support/contain orientations differ by less than this factor
# is genuinely ambiguous; anything above it is one real orientation plus force
# -sign noise, which is what the collected data actually looks like.
ORIENTATION_RATIO = 5.0


def _unit(v) -> Optional[List[float]]:
    arr = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(arr))
    return None if n <= 0 else (arr / n).tolist()


def _round(v, nd=6) -> List[float]:
    return [round(float(x), nd) for x in np.asarray(v, dtype=float).reshape(-1)]


def _parse_bucket(text: str) -> Tuple[str, str, str]:
    relation, src, dst = [p.strip() for p in text.split("/", 2)]
    return relation, src, dst


def load_shards(env_id: str, root: Path) -> Dict[str, Any]:
    """Merge every shard for one env. Samples concatenate; maxima take the max."""
    paths = sorted((root / env_id).glob("shard_*.pkl"))
    if not paths:
        raise SystemExit(f"no shards under {root / env_id}")
    merged: Dict[str, Any] = {
        "env_id": env_id, "episodes": 0, "target": 0,
        "samples": defaultdict(list), "presence": {}, "excluded": {},
        "symmetry": {}, "bin_stats": defaultdict(float), "capability": None,
        "bin_samples": defaultdict(list),
        "bin_pose_pairs": [],
        "bin_keyed_pairs": [],
        "extents": {},
        # The oldest shard in the set decides what this merge can calibrate.
        "schema_version": None,
        "episode_presence": defaultdict(int),
        # One entry per successful episode, never flattened: the "present in
        # every successful rollout" rule counts episodes, so concatenating
        # them would destroy the denominator.
        "traces": [],
        "complete": set(),
    }
    for path in paths:
        with open(path, "rb") as f:
            shard = pickle.load(f)
        version = int(shard.get("_schema_version", 0))
        if version < SHARD_SCHEMA_MIN:
            raise SystemExit(
                f"{path} is schema v{version}; v{SHARD_SCHEMA_MIN} or newer is "
                "required (older shards carry no object-pair pose reservoir, "
                "so object-object bins would fall back to origins). "
                "Re-collect this task."
            )
        merged["episodes"] += int(shard.get("episodes", 0))
        merged["traces"].extend(shard.get("traces") or [])
        for key, n in (shard.get("episode_presence") or {}).items():
            merged["episode_presence"][key] += int(n)
        merged["target"] = max(merged["target"], int(shard.get("target", 0)))
        merged["capability"] = merged["capability"] or shard.get("capability")
        merged["symmetry"].update(shard.get("symmetry") or {})
        merged["presence"].update(shard.get("presence") or {})
        merged["excluded"].update(shard.get("excluded") or {})
        merged["complete"] |= set(shard.get("complete") or [])
        for key, arr in (shard.get("bin_samples") or {}).items():
            merged["bin_samples"][key].append(np.asarray(arr))
        merged["bin_pose_pairs"].extend(shard.get("bin_pose_pairs") or [])
        merged["bin_keyed_pairs"].extend(shard.get("bin_keyed_pairs") or [])
        merged["extents"].update(shard.get("extents") or {})
        merged["schema_version"] = (
            version if merged["schema_version"] is None
            else min(merged["schema_version"], version)
        )
        for key, value in (shard.get("bin_stats") or {}).items():
            merged["bin_stats"][key] = max(merged["bin_stats"][key],
                                           float(value))
        for bucket, samples in (shard.get("samples") or {}).items():
            merged["samples"][bucket].extend(samples)
    # A rate is not mergeable -- the per-shard value would just be the last
    # one written, measured over that shard's episodes while the denominator
    # here counts all of them. Recompute from the summed counts. Excluded
    # buckets keep their frozen-in rate: they stop accumulating presence, so a
    # recomputed rate would decay toward zero and misreport why they went.
    if merged["episodes"]:
        for bucket, n in merged["episode_presence"].items():
            if bucket not in merged["excluded"]:
                merged["presence"][bucket] = n / merged["episodes"]
    print(f"[mine] {len(paths)} shard(s), {merged['episodes']} episodes, "
          f"{len(merged['samples'])} buckets, {len(merged['traces'])} traces")
    return merged


def usable_buckets(merged: Dict[str, Any], target: int,
                   min_presence: float = 0.0) -> Dict[str, List[Dict]]:
    """Complete, non-incidental buckets only.

    Presence is re-checked here against the whole run. The collector can only
    judge it at freeze time, over the few dozen episodes discovery took, and a
    bucket that looked frequent then can drift well below the gate by the end
    while still filling to target.
    """
    out, dropped = {}, []
    for bucket, samples in merged["samples"].items():
        presence = float(merged["presence"].get(bucket, 1.0))
        if bucket in merged["excluded"]:
            dropped.append((bucket, "incidental at freeze"))
        elif min_presence > 0.0 and presence < min_presence:
            dropped.append(
                (bucket, f"final presence {presence:.0%} < {min_presence:.0%}"))
        elif len(samples) < target:
            dropped.append((bucket, f"only {len(samples)}/{target}"))
        else:
            out[bucket] = samples[:target]
    for bucket, why in sorted(dropped):
        print(f"[mine] excluded ({why}): {bucket}")
    return out


# --------------------------------------------------------------------------- #
# Component derivation
# --------------------------------------------------------------------------- #
def _grasp_components(samples, symmetry) -> List[Dict[str, Any]]:
    """TCP pose in the object frame, one component per sample.

    A spherical object gets a single radial component instead: its orientation
    is meaningless, so 300 poses in its frame would record 300 arbitrary
    angular coordinates that no runtime comparison can use.
    """
    anchors, dirs, widths = [], [], []
    for s in samples:
        p = s["payload"]
        tcp, obj = p.get("tcp_pose"), p.get("obj_pose")
        if tcp is None or obj is None:
            continue
        anchors.append(to_local(np.asarray(tcp[:3])[None, :], obj)[0])
        # Gripper approach is the TCP local z axis.
        zaxis = _rot(tcp)[:, 2]
        dirs.append(directions_to_local(zaxis[None, :], obj)[0])
        if p.get("gripper_width") is not None:
            widths.append(float(p["gripper_width"]))
    if not anchors:
        return []

    if symmetry.get("symmetry") == "spherical":
        radii = [float(np.linalg.norm(a)) for a in anchors]
        return [{
            "anchor": [0.0, 0.0, 0.0],
            "radial_offset": round(float(np.mean(radii)), 6),
            "radial_spread": round(float(np.std(radii)), 6),
            "width": round(float(np.mean(widths)), 6) if widths else None,
            "orientation_invariant": True,
            "n_samples": len(anchors),
        }]
    return [{
        "anchor": _round(a),
        "approach_dir": _unit(d),
        "width": round(w, 6) if w is not None else None,
    } for a, d, w in zip(anchors, dirs,
                         widths or [None] * len(anchors))]


def _rot(pose) -> np.ndarray:
    w, x, y, z = [float(v) for v in np.asarray(pose, dtype=float)[3:7]]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def _paired_contact(samples, a_key: str, b_key: str
                    ) -> Tuple[List[Dict], List[Dict]]:
    """Index-aligned contact components for both endpoints of one pair.

    Both sides come from the same physical event, so the runtime compares
    ``a[i]`` against ``b[i]`` -- N comparisons, never an N-by-N product.
    """
    a_comps, b_comps = [], []
    for s in samples:
        p = s["payload"]
        if "anchor_a_local" not in p or "anchor_b_local" not in p:
            continue
        # ``partner`` is what keeps the two lists index-aligned once an
        # object touches more than one thing.
        a_comps.append({"anchor": _round(p["anchor_a_local"]),
                        "outward_normal": _unit(p.get("normal_a_local")
                                                or [0, 0, 1]),
                        "partner": b_key})
        b_comps.append({"anchor": _round(p["anchor_b_local"]),
                        "outward_normal": _unit(p.get("normal_b_local")
                                                or [0, 0, -1]),
                        "partner": a_key})
    return a_comps, b_comps


def _support_components(samples, supporter_key, supported_key, symmetry
                        ) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Pair-specific surface and bottom descriptors from paired contact anchors.

    Origins are not contact points -- a table's sits ~0.9m below its own top --
    so averaging them described the surface as being under the floor and made a
    resting bin read as far-above. Both sides come from the same event, and
    each carries the partner it was mined against.
    """
    surface, s_normals, bottom, b_normals = [], [], [], []
    for s in samples:
        p = s["payload"]
        if "anchor_a_local" not in p or "anchor_b_local" not in p:
            continue
        ka, kb = p.get("key_a"), p.get("key_b")
        if ka == supporter_key and kb == supported_key:
            sa, sn = p["anchor_a_local"], p.get("normal_a_local")
            ba, bn = p["anchor_b_local"], p.get("normal_b_local")
        elif kb == supporter_key and ka == supported_key:
            sa, sn = p["anchor_b_local"], p.get("normal_b_local")
            ba, bn = p["anchor_a_local"], p.get("normal_a_local")
        else:
            continue
        surface.append(sa)
        s_normals.append(sn or [0.0, 0.0, 1.0])
        bottom.append(ba)
        b_normals.append(bn or [0.0, 0.0, -1.0])
    if not surface:
        return None, None

    arr = np.asarray(surface, dtype=float)
    mean = arr.mean(axis=0)
    spread = (float(np.max(np.linalg.norm(arr[:, :2] - mean[:2], axis=1)))
              if len(arr) > 1 else 0.01)
    surface_comp = {
        "surface_anchor": _round(mean),
        "surface_normal": (_unit(np.asarray(s_normals, dtype=float).mean(axis=0))
                           or [0.0, 0.0, 1.0]),
        "footprint_radius": round(max(0.01, spread), 6),
        "partner": supported_key,
        "n_samples": len(arr),
    }
    barr = np.asarray(bottom, dtype=float)
    bottom_comp = {
        "bottom_anchor": _round(barr.mean(axis=0)),
        "bottom_normal": (_unit(np.asarray(b_normals, dtype=float).mean(axis=0))
                          or [0.0, 0.0, -1.0]),
        "partner": supporter_key,
        "n_samples": len(barr),
    }
    if (symmetry.get(supported_key) or {}).get("symmetry") == "spherical":
        # A fixed local point orbits with the ball, so an unmoved but spinning
        # sphere would change its spatial relations. Store the radius instead.
        bottom_comp["bottom_anchor"] = [0.0, 0.0, 0.0]
        bottom_comp["bottom_normal"] = [0.0, 0.0, -1.0]
        bottom_comp["radial_offset"] = round(
            float(np.mean(np.linalg.norm(barr, axis=1))), 6)
        bottom_comp["orientation_invariant"] = True
    return surface_comp, bottom_comp


def _contain_components(samples) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Entry descriptor for the container, key descriptor for the containee.

    Both are expressed in the hole/slot frame rather than the container actor's
    origin, because that is where the feature actually is.
    """
    entries, keys, radii, depths = [], [], [], []
    for s in samples:
        p = s["payload"]
        hole, con_pose = p.get("hole_pose"), p.get("container_pose")
        key_pose, tee_pose = p.get("key_pose"), p.get("containee_pose")
        if hole is None or con_pose is None:
            continue
        entries.append(to_local(np.asarray(hole[:3])[None, :], con_pose)[0])
        axis = directions_to_local(_rot(hole)[:, 0][None, :], con_pose)[0]
        if p.get("hole_half_width") is not None:
            radii.append(float(p["hole_half_width"]))
        depths.append(abs(float(p.get("axial", 0.0))))
        if key_pose is not None and tee_pose is not None:
            keys.append(to_local(np.asarray(key_pose[:3])[None, :],
                                 tee_pose)[0])
    if not entries:
        return None, None
    entry_comp = {
        "entry_anchor": _round(np.asarray(entries).mean(axis=0)),
        "entry_axis": _unit(axis),
        "opening_radius": round(float(np.mean(radii)), 6) if radii else 0.02,
        "depth": round(float(np.max(depths)), 6) if depths else 0.05,
        "n_samples": len(entries),
    }
    key_comp = None
    if keys:
        key_comp = {
            "key_anchor": _round(np.asarray(keys).mean(axis=0)),
            "key_axis": _unit(axis),
            "n_samples": len(keys),
        }
    return entry_comp, key_comp


def _dominant_orientation(counts_by_bucket: Dict[str, int], relation: str,
                          ) -> Dict[Tuple[str, str], Tuple[str, str]]:
    """Pick one role orientation per unordered pair.

    Collection emits ``A->B`` and ``B->A`` as separate buckets, but the runtime
    stores one edge and names the role in the label, so a pair must claim one
    orientation. Real data splits these hugely lopsidedly -- 300 against 5 --
    because the force-sign test flips transiently mid-insertion. Raising on
    that would reject good evidence, so the dominant side wins and only a
    genuinely close split is an error.
    """
    # Counted over every discovered bucket, not only the complete ones.
    # Filtering incompleteness first would let a near-tie resolve silently
    # whenever one side happened to fall short of target.
    counts: Dict[Tuple[str, str], Dict[Tuple[str, str], int]] = defaultdict(dict)
    for bucket, n in counts_by_bucket.items():
        rel, src, dst = _parse_bucket(bucket)
        if rel != relation or EE_KEY in (src, dst):
            continue
        counts[tuple(sorted((src, dst)))][(src, dst)] = int(n)

    chosen = {}
    for pair, options in counts.items():
        ranked = sorted(options.items(), key=lambda kv: -kv[1])
        best, best_n = ranked[0]
        if len(ranked) > 1:
            runner, runner_n = ranked[1]
            if runner_n * ORIENTATION_RATIO > best_n:
                raise SystemExit(
                    f"{relation} orientation is ambiguous for {pair}: "
                    f"{best}={best_n} vs {runner}={runner_n}. Extend the "
                    "schema rather than picking one silently."
                )
            print(f"[mine] {relation} {pair}: kept {best} ({best_n}), "
                  f"dropped {runner} ({runner_n}) as force-sign noise")
        chosen[pair] = best
    return chosen


# --------------------------------------------------------------------------- #
# Assets
# --------------------------------------------------------------------------- #
def orientation_counts(merged: Dict[str, Any]) -> Dict[str, int]:
    """Sample counts per bucket, minus incidental ones. A brush must not make
    a pair look ambiguous."""
    return {b: len(s) for b, s in merged["samples"].items()
            if b not in merged["excluded"]}


def build_assets(merged: Dict[str, Any], buckets: Dict[str, List[Dict]],
                 env_id: str, sites_dir: Path = SITES_DIR
                 ) -> Tuple[Dict, Dict]:
    objects: Dict[str, Dict[str, list]] = defaultdict(
        lambda: defaultdict(list))
    members: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"roles": set(), "interaction_types": set(), "kind": "actor"})
    symmetry = merged["symmetry"]

    empty: List[str] = []
    counts = orientation_counts(merged)
    support_roles = _dominant_orientation(counts, "support")
    contain_roles = _dominant_orientation(counts, "contain")

    for bucket, samples in sorted(buckets.items()):
        rel, src, dst = _parse_bucket(bucket)
        pair = tuple(sorted((src, dst)))

        if rel == "grasp" and src == EE_KEY:
            comps = _grasp_components(samples, symmetry.get(dst, {}))
            objects[dst]["grasp_components"].extend(comps)
            members[dst]["interaction_types"].add("grasp")
            members[dst]["roles"].add("interacted")
        elif rel == "contact" and src == EE_KEY:
            members[dst]["interaction_types"].add("contact")
            members[dst]["roles"].add("interacted")
        elif rel == "contact":
            a_comps, b_comps = _paired_contact(samples, src, dst)
            if not a_comps:
                empty.append(f"{bucket} (no anchor_a_local in payload)")
            objects[src]["contact_components"].extend(a_comps)
            objects[dst]["contact_components"].extend(b_comps)
            for key in (src, dst):
                members[key]["interaction_types"].add("contact")
                members[key]["roles"].add("interacted")
        elif rel == "support":
            if support_roles.get(pair) != (src, dst):
                continue
            surface, bottom = _support_components(samples, src, dst, symmetry)
            if surface is None:
                empty.append(f"{bucket} (no paired contact anchors)")
            if surface:
                objects[src]["support_components"].append(surface)
            if bottom:
                objects[dst]["bottom_components"].append(bottom)
            members[src]["roles"].add("support")
            for key in (src, dst):
                members[key]["interaction_types"].add("support")
        elif rel == "contain":
            if contain_roles.get(pair) != (src, dst):
                continue
            entry, key_comp = _contain_components(samples)
            if not entry:
                empty.append(f"{bucket} (no hole_pose/container_pose)")
            if entry:
                objects[src]["contain_components"].append(entry)
            if key_comp:
                objects[dst]["key_components"].append(key_comp)
            for key in (src, dst):
                members[key]["interaction_types"].add("contain")

    # A bucket with 300 samples that yields no component means the payload
    # lost a field between collection and here. Emitting the asset anyway
    # produces a runtime that quietly scores every such relation "unobserved".
    if empty:
        raise SystemExit(
            "these buckets had evidence but produced no components, so "
            "the payload is missing fields the miner reads: "
            + "; ".join(empty)
            + ". Re-collect with the current collector before mining."
        )

    for key, sym in symmetry.items():
        if key in objects and sym.get("symmetry") != "none":
            objects[key]["symmetry"] = sym

    # Scene entities that never interacted. A goal marker has no collision
    # geometry, so it produces no contact, grasp, support or contain bucket and
    # would be dropped -- but a task whose success is "the object reaches
    # *there*" has nowhere to point without it. Admitted spatially only: no
    # components, no interaction types, so the physical and affordance families
    # skip it and only planar-distance and height-offset are emitted.
    for key in sorted(symmetry):
        if key not in members:
            members[key]["roles"].add("spatial")

    plain_members = {
        k: {"roles": sorted(v["roles"]) or ["interacted"],
            "interaction_types": sorted(v["interaction_types"]),
            "kind": "link" if k.startswith("link:") else "actor"}
        for k, v in sorted(members.items())
    }

    # A v4 shard has no extents, no keyed reservoir and no site samples, so
    # it can still mine everything it always could -- it just cannot classify
    # surfaces or families, and the runtime falls back to the single shared
    # end-effector height scale. What it must not do is claim a classification
    # it has no evidence for.
    keyed = int(merged.get("schema_version") or 0) >= SHARD_SCHEMA_KEYED
    if not keyed:
        return _legacy_assets(merged, objects, plain_members, env_id)

    surfaces = structural_surfaces(merged, plain_members)
    unreadable = unclassified_supporters(merged, plain_members)
    if unreadable:
        raise SystemExit(
            f"no collision extents were recorded for supporters {unreadable}, "
            "so a tabletop among them cannot be told from a bin and would be "
            "measured from its actor origin -- roughly 0.9m below its own top. "
            "Re-collect with the current collector (shard schema "
            f"v{SHARD_SCHEMA_KEYED}) before mining."
        )

    families = object_families(plain_members, buckets, set(surfaces))
    ambiguous = ambiguous_families(families)
    if ambiguous:
        raise SystemExit(
            f"no end-effector height family fits {ambiguous}: each took part "
            "in interactions but is neither grasped, nor a supporter or "
            "container, nor an extended surface. Assigning one anyway would "
            "give it another family's deadband. Extend the classification "
            "rather than letting the asset guess."
        )

    for key, reason in surfaces.items():
        surface = _reference_surface(merged, key, objects)
        if surface is None:
            raise SystemExit(
                f"{key} classified as a structural surface ({reason}) but no "
                "support-surface anchors were mined against it, so its top "
                "face is unknown and every height would fall back to the "
                "actor origin."
            )
        objects[key]["reference_surface"] = surface
        plain_members[key]["structural_surface"] = True
        plain_members[key]["structural_surface_reason"] = reason
    for key, family in families.items():
        plain_members[key]["family"] = family

    sites = site_declarations(merged, plain_members, env_id, sites_dir)

    affordances = {
        "_schema_version": AFFORDANCE_SCHEMA_VERSION,
        "env_id": env_id,
        "objects": {k: {kk: vv for kk, vv in v.items()}
                    for k, v in sorted(objects.items())},
    }
    whitelist = {
        "_schema_version": WHITELIST_SCHEMA_VERSION,
        "subtask": TASK_SUBTASK,
        "task_group": env_id,
        "target": TASK_TARGET,
        "members": plain_members,
        "sites": sites,
        "bin_edges": _bin_edges(merged, objects, families, sites),
        "episodes": merged["episodes"],
    }
    return affordances, whitelist


def _legacy_assets(merged, objects, plain_members, env_id):
    """Assets from a pre-keyed shard: no families, no surfaces, no sites."""
    return (
        {
            "_schema_version": AFFORDANCE_SCHEMA_VERSION,
            "env_id": env_id,
            "objects": {k: {kk: vv for kk, vv in v.items()}
                        for k, v in sorted(objects.items())},
        },
        {
            "_schema_version": WHITELIST_SCHEMA_VERSION,
            "subtask": TASK_SUBTASK,
            "task_group": env_id,
            "target": TASK_TARGET,
            "members": plain_members,
            "bin_edges": _bin_edges(merged, objects),
            "episodes": merged["episodes"],
        },
    )


def _anchor_index(objects):
    """``(supporter, supported) -> (surface_anchor, bottom_anchor, radial)``."""
    index = {}
    bottoms = {}
    for key, entry in objects.items():
        for comp in entry.get("bottom_components", []) or []:
            partner = comp.get("partner")
            if partner:
                bottoms[(str(partner), key)] = (
                    comp.get("bottom_anchor"), comp.get("radial_offset"))
    for key, entry in objects.items():
        for comp in entry.get("support_components", []) or []:
            partner = comp.get("partner")
            if not partner:
                continue
            bottom = bottoms.get((key, str(partner)))
            if bottom is None:
                continue
            index[(key, str(partner))] = (
                comp.get("surface_anchor"), bottom[0], bottom[1])
    return index


def _pair_measures(index, key_a, key_b, pose_a, pose_b):
    """``(planar, height)`` in ``a - b`` order, anchored where mined."""
    spec = index.get((key_a, key_b))
    if spec is not None:
        anchor_a, anchor_b, radial_b = spec
        radial_a = None
    else:
        spec = index.get((key_b, key_a))
        if spec is None:
            anchor_a = anchor_b = radial_a = radial_b = None
        else:
            anchor_b, anchor_a, radial_a = spec
            radial_b = None
    points = spatial_metrics.pair_points(
        pose_a, pose_b, anchor_a, anchor_b, radial_a, radial_b)
    return spatial_metrics.measures(*points)


def _object_pair_stats(merged, objects):
    """Object-object maxima, measured the way the runtime will measure them.

    The pose reservoir travels raw because the anchors do not exist until this
    point in the pipeline. Reprojecting here is what stops the scale being
    calibrated on origins while the labels are read off surfaces.
    """
    index = _anchor_index(objects)
    stats = defaultdict(float)
    pd_key = stat_key(OBJECT_OBJECT_SCOPE, "planar-distance")
    ho_key = stat_key(OBJECT_OBJECT_SCOPE, "height-offset")
    planar_samples = []
    signed = defaultdict(list)
    for rec in merged.get("bin_pose_pairs") or []:
        a, b = rec.get("key_a"), rec.get("key_b")
        pose_a, pose_b = rec.get("pose_a"), rec.get("pose_b")
        if not a or not b or pose_a is None or pose_b is None:
            continue
        planar, height = _pair_measures(index, a, b, pose_a, pose_b)
        stats[pd_key] = max(stats[pd_key], abs(planar))
        stats[ho_key] = max(stats[ho_key], abs(height))
        signed[ho_key].append(height)
        planar_samples.append(planar)
        prev_a, prev_b = rec.get("prev_pose_a"), rec.get("prev_pose_b")
        if prev_a is None or prev_b is None:
            continue
        old_planar, old_height = _pair_measures(index, a, b, prev_a, prev_b)
        stats[pd_key + "_change"] = max(
            stats[pd_key + "_change"], abs(planar - old_planar))
        stats[ho_key + "_change"] = max(
            stats[ho_key + "_change"], abs(height - old_height))
        signed[pd_key + "_change"].append(planar - old_planar)
        signed[ho_key + "_change"].append(height - old_height)

    for key, values in signed.items():
        robust = _robust_scale(values, key.endswith("_change"))
        if robust is not None:
            stats[key] = robust
    return dict(stats), np.asarray(planar_samples, dtype=np.float32)


def structural_surfaces(merged, members) -> Dict[str, str]:
    """Which members are extended support planes, and why.

    Two conditions, both necessary. It has to actually support something --
    a large object nothing ever rests on is scenery, not a surface -- and its
    smaller horizontal half-extent has to reach
    ``STRUCTURAL_SURFACE_MIN_HALF_EXTENT``.

    Size is the only available discriminator. Roles cannot do it: in
    PlaceSphere the bin and the table carry byte-identical ``roles`` and
    ``interaction_types``, because both are kinematic and both support the
    sphere. An actor whose collision geometry could not be read is left
    unclassified rather than assumed small -- a missing measurement is not
    evidence of absence, and a table quietly demoted to an ordinary object
    reinstates the ~0.9m origin error this is here to remove.
    """
    extents = merged.get("extents") or {}
    out: Dict[str, str] = {}
    for key, entry in sorted(members.items()):
        if "support" not in (entry.get("interaction_types") or ()):
            continue
        half = extents.get(key)
        if not half or len(half) < 3:
            continue
        horizontal = min(float(half[0]), float(half[1]))
        if horizontal >= STRUCTURAL_SURFACE_MIN_HALF_EXTENT:
            out[key] = (
                f"collision half-extent {horizontal:.3f}m horizontal >= "
                f"{STRUCTURAL_SURFACE_MIN_HALF_EXTENT}m"
            )
    return out


def unclassified_supporters(merged, members) -> List[str]:
    """Supporters whose collision geometry the shard never recorded.

    Reported rather than defaulted. Every one of them is a member the runtime
    will measure from its actor origin, and if one of them is a tabletop that
    is exactly the failure this change exists to remove.
    """
    extents = merged.get("extents") or {}
    return sorted(
        key for key, entry in members.items()
        if "support" in (entry.get("interaction_types") or ())
        and not extents.get(key)
    )


def declared_sites(env_id: str, sites_dir: Path) -> Dict[str, Any]:
    """The reviewed site declarations for one task, or none.

    Sites are task semantics, not something a pose stream reveals. The
    collector records region pairs against *every* movable actor, because it
    cannot know which one a task drags into its goal -- so letting the miner
    turn that into a declaration invented a goal region for PlaceSphere, a
    task with no goal region at all.

    So the declaration is written and reviewed, like a schedule, and the miner
    only checks the evidence agrees with it. Absent file means no sites.
    """
    path = Path(sites_dir) / f"{env_id}.json"
    if not path.is_file():
        return {}
    with open(path) as handle:
        raw = json.load(handle)
    return dict(raw.get("sites") or {})


def keyed_pair_kinds(merged) -> Dict[str, set]:
    """``kind -> {(src_key, dst_key)}`` actually present in the reservoir."""
    kinds: Dict[str, set] = defaultdict(set)
    for record in merged.get("bin_keyed_pairs") or []:
        kinds[record.get("kind")].add(
            (record.get("src_key"), record.get("dst_key")))
    return kinds


def site_declarations(merged, members, env_id, sites_dir) -> Dict[str, Any]:
    """Reviewed declarations, checked against what the shard recorded.

    Every declared site has to be backed by samples and by members that exist,
    and every kind of evidence has to be claimed by some declaration. A site
    nothing measured would calibrate no scale and label nothing; evidence
    nothing declared is a task whose goal geometry was collected and then
    silently dropped. Both are errors here rather than surprises at runtime.
    """
    declared = declared_sites(env_id, sites_dir)
    kinds = keyed_pair_kinds(merged)
    site_pairs = kinds.get(KIND_OBJECT_SITE, set())
    region_pairs = kinds.get(KIND_OBJECT_REGION, set())

    # Checked before the empty-declaration shortcut, not after. Returning
    # early on "no declarations" is exactly how a task that *does* expose goal
    # geometry ends up silently dropping it -- which is what happened to
    # PegInsertionSide when the declaration directory was pointed at the
    # asset output directory instead of the repo.
    if site_pairs and not any(
            e.get("site_type") == "surface" for e in declared.values()):
        raise SystemExit(
            f"{env_id}: the shard holds object-site samples "
            f"{sorted(site_pairs)} that no declaration claims. The live hole "
            f"geometry was collected and would be thrown away. Declare the "
            f"site in {sites_dir}/{env_id}.json, or pass --sites."
        )
    if not declared:
        return {}

    for key, entry in sorted(declared.items()):
        subject = entry.get("subject")
        if subject not in members:
            raise SystemExit(
                f"{env_id}: site {key!r} names subject {subject!r}, which is "
                "not a whitelist member of this task."
            )
        kind = entry.get("site_type")
        if kind == "surface" and (subject, key) not in site_pairs:
            raise SystemExit(
                f"{env_id}: site {key!r} is declared but the shard holds no "
                f"object-site samples for ({subject}, {key}). Its ladder "
                "would have no scale. Re-collect, or drop the declaration."
            )
        if kind == "region" and (subject, key) not in region_pairs:
            raise SystemExit(
                f"{env_id}: site {key!r} is declared but the shard holds no "
                f"object-region samples for ({subject}, {key})."
            )
    return declared


def object_families(members, buckets, structural) -> Dict[str, Optional[str]]:
    """The end-effector height family for every member, in strict precedence.

    1. A structural surface is one, whatever else it does. A tabletop is also
       grasped by nothing and supports everything, so the later rules would
       reach it too -- and give it the wrong answer.
    2. Anything the demos grasped is a manipuland. This is what the gripper
       approaches and lifts, and its heights are the ones the deadband has to
       resolve.
    3. Anything that held something else -- the supporter of a support bucket
       or the container of a contain bucket -- is a receptacle. Read from the
       directed buckets, not from ``interaction_types``, which is a flat set
       per member and says only that the object took part.
    4. A member with no interactions at all is a goal marker: PickCube's
       ``goal_site`` has no collision geometry, so it appears in no bucket and
       exists purely to be measured against.

    Anything else is left ``None``. That is not a family and must not be
    treated as one -- see ``ambiguous_families``.
    """
    holders: set = set()
    for bucket in buckets:
        parts = [p.strip() for p in str(bucket).split("/")]
        if len(parts) == 3 and parts[0] in ("support", "contain"):
            holders.add(parts[1])

    out: Dict[str, Optional[str]] = {}
    for key, entry in members.items():
        types = set((entry or {}).get("interaction_types") or ())
        if key in structural:
            out[key] = spatial_metrics.FAMILY_STRUCTURAL
        elif "grasp" in types:
            out[key] = spatial_metrics.FAMILY_MANIPULAND
        elif key in holders:
            out[key] = spatial_metrics.FAMILY_RECEPTACLE
        elif not types:
            out[key] = spatial_metrics.FAMILY_GOAL_MARKER
        else:
            out[key] = None
    return out


def ambiguous_families(families) -> List[str]:
    """Members no rule classified.

    A member that took part in interactions but is neither grasped, nor a
    holder, nor structural. Falling back to a family would give it another
    family's deadband, which is how one token comes to mean two heights; the
    miner refuses to write the asset instead.
    """
    return sorted(key for key, family in families.items() if not family)


def _reference_surface(merged, key, objects) -> Optional[Dict[str, Any]]:
    """The top face of a structural surface, in its own object frame.

    Derived from the support anchors already mined against it -- those points
    lie on the face things rest on, which is the plane by definition. The
    normal is forced upward here rather than at read time as well, so the
    stored asset means what it says.
    """
    supports = (objects.get(key) or {}).get("support_components") or []
    anchors = [np.asarray(c["surface_anchor"], dtype=float)
               for c in supports if c.get("surface_anchor") is not None]
    if not anchors:
        return None
    normals = [np.asarray(c["surface_normal"], dtype=float)
               for c in supports if c.get("surface_normal") is not None]
    normal = np.array([0.0, 0.0, 1.0])
    if normals:
        mean = np.mean(np.vstack(normals), axis=0)
        oriented = spatial_metrics.oriented_normal(mean)
        if oriented is not None:
            normal = oriented
    return {
        "anchor": _round(np.mean(np.vstack(anchors), axis=0)),
        "outward_normal": _round(normal),
        "n_samples": len(anchors),
        "provenance": "mean of mined support-surface anchors, normal forced "
                      "outward (support normals are mined from contact "
                      "forces and point into the supporter)",
    }


def _ee_structural_height(src, dst_pose, surface) -> Optional[float]:
    """End-effector height above a structural surface's plane, not its origin.

    The reservoir travels raw because nothing knows which members are surfaces
    until the extents have been read and the anchors mined, so the correction
    happens here -- the same reprojection the object-pair scales already get.
    Without it the table's scale is calibrated on the ~1m to its own origin
    while the runtime labels the ~0.15m to its top, and the deadband comes out
    an order of magnitude too wide.
    """
    anchor = spatial_metrics.anchor_world(dst_pose, surface.get("anchor"))
    normal = transform_dir(dst_pose, surface.get("outward_normal"))
    if anchor is None or normal is None:
        return None
    return spatial_metrics.surface_height(src[:3], anchor, normal)


def _declared_pairs(sites) -> set:
    """``(subject, site)`` pairs a declaration claims.

    Only these calibrate. The collector records region samples against every
    movable actor because it cannot know the subject, so calibrating on all of
    them would set PullCubeTool's goal scale from the tool as well as the cube
    -- and would invent a scale entirely for tasks with no region at all.
    """
    return {(entry.get("subject"), key) for key, entry in (sites or {}).items()}


def _keyed_stats(merged, families, objects=None, declared=None
                 ) -> Dict[str, float]:
    """Scales that only the keyed reservoir can produce.

    Per-family end-effector heights, the object-to-site ladder and the
    object-to-region ladder. All three are unavailable from a v4 shard: its
    end-effector streams record a height and discard which object produced it,
    and it has no site or region samples at all.
    """
    signed: Dict[str, List[float]] = defaultdict(list)
    unsigned: Dict[str, List[float]] = defaultdict(list)

    def _pair(record, key):
        src = np.asarray(record.get("src_pose"), dtype=float)
        dst = np.asarray(record.get("dst_pose"), dtype=float)
        if src.size < 3 or dst.size < 3:
            return None, None
        return src, dst

    for record in merged.get("bin_keyed_pairs") or []:
        kind = record.get("kind")
        if kind in (KIND_OBJECT_SITE, KIND_OBJECT_REGION):
            if declared is None or (
                    record.get("src_key"), record.get("dst_key")
            ) not in declared:
                continue
        src, dst = _pair(record, kind)
        if src is None:
            continue
        planar = float(np.linalg.norm(src[:2] - dst[:2]))
        height = float(src[2] - dst[2])

        if kind == KIND_EE_OBJECT:
            dst_key = record.get("dst_key")
            family = families.get(dst_key)
            if not family:
                continue
            if family == spatial_metrics.FAMILY_STRUCTURAL:
                surface = ((objects or {}).get(dst_key) or {}).get(
                    "reference_surface")
                if surface is None:
                    continue
                surface_h = _ee_structural_height(src, dst, surface)
                if surface_h is None:
                    continue
                height = surface_h
            signed[spatial_metrics.ee_family_bin_key(family)].append(height)
        elif kind == KIND_OBJECT_SITE:
            unsigned[spatial_metrics.OBJECT_SITE_PLANAR_KEY].append(planar)
            signed[spatial_metrics.OBJECT_SITE_HEIGHT_KEY].append(height)
        elif kind == KIND_OBJECT_REGION:
            unsigned[spatial_metrics.OBJECT_REGION_PLANAR_KEY].append(planar)
        else:
            continue

        prev_src, prev_dst = record.get("prev_src_pose"), record.get("prev_dst_pose")
        if prev_src is None or prev_dst is None:
            continue
        ps = np.asarray(prev_src, dtype=float)
        pd = np.asarray(prev_dst, dtype=float)
        old_planar = float(np.linalg.norm(ps[:2] - pd[:2]))
        old_height = float(ps[2] - pd[2])
        if kind == KIND_EE_OBJECT:
            dst_key = record.get("dst_key")
            family = families.get(dst_key)
            if not family:
                continue
            if family == spatial_metrics.FAMILY_STRUCTURAL:
                surface = ((objects or {}).get(dst_key) or {}).get(
                    "reference_surface")
                if surface is None:
                    continue
                old_surface = _ee_structural_height(ps, pd, surface)
                if old_surface is None:
                    continue
                old_height = old_surface
            signed[spatial_metrics.change_bin_key(
                spatial_metrics.ee_family_bin_key(family))].append(
                    height - old_height)
        elif kind == KIND_OBJECT_SITE:
            signed[spatial_metrics.change_bin_key(
                spatial_metrics.OBJECT_SITE_PLANAR_KEY)].append(
                    planar - old_planar)
            signed[spatial_metrics.change_bin_key(
                spatial_metrics.OBJECT_SITE_HEIGHT_KEY)].append(
                    height - old_height)
        elif kind == KIND_OBJECT_REGION:
            signed[spatial_metrics.change_bin_key(
                spatial_metrics.OBJECT_REGION_PLANAR_KEY)].append(
                    planar - old_planar)

    stats: Dict[str, float] = {}
    for key, values in signed.items():
        robust = _robust_scale(values, key.endswith("-change"))
        if robust is None:
            robust = float(np.max(np.abs(values))) if values else 0.0
        if robust > 0.0:
            stats[key.replace("-", "_")] = robust
    for key, values in unsigned.items():
        if values:
            stats[key.replace("-", "_")] = float(np.max(values))
    return stats


def _keyed_planar_samples(merged, declared=None) -> Dict[str, np.ndarray]:
    """Raw planar samples per unsigned keyed scale, for equal-population edges."""
    out: Dict[str, List[float]] = defaultdict(list)
    for record in merged.get("bin_keyed_pairs") or []:
        kind = record.get("kind")
        if kind == KIND_OBJECT_SITE:
            key = spatial_metrics.OBJECT_SITE_PLANAR_KEY
        elif kind == KIND_OBJECT_REGION:
            key = spatial_metrics.OBJECT_REGION_PLANAR_KEY
        else:
            continue
        if declared is None or (
                record.get("src_key"), record.get("dst_key")) not in declared:
            continue
        src = np.asarray(record.get("src_pose"), dtype=float)
        dst = np.asarray(record.get("dst_pose"), dtype=float)
        if src.size >= 3 and dst.size >= 3:
            out[key].append(float(np.linalg.norm(src[:2] - dst[:2])))
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items() if v}


def _bin_edges(merged, objects, families=None, sites=None):
    """Scoped edges: EE from recorded stats, object-object from reprojection.

    Quantiles apply only to non-negative planar distance, where they stop an
    outlier setting the whole scale. Signed height and change keep symmetric
    zero-centred deadbands, so exactly zero is always level / stable.
    """
    stats = dict(merged["bin_stats"])
    object_stats, object_planar = _object_pair_stats(merged, objects)

    samples = {k: np.concatenate([np.asarray(c).reshape(-1) for c in chunks])
               for k, chunks in (merged.get("bin_samples") or {}).items()
               if len(chunks)}
    # Same guard on the ee-object side: a dropped object contaminates its
    # end-effector heights too.
    for key, values in samples.items():
        if key.endswith("planar_distance"):
            continue
        robust = _robust_scale(values, key.endswith("_change"))
        if robust is not None:
            stats[key] = robust
    stats.update(object_stats)
    declared = _declared_pairs(sites)
    stats.update(_keyed_stats(merged, families or {}, objects, declared))
    edges = derive_bin_edges(stats)
    if object_planar.size:
        samples[stat_key(OBJECT_OBJECT_SCOPE, "planar-distance")] = object_planar

    quantile_keys = dict(_QUANTILE_KEYS)
    for key, values in _keyed_planar_samples(merged, declared).items():
        samples[key.replace("-", "_")] = values
        quantile_keys[key.replace("-", "_")] = key

    if families:
        # Superseded. Leaving it in the asset invites something to read a
        # scale nothing calibrates against any more.
        shared = spatial_bin_key(EE_OBJECT_SCOPE, "height-offset")
        edges.pop(shared, None)
        edges.pop(spatial_metrics.change_bin_key(shared), None)

    for stat, key in quantile_keys.items():
        values = samples.get(stat)
        if values is None or values.size < 100:
            continue
        cut = sorted(float(np.quantile(values, p)) for p in _EDGE_PROBS)
        # Degenerate when a statistic barely varies; keep the max-derived scale.
        if len(set(round(c, 9) for c in cut)) < len(cut):
            continue
        edges[key] = cut
    return edges


def write_assets(affordances, whitelist, env_id: str, configs: Path) -> None:
    aff_dir = configs / "affordances"
    wl_dir = configs / "subtask_whitelists" / env_id
    aff_dir.mkdir(parents=True, exist_ok=True)
    wl_dir.mkdir(parents=True, exist_ok=True)
    aff_path = aff_dir / f"{env_id}.json"
    wl_path = wl_dir / f"{TASK_SUBTASK}_{TASK_TARGET}.json"
    with open(aff_path, "w") as f:
        json.dump(affordances, f, indent=2)
    with open(wl_path, "w") as f:
        json.dump(whitelist, f, indent=2)
    print(f"[mine] wrote {aff_path}")
    print(f"[mine] wrote {wl_path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Mine ManiSkill shards into assets")
    p.add_argument("--env-id", required=True)
    p.add_argument("--shards", default="data/maniskill_evidence")
    p.add_argument("--configs", default=str(CONFIGS),
                   help="where the mined assets are written")
    p.add_argument("--sites", default=str(SITES_DIR),
                   help="reviewed site declarations to validate against; "
                        "input, not output")
    p.add_argument("--target", type=int, default=0,
                   help="0 uses the target recorded in the shard")
    p.add_argument("--min-presence", type=float, default=0.2,
                   help="drop buckets appearing in fewer than this fraction "
                        "of successful episodes, measured over the whole run")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    merged = load_shards(args.env_id, Path(args.shards))
    target = args.target or merged["target"]
    buckets = usable_buckets(merged, target, args.min_presence)
    if not buckets:
        raise SystemExit("no complete buckets; nothing to mine")
    affordances, whitelist = build_assets(
        merged, buckets, args.env_id, Path(args.sites))

    print(f"[mine] {len(affordances['objects'])} objects, "
          f"{len(whitelist['members'])} whitelist members")
    # What this asset must calibrate depends on what it classified. A
    # families asset has no shared end-effector height scale by design -- it
    # was dropped as superseded -- so asking for one here would reject exactly
    # the assets the split was built to produce.
    # Asked of the runtime's own rule, not of a copy of it. A second copy
    # here immediately diverged: it demanded the object-site ladder for
    # PickCube's goal marker, which is a real actor riding the ordinary
    # object-object scale, and rejected a perfectly good asset.
    edges = whitelist["bin_edges"]
    families = {
        key: entry.get("family")
        for key, entry in whitelist["members"].items()
        if isinstance(entry, dict) and entry.get("family")
    }
    required = required_bin_keys({
        "families": families,
        "site_declarations": parse_site_declarations(
            whitelist.get("sites"), where=args.env_id),
        "object_object_spatial": True,
    })
    missing = sorted({k for k in required if not edges.get(k)})
    if missing:
        raise SystemExit(
            f"bin_stats calibrate no edges for {missing}; those relations "
            "would emit nothing for the whole run. Re-collect before mining."
        )
    print(f"[mine] families: "
          f"{sorted(set(families.values())) or ['(none -- legacy shard)']}")
    if whitelist.get("sites"):
        print(f"[mine] sites: {sorted(whitelist['sites'])}")
    write_assets(affordances, whitelist, args.env_id, Path(args.configs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
