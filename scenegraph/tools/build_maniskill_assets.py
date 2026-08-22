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
from scenegraph.adapters.interaction_events import EE_KEY
from scenegraph.core.whitelist import WHITELIST_SCHEMA_VERSION, derive_bin_edges

# Equal-population edges for the five-label scales.
_EDGE_PROBS = (0.2, 0.4, 0.6, 0.8)
_QUANTILE_RELATIONS = {
    "planar_distance": "planar-distance",
    "height_offset": "height-offset",
    "planar_distance_change": "planar-distance-change",
    "height_offset_change": "height-offset-change",
}

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "scenegraph" / "configs"
AFFORDANCE_SCHEMA_VERSION = 3
# Shards below this carry no interaction traces, no env predicate traces
# and no raw presence counts.
SHARD_SCHEMA_MIN = 3
TASK_SUBTASK = "task"
TASK_TARGET = "all"

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
                "required (older shards carry no per-episode traces and no "
                "raw presence counts). Re-collect this task."
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


def _support_components(samples, supporter_key) -> Tuple[Optional[Dict],
                                                         Optional[Dict]]:
    """One averaged surface component and one bottom component.

    Averaged deliberately: support is a surface, not a set of points, and the
    footprint radius carries the spread so a large table still matches an
    object resting anywhere on it.
    """
    surface, bottom = [], []
    for s in samples:
        p = s["payload"]
        pose_a, pose_b = p.get("pose_a"), p.get("pose_b")
        if pose_a is None or pose_b is None:
            continue
        if p.get("key_a") == supporter_key:
            sup_pose, sub_pose = pose_a, pose_b
        else:
            sup_pose, sub_pose = pose_b, pose_a
        surface.append(to_local(np.asarray(sub_pose[:3])[None, :], sup_pose)[0])
        bottom.append(to_local(np.asarray(sup_pose[:3])[None, :], sub_pose)[0])
    if not surface:
        return None, None

    arr = np.asarray(surface)
    mean = arr.mean(axis=0)
    spread = (float(np.max(np.linalg.norm(arr[:, :2] - mean[:2], axis=1)))
              if len(arr) > 1 else 0.01)
    surface_comp = {
        "surface_anchor": _round(mean),
        "surface_normal": [0.0, 0.0, 1.0],
        "footprint_radius": round(max(0.01, spread), 6),
        "n_samples": len(arr),
    }
    bmean = np.asarray(bottom).mean(axis=0)
    bottom_comp = {
        "bottom_anchor": _round(bmean),
        "bottom_normal": [0.0, 0.0, 1.0],
        "n_samples": len(bottom),
    }
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
                 env_id: str) -> Tuple[Dict, Dict]:
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
            if samples and "key_a" not in samples[0].get("payload", {}):
                empty.append(f"{bucket} (no key_a in payload)")
            surface, bottom = _support_components(samples, src)
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
        "members": {
            k: {"roles": sorted(v["roles"]) or ["interacted"],
                "interaction_types": sorted(v["interaction_types"]),
                "kind": "link" if k.startswith("link:") else "actor"}
            for k, v in sorted(members.items())
        },
        "bin_edges": _bin_edges(merged),
        "episodes": merged["episodes"],
    }
    return affordances, whitelist


def _bin_edges(merged: Dict[str, Any]) -> Dict[str, List[float]]:
    """Quantile edges where the distribution was recorded, max-derived else.

    Equal-width bins over a maximum are wrong for these scenes: the table
    origin sits ~0.9m below its own surface, so height offsets are bimodal and
    every object pair lands in one bin. Equal-population edges separate the
    modes instead of collapsing them, and outliers stop setting the scale.
    """
    edges = derive_bin_edges(dict(merged["bin_stats"]))
    for stat, relation in _QUANTILE_RELATIONS.items():
        chunks = merged.get("bin_samples", {}).get(stat) or []
        if not len(chunks):
            continue
        values = np.concatenate([np.asarray(c).reshape(-1) for c in chunks])
        if values.size < 100:
            continue
        cut = sorted(float(np.quantile(values, p)) for p in _EDGE_PROBS)
        # Degenerate when a statistic barely varies; keep the max-derived scale.
        if len(set(round(c, 9) for c in cut)) < len(cut):
            continue
        edges[relation] = cut
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
    p.add_argument("--configs", default=str(CONFIGS))
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
    affordances, whitelist = build_assets(merged, buckets, args.env_id)

    print(f"[mine] {len(affordances['objects'])} objects, "
          f"{len(whitelist['members'])} whitelist members")
    missing = [r for r in ("planar-distance", "height-offset")
               if not whitelist["bin_edges"].get(r)]
    if missing:
        raise SystemExit(
            f"bin_stats calibrate no edges for {missing}; the shard predates "
            "spatial-statistic collection. Re-collect before mining."
        )
    write_assets(affordances, whitelist, args.env_id, Path(args.configs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
