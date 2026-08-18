"""Offline miner: success rollouts -> ``affordances.json`` (schema v3).

For each canonical object the miner emits up to six per-relation component
lists:

* ``grasp_components`` -- ``anchor`` + ``approach_dir`` + ``width``, mined from
  ee--object success grasp poses (existing path).
* ``contact_components`` -- ``anchor`` + ``outward_normal``, mined from obj-obj
  contact events recorded by the schema-v6 collector.
* ``support_components`` -- ``surface_anchor`` + ``surface_normal`` +
  ``footprint_radius``, mined from obj-obj support events on the supporter
  side.
* ``bottom_components`` -- ``bottom_anchor`` + ``bottom_normal``, mined from
  the supported side of the same support events.
* ``contain_components`` / ``key_components`` -- PegInsertionSide-style entry +
  key descriptors. MS-HAB has no containment env so these stay empty for
  MS-HAB rollouts; collecting from PegInsertionSide-v1 would populate them.

Grasp components require schema-v4 ``tcp_pose_wrt_base`` (or a SAPIEN FK
fallback on Fetch). PLACE is excluded for grasp mining: its success requires
the TCP at the rest pose (mshab/envs/sequential_task.py:1138), so the anchor
would learn the rest pose.

Usage::

    python -m scenegraph.tools.build_affordances \\
        --success-states-dir $MS_ASSET_DIR/data/robot_success_states \\
        --out scenegraph/configs/affordances.json --robot fetch
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


log = logging.getLogger("build_affordances")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _stable_key(value: Optional[str]) -> Optional[str]:
    """Validate a pre-prefixed stable entity key from the collector.

    The collector emits ``actor:<id>`` / ``link:<art>/<link>`` / ``object:<n>``;
    anything else is treated as malformed and dropped.
    """
    if isinstance(value, str) and value.startswith(("actor:", "link:", "object:")):
        return value
    return None


def _normalize(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return v / n


def _quat_wxyz_to_rotmat(q: np.ndarray) -> Optional[np.ndarray]:
    qn = _normalize(np.asarray(q, dtype=float))
    if qn is None:
        return None
    w, x, y, z = qn
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def _inv_transform_point(
    pose_wxyz: np.ndarray, point: np.ndarray
) -> Optional[np.ndarray]:
    """``inv(pose) * point`` for pose ``[xyz, qw, qx, qy, qz]``."""
    if pose_wxyz is None or len(pose_wxyz) < 7:
        return None
    p = np.asarray(pose_wxyz[:3], dtype=float)
    q = np.asarray(pose_wxyz[3:7], dtype=float)
    R = _quat_wxyz_to_rotmat(q)
    if R is None or not np.all(np.isfinite(p)) or not np.all(np.isfinite(point)):
        return None
    return R.T @ (np.asarray(point, dtype=float) - p)


def _inv_rotate_dir(
    pose_wxyz: np.ndarray, dir_world: np.ndarray
) -> Optional[np.ndarray]:
    """World direction -> OBJECT frame unit (``R_obj.T @ d``)."""
    if pose_wxyz is None or len(pose_wxyz) < 7:
        return None
    R = _quat_wxyz_to_rotmat(np.asarray(pose_wxyz[3:7], dtype=float))
    if R is None:
        return None
    d = R.T @ np.asarray(dir_world, dtype=float).reshape(3)
    n = float(np.linalg.norm(d))
    if n < 1e-9 or not np.all(np.isfinite(d)):
        return None
    return d / n


# SAPIEN-based FK, Fetch only.
class _FetchFK:
    """Loads the Fetch URDF once; ``set_qpos`` + ``gripper_link.pose`` per sample.
    Root is fixed at identity so world == base frame."""

    def __init__(self, urdf_path: str, tcp_link_name: str = "gripper_link"):
        # Local imports so ``--help`` works without SAPIEN installed.
        try:
            import sapien                                       # noqa: F401
            from sapien import physx                            # noqa: F401
            import sapien.render                                # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "SAPIEN is required for FK. Install with the project's normal "
                "ManiSkill setup. Underlying error: " + repr(exc)
            ) from exc

        import sapien
        from sapien import physx
        import sapien.render

        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"Fetch URDF not found at {urdf_path}")

        self._sys = physx.PhysxCpuSystem()
        # URDF loader requires a render system even in headless FK
        # (see ManiSkill/mani_skill/envs/sapien_env.py:1213-1217).
        try:
            render_sys = sapien.render.RenderSystem()
        except Exception as exc:
            raise RuntimeError(
                "sapien.render.RenderSystem() failed -- it's required by the "
                f"URDF loader even in headless FK. Original error: {exc!r}"
            ) from exc
        self._scene = sapien.Scene([self._sys, render_sys])
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self._robot = loader.load(urdf_path)
        if self._robot is None:
            raise RuntimeError(f"Failed to load URDF at {urdf_path}")
        try:
            self._robot.set_root_pose(sapien.Pose())
        except Exception:
            pass  # Some SAPIEN versions lack set_root_pose; fix_root_link suffices.

        self._tcp_link = None
        for link in self._robot.get_links():
            if getattr(link, "name", None) == tcp_link_name:
                self._tcp_link = link
                break
        if self._tcp_link is None:
            raise RuntimeError(
                f"{tcp_link_name!r} not found in URDF links: "
                + ", ".join(getattr(l, "name", "?") for l in self._robot.get_links())
            )

        # qpos length may differ from MS-HAB's if joints are fixed/mimic.
        try:
            self._qpos_dim = int(self._robot.dof)
        except Exception:
            self._qpos_dim = -1

    def tcp_in_base(self, qpos: np.ndarray) -> Optional[np.ndarray]:
        """TCP pose ``[xyz, qw, qx, qy, qz]`` in base (= world) frame."""
        try:
            q = np.asarray(qpos, dtype=float).reshape(-1)
            if self._qpos_dim > 0 and q.shape[0] != self._qpos_dim:
                if q.shape[0] < self._qpos_dim:
                    return None
                q = q[: self._qpos_dim]
            self._robot.set_qpos(q)
            # Some SAPIEN versions need an explicit kinematic update.
            for refresh in ("compute_forward_kinematics",
                            "compute_kinematic_pass"):
                fn = getattr(self._robot, refresh, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
                    break
            pose = self._tcp_link.pose
            p = np.asarray(pose.p, dtype=float).reshape(-1)[:3]
            qw = np.asarray(pose.q, dtype=float).reshape(-1)[:4]
            out = np.concatenate([p, qw])
            if not np.all(np.isfinite(out)):
                return None
            return out
        except Exception as exc:
            log.debug("FK failure: %r", exc)
            return None


def _default_fetch_urdf() -> Optional[str]:
    """``fetch.urdf`` via ManiSkill's PACKAGE_ASSET_DIR, or None."""
    try:
        from mani_skill import PACKAGE_ASSET_DIR
    except Exception:
        return None
    return os.path.join(PACKAGE_ASSET_DIR, "robots", "fetch", "fetch.urdf")


# --------------------------------------------------------------------------- #
# Per-object mining
# --------------------------------------------------------------------------- #
def _load_pkl(path: Path) -> Optional[Dict]:
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        log.error("failed to read %s: %r", path, exc)
        return None
    if not isinstance(data, dict):
        log.error("unexpected pkl shape in %s: %s", path, type(data))
        return None
    required = ("obj_id", "entity_key", "robot_qpos", "obj_pose_wrt_base")
    if not all(r in data for r in required):
        log.error("missing keys in %s: have=%s, need=%s",
                  path, sorted(data), required)
        return None
    # Prefer schema-v4 simulator TCP poses. FK is only a legacy fallback for
    # older pickles that do not carry tcp_pose_wrt_base.
    return data


def _mine_object(
    pkl_path: Path, data: Dict, fk: Optional[_FetchFK], max_samples: int,
    approach_axis_local: np.ndarray,
) -> Optional[Dict]:
    qpos_list = np.asarray(data["robot_qpos"], dtype=float)
    pose_list = np.asarray(data["obj_pose_wrt_base"], dtype=float)
    tcp_in_base_list: Optional[np.ndarray] = None
    if "tcp_pose_wrt_base" in data and data["tcp_pose_wrt_base"]:
        tcp_in_base_list = np.asarray(data["tcp_pose_wrt_base"], dtype=float)
    raw_obj_id = data.get("entity_key")
    canonical = _stable_key(raw_obj_id)
    if canonical is None:
        log.error("%s: entity_key %r did not canonicalize", pkl_path, raw_obj_id)
        return None

    if qpos_list.ndim != 2 or pose_list.ndim != 2:
        log.error("%s: expected 2D arrays, got qpos=%s pose=%s",
                  pkl_path, qpos_list.shape, pose_list.shape)
        return None
    if qpos_list.shape[0] != pose_list.shape[0]:
        log.error("%s: qpos and pose lengths disagree (%d vs %d)",
                  pkl_path, qpos_list.shape[0], pose_list.shape[0])
        return None
    if pose_list.shape[1] < 7:
        log.error("%s: obj_pose_wrt_base needs 7-D rows, got %d",
                  pkl_path, pose_list.shape[1])
        return None
    if qpos_list.shape[1] < 2:
        log.error("%s: qpos width %d < 2 (need gripper joints)",
                  pkl_path, qpos_list.shape[1])
        return None
    if tcp_in_base_list is not None:
        if (
            tcp_in_base_list.ndim != 2
            or tcp_in_base_list.shape[0] != qpos_list.shape[0]
            or tcp_in_base_list.shape[1] < 7
        ):
            log.error(
                "%s: tcp_pose_wrt_base shape %s incompatible with qpos %s",
                pkl_path, tcp_in_base_list.shape, qpos_list.shape,
            )
            return None
    elif fk is None:
        log.error(
            "%s: no tcp_pose_wrt_base (schema-v4) and no FK fallback available",
            pkl_path,
        )
        return None

    n_samples = qpos_list.shape[0]
    if n_samples == 0:
        log.warning("%s: 0 samples, skipping", pkl_path)
        return None

    # Subsample if too many.
    if 0 < max_samples < n_samples:
        sel = np.random.default_rng(0).choice(n_samples, size=max_samples, replace=False)
        qpos_list = qpos_list[sel]
        pose_list = pose_list[sel]
        if tcp_in_base_list is not None:
            tcp_in_base_list = tcp_in_base_list[sel]
        n_samples = max_samples

    anchors_obj: List[np.ndarray] = []
    widths: List[float] = []
    approaches_obj: List[Optional[np.ndarray]] = []
    fk_failures = 0
    for i in range(n_samples):
        qpos = qpos_list[i]
        obj_pose = pose_list[i, :7]
        if tcp_in_base_list is not None:
            tcp_in_base = tcp_in_base_list[i, :7]
        else:
            tcp_in_base = fk.tcp_in_base(qpos)
            if tcp_in_base is None:
                fk_failures += 1
                continue
        anchor = _inv_transform_point(obj_pose, tcp_in_base[:3])
        if anchor is None or not np.all(np.isfinite(anchor)):
            continue
        # World approach direction = R_tcp @ axis_local; express in object frame.
        R_tcp = _quat_wxyz_to_rotmat(tcp_in_base[3:7])
        approach_obj: Optional[np.ndarray] = None
        if R_tcp is not None:
            dir_world = R_tcp @ approach_axis_local
            approach_obj = _inv_rotate_dir(obj_pose, dir_world)
        width = float(qpos[-2] + qpos[-1])
        if not np.isfinite(width):
            continue
        anchors_obj.append(anchor)
        widths.append(width)
        approaches_obj.append(approach_obj)   # may be None for this sample

    if fk_failures:
        log.warning("%s: FK failed on %d/%d samples", pkl_path, fk_failures, n_samples)

    if not anchors_obj:
        log.error("%s: no usable success samples -- skipping", pkl_path)
        return None

    components: List[Dict] = []
    for idx, (anchor, width, approach_obj) in enumerate(
        zip(anchors_obj, widths, approaches_obj)
    ):
        approach_field: Dict[str, List[float]] = {}
        if approach_obj is not None:
            n = float(np.linalg.norm(approach_obj))
            if n > 1e-9:
                approach_field = {
                    "approach_dir": [round(float(x), 6) for x in (approach_obj / n)]
                }
        components.append({
            "anchor": [round(float(x), 6) for x in anchor],
            **approach_field,
            "width": round(float(width), 6),
            "sample_index": int(idx),
        })

    log.info("%s -> %s : %d raw affordance candidates",
             pkl_path.name, canonical, len(components))
    return {
        "canonical_key": canonical,
        "raw_obj_id": str(raw_obj_id),
        "n_samples": int(len(components)),
        "components": components,
    }


# --------------------------------------------------------------------------- #
# Obj-obj component mining (schema-v6 collector output)
# --------------------------------------------------------------------------- #
def _ensure_obj_obj_entry(by_object: Dict[str, Dict], key: str) -> Dict:
    entry = by_object.setdefault(key, {})
    entry.setdefault("contact_components", [])
    entry.setdefault("support_components", [])
    entry.setdefault("bottom_components", [])
    return entry


def _accumulate_obj_obj_samples(
    rollouts: List[Dict], by_object: Dict[str, Dict],
) -> None:
    """Aggregate obj-obj contact + support pose samples per canonical key.

    For each (canonical_key, role) we collect raw observations in object-local
    frames; the final component lists are computed downstream from these
    aggregates so we can apply a single mean / spread per object.
    """
    # canonical_key -> list of (anchor_obj, outward_normal_obj_unit)
    contact_obs: Dict[str, List[Tuple[np.ndarray, Optional[np.ndarray]]]] = {}
    # canonical_key -> list of (supported_pos_in_supporter_frame, force_along_normal_sign)
    support_obs: Dict[str, List[np.ndarray]] = {}
    # canonical_key -> list of supporter_pos_in_supported_frame
    bottom_obs: Dict[str, List[np.ndarray]] = {}

    for rollout in rollouts:
        for ev in rollout.get("obj_contacts", []) or []:
            if not isinstance(ev, dict):
                continue
            a_pose = ev.get("a_pose")
            b_pose = ev.get("b_pose")
            force = ev.get("force_vector")
            a_key = _stable_key(ev.get("a_key"))
            b_key = _stable_key(ev.get("b_key"))
            if a_pose is None or b_pose is None or force is None:
                continue
            a_pose = np.asarray(a_pose, dtype=float)
            b_pose = np.asarray(b_pose, dtype=float)
            f = np.asarray(force, dtype=float).reshape(3)
            f_norm = float(np.linalg.norm(f))
            if f_norm < 1e-6:
                continue
            f_unit_world = f / f_norm
            midpoint_world = 0.5 * (a_pose[:3] + b_pose[:3])
            if a_key:
                anchor_a = _inv_transform_point(a_pose, midpoint_world)
                # Outward normal on A is OPPOSITE to the contact force that B
                # exerts on A (force_vector is the force on the first body in
                # SAPIEN; we pick the convention that the outward normal is the
                # one pointing AWAY from the midpoint relative to A's center).
                outward_world = -f_unit_world
                outward_obj_a = _inv_rotate_dir(a_pose, outward_world)
                if anchor_a is not None:
                    contact_obs.setdefault(a_key, []).append((anchor_a, outward_obj_a))
            if b_key:
                anchor_b = _inv_transform_point(b_pose, midpoint_world)
                outward_world_b = f_unit_world  # opposite of A's outward
                outward_obj_b = _inv_rotate_dir(b_pose, outward_world_b)
                if anchor_b is not None:
                    contact_obs.setdefault(b_key, []).append((anchor_b, outward_obj_b))

        for rec in rollout.get("supports", []) or []:
            if not isinstance(rec, dict):
                continue
            supporter_pose = rec.get("supporter_pose")
            supported_pose = rec.get("supported_pose")
            if supporter_pose is None or supported_pose is None:
                continue
            supporter = rec.get("supporter") or {}
            supporter_key = _stable_key(supporter.get("key"))
            supported_key = _stable_key(rec.get("supported_key"))
            sp_sup = np.asarray(supporter_pose, dtype=float)
            sp_subj = np.asarray(supported_pose, dtype=float)
            # Surface anchor on supporter: supported center projected into
            # supporter's local frame. We trust the supporter's surface to be
            # the local +z plane (matches MS-HAB drawer/counter convention).
            if supporter_key:
                anchor = _inv_transform_point(sp_sup, sp_subj[:3])
                if anchor is not None:
                    support_obs.setdefault(supporter_key, []).append(anchor)
            # Bottom anchor on supported: supporter center in supported frame.
            if supported_key:
                anchor = _inv_transform_point(sp_subj, sp_sup[:3])
                if anchor is not None:
                    bottom_obs.setdefault(supported_key, []).append(anchor)

    # Emit components: one per canonical key per relation. Clustering is
    # deliberately kept simple (mean + spread) -- can be upgraded to k-means
    # later if multi-modality is needed.
    for key, samples in contact_obs.items():
        anchors = np.stack([s[0] for s in samples], axis=0)
        anchor_mean = anchors.mean(axis=0)
        normals = [s[1] for s in samples if s[1] is not None]
        normal_field: Dict[str, List[float]] = {}
        if normals:
            n_arr = np.stack(normals, axis=0)
            n_mean = n_arr.mean(axis=0)
            nn = float(np.linalg.norm(n_mean))
            if nn > 1e-9:
                n_unit = n_mean / nn
                normal_field = {"outward_normal":
                                [round(float(x), 6) for x in n_unit]}
        comp = {
            "anchor": [round(float(x), 6) for x in anchor_mean],
            **normal_field,
            "n_samples": int(len(samples)),
        }
        entry = _ensure_obj_obj_entry(by_object, key)
        entry["contact_components"].append(comp)

    for key, samples in support_obs.items():
        arr = np.stack(samples, axis=0)
        anchor_mean = arr.mean(axis=0)
        # Footprint radius: max in-plane distance from the mean (xy in
        # supporter local frame). Floor at 1 cm so a single observation still
        # produces a non-degenerate radius.
        xy = arr[:, :2] - anchor_mean[:2]
        spread = float(np.max(np.linalg.norm(xy, axis=1))) if len(arr) > 1 else 0.01
        footprint = max(0.01, spread)
        comp = {
            "surface_anchor": [
                round(float(anchor_mean[0]), 6),
                round(float(anchor_mean[1]), 6),
                round(float(anchor_mean[2]), 6),
            ],
            "surface_normal": [0.0, 0.0, 1.0],
            "footprint_radius": round(footprint, 6),
            "n_samples": int(len(samples)),
        }
        entry = _ensure_obj_obj_entry(by_object, key)
        entry["support_components"].append(comp)

    for key, samples in bottom_obs.items():
        arr = np.stack(samples, axis=0)
        anchor_mean = arr.mean(axis=0)
        comp = {
            "bottom_anchor": [
                round(float(anchor_mean[0]), 6),
                round(float(anchor_mean[1]), 6),
                round(float(anchor_mean[2]), 6),
            ],
            "bottom_normal": [0.0, 0.0, -1.0],
            "n_samples": int(len(samples)),
        }
        entry = _ensure_obj_obj_entry(by_object, key)
        entry["bottom_components"].append(comp)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
# Known MS-HAB YCB pickable pool (from mshab/evaluate.py:88-148). Used only for
# coverage warnings -- the miner walks whatever .pkl files exist.
_KNOWN_YCB_POOL = (
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "013_apple",
    "024_bowl",
)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--success-states-dir",
        required=True,
        help="Path to robot_success_states/ (parent of "
             "<robot_uid>/<task_group>/<subtask>/).",
    )
    parser.add_argument(
        "--robot", default="fetch",
        help="Robot uid subdirectory (default: fetch).",
    )
    parser.add_argument(
        "--task-group", required=True,
        help="MS-HAB task whose rollouts to mine. Only "
             "<success-states-dir>/<robot>/<task-group>/<subtask>/ is read.",
    )
    parser.add_argument(
        "--subtask", default="pick", choices=["pick", "open", "close"],
        help="Successful manipulation subtask. Place is excluded because its "
             "success pose is the robot rest pose.",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output affordances.json path.",
    )
    parser.add_argument(
        "--merge-existing", action="store_true",
        help="Preserve entities already present in --out and update/add the "
             "entities mined by this run.",
    )
    parser.add_argument(
        "--n-components", type=int, default=4,
        help="Deprecated compatibility flag. Raw success samples are emitted; "
             "use --max-samples to cap candidate count.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=2000,
        help="Cap per-object samples after random subsampling (default 2000; "
             "0 disables capping).",
    )
    parser.add_argument(
        "--urdf",
        default=None,
        help="Path to fetch.urdf. Defaults to ManiSkill's packaged Fetch URDF.",
    )
    parser.add_argument(
        "--tcp-link", default="gripper_link",
        help="Link name to read as TCP (default: gripper_link).",
    )
    parser.add_argument(
        "--tcp-approach-axis", default="0,0,1",
        help="Gripper-link local approach axis as 'x,y,z' (default 0,0,1).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    approach_axis_local = np.asarray(
        [float(x) for x in args.tcp_approach_axis.split(",")], dtype=float
    )
    _setup_logging(args.verbose)

    root = (
        Path(args.success_states_dir) / args.robot / args.task_group
        / args.subtask
    )
    if not root.is_dir():
        log.error("expected directory %s does not exist", root)
        return 2

    # FK is only a fallback for legacy (schema < 4) PKLs that do not carry
    # ``tcp_pose_wrt_base``. Schema-v4 PKLs are anchor-correct without FK and
    # do not need a URDF, so we set ``fk`` to None when FK setup fails rather
    # than aborting.
    urdf_path = args.urdf or _default_fetch_urdf()
    fk: Optional[_FetchFK] = None
    if urdf_path:
        log.info("FK URDF (legacy fallback): %s", urdf_path)
        try:
            fk = _FetchFK(urdf_path, tcp_link_name=args.tcp_link)
        except Exception as exc:
            log.warning(
                "FK setup failed (%s); legacy PKLs without tcp_pose_wrt_base "
                "will be skipped",
                exc,
            )
    else:
        log.warning(
            "Could not locate fetch.urdf; legacy PKLs without "
            "tcp_pose_wrt_base will be skipped"
        )

    pkls = sorted(root.glob("*.pkl"))
    if not pkls:
        log.error("no .pkl files under %s", root)
        return 2

    seen_canonical: set = set()
    by_object: Dict[str, Dict] = {}
    all_rollouts: List[Dict] = []
    for pkl_path in pkls:
        data = _load_pkl(pkl_path)
        if data is None:
            continue
        # Same provenance gate as the whitelist miner. Without it a pickle
        # misplaced into this group's tree contaminates the group's affordance
        # asset first, and the whitelist miner's later rejection comes too late
        # to undo it -- the affordance file is what the whitelists are then
        # mined against.
        recorded = str((data.get("provenance") or {}).get("task_group") or "")
        if not recorded:
            log.error(
                "%s carries no provenance.task_group; it predates the "
                "task-namespaced collector and cannot be attributed to a task. "
                "Recollect it with --no-skip-done", pkl_path,
            )
            return 2
        if recorded != args.task_group:
            log.error(
                "%s sits under task group %r but records %r; refusing to mine "
                "one task's poses into another's affordance asset",
                pkl_path, args.task_group, recorded,
            )
            return 2
        # Pass 1: grasp components.
        rec = _mine_object(
            pkl_path, data, fk=fk,
            max_samples=args.max_samples,
            approach_axis_local=approach_axis_local,
        )
        if rec is not None and rec["canonical_key"] not in seen_canonical:
            seen_canonical.add(rec["canonical_key"])
            by_object[rec["canonical_key"]] = {
                "raw_obj_id": rec["raw_obj_id"],
                "n_samples": rec["n_samples"],
                "grasp_components": rec["components"],
            }
        elif rec is not None:
            log.warning("duplicate canonical key %s from %s -- ignoring second",
                        rec["canonical_key"], pkl_path)

        # Stash rollouts for the obj-obj pass below.
        rollouts = data.get("interaction_rollouts") or []
        if isinstance(rollouts, list):
            all_rollouts.extend(r for r in rollouts if isinstance(r, dict))

    # Pass 2: obj-obj contact / support component mining across all rollouts.
    _accumulate_obj_obj_samples(all_rollouts, by_object)

    # Coverage warnings apply only to the known pickable actor pool.
    if args.subtask == "pick":
        for ycb in _KNOWN_YCB_POOL:
            if f"actor:{ycb}" not in by_object:
                log.warning("no data for canonical key %s -- runtime will emit no "
                            "affordance edges for it", ycb)

    if args.merge_existing and os.path.isfile(args.out):
        try:
            with open(args.out) as stream:
                existing = json.load(stream)
            existing_objects = existing.get("objects", {})
            if isinstance(existing_objects, dict):
                by_object = {**existing_objects, **by_object}
        except Exception as exc:
            log.error("failed to merge existing affordance asset %s: %r", args.out, exc)
            return 2

    payload = {
        "_README": (
            "Per canonical object key, up to six per-relation component lists "
            "(all coordinates in OBJECT frame, metres / unit-vector). "
            "grasp_components: ee--object grasp anchors. "
            "contact_components: obj-obj contact anchors + outward normals. "
            "support_components: supporter surface anchors + normal + "
            "footprint radius. bottom_components: supported bottom anchors. "
            "contain_components / key_components: PegInsertionSide-style "
            "container entry + containee key descriptors (empty for MS-HAB)."
        ),
        "_schema_version": 3,
        "objects": by_object,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    log.info("wrote %d objects (%d components total) to %s",
             len(by_object),
             sum(len(lst)
                 for v in by_object.values()
                 for k, lst in v.items()
                 if k.endswith("_components") and isinstance(lst, list)),
             out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
