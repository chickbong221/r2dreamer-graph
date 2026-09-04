"""Mine per-subtask whitelists from successful rollout interactions.

Mining is scoped to one MS-HAB task group: the tool reads
``<success-states-dir>/<robot>/<task-group>/<subtask>/`` and nothing else, so a
run for ``set_table`` cannot see, read or overwrite ``tidy_house`` evidence.
Every pickle's recorded ``provenance.task_group`` is cross-checked against the
directory it was found under, and a disagreement aborts the run rather than
quietly folding one task's scene into another's whitelist.

``--membership-policy`` decides what reaches ``members``:

``full-evidence``
    Every entity the robot interacted with, plus their direct supporters, with
    counts, interaction types and support references intact. This is the raw
    asset -- expensive to collect, cheap to prune -- and is what
    ``prune_whitelists.py`` later shortens without re-running rollouts.

``target-supporters``
    The target plus whatever directly supports it. Nothing else: a rollout
    contacts whatever is in the way, so admitting every contacted entity would
    fill a pick-the-bowl graph with groceries the arm brushed past. Support is
    never expanded recursively.

Frequency counts are emitted for audit but do not filter membership. The asset
also records, per member, the ee-driven interaction types (``contact`` and/or
``grasp``) seen across rollouts and, at the asset level, the per-relation bin
edges derived from the collector's per-rollout running maxes. Bin statistics do
not depend on the membership policy, so pruning never changes the bins.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import numpy as np

from scenegraph.core.affordance import (
    AffordanceSet,
    compatibility_components,
    load_affordance_set,
    lookup_bottom_components,
    lookup_components,
    lookup_contact_components,
    lookup_contain_components,
    lookup_key_components,
    lookup_reference_surface,
    lookup_support_components,
    select_active_component,
    transform_anchors,
    transform_dir,
)
from scenegraph.core.containment import (
    contain_compatibility,
    obj_contact_compatibility,
    support_compatibility,
)
from scenegraph.configs.loader import default_temporal_k
from scenegraph.core import families as families_rules
from scenegraph.core import spatial_metrics
from scenegraph.core.sites import (
    SITE_PREFIX,
    admit_site_members,
    declared_sites,
    parse_site_declarations,
)
from scenegraph.core.affordance import components_for_partner
from scenegraph.core.entity_identity import normalize_asset_key
from scenegraph.core.relation_rules import (
    SPATIAL_LABELS,
    _compat_norm,
    _compatibility_score,
    _mean_normalized,
    bin_label,
)
from scenegraph.core.schema import Node
from scenegraph.core.spatial_metrics import (
    EE_OBJECT_SCOPE,
    OBJECT_OBJECT_SCOPE,
    spatial_bin_key,
    stat_key,
)
from scenegraph.core.whitelist import (
    INTERACTION_CONTACT,
    INTERACTION_CONTAIN,
    INTERACTION_GRASP,
    INTERACTION_SUPPORT,
    WHITELIST_SCHEMA_VERSION,
    derive_bin_edges,
    whitelist_target_slug,
)


MEMBERSHIP_FULL_EVIDENCE = "full-evidence"
MEMBERSHIP_TARGET_SUPPORTERS = "target-supporters"
MEMBERSHIP_POLICIES = (MEMBERSHIP_FULL_EVIDENCE, MEMBERSHIP_TARGET_SUPPORTERS)


# Change relations use a low quantile to shed transient outliers; absolute
# relations use a high one to still cover the demo's operating range.
_MINER_QUANTILE_CHANGE = 0.6
_MINER_QUANTILE_ABSOLUTE = 0.9


# Pre-split shards named the ee-object streams without a scope. They only
# ever held ee-object samples, so the rename is exact and re-mining is
# enough -- no recollection.
_LEGACY_STAT_ALIASES = {
    "planar_distance": "ee_object_planar_distance",
    "height_offset": "ee_object_height_offset",
    "planar_distance_change": "ee_object_planar_distance_change",
    "height_offset_change": "ee_object_height_offset_change",
}


def _miner_quantile(relation: str) -> float:
    return (
        _MINER_QUANTILE_CHANGE
        if relation.endswith("_change")
        else _MINER_QUANTILE_ABSOLUTE
    )


# Per-relation sanity ceiling. Anything above this is treated as a numerical
# blow-up rather than a meaningful EE-operating range. Tuned for Fetch in a
# kitchen scene (max reach ~1.5 m planar; ~1.2 m vertical).
_BIN_VALUE_CEILING: Dict[str, float] = {
    "ee_object_planar_distance": 2.0,
    "ee_object_height_offset": 1.5,
    "ee_object_planar_distance_change": 2.0,
    "ee_object_height_offset_change": 1.5,
    "object_object_planar_distance": 2.0,
    "object_object_height_offset": 1.5,
    "object_object_planar_distance_change": 2.0,
    "object_object_height_offset_change": 1.5,
    "grasp_compatibility_change": 1.0,
    "contact_compatibility_change": 1.0,
    "support_compatibility_change": 1.0,
    "contain_compatibility_change": 1.0,
}

_DEFAULT_TCP_AXIS_LOCAL = [0.0, 0.0, 1.0]
_DEFAULT_ORIENTATION_SELECTION_WEIGHT = 0.10


# The evidence schema this miner needs. Raised from 3 with the v9 collector:
# extents, end-effector rest calibration and per-rollout build configuration
# are all newly *recorded*, not newly derived, so an older pickle is not a
# degraded input -- it is the wrong input, and mining it would silently
# produce assets missing exactly the fields the MS-HAB schedule reads.
MIN_ROLLOUT_SCHEMA = 9

log = logging.getLogger("build_subtask_whitelists")


def _iter_pickles(root: Path):
    for path in sorted(root.rglob("*.pkl")):
        try:
            with open(path, "rb") as stream:
                payload = pickle.load(stream)
        except Exception as exc:
            log.warning("skip %s: %r", path, exc)
            continue
        if isinstance(payload, dict):
            yield path, payload


class _WhitelistBuilder:
    def __init__(
        self,
        subtask: str,
        target: str,
        *,
        task_group: str = "",
        membership_policy: str = MEMBERSHIP_TARGET_SUPPORTERS,
        affordance_set: Optional[AffordanceSet] = None,
        temporal_k: Optional[int] = None,
        sites: Optional[Dict[str, Any]] = None,
    ):
        if membership_policy not in MEMBERSHIP_POLICIES:
            raise ValueError(
                f"unknown membership policy {membership_policy!r}; "
                f"have {list(MEMBERSHIP_POLICIES)}"
            )
        self.subtask = subtask
        self.target = target
        self.task_group = str(task_group or "")
        self.membership_policy = membership_policy
        self.affordance_set = affordance_set or AffordanceSet()
        self.temporal_k = max(
            1, int(temporal_k if temporal_k is not None
                   else default_temporal_k()))
        self.roles: Dict[str, Set[str]] = defaultdict(set)
        self.kinds: Dict[str, str] = {}
        self.names: Dict[str, str] = {}
        self.interaction_types: Dict[str, Set[str]] = defaultdict(set)
        self.rollout_count = 0
        self.interaction_count: Dict[str, int] = defaultdict(int)
        self.support_count: Dict[str, int] = defaultdict(int)
        self.supports: Dict[str, Set[str]] = defaultdict(set)
        # Aggregated robust value per relation. Filled at payload() time.
        self.bin_value: Dict[str, float] = {}
        # Raw per-relation sample pool across rollouts.
        self.bin_samples: Dict[str, List[float]] = defaultdict(list)
        # Per-rollout pose traces used to mine compatibility-change bins once
        # affordance components are available.
        self.pose_rollouts: List[List[Dict[str, Any]]] = []
        # Collision geometry per member, first reading wins. It is a property
        # of the built scene rather than of a rollout, and the classification
        # it feeds -- surface or receptacle -- cannot be recovered from roles:
        # a bin and a tabletop carry identical ones.
        self.extents: Dict[str, Dict[str, Any]] = {}
        # Reviewed site declarations for this task group, injected by the
        # caller. The miner never invents one: letting it derive a site from
        # evidence is how PlaceSphere acquired a goal region it does not have.
        self.sites: Dict[str, Any] = dict(sites or {})

    def absorb(self, rollout: Dict[str, Any]) -> None:
        self.rollout_count += 1
        interacted_this_rollout: Set[str] = set()
        ee_source_keys: Set[str] = set()
        for item in rollout.get("interacted", []) or []:
            if not isinstance(item, dict):
                continue
            key = normalize_asset_key(item.get("key"), item.get("kind"))
            if not key:
                continue
            self.roles[key].add("interacted")
            self.kinds.setdefault(key, str(item.get("kind") or "other"))
            self.names.setdefault(key, str(item.get("name") or key))
            # Every interacted record means an ee link touched the object;
            # grasped=True additionally implies the grasp predicate fired.
            self.interaction_types[key].add(INTERACTION_CONTACT)
            if bool(item.get("grasped")):
                self.interaction_types[key].add(INTERACTION_GRASP)
            interacted_this_rollout.add(key)
            # Only ee-direct (real ee force or an active grasp) entries can
            # propagate one-hop through an obj-obj contact below. Entries
            # already elevated by the collector have max_ee_force=0 and no
            # grasped flag, so they can't propagate further.
            if (
                float(item.get("max_ee_force", 0.0) or 0.0) > 0.0
                or bool(item.get("grasped"))
            ):
                ee_source_keys.add(key)
        for key in interacted_this_rollout:
            self.interaction_count[key] += 1

        # One-hop elevation via obj-obj contact. Mirrors the collector rule;
        # also a safety net for pkls collected before elevation was wired in.
        # Elevated entries pick up the ``contact`` interaction type but not
        # ``grasp``; supporters of elevated entities are admitted below.
        for ev in rollout.get("obj_contacts", []) or []:
            if not isinstance(ev, dict):
                continue
            a_key = normalize_asset_key(ev.get("a_key"))
            b_key = normalize_asset_key(ev.get("b_key"))
            for src, dst in ((a_key, b_key), (b_key, a_key)):
                if (
                    src in ee_source_keys
                    and dst
                    and dst not in interacted_this_rollout
                ):
                    self.roles[dst].add("interacted")
                    self.interaction_types[dst].add(INTERACTION_CONTACT)
                    self.kinds.setdefault(dst, "other")
                    self.names.setdefault(dst, dst)
                    interacted_this_rollout.add(dst)
                    self.interaction_count[dst] += 1

        # One hop only: supporters are kept only when they directly support an
        # entity that was actually contacted in this successful rollout.  The
        # task target is metadata for file selection, not an injected member.
        supported_roots = set(interacted_this_rollout)
        supported_pairs_this_rollout: Set[Tuple[str, str]] = set()
        for relation in rollout.get("supports", []) or []:
            if not isinstance(relation, dict):
                continue
            supported = normalize_asset_key(relation.get("supported_key"))
            supporter = relation.get("supporter")
            if supported not in supported_roots or not isinstance(supporter, dict):
                continue
            supporter_key = normalize_asset_key(
                supporter.get("key"), supporter.get("kind")
            )
            if not supporter_key or supporter_key == supported:
                continue
            self.roles[supporter_key].add("support")
            self.kinds.setdefault(
                supporter_key, str(supporter.get("kind") or "other")
            )
            self.names.setdefault(
                supporter_key, str(supporter.get("name") or supporter_key)
            )
            self.supports[supporter_key].add(supported)
            # Both endpoints of a support pair carry the ``support`` token so
            # the runtime can gate obj-obj support-compatibility on it.
            self.interaction_types[supporter_key].add(INTERACTION_SUPPORT)
            self.interaction_types[supported].add(INTERACTION_SUPPORT)
            supported_pairs_this_rollout.add((supporter_key, supported))
        for supporter_key, _supported in supported_pairs_this_rollout:
            self.support_count[supporter_key] += 1

        # Obj-obj contact events (schema-v6 collector). Both endpoints get the
        # ``contact`` token so the runtime can emit obj-obj
        # contact-compatibility for whitelisted pairs. A separate ``contain``
        # token is opt-in: it's added only when the source data explicitly
        # marks an event as a containment (no MS-HAB env does this today).
        for ev in rollout.get("obj_contacts", []) or []:
            if not isinstance(ev, dict):
                continue
            a_key = normalize_asset_key(ev.get("a_key"))
            b_key = normalize_asset_key(ev.get("b_key"))
            for k in (a_key, b_key):
                if not k:
                    continue
                self.interaction_types[k].add(INTERACTION_CONTACT)
            if bool(ev.get("contain")):
                for k in (a_key, b_key):
                    if k:
                        self.interaction_types[k].add(INTERACTION_CONTAIN)

        raw_samples = rollout.get("bin_samples")
        if isinstance(raw_samples, dict):
            for k, values in raw_samples.items():
                if not isinstance(values, (list, tuple)):
                    continue
                name = _LEGACY_STAT_ALIASES.get(str(k), str(k))
                bucket = self.bin_samples[name]
                for v in values:
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(fv):
                        bucket.append(fv)

        raw_extents = rollout.get("extents")
        if isinstance(raw_extents, dict):
            for raw_key, entry in raw_extents.items():
                if not isinstance(entry, dict):
                    continue
                key = normalize_asset_key(raw_key, entry.get("kind"))
                if key and key not in self.extents:
                    self.extents[key] = dict(entry)

        raw_pose_samples = rollout.get("pose_samples")
        if isinstance(raw_pose_samples, list):
            self.pose_rollouts.append(raw_pose_samples)

    def _object_anchor_spec(self, a: Node, b: Node):
        """``(anchor_a, radial_a, anchor_b, radial_b)`` for the pair, or None."""
        if self.affordance_set is None or self.affordance_set.is_empty():
            return None
        for supporter, supported in ((a, b), (b, a)):
            sup = components_for_partner(
                lookup_support_components(self.affordance_set, supporter),
                supported.node_id)
            bot = components_for_partner(
                lookup_bottom_components(self.affordance_set, supported),
                supporter.node_id)
            if len(sup) != 1 or len(bot) != 1:
                continue
            s_anchor = sup[0].surface_anchor_obj_frame
            b_anchor = bot[0].bottom_anchor_obj_frame
            radial = bot[0].radial_offset
            if supporter is a:
                return s_anchor, None, b_anchor, radial
            return b_anchor, radial, s_anchor, None
        return None

    def _mine_family_height_samples(
        self, families: Dict[str, Optional[str]],
    ) -> Dict[str, List[float]]:
        """One end-effector height scale per family, from the pose trace.

        The collector pools every end-effector height into one
        ``ee_object_height_offset`` stream -- the target and the counter it
        sits on alike -- so that stream cannot calibrate a family. It does not
        have to: ``pose_samples`` already carries ``tcp_pose`` and each
        member's pose keyed by canonical key, which is enough to reproject
        every sample onto the scale of the member it was measured against.
        Reprojection has to happen here rather than at collection because
        nothing knows which members are surfaces until the extents have been
        read and classified.

        A structural surface is measured against its mined plane, exactly as
        ``surface_relative_height`` will at runtime. Everything else uses the
        signed offset from the member's own origin. Calibrating the table on
        the metre to its origin while the runtime labels the 15cm to its top
        is the deadband error this exists to remove.
        """
        samples: Dict[str, List[float]] = defaultdict(list)
        if not self.pose_rollouts or not families:
            return samples
        for rollout in self.pose_rollouts:
            history: Dict[str, Deque[float]] = {}
            for snap in rollout:
                if not isinstance(snap, dict):
                    continue
                tcp = snap.get("tcp_pose")
                raw_entities = snap.get("entities")
                if not isinstance(raw_entities, dict) or tcp is None:
                    continue
                try:
                    tcp_xyz = np.asarray(tcp, dtype=float).reshape(-1)[:3]
                except (TypeError, ValueError):
                    continue
                if tcp_xyz.size < 3 or not np.all(np.isfinite(tcp_xyz)):
                    continue
                for raw_key, raw in raw_entities.items():
                    key = normalize_asset_key(str(raw_key))
                    family = families.get(key) if key else None
                    if not family:
                        continue
                    node = self._trace_node(str(raw_key), raw)
                    if node is None or node.pose_world is None:
                        continue
                    dz = self._family_height(family, tcp_xyz, node)
                    if dz is None or not np.isfinite(dz):
                        continue
                    stat = spatial_metrics.ee_family_bin_key(family).replace(
                        "-", "_")
                    # Magnitude, not the signed value. ``derive_bin_edges``
                    # builds a symmetric band from a positive half-width, so a
                    # negative robust value yields no edges at all -- and a
                    # gripper that spends the episode *below* a member's
                    # reference height produces only negative offsets. The
                    # collector's own ee-object stream stores abs() for the
                    # same reason. Sign is kept in the history below, where
                    # the change sample needs it.
                    samples[stat].append(abs(float(dz)))
                    buf = history.get(key)
                    if buf is None:
                        buf = deque(maxlen=self.temporal_k + 1)
                        history[key] = buf
                    buf.append(float(dz))
                    if len(buf) > self.temporal_k:
                        samples[f"{stat}_change"].append(abs(buf[-1] - buf[0]))
        return samples

    def _family_height(self, family: str, tcp_xyz, node) -> Optional[float]:
        """Signed end-effector height on that family's scale, or None."""
        if family != spatial_metrics.FAMILY_STRUCTURAL:
            return float(tcp_xyz[2]) - float(node.pose_world[2])
        surface = lookup_reference_surface(self.affordance_set, node)
        if surface is None:
            # Reported by the caller, which knows the member key. Silently
            # falling back to the origin is the metre of error itself.
            return None
        anchor = spatial_metrics.anchor_world(
            node.pose_world, surface.anchor_obj_frame)
        normal = transform_dir(
            node.pose_world, surface.outward_normal_obj_frame)
        if anchor is None or normal is None:
            return None
        return spatial_metrics.surface_height(tcp_xyz, anchor, normal)

    def _mine_object_pair_samples(self) -> Dict[str, List[float]]:
        """Object-object scales, measured the way the runtime will measure them.

        MS-HAB emits no object-object spatial edges, but its object-object
        compatibility near gate reads the planar scale, so it still has to be
        calibrated -- and on surface anchors, not link origins.
        """
        samples: Dict[str, List[float]] = defaultdict(list)
        if not self.pose_rollouts:
            return samples
        pd_key = stat_key(OBJECT_OBJECT_SCOPE, "planar-distance")
        ho_key = stat_key(OBJECT_OBJECT_SCOPE, "height-offset")
        for rollout in self.pose_rollouts:
            history: Dict[Tuple[str, str], Deque[Tuple[float, float]]] = {}
            for snap in rollout:
                if not isinstance(snap, dict):
                    continue
                raw_entities = snap.get("entities")
                if not isinstance(raw_entities, dict):
                    continue
                nodes = {
                    key: node
                    for key, raw in raw_entities.items()
                    if (node := self._trace_node(str(key), raw)) is not None
                }
                keys = sorted(nodes)
                for i in range(len(keys)):
                    for j in range(i + 1, len(keys)):
                        a, b = nodes[keys[i]], nodes[keys[j]]
                        spec = (self._object_anchor_spec(a, b)
                                or (None, None, None, None))
                        points = spatial_metrics.pair_points(
                            a.pose_world, b.pose_world,
                            spec[0], spec[2], spec[1], spec[3])
                        planar, height = spatial_metrics.measures(*points)
                        samples[pd_key].append(abs(planar))
                        samples[ho_key].append(abs(height))
                        hk = (keys[i], keys[j])
                        buf = history.get(hk)
                        if buf is None:
                            buf = deque(maxlen=self.temporal_k + 1)
                            history[hk] = buf
                        buf.append((planar, height))
                        if len(buf) > self.temporal_k:
                            samples[pd_key + "_change"].append(
                                abs(planar - buf[0][0]))
                            samples[ho_key + "_change"].append(
                                abs(height - buf[0][1]))
        return samples

    def _aggregate_bins(
        self,
        extra_samples: Optional[Dict[str, List[float]]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Return ``(robust_value, observed_max)`` per relation.

        Change relations use ``_MINER_QUANTILE_CHANGE``; absolute relations use
        ``_MINER_QUANTILE_ABSOLUTE``. A per-relation ceiling caps numerical
        blow-ups so a single bad pickle cannot push the bin edges to absurd
        ranges.
        """
        robust: Dict[str, float] = {}
        observed: Dict[str, float] = {}
        keys = set(self.bin_samples)
        if extra_samples:
            keys.update(extra_samples)
        for k in sorted(keys):
            samples = list(self.bin_samples.get(k, ()))
            if extra_samples:
                samples.extend(extra_samples.get(k, ()))
            if not samples:
                continue
            value = float(np.quantile(samples, _miner_quantile(k)))
            obs = float(np.max(samples))
            ceiling = _BIN_VALUE_CEILING.get(k)
            if ceiling is not None and value > ceiling:
                log.warning(
                    "bin '%s' for subtask=%s target=%s capped %.3f -> %.3f "
                    "(observed max=%.3f); raw samples likely contain an "
                    "outlier",
                    k, self.subtask, self.target, value, ceiling, obs,
                )
                value = ceiling
            robust[k] = value
            observed[k] = obs
        return robust, observed

    @staticmethod
    def _planar_near_labels() -> Set[str]:
        labels = SPATIAL_LABELS["planar-distance"]
        if len(labels) >= 5:
            return set(labels[:2])
        return {labels[0]}

    @staticmethod
    def _trace_node(raw_key: str, raw: Any) -> Optional[Node]:
        if not isinstance(raw, dict):
            return None
        key = normalize_asset_key(raw_key, raw.get("kind"))
        if not key:
            return None
        pose = raw.get("pose")
        if not isinstance(pose, (list, tuple)) or len(pose) < 7:
            return None
        try:
            pose7 = [float(x) for x in pose[:7]]
        except (TypeError, ValueError):
            return None
        if not np.all(np.isfinite(pose7)):
            return None
        kind = str(raw.get("kind") or ("actor" if key.startswith("actor:") else "other"))
        return Node(
            node_id=key,
            node_type="object",
            name=str(raw.get("name") or key),
            pose_world=pose7,
            attributes={
                "whitelist_key": key,
                "entity_key": key,
                "entity_kind": kind,
                "is_actor": key.startswith("actor:"),
            },
        )

    @staticmethod
    def _is_near(
        a_xyz: np.ndarray,
        b_xyz: np.ndarray,
        pd_edges: List[float],
        near_labels: Set[str],
    ) -> bool:
        d = float(np.linalg.norm(np.asarray(a_xyz[:2]) - np.asarray(b_xyz[:2])))
        return bin_label(d, pd_edges, SPATIAL_LABELS["planar-distance"]) in near_labels

    def _push_compat_history(
        self,
        samples: Dict[str, List[float]],
        history: Dict[Tuple[str, str, str], Deque[float]],
        present: Set[Tuple[str, str, str]],
        key: Tuple[str, str, str],
        value: float,
    ) -> None:
        if not np.isfinite(value):
            return
        present.add(key)
        absolute_key = key[2].replace('-', '_')
        samples[absolute_key].append(float(value))
        buf = history.get(key)
        if buf is None:
            buf = deque(maxlen=self.temporal_k + 1)
            history[key] = buf
        buf.append(float(value))
        if len(buf) > self.temporal_k:
            samples[f"{absolute_key}_change"].append(abs(buf[-1] - buf[0]))

    def _score_ee_object_compatibility(
        self,
        node: Node,
        tcp_pose: np.ndarray,
        gripper_width: Optional[float],
        anchor_cache: Dict[str, int],
    ) -> Optional[Tuple[float, float]]:
        comps = lookup_components(self.affordance_set, node)
        if not comps:
            return None
        anchors_world = transform_anchors(node.pose_world, comps)
        if anchors_world is None:
            return None
        cached = anchor_cache.get(node.node_id)
        if isinstance(cached, int) and 0 <= cached < len(comps):
            a_star = cached
        else:
            a_star = select_active_component(
                tcp_pose[:3],
                anchors_world,
                components=comps,
                obj_pose_world=node.pose_world,
                tcp_pose_world=tcp_pose,
                tcp_axis_local=_DEFAULT_TCP_AXIS_LOCAL,
                orientation_weight=_DEFAULT_ORIENTATION_SELECTION_WEIGHT,
            )
            if a_star is None:
                return None
            anchor_cache[node.node_id] = int(a_star)
        norm = _compat_norm({})
        meas = compatibility_components(
            comps[a_star],
            int(a_star),
            anchors_world[a_star],
            obj_pose_world=node.pose_world,
            tcp_pose_world=tcp_pose,
            tcp_axis_local=_DEFAULT_TCP_AXIS_LOCAL,
            gripper_width=gripper_width,
        )
        grasp_score = _compatibility_score(meas, norm, include_width=True)
        contact_score = _compatibility_score(meas, norm, include_width=False)
        return grasp_score, contact_score

    def _mine_compatibility_samples(
        self,
        bin_edges: Dict[str, List[float]],
    ) -> Dict[str, List[float]]:
        samples: Dict[str, List[float]] = defaultdict(list)
        if self.affordance_set.is_empty() or not self.pose_rollouts:
            return samples
        ee_pd_edges = bin_edges.get(
            spatial_bin_key(EE_OBJECT_SCOPE, "planar-distance"))
        obj_pd_edges = bin_edges.get(
            spatial_bin_key(OBJECT_OBJECT_SCOPE, "planar-distance"))
        # Both scopes or nothing: gating object pairs by the end-effector
        # scale would score a different population than the runtime does.
        if not ee_pd_edges or not obj_pd_edges:
            return samples

        near_labels = self._planar_near_labels()
        norm = _compat_norm({})

        for rollout in self.pose_rollouts:
            history: Dict[Tuple[str, str, str], Deque[float]] = {}
            anchor_cache: Dict[str, int] = {}
            for snap in rollout:
                if not isinstance(snap, dict):
                    continue
                tcp_raw = snap.get("tcp_pose")
                if not isinstance(tcp_raw, (list, tuple)) or len(tcp_raw) < 7:
                    continue
                try:
                    tcp_pose = np.asarray([float(x) for x in tcp_raw[:7]], dtype=float)
                except (TypeError, ValueError):
                    continue
                if not np.all(np.isfinite(tcp_pose[:3])):
                    continue
                gripper_width = snap.get("gripper_width")
                try:
                    gripper_width = (
                        float(gripper_width)
                        if gripper_width is not None
                        else None
                    )
                except (TypeError, ValueError):
                    gripper_width = None

                raw_entities = snap.get("entities")
                if not isinstance(raw_entities, dict):
                    continue
                nodes = {
                    key: node
                    for key, raw in raw_entities.items()
                    if (node := self._trace_node(str(key), raw)) is not None
                }
                present: Set[Tuple[str, str, str]] = set()

                # EE-object compatibility mirrors runtime gating: only near
                # objects with matching whitelist interaction types emit.
                for key, node in nodes.items():
                    types = self.interaction_types.get(key, set())
                    if not (INTERACTION_GRASP in types or INTERACTION_CONTACT in types):
                        continue
                    obj_xyz = np.asarray(node.pose_world[:3], dtype=float)
                    if not self._is_near(tcp_pose[:3], obj_xyz, ee_pd_edges,
                                         near_labels):
                        continue
                    scored = self._score_ee_object_compatibility(
                        node, tcp_pose, gripper_width, anchor_cache,
                    )
                    if scored is None:
                        continue
                    grasp_score, contact_score = scored
                    if INTERACTION_GRASP in types:
                        self._push_compat_history(
                            samples, history, present,
                            ("ee", key, "grasp-compatibility"),
                            grasp_score,
                        )
                    if INTERACTION_CONTACT in types:
                        self._push_compat_history(
                            samples, history, present,
                            ("ee", key, "contact-compatibility"),
                            contact_score,
                        )

                keys = sorted(nodes)
                for i in range(len(keys)):
                    for j in range(i + 1, len(keys)):
                        a = nodes[keys[i]]
                        b = nodes[keys[j]]
                        a_xyz = np.asarray(a.pose_world[:3], dtype=float)
                        b_xyz = np.asarray(b.pose_world[:3], dtype=float)
                        if not self._is_near(a_xyz, b_xyz, obj_pd_edges,
                                             near_labels):
                            continue
                        a_types = self.interaction_types.get(a.node_id, set())
                        b_types = self.interaction_types.get(b.node_id, set())

                        if (
                            INTERACTION_CONTACT in a_types
                            and INTERACTION_CONTACT in b_types
                        ):
                            a_comps = lookup_contact_components(self.affordance_set, a)
                            b_comps = lookup_contact_components(self.affordance_set, b)
                            if a_comps and b_comps:
                                meas = obj_contact_compatibility(
                                    a.pose_world, a_comps,
                                    b.pose_world, b_comps,
                                    a.node_id, b.node_id,
                                )
                                if meas is not None:
                                    parts = [meas.pos_mismatch / norm["pos"]]
                                    if meas.orient_mismatch is not None:
                                        parts.append(meas.orient_mismatch / norm["orient"])
                                    self._push_compat_history(
                                        samples, history, present,
                                        (a.node_id, b.node_id, "contact-compatibility"),
                                        _mean_normalized(parts),
                                    )

                        if (
                            INTERACTION_SUPPORT in a_types
                            and INTERACTION_SUPPORT in b_types
                        ):
                            for supporter, supported in ((a, b), (b, a)):
                                sup_comps = components_for_partner(
                                    lookup_support_components(
                                        self.affordance_set, supporter),
                                    supported.node_id,
                                )
                                bot_comps = components_for_partner(
                                    lookup_bottom_components(
                                        self.affordance_set, supported),
                                    supporter.node_id,
                                )
                                if not sup_comps or not bot_comps:
                                    continue
                                meas = support_compatibility(
                                    supporter.pose_world, sup_comps,
                                    supported.pose_world, bot_comps,
                                )
                                if meas is None:
                                    continue
                                parts = [
                                    meas.xy_mismatch / norm["xy"],
                                    meas.vertical_mismatch / norm["vertical"],
                                ]
                                if meas.orient_mismatch is not None:
                                    parts.append(meas.orient_mismatch / norm["orient"])
                                self._push_compat_history(
                                    samples, history, present,
                                    (
                                        supporter.node_id,
                                        supported.node_id,
                                        "support-compatibility",
                                    ),
                                    _mean_normalized(parts),
                                )

                        if (
                            INTERACTION_CONTAIN in a_types
                            and INTERACTION_CONTAIN in b_types
                        ):
                            for container, containee in ((a, b), (b, a)):
                                con_comps = lookup_contain_components(
                                    self.affordance_set, container,
                                )
                                key_comps = lookup_key_components(
                                    self.affordance_set, containee,
                                )
                                if not con_comps or not key_comps:
                                    continue
                                meas = contain_compatibility(
                                    container.pose_world, con_comps,
                                    containee.pose_world, key_comps,
                                )
                                if meas is None:
                                    continue
                                parts = [
                                    meas.radial_mismatch / norm["radial"],
                                    meas.axial_mismatch / norm["axial"],
                                ]
                                if meas.orient_mismatch is not None:
                                    parts.append(meas.orient_mismatch / norm["orient"])
                                self._push_compat_history(
                                    samples, history, present,
                                    (
                                        container.node_id,
                                        containee.node_id,
                                        "contain-compatibility",
                                    ),
                                    _mean_normalized(parts),
                                )

                for key in list(history):
                    if key not in present:
                        del history[key]
        return samples

    def _admitted(self) -> Set[str]:
        """The keys that reach ``members`` under the active policy.

        ``target-supporters`` keeps the target plus whatever directly supports
        it. Contact alone is not enough there: a rollout touches whatever is in
        the way, so admitting every contacted entity fills a pick-the-bowl graph
        with the groceries the arm brushed past, which have nothing to do with
        the bowl. Direct support is the relation that makes an entity part of
        the task -- it is what the target rests on and must be lifted off.

        ``full-evidence`` keeps everything the collector saw. It is not a
        runtime asset; it is the record a pruning policy is applied to later,
        so that trying a different policy costs a re-prune instead of another
        collection run.
        """
        if self.membership_policy == MEMBERSHIP_FULL_EVIDENCE:
            keep = set(self.roles)
            if self.target:
                keep.add(self.target)
            return keep
        keep = {self.target} if self.target else set()
        for key, supported in self.supports.items():
            if self.target in supported:
                keep.add(key)
        return keep

    def _resolve_or_refuse(self, unresolved: Dict[str, str]) -> None:
        """Refuse a runtime asset with unresolved members; record a raw one.

        The two policies write different artifacts and want different answers.

        ``target-supporters`` is the shape the runtime loads, so a member it
        admits without a usable height scale has to stop the mine. Every one
        of them would be labelled on another family's deadband, and ``level``
        has to mean one thing per scale.

        ``full-evidence`` is not a runtime asset -- it is the record a pruning
        policy is applied to later, and the members that fail to classify are
        usually the ones pruning removes anyway. A sofa the arm brushed past
        on its way to the can is neither grasped, nor a holder, nor an
        extended surface, so no rule reaches it; refusing there would discard
        nine targets' evidence over furniture no runtime asset can contain.
        It is recorded instead, and rejected at the stage that decides what
        reaches the runtime -- see ``prune_whitelists``.
        """
        if not unresolved:
            return
        if self.membership_policy == MEMBERSHIP_FULL_EVIDENCE:
            for key, reason in sorted(unresolved.items()):
                log.warning(
                    "subtask=%s target=%s: %s is unresolved (%s). The raw "
                    "asset keeps it, marked %r; no runtime asset may contain "
                    "it.",
                    self.subtask, self.target, key, reason,
                    families_rules.UNRESOLVED_FIELD,
                )
            return
        detail = "; ".join(
            f"{key}: {reason}" for key, reason in sorted(unresolved.items()))
        raise ValueError(
            f"subtask={self.subtask} target={self.target}: "
            f"{len(unresolved)} member(s) admitted by the "
            f"{self.membership_policy!r} policy have no usable end-effector "
            f"height family -- {detail}. Labelling them would borrow another "
            "family's deadband, which is how one token comes to mean two "
            "heights."
        )

    def payload(self) -> Dict[str, Any]:
        admitted = self._admitted()
        members: Dict[str, Dict[str, Any]] = {}
        for key in sorted(self.roles):
            if key not in admitted:
                continue
            entry: Dict[str, Any] = {
                "roles": sorted(self.roles[key]),
                "interaction_types": sorted(self.interaction_types.get(key, set())),
                "kind": self.kinds.get(key, "other"),
            }
            if key in self.names:
                entry["name"] = self.names[key]
            if self.interaction_count.get(key):
                entry["interaction_rollouts"] = self.interaction_count[key]
            if self.support_count.get(key):
                entry["support_rollouts"] = self.support_count[key]
                # Only references that survived admission, so no member points
                # at a key the file does not contain.
                supports = sorted(self.supports[key] & admitted)
                if supports:
                    entry["supports"] = supports
            members[key] = entry

        # Surface the missing-supporter regression loudly. A pick target that
        # is interacted across every rollout but has zero supporters usually
        # means the collector never observed the resting contact before the
        # arm broke it -- widen the pre-grasp observation window (or lower
        # _RESET_WARMUP_TICKS / observe_stride) so at least one tick lands
        # while the target is still on its receptacle.
        has_supporter = any(
            "support" in entry["roles"] for entry in members.values())
        if not has_supporter and self.membership_policy != MEMBERSHIP_FULL_EVIDENCE:
            log.warning(
                "subtask=%s target=%s: interacted across %d rollouts but no "
                "supporter of it was recorded, so the whitelist is the target "
                "alone; the collector likely missed the resting contact window "
                "before the arm broke it",
                self.subtask, self.target,
                self.interaction_count.get(self.target, 0),
            )

        # Classification, before the bins: what a member is decides which
        # height scale labels it, and an unclassified one would be labelled on
        # another family's deadband -- which is how one token comes to mean
        # two heights.
        structural = families_rules.structural_surfaces(self.extents, members)
        # Collected rather than raised one at a time. Whether an unresolved
        # member is fatal depends on which asset is being written, and that
        # decision belongs in one place: see ``_resolve_or_refuse``.
        unresolved: Dict[str, str] = {
            key: families_rules.UNRESOLVED_NO_EXTENT
            for key in families_rules.unclassified_supporters(
                self.extents, members)
        }
        holders = {k for k, v in self.supports.items() if v}
        supported = {s for values in self.supports.values() for s in values}
        members = admit_site_members(members, self.sites)
        # Two exclusions from classification, for different reasons.
        #
        # Virtual sites are excluded because they carry no interaction types,
        # so rule 5 would call a rest position a goal marker and hand it that
        # family's height scale -- silently, because rule 5 returns a family
        # rather than None.
        #
        # A supporter with no readable extent is excluded because the holder
        # rule would call it a receptacle -- a real family, which then hides
        # it from every later check that looks for a missing one. Whether it
        # is an extended surface is undecidable without the extent, and
        # guessing "not a surface" is the ~0.9m origin error the
        # classification exists to remove.
        families = families_rules.object_families(
            {k: v for k, v in members.items()
             if not k.startswith(SITE_PREFIX) and k not in unresolved},
            holders, supported, structural,
        )
        for key in families_rules.ambiguous_families(families):
            unresolved.setdefault(key, families_rules.UNRESOLVED_NO_FAMILY)
        # A surface whose plane was never mined is classified and still
        # unusable: the runtime measures its height against that plane and
        # raises without it, and calibrating against the actor origin instead
        # is the ~0.9m error the classification exists to remove.
        for key, family in sorted(families.items()):
            if family != spatial_metrics.FAMILY_STRUCTURAL:
                continue
            if lookup_reference_surface(
                    self.affordance_set,
                    Node(node_id=key, node_type="object", name=key,
                         pose_world=[0.0] * 3 + [1.0, 0.0, 0.0, 0.0],
                         attributes={"whitelist_key": key})) is None:
                unresolved.setdefault(key, families_rules.UNRESOLVED_NO_PLANE)

        self._resolve_or_refuse(unresolved)
        for key, entry in members.items():
            if key in structural:
                entry["structural_surface"] = True
                entry["structural_surface_reason"] = structural[key]
            family = families.get(key)
            if family:
                entry["family"] = family
            if key in unresolved:
                entry[families_rules.UNRESOLVED_FIELD] = unresolved[key]

        object_samples = self._mine_object_pair_samples()
        # After classification, because which scale a sample belongs on is
        # exactly what the classification decides -- and an unresolved member
        # contributes nothing, because there is no scale it belongs on. Its
        # heights would otherwise land on whichever family the rules happened
        # to reach it with, which is the deadband error in a new place.
        family_samples = self._mine_family_height_samples(
            {key: family for key, family in families.items()
             if key not in unresolved})
        for key, values in family_samples.items():
            object_samples[key].extend(values)
        robust, _observed = self._aggregate_bins(object_samples)
        compatibility_samples = self._mine_compatibility_samples(
            derive_bin_edges(robust)
        )
        for key, values in object_samples.items():
            compatibility_samples[key].extend(values)
        robust, observed = self._aggregate_bins(compatibility_samples)
        bin_edges = derive_bin_edges(robust)
        out = {
            "_schema_version": WHITELIST_SCHEMA_VERSION,
            "subtask": self.subtask,
            "task_group": self.task_group,
            "membership_policy": self.membership_policy,
            "target": self.target,
            "members": members,
            "sites": dict(self.sites),
            "bin_edges": bin_edges,
            "bin_stats_robust": robust,
            "bin_stats_observed": observed,
            "_n_successful_rollouts": self.rollout_count,
        }
        # An index of what the member entries already say, so the state is
        # visible without walking them. Recomputed wherever membership
        # changes -- a copied one would go on naming members a prune dropped.
        if unresolved:
            out["_unresolved_members"] = dict(sorted(unresolved.items()))
        return out


def _target_key(data: Dict[str, Any]) -> Optional[str]:
    return normalize_asset_key(data.get("entity_key"))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--success-states-dir", required=True,
        help="robot_success_states/ root (parent of <robot>/<task-group>/).",
    )
    parser.add_argument(
        "--task-group", required=True,
        help="MS-HAB task whose rollouts to mine (set_table, tidy_house, ...). "
             "Only this group's subtree is read, so mining one group can never "
             "touch another's evidence or output.",
    )
    parser.add_argument(
        "--robot", default="fetch",
        help="Robot uid subdirectory (default: fetch).",
    )
    parser.add_argument(
        "--membership-policy", default=MEMBERSHIP_TARGET_SUPPORTERS,
        choices=list(MEMBERSHIP_POLICIES),
        help="full-evidence keeps every interacted entity and its supporters "
             "(the raw asset a later pruning pass consumes); target-supporters "
             "keeps the target plus its direct supporters (runtime shape).",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--sites-dir",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "sites"),
        help="Reviewed site declarations, read as <dir>/<task group>.json. "
             "The miner never derives a site from evidence: letting it do so "
             "invented a goal region for a task that has none.",
    )
    parser.add_argument(
        "--affordance-json",
        default=None,
        help=(
            "Path to affordances.json. Defaults to <out-dir>/../affordances.json; "
            "compatibility-change bins are omitted when unavailable."
        ),
    )
    parser.add_argument(
        "--expect-targets", nargs="+", default=None, metavar="TARGET",
        help="The targets this mine must produce, as canonical keys or bare "
             "object names (004_sugar_box). The run fails if the rollouts "
             "yield a different set. Without it a collection that lost a "
             "target mines cleanly and simply writes one file fewer, which "
             "is indistinguishable on disk from one that never wanted it.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    root = Path(args.success_states_dir) / args.robot / args.task_group
    if not root.is_dir():
        log.error(
            "no rollouts for task group %r under %s (looked in %s); collect "
            "them first with collect_robot_success_states --task %s",
            args.task_group, args.success_states_dir, root, args.task_group,
        )
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    affordance_path = (
        Path(args.affordance_json)
        if args.affordance_json
        else out_dir.parent / "affordances.json"
    )
    affordance_set = load_affordance_set(str(affordance_path))

    # One declaration set per subtask of the group, because a site belongs to
    # the subtask whose success predicate defines it. The end-effector rest
    # position is pick's: no other subtask defines an ``ee_rest_thresh``, and
    # routing by group alone would inject it into a place or open mine.
    # The subtask is a property of each pickle, so this resolves lazily.
    declared_cache: Dict[str, Dict[str, Any]] = {}

    def declared_for(subtask: str) -> Dict[str, Any]:
        if subtask not in declared_cache:
            found = declared_sites(
                args.task_group, Path(args.sites_dir), subtask)
            if found:
                # Parsed here rather than at first use: a malformed
                # declaration should stop the mine, not surface as an
                # unresolvable pair at frame one.
                parse_site_declarations(
                    found, where=f"{args.task_group}/{subtask}")
                log.info("site declarations for %s/%s: %s",
                         args.task_group, subtask, sorted(found))
            declared_cache[subtask] = found
        return declared_cache[subtask]

    builders: Dict[Tuple[str, str], _WhitelistBuilder] = {}
    n_rollouts = 0
    for path, data in _iter_pickles(root):
        rollouts = data.get("interaction_rollouts") or []
        version = int(data.get("_schema_version", 0))
        if version < MIN_ROLLOUT_SCHEMA or not rollouts:
            log.error(
                "skip %s: schema v%d, need v%d+. The collision extents, the "
                "end-effector rest calibration and the per-rollout build "
                "configuration were never recorded in this pickle, so no "
                "amount of re-mining can produce them. Recollect with "
                "--no-skip-done.",
                path, version, MIN_ROLLOUT_SCHEMA,
            )
            continue
        recorded = str(
            (data.get("provenance") or {}).get("task_group") or "")
        if not recorded:
            log.error(
                "%s carries no provenance.task_group; it predates the "
                "task-namespaced collector and cannot be attributed to a task. "
                "Recollect it with --no-skip-done",
                path,
            )
            return 2
        if recorded != args.task_group:
            log.error(
                "%s sits under task group %r but records %r. A pickle moved or "
                "copied between task trees would mine one task's scene into "
                "another's whitelist; refusing",
                path, args.task_group, recorded,
            )
            return 2
        subtask = str(data.get("subtask_type") or path.parent.name)
        # Group by each rollout's own target rather than the file's. A
        # multi-object collection (the ``all`` policy) puts many targets in one
        # pkl, where the file-level entity_key is only the first rollout's.
        fallback = _target_key(data)
        for rollout in rollouts:
            if not isinstance(rollout, dict):
                continue
            target = (
                normalize_asset_key(rollout.get("target_key")) or fallback)
            if not target:
                log.warning("skip a rollout in %s: no target_key", path)
                continue
            builder = builders.setdefault(
                (subtask, target),
                _WhitelistBuilder(
                    subtask,
                    target,
                    task_group=args.task_group,
                    membership_policy=args.membership_policy,
                    affordance_set=affordance_set,
                    sites=declared_for(subtask),
                ),
            )
            builder.absorb(rollout)
            n_rollouts += 1

    if not builders:
        log.error("no successful interaction rollouts found under %s", root)
        return 2

    if args.expect_targets is not None:
        def _expected(name: str) -> str:
            # ``004_sugar_box`` is how a person writes it on the command
            # line; the canonical key is ``actor:004_sugar_box``. Both are
            # accepted, and a key naming its own kind is left alone.
            key = normalize_asset_key(name) or str(name)
            return key if ":" in key else f"actor:{key}"

        mined = {target for _subtask, target in builders}
        wanted = {_expected(name) for name in args.expect_targets}
        missing, extra = sorted(wanted - mined), sorted(mined - wanted)
        if missing or extra:
            log.error(
                "the rollouts under %s yield %d target(s), not the %d asked "
                "for. Missing: %s. Unexpected: %s. A mine of the wrong set "
                "writes a directory that looks finished, so this stops "
                "before anything is written.",
                root, len(mined), len(wanted), missing or "none",
                extra or "none",
            )
            return 2

    empty = [
        (subtask, target)
        for (subtask, target), builder in sorted(builders.items())
        if not builder.roles
    ]
    if empty:
        for subtask, target in empty:
            log.error(
                "empty whitelist for subtask=%s target=%s; collection recorded "
                "no robot-interacted entities for successful rollouts",
                subtask, target,
            )
        log.error(
            "refusing to write invalid whitelist assets; recollect with the "
            "current collector and --no-skip-done"
        )
        return 2

    # Every payload before any file. Mining is where an asset can still be
    # refused -- an unresolved member, a surface with no plane -- and writing
    # as we went left a directory holding four of nine targets that looked
    # exactly like a finished mine. Nothing is written unless all of it can be.
    staged: List[Tuple[Path, Dict[str, Any], int]] = []
    for (subtask, target), builder in sorted(builders.items()):
        filename_target = whitelist_target_slug(target)
        out_path = out_dir / f"{subtask}_{filename_target}.json"
        staged.append((out_path, builder.payload(), len(builder.roles)))

    for out_path, payload, seen in staged:
        with open(out_path, "w") as stream:
            json.dump(payload, stream, indent=2)
        log.info(
            "wrote %s (%d members of %d seen)",
            out_path.name, len(payload["members"]), seen,
        )
    flagged = {
        path.name: sorted(payload["_unresolved_members"])
        for path, payload, _seen in staged
        if payload.get("_unresolved_members")
    }
    if flagged:
        log.warning(
            "%d of %d whitelist(s) carry unresolved members. They are kept as "
            "evidence and marked %r, and prune_whitelists refuses any runtime "
            "asset that still contains one: %s",
            len(flagged), len(staged), families_rules.UNRESOLVED_FIELD,
            flagged,
        )
    log.info(
        "mined %d %s whitelists for task group %r from %d successful rollouts",
        len(builders), args.membership_policy, args.task_group, n_rollouts,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
