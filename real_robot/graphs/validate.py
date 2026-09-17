"""Gemini's episode answers -> per-frame facts, or an exact list of why not.

Gemini decides every label: which bin a distance falls in, whether a grasp
holds, how fast a gap is closing, when an event happens. This module decides
none of them. It

* resolves entity names and fact ids and stores each fact in canonical
  orientation, mirroring direction-carrying labels when the answer named the
  pair the other way round;
* expands frame intervals into per-frame labels;
* refuses illegal relation/label combinations, facts that conflict with
  themselves, frames outside the episode, and gaps in required coverage;
* removes temporal labels before frame ``K``, where ``v_t - v_{t-K}`` has no
  value to difference against, and records that it did;
* checks the tracking anchors against what tracking and geometry need: an
  anchor for every entity in every camera at frame 0, every named point listed
  (visible or explicitly hidden), anchors at every event, bounded gaps while
  something moves in the image, and a table keyframe whose surface points are
  far enough apart, and far enough from a line, to define a plane;
* checks the events and the outcome against each other (a completion needs a
  success and a seated lid first; nothing is released before it is grasped) --
  these need only the events pass, so they are checked before relations are
  asked for;
* checks the task-critical facts against each other: grasp and release events
  against the grasp labels, placement and seating against containment and
  support, the completion and the outcome against all of them, and the active
  target against where the banana is. A disagreement is an issue, not a
  warning -- the annotation is not valid until Gemini resolves it from the
  video.

Problems come back as :class:`Issue` records grouped by what has to be asked
again -- the events pass, specific anchors, a reconciliation of contradicting
answers, or specific facts over specific frames -- so the caller can request
exactly that instead of re-annotating the episode.
"""

from __future__ import annotations

import collections
import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from scenegraph.core.relation_rules import CHANGE_LABELS, DST_HOLDS, SRC_HOLDS, TEMPORAL_RELATIONS

from .schema import EE_ID, RELATION_SCOPES, GraphSpec, SpecError

ANNOTATION_FORMAT = "real_robot/kitchen-annotation-v2"
EARLIER_FORMATS = ("real_robot/kitchen-annotation-v1",)
MODES = ("full_episode", "past_only")
PASSES = ("events", "relations")
# Repair order: upstream answers are corrected before what depends on them.
STAGES = ("events", "anchors", "consistency", "relations")
EVENT_TYPES = ("grasp", "release", "place", "drop", "lid_seated", "task_complete")
DISAGREEMENT_KINDS = ("event", "active_target", "outcome")

_EE_ALIASES = ("ee", "gripper", "right_gripper", "end_effector", "right_end_effector", "robot_gripper")


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class Issue:
    code: str
    message: str
    stage: str
    fact: Optional[Tuple[str, str, str]] = None
    frames: List[Tuple[int, int]] = field(default_factory=list)
    # Further facts a contradiction involves, and the anchors to request.
    facts: List[Tuple[str, str, str]] = field(default_factory=list)
    anchors: List[Dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "stage": self.stage,
            "message": self.message,
            "fact": list(self.fact) if self.fact else None,
            "frames": [list(r) for r in self.frames],
            "facts": [list(f) for f in self.facts],
            "anchors": [dict(a) for a in self.anchors],
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Issue":
        return cls(code=data["code"], message=data["message"], stage=data["stage"],
                   fact=tuple(data["fact"]) if data.get("fact") else None,
                   frames=[tuple(r) for r in data.get("frames", [])],
                   facts=[tuple(f) for f in data.get("facts", [])],
                   anchors=[dict(a) for a in data.get("anchors", [])])


@dataclass
class ValidationSettings:
    """What a complete annotation has to contain. Part of every annotation's input identity."""

    keyframe_every: int = 15
    consistency_tolerance: int = 5
    event_anchor_tolerance: int = 1
    moving_cameras: Tuple[str, ...] = ("wrist_right",)
    check_event_anchors: bool = True
    check_anchor_gaps: bool = True
    # Table surface points that define a plane: every side of their triangle at
    # least this long, and its area at least this, in normalised image units.
    table_min_separation: float = 0.08
    table_min_area: float = 0.005

    @classmethod
    def from_config(cls, annotation_cfg: Mapping[str, Any], mode: str) -> "ValidationSettings":
        section = annotation_cfg["annotation"]
        anchors = section["anchors"]
        plane = anchors["table_plane"]
        full = mode == "full_episode"
        return cls(keyframe_every=int(anchors["max_gap_frames"]),
                   consistency_tolerance=int(section["consistency_tolerance_frames"]),
                   event_anchor_tolerance=int(anchors["event_tolerance_frames"]),
                   moving_cameras=tuple(anchors["moving_cameras"]),
                   check_event_anchors=full, check_anchor_gaps=full,
                   table_min_separation=float(plane["min_separation"]), table_min_area=float(plane["min_area"]))

    def identity(self) -> Dict[str, Any]:
        return {"keyframe_every": self.keyframe_every, "consistency_tolerance": self.consistency_tolerance,
                "event_anchor_tolerance": self.event_anchor_tolerance, "moving_cameras": list(self.moving_cameras),
                "check_event_anchors": self.check_event_anchors, "check_anchor_gaps": self.check_anchor_gaps,
                "table_min_separation": self.table_min_separation, "table_min_area": self.table_min_area}


@dataclass
class EpisodeAnnotation:
    """One validated episode. Per-frame lists are indexed ``[fact][frame]``."""

    episode_index: int
    n_frames: int
    fps: float
    mode: str
    temporal_window: int
    absolute: List[List[Optional[str]]]
    temporal: List[List[Optional[str]]]
    active_target: List[Optional[str]]
    events: List[Dict[str, Any]]
    outcome: Dict[str, Any]
    keyframes: List[Dict[str, Any]]
    provenance: Dict[str, Any]
    spec_identity: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)
    input_identity: Dict[str, Any] = field(default_factory=dict)
    repair_log: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.issues

    # ---------------------------------------------------------- per frame
    def label(self, spec: GraphSpec, src: str, dst: str, relation: str) -> List[Optional[str]]:
        """Labels for one fact, named in either orientation, read back canonically."""
        csrc, cdst, _, _ = spec.canonicalize(relation, src, dst)
        index = spec.fact_index(csrc, cdst, relation)
        if index is None:
            raise KeyError(f"{relation}({src}, {dst}) is not a fact in this graph")
        labels = self.absolute[index]
        if (csrc, cdst) == (src, dst):
            return list(labels)
        from .schema import mirror_absolute
        return [None if v is None else mirror_absolute(relation, v) for v in labels]

    def holds(self, spec: GraphSpec, relation: str, a: str, b: str) -> np.ndarray:
        """Undirected predicate: the label is ``holds``."""
        return np.array([v == "holds" for v in self.label(spec, a, b, relation)], dtype=bool)

    def held_by(self, spec: GraphSpec, relation: str, holder: str, held: str) -> np.ndarray:
        """Directed predicate: ``holder`` supports/contains ``held``."""
        want = holder_label(spec, holder, held)
        src, dst, _ = spec.canonical_order(holder, held)
        index = spec.fact_index(src, dst, relation)
        if index is None:
            raise KeyError(f"{relation}({holder}, {held}) is not a fact in this graph")
        return np.array([v == want for v in self.absolute[index]], dtype=bool)

    def target_array(self) -> List[Optional[str]]:
        return list(self.active_target)

    # --------------------------------------------------------- serialise
    def to_json(self, spec: GraphSpec) -> Dict[str, Any]:
        from ..common import stable_hash

        return {
            "format": ANNOTATION_FORMAT,
            "episode_index": int(self.episode_index),
            "n_frames": int(self.n_frames),
            "fps": float(self.fps),
            "mode": self.mode,
            "temporal_window": int(self.temporal_window),
            "status": "valid" if self.valid else "invalid",
            "issues": [issue.to_json() for issue in self.issues],
            "warnings": list(self.warnings),
            "outcome": self.outcome,
            "events": self.events,
            "active_target": [
                {"object": value, "start": start, "end": end}
                for start, end, value in runs(self.active_target) if value is not None
            ],
            "absolute": _fact_runs(spec, self.absolute),
            "temporal": _fact_runs(spec, self.temporal),
            "keyframes": self.keyframes,
            "provenance": self.provenance,
            "spec_identity": self.spec_identity,
            "input_identity": self.input_identity,
            "input_hash": stable_hash(self.input_identity) if self.input_identity else None,
            "repair_log": self.repair_log,
        }

    @classmethod
    def from_json(cls, spec: GraphSpec, data: Mapping[str, Any]) -> "EpisodeAnnotation":
        if data.get("format") in EARLIER_FORMATS:
            raise ValueError("annotation written by an earlier version of this package (three passes, no anchor "
                             "or consistency checks); annotate the episode again")
        if data.get("format") != ANNOTATION_FORMAT:
            raise ValueError(f"not a kitchen annotation: format={data.get('format')!r}")
        n = int(data["n_frames"])
        absolute = [[None] * n for _ in spec.facts]
        temporal = [[None] * n for _ in spec.facts]
        for store, key in ((absolute, "absolute"), (temporal, "temporal")):
            for item in data[key]:
                index = spec.fact_index(item["src"], item["dst"], item["relation"])
                if index is None:
                    raise ValueError(
                        f"stored fact {item['relation']}({item['src']}, {item['dst']}) is not "
                        "in the current graph configuration; the annotation was made under "
                        "another contract"
                    )
                for t in range(int(item["start"]), int(item["end"]) + 1):
                    store[index][t] = item["label"]
        target: List[Optional[str]] = [None] * n
        for item in data["active_target"]:
            for t in range(int(item["start"]), int(item["end"]) + 1):
                target[t] = item["object"]
        return cls(
            episode_index=int(data["episode_index"]), n_frames=n, fps=float(data["fps"]),
            mode=str(data["mode"]), temporal_window=int(data["temporal_window"]),
            absolute=absolute, temporal=temporal, active_target=target,
            events=list(data["events"]), outcome=dict(data["outcome"]),
            keyframes=list(data["keyframes"]), provenance=dict(data.get("provenance", {})),
            spec_identity=dict(data.get("spec_identity", {})),
            warnings=list(data.get("warnings", [])),
            issues=[Issue.from_json(i) for i in data.get("issues", [])],
            input_identity=dict(data.get("input_identity") or {}),
            repair_log=list(data.get("repair_log") or []),
        )


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def runs(values: Sequence[Any]) -> List[Tuple[int, int, Any]]:
    """Maximal runs of equal values as ``(start, end_inclusive, value)``."""
    out: List[Tuple[int, int, Any]] = []
    start = 0
    for t in range(1, len(values) + 1):
        if t == len(values) or values[t] != values[start]:
            out.append((start, t - 1, values[start]))
            start = t
    return out


def frame_ranges(frames: Iterable[int]) -> List[Tuple[int, int]]:
    ordered = sorted(set(int(f) for f in frames))
    out: List[Tuple[int, int]] = []
    for frame in ordered:
        if out and frame == out[-1][1] + 1:
            out[-1] = (out[-1][0], frame)
        else:
            out.append((frame, frame))
    return out


def format_ranges(ranges: Sequence[Tuple[int, int]], limit: int = 8) -> str:
    text = ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in ranges[:limit])
    if len(ranges) > limit:
        text += f", ... ({len(ranges) - limit} more)"
    return text


def _fact_runs(spec: GraphSpec, labels: List[List[Optional[str]]]) -> List[Dict[str, Any]]:
    out = []
    for fact, series in zip(spec.facts, labels):
        for start, end, value in runs(series):
            if value is not None:
                out.append({"src": fact.src, "dst": fact.dst, "relation": fact.relation,
                            "label": value, "start": start, "end": end})
    return out


def holder_label(spec: GraphSpec, holder: str, held: str) -> str:
    """The directional label meaning ``holder`` holds ``held`` in stored order."""
    src, _, _ = spec.canonical_order(holder, held)
    return SRC_HOLDS if src == holder else DST_HOLDS


def normalize_entity(spec: GraphSpec, raw: Any) -> Optional[str]:
    """Entity id for a name Gemini used, or None."""
    if raw is None:
        return None
    text = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    if text in _EE_ALIASES:
        return EE_ID
    for entity in spec.entities:
        names = {
            entity.id.lower(),
            entity.name.lower().replace(" ", "_"),
            entity.key.split(":", 1)[-1].lower(),
            entity.key.lower(),
        }
        if text in names:
            return entity.id
    return None


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number != int(number):
        return None
    return int(number)


# --------------------------------------------------------------------------- #
# Fact ids
# --------------------------------------------------------------------------- #
def fact_ids(spec: GraphSpec) -> List[str]:
    """Stable short ids in configuration order: ``F00``, ``F01``, ..."""
    return [f"F{index:02d}" for index in range(len(spec.facts))]


def fact_id_of(spec: GraphSpec, key: Sequence[str]) -> str:
    for fid, fact in zip(fact_ids(spec), spec.facts):
        if tuple(fact.key) == tuple(key):
            return fid
    raise KeyError(f"no fact {tuple(key)}")


def relations_to_intervals(spec: GraphSpec, raw: Mapping[str, Any]
                           ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Issue], List[Dict[str, Any]]]:
    """A relations answer (per fact id, absolute and temporal interval lists) -> named intervals.

    Returns ``(absolute, temporal, issues, disagreements)``. The two interval
    lists of a fact are independent: their boundaries need not coincide.
    """
    ids = dict(zip(fact_ids(spec), spec.facts))
    absolute: List[Dict[str, Any]] = []
    temporal: List[Dict[str, Any]] = []
    issues: List[Issue] = []
    for number, item in enumerate((raw or {}).get("facts") or ()):
        fid = str((item or {}).get("fact", "")).strip()
        fact = ids.get(fid)
        if fact is None:
            issues.append(Issue("unknown_fact", f"relations entry {number}: unknown fact id {fid!r}", "relations"))
            continue
        for kind, target in (("absolute", absolute), ("temporal", temporal)):
            for interval in item.get(kind) or ():
                interval = interval or {}
                target.append({"relation": fact.relation, "src": fact.src, "dst": fact.dst,
                               "label": interval.get("label"), "start_frame": interval.get("start"),
                               "end_frame": interval.get("end")})
    return absolute, temporal, issues, validate_disagreements((raw or {}).get("pass1_disagreements"))


def validate_disagreements(raw: Any) -> List[Dict[str, Any]]:
    out = []
    for item in raw or ():
        item = item or {}
        kind = str(item.get("kind", "")).strip()
        start, end = _as_int(item.get("start_frame")), _as_int(item.get("end_frame"))
        description = str(item.get("description", "")).strip()
        if kind not in DISAGREEMENT_KINDS or not description:
            continue
        out.append({"kind": kind, "start_frame": start, "end_frame": end, "description": description})
    return out


# --------------------------------------------------------------------------- #
# Relation intervals
# --------------------------------------------------------------------------- #
def expand_intervals(
    spec: GraphSpec,
    intervals: Sequence[Mapping[str, Any]],
    n_frames: int,
    kind: str,
) -> Tuple[List[List[Optional[str]]], List[Issue], List[str]]:
    """Per-frame labels for every configured fact, plus problems."""
    if kind not in ("absolute", "temporal"):
        raise ValueError(kind)
    labels: List[List[Optional[str]]] = [[None] * n_frames for _ in spec.facts]
    issues: List[Issue] = []
    warnings: List[str] = []
    conflicts: Dict[Tuple[int, str, str], List[int]] = {}
    ignored = set()

    for number, item in enumerate(intervals or ()):
        relation = str(item.get("relation", "")).strip()
        raw_src, raw_dst = item.get("src"), item.get("dst")
        label = str(item.get("label", "")).strip()
        start = _as_int(item.get("start_frame", item.get("start")))
        end = _as_int(item.get("end_frame", item.get("end")))
        where = f"{kind} interval {number} [{relation} {raw_src}->{raw_dst} {label!r} {start}-{end}]"

        src, dst = normalize_entity(spec, raw_src), normalize_entity(spec, raw_dst)
        if src is None or dst is None:
            issues.append(Issue("unknown_entity", f"{where}: unknown entity", "relations"))
            continue
        if src == dst:
            issues.append(Issue("self_relation", f"{where}: relates an entity to itself", "relations"))
            continue
        if relation not in RELATION_SCOPES:
            issues.append(Issue("unknown_relation", f"{where}: unknown relation", "relations"))
            continue
        if kind == "temporal" and relation not in TEMPORAL_RELATIONS:
            issues.append(Issue(
                "temporal_not_applicable",
                f"{where}: {relation} carries no temporal-change label", "relations"))
            continue
        try:
            if kind == "absolute":
                csrc, cdst, clabel, _ = spec.canonicalize(relation, src, dst, label, None)
            else:
                csrc, cdst, _, clabel = spec.canonicalize(relation, src, dst, None, label)
        except SpecError as exc:
            issues.append(Issue("illegal_label", f"{where}: {exc}", "relations"))
            continue
        index = spec.fact_index(csrc, cdst, relation)
        if index is None:
            if (csrc, cdst, relation) not in ignored:
                ignored.add((csrc, cdst, relation))
                warnings.append(
                    f"{kind}: ignored {relation}({csrc}, {cdst}), which is not a configured fact"
                )
            continue
        fact = spec.facts[index]
        legal = spec.legal_labels(relation) if kind == "absolute" else list(CHANGE_LABELS)
        if clabel not in legal:
            issues.append(Issue(
                "illegal_label",
                f"{where}: {clabel!r} is not a legal {kind} label for {relation}; "
                f"legal labels are {legal}", "relations", fact=fact.key))
            continue
        if start is None or end is None or start > end:
            issues.append(Issue("bad_interval", f"{where}: start/end must be integers with start <= end",
                                "relations", fact=fact.key))
            continue
        if start < 0 or end >= n_frames:
            issues.append(Issue(
                "out_of_range",
                f"{where}: the episode has frames 0-{n_frames - 1}", "relations", fact=fact.key,
                frames=[(max(start, 0), min(end, n_frames - 1))] if start < n_frames and end >= 0 else []))
            continue
        series = labels[index]
        for t in range(start, end + 1):
            if series[t] is None:
                series[t] = clabel
            elif series[t] != clabel:
                conflicts.setdefault((index, series[t], clabel), []).append(t)

    for (index, first, second), frames in conflicts.items():
        fact = spec.facts[index]
        ranges = frame_ranges(frames)
        issues.append(Issue(
            "conflict",
            f"{kind}: {fact.label()} is both {first!r} and {second!r} on frames {format_ranges(ranges)}",
            "relations", fact=fact.key, frames=ranges))

    if kind == "temporal":
        K = spec.temporal_window
        dropped = 0
        for series in labels:
            for t in range(min(K, n_frames)):
                if series[t] is not None:
                    series[t] = None
                    dropped += 1
        if dropped:
            warnings.append(
                f"temporal: removed {dropped} label(s) before frame {K}, where "
                f"v_t - v_(t-{K}) has no earlier value in this episode"
            )
    return labels, issues, warnings


def coverage_issues(spec: GraphSpec, labels: List[List[Optional[str]]], n_frames: int,
                    kind: str) -> List[Issue]:
    """Every configured fact labelled on every frame it is defined."""
    issues = []
    first = spec.temporal_window if kind == "temporal" else 0
    for index, fact in enumerate(spec.facts):
        if kind == "temporal" and not fact.temporal:
            continue
        missing = [t for t in range(first, n_frames) if labels[index][t] is None]
        if missing:
            ranges = frame_ranges(missing)
            issues.append(Issue(
                "missing_coverage",
                f"{kind}: {fact.label()} has no label on frames {format_ranges(ranges)}",
                "relations", fact=fact.key, frames=ranges))
    return issues


# --------------------------------------------------------------------------- #
# Events, target, keyframes, outcome
# --------------------------------------------------------------------------- #
def validate_events(spec: GraphSpec, raw: Sequence[Mapping[str, Any]], n_frames: int
                    ) -> Tuple[List[Dict[str, Any]], List[Issue], List[str]]:
    events, issues, warnings = [], [], []
    for number, item in enumerate(raw or ()):
        kind = str(item.get("type", "")).strip()
        obj = normalize_entity(spec, item.get("object"))
        frame = _as_int(item.get("frame"))
        where = f"event {number} [{kind} {item.get('object')} @ {item.get('frame')}]"
        if kind not in EVENT_TYPES:
            issues.append(Issue("bad_event", f"{where}: type must be one of {EVENT_TYPES}", "events"))
            continue
        if obj is None or obj == EE_ID:
            issues.append(Issue("bad_event", f"{where}: object must be one of {spec.object_ids}", "events"))
            continue
        if frame is None or not 0 <= frame < n_frames:
            issues.append(Issue("bad_event", f"{where}: frame must be in 0-{n_frames - 1}", "events"))
            continue
        events.append({"type": kind, "object": obj, "frame": frame,
                       "evidence": str(item.get("evidence", ""))})
    events.sort(key=lambda e: (e["frame"], EVENT_TYPES.index(e["type"])))
    return events, issues, warnings


def validate_target(spec: GraphSpec, raw: Sequence[Mapping[str, Any]], n_frames: int
                    ) -> Tuple[List[Optional[str]], List[Issue]]:
    target: List[Optional[str]] = [None] * n_frames
    issues: List[Issue] = []
    conflicts: List[int] = []
    for number, item in enumerate(raw or ()):
        obj = normalize_entity(spec, item.get("object"))
        start = _as_int(item.get("start_frame", item.get("start")))
        end = _as_int(item.get("end_frame", item.get("end")))
        where = f"active_target interval {number} [{item.get('object')} {start}-{end}]"
        if obj not in spec.targets:
            issues.append(Issue("bad_target", f"{where}: target must be one of {spec.targets}", "events"))
            continue
        if start is None or end is None or start > end or start < 0 or end >= n_frames:
            issues.append(Issue("bad_target", f"{where}: frames must lie in 0-{n_frames - 1}", "events"))
            continue
        for t in range(start, end + 1):
            if target[t] is None:
                target[t] = obj
            elif target[t] != obj:
                conflicts.append(t)
    if conflicts:
        issues.append(Issue("target_conflict",
                            f"active_target names two objects on frames {format_ranges(frame_ranges(conflicts))}",
                            "events", frames=frame_ranges(conflicts)))
    missing = [t for t in range(n_frames) if target[t] is None]
    if missing:
        issues.append(Issue("target_coverage",
                            f"active_target is missing on frames {format_ranges(frame_ranges(missing))}",
                            "events", frames=frame_ranges(missing)))
    return target, issues


def gemini_box_to_normalized(box: Sequence[Any]) -> Optional[List[float]]:
    """Gemini ``[ymin, xmin, ymax, xmax]`` in 0-1000 -> ``[x0, x1, y0, y1]`` in [0, 1].

    The repository's node boxes are ``[x0, x1, y0, y1]`` with exclusive maxima;
    a box whose maxima do not exceed its minima reads back as "not visible".
    """
    if box is None or len(box) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite([ymin, xmin, ymax, xmax])):
        return None
    clip = lambda v: float(min(max(v / 1000.0, 0.0), 1.0))
    return [clip(xmin), clip(xmax), clip(ymin), clip(ymax)]


def gemini_point_to_normalized(point: Sequence[Any]) -> Optional[List[float]]:
    """Gemini ``[y, x]`` in 0-1000 -> ``[x, y]`` in [0, 1]."""
    if point is None or len(point) != 2:
        return None
    try:
        y, x = (float(v) for v in point)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite([y, x])) or not (0 <= y <= 1000 and 0 <= x <= 1000):
        return None
    return [x / 1000.0, y / 1000.0]


def validate_keyframes(spec: GraphSpec, raw: Sequence[Mapping[str, Any]], n_frames: int
                       ) -> Tuple[List[Dict[str, Any]], List[Issue], List[str]]:
    """Well-formed anchors. Whether there are enough of them is :func:`anchor_issues`."""
    by_slot: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
    issues, warnings = [], []
    for number, item in enumerate(raw or ()):
        frame = _as_int(item.get("frame"))
        camera = str(item.get("camera", "")).strip()
        obj = normalize_entity(spec, item.get("object"))
        where = f"keyframe {number} [{item.get('object')} {camera} @ {item.get('frame')}]"
        if frame is None or not 0 <= frame < n_frames:
            issues.append(Issue("bad_keyframe", f"{where}: frame must be in 0-{n_frames - 1}", "anchors",
                                anchors=[{"object": obj, "camera": camera, "frames": [[0, n_frames - 1]],
                                          "rule": "well-formed"}] if obj and camera in spec.cameras else []))
            continue
        if camera not in spec.cameras:
            issues.append(Issue("bad_keyframe", f"{where}: camera must be one of {spec.cameras}", "anchors"))
            continue
        if obj is None:
            issues.append(Issue("bad_keyframe", f"{where}: unknown object", "anchors"))
            continue
        visible = bool(item.get("visible", False))
        box = None
        anchor = {"object": obj, "camera": camera, "frames": [[frame, frame]], "rule": "well-formed"}
        if visible:
            box = gemini_box_to_normalized(item.get("box_2d"))
            if box is None or not (box[1] > box[0] and box[3] > box[2]):
                issues.append(Issue(
                    "bad_keyframe",
                    f"{where}: a visible object needs box_2d [ymin, xmin, ymax, xmax] in 0-1000 "
                    "with ymin < ymax and xmin < xmax", "anchors", frames=[(frame, frame)], anchors=[anchor]))
                continue
        points: Dict[str, List[float]] = {}
        hidden: List[str] = []
        allowed = spec.points.get(obj, ())
        for entry in item.get("points") or ():
            entry = entry or {}
            name = str(entry.get("name", "")).strip()
            if name not in allowed:
                warnings.append(f"{where}: ignored point {name!r}; {obj} points are {list(allowed)}")
                continue
            if not visible or entry.get("visible") is False:
                hidden.append(name)
                continue
            value = gemini_point_to_normalized(entry.get("point"))
            if value is None:
                issues.append(Issue("bad_keyframe", f"{where}: visible point {name!r} must be [y, x] in 0-1000",
                                    "anchors", frames=[(frame, frame)], anchors=[{**anchor, "points": [name]}]))
                continue
            points[name] = value
        if not visible:
            hidden = list(allowed)
        slot = (frame, camera, obj)
        if slot in by_slot:
            warnings.append(f"{where}: duplicate keyframe; the later one is kept")
        by_slot[slot] = {"frame": frame, "camera": camera, "object": obj, "visible": visible,
                         "box": box, "points": points, "hidden_points": sorted(set(hidden) - set(points))}
    keyframes = sorted(by_slot.values(), key=lambda k: (k["frame"], k["camera"], k["object"]))
    return keyframes, issues, warnings


def plane_triangle(points: Mapping[str, Sequence[float]], min_separation: float
                   ) -> Tuple[float, Optional[Tuple[str, str, str]]]:
    """The largest triangle among named image points whose sides are all at least ``min_separation``.

    Returns ``(area, names)`` in normalised image units, or ``(0.0, None)``
    when no three points are that far apart. Area near zero means the points
    lie along a line.
    """
    best, chosen = 0.0, None
    for names in itertools.combinations(sorted(points), 3):
        a, b, c = (np.asarray(points[name], dtype=np.float64) for name in names)
        if min(np.linalg.norm(a - b), np.linalg.norm(b - c), np.linalg.norm(a - c)) < min_separation:
            continue
        area = 0.5 * abs(float((b - a)[0] * (c - a)[1] - (b - a)[1] * (c - a)[0]))
        if area > best:
            best, chosen = area, names
    return best, chosen


def table_plane_keyframes(keyframes: Sequence[Mapping[str, Any]], camera: str, settings: ValidationSettings
                          ) -> List[Mapping[str, Any]]:
    """Visible table keyframes in ``camera`` whose surface points span a plane, the widest spread first.

    Validation requires one; geometry fits the table plane on the first of them,
    so the two can never disagree about which keyframe is usable.
    """
    scored = []
    for key in keyframes:
        if key["object"] != "table" or key["camera"] != camera or not key["visible"]:
            continue
        area, names = plane_triangle(key["points"], settings.table_min_separation)
        if names is not None and area >= settings.table_min_area:
            scored.append((-area, int(key["frame"]), key))
    return [key for _, _, key in sorted(scored, key=lambda item: item[:2])]


def _gaps(frames: Sequence[int], lo: int, hi: int, limit: int) -> List[Tuple[int, int]]:
    """Stretches of ``[lo, hi]`` longer than ``limit`` frames without an anchor."""
    if hi < lo:
        return []
    inside = sorted(set(f for f in frames if lo <= f <= hi))
    bounds = [lo] + inside + [hi]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a > limit:
            out.append((a, b))
    return out


def anchor_issues(spec: GraphSpec, keyframes: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]],
                  n_frames: int, settings: ValidationSettings) -> List[Issue]:
    """Whether the anchors give tracking and geometry what they need."""
    issues: List[Issue] = []
    primary = spec.cameras[0]
    slots = {(k["frame"], k["camera"], k["object"]): k for k in keyframes}
    tracks: Dict[Tuple[str, str], List[Mapping[str, Any]]] = collections.defaultdict(list)
    for key in keyframes:
        tracks[(key["object"], key["camera"])].append(key)
    limit = int(settings.keyframe_every)

    missing = [(e.id, camera) for e in spec.entities for camera in spec.cameras if (0, camera, e.id) not in slots]
    if missing:
        issues.append(Issue(
            "anchor_initial",
            "anchors: frame 0 needs a keyframe (visible or not) for "
            + ", ".join(f"{obj} in {camera}" for obj, camera in missing),
            "anchors", frames=[(0, 0)],
            anchors=[{"object": obj, "camera": camera, "frames": [[0, 0]], "rule": "initial"} for obj, camera in missing]))

    unlisted: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key in keyframes:
        if not key["visible"]:
            continue
        names = spec.points.get(key["object"], ())
        absent = [p for p in names if p not in key["points"] and p not in key.get("hidden_points", ())]
        if absent:
            entry = unlisted.setdefault((key["object"], key["camera"]), {"frames": [], "points": set()})
            entry["frames"].append(key["frame"])
            entry["points"].update(absent)
    for (obj, camera), entry in sorted(unlisted.items()):
        ranges = frame_ranges(entry["frames"])
        issues.append(Issue(
            "anchor_points",
            f"anchors: {obj} in {camera} is visible at frames {format_ranges(ranges)} without listing points "
            f"{sorted(entry['points'])} (give each point, or mark it hidden)",
            "anchors", frames=ranges,
            anchors=[{"object": obj, "camera": camera, "frames": [[f, f] for f in sorted(set(entry["frames"]))],
                      "rule": "points", "points": sorted(entry["points"])}]))

    if spec.has_entity("table"):
        surface = spec.points.get("table", ())
        if len(surface) >= 3 and not table_plane_keyframes(tracks[("table", primary)], primary, settings):
            issues.append(Issue(
                "anchor_table_plane",
                f"anchors: no {primary} keyframe shows three table surface points at least "
                f"{settings.table_min_separation:.0%} of the image apart and not in a line; the table plane "
                "cannot be measured", "anchors", frames=[(0, 0)],
                anchors=[{"object": "table", "camera": primary, "frames": [[0, 0]], "rule": "table_plane",
                          "points": list(surface)}]))

    if settings.check_event_anchors:
        lacking: Dict[Tuple[str, str], List[int]] = collections.defaultdict(list)
        for event in events:
            for obj in (event["object"], EE_ID):
                for camera in spec.cameras:
                    frames = [k["frame"] for k in tracks[(obj, camera)]]
                    if not any(abs(f - event["frame"]) <= settings.event_anchor_tolerance for f in frames):
                        lacking[(obj, camera)].append(int(event["frame"]))
        for (obj, camera), frames in sorted(lacking.items()):
            issues.append(Issue(
                "anchor_event",
                f"anchors: {obj} in {camera} has no keyframe at event frame(s) {sorted(set(frames))}",
                "anchors", frames=[(f, f) for f in sorted(set(frames))],
                anchors=[{"object": obj, "camera": camera, "frames": [[f, f] for f in sorted(set(frames))],
                          "rule": "event"}]))

    if settings.check_anchor_gaps:
        spans: Dict[Tuple[str, str], List[Tuple[int, int]]] = collections.defaultdict(list)
        # The gripper moves throughout, in every camera.
        for camera in spec.cameras:
            frames = [k["frame"] for k in tracks[(EE_ID, camera)]]
            spans[(EE_ID, camera)] += _gaps(frames, 0, n_frames - 1, limit)
        # A carried object moves in every camera between its grasp and its release or drop.
        for obj in spec.targets:
            ordered = [e for e in events if e["object"] == obj]
            for position, event in enumerate(ordered):
                if event["type"] != "grasp":
                    continue
                end = next((e["frame"] for e in ordered[position + 1:] if e["type"] in ("release", "drop")),
                           n_frames - 1)
                for camera in spec.cameras:
                    frames = [k["frame"] for k in tracks[(obj, camera)]]
                    spans[(obj, camera)] += _gaps(frames, event["frame"], end, limit)
        # A moving camera moves everything it sees in its image.
        for camera in settings.moving_cameras:
            if camera not in spec.cameras:
                continue
            for entity in spec.entities:
                if entity.id == EE_ID:
                    continue
                ordered = sorted(tracks[(entity.id, camera)], key=lambda k: k["frame"])
                for position, key in enumerate(ordered):
                    if not key["visible"]:
                        continue
                    following = ordered[position + 1]["frame"] if position + 1 < len(ordered) else n_frames - 1
                    if following - key["frame"] > limit:
                        spans[(entity.id, camera)].append((key["frame"], following))
        for (obj, camera), gaps in sorted(spans.items()):
            gaps = sorted(set(gaps))
            if not gaps:
                continue
            issues.append(Issue(
                "anchor_gap",
                f"anchors: {obj} in {camera} moves in the image but goes more than {limit} frames without a "
                f"keyframe over {format_ranges(gaps)}",
                "anchors", frames=gaps,
                anchors=[{"object": obj, "camera": camera, "frames": [[a, b] for a, b in gaps], "rule": "gap",
                          "max_gap": limit}]))
    return issues


def validate_outcome(raw: Mapping[str, Any], n_frames: int) -> Tuple[Dict[str, Any], List[Issue]]:
    raw = raw or {}
    issues: List[Issue] = []
    completion = _as_int(raw.get("completion_frame", -1))
    outcome = {
        "success": bool(raw.get("success", False)),
        "banana_in_pot_at_end": bool(raw.get("banana_in_pot_at_end", False)),
        "lid_closed_at_end": bool(raw.get("lid_closed_at_end", False)),
        "completion_frame": completion if completion is not None else -1,
        "failure_reason": str(raw.get("failure_reason", "")),
    }
    if completion is None or not -1 <= completion < n_frames:
        issues.append(Issue("bad_outcome", f"outcome: completion_frame must be -1 or in 0-{n_frames - 1}",
                            "events"))
    elif outcome["success"] and completion < 0:
        issues.append(Issue("bad_outcome", "outcome: a successful episode needs its completion_frame",
                            "events"))
    elif not outcome["success"] and completion >= 0:
        issues.append(Issue("bad_outcome", "outcome: an unsuccessful episode cannot have a completion_frame",
                            "events"))
    return outcome, issues


# --------------------------------------------------------------------------- #
# Consistency of the task-critical facts
# --------------------------------------------------------------------------- #
def _changes(mask: np.ndarray) -> Tuple[List[int], List[int]]:
    """``(onsets, offsets)``: first frames a series turns true, and first frames it turns false again."""
    step = np.diff(np.asarray(mask, dtype=np.int8))
    return [int(t) + 1 for t in np.flatnonzero(step == 1)], [int(t) + 1 for t in np.flatnonzero(step == -1)]


def _near(frames: Iterable[int], target: int, tolerance: int) -> bool:
    return any(abs(int(f) - int(target)) <= tolerance for f in frames)


def event_consistency_issues(spec: GraphSpec, events: Sequence[Mapping[str, Any]], outcome: Mapping[str, Any],
                             n_frames: int, tolerance: int) -> List[Issue]:
    """Contradictions inside the events pass itself: events against each other and against the outcome.

    They need no relation labels, so they are checked -- and sent back to the
    events pass -- before relations are asked for.
    """
    issues: List[Issue] = []

    def window(frame: int) -> Tuple[int, int]:
        return (max(0, int(frame) - tolerance), min(n_frames - 1, int(frame) + tolerance))

    def add(code: str, message: str, frame: int) -> None:
        issues.append(Issue(code, message, "events", frames=[window(frame)]))

    def of(kind: str, obj: str) -> List[int]:
        return [e["frame"] for e in events if e["type"] == kind and e["object"] == obj]

    for obj in spec.targets:
        for frame in of("release", obj):
            if not any(g <= frame for g in of("grasp", obj)):
                add("release_before_grasp", f"events: {obj} is released at frame {frame} before any grasp", frame)
    completion = int(outcome.get("completion_frame", -1))
    complete_events = of("task_complete", "lid")
    seated = of("lid_seated", "lid")
    if outcome.get("success"):
        if not _near(complete_events, completion, tolerance):
            add("completion_event", f"events: the outcome completes at frame {completion}, but no task_complete "
                f"event is reported within {tolerance} frames", completion)
    elif complete_events:
        add("completion_without_success", f"events: task_complete is reported at frame {complete_events[0]}, "
            "but the outcome is not a success", complete_events[0])
    for frame in complete_events:
        if not any(s <= frame + tolerance for s in seated):
            add("completion_before_seated", f"events: task_complete at frame {frame} precedes any lid_seated event",
                frame)
    return issues


def disagreement_issues(n_frames: int, disagreements: Sequence[Mapping[str, Any]]) -> List[Issue]:
    """Problems the relations pass reports with the events pass's answer, to be resolved from the video."""
    issues = []
    for item in disagreements:
        start, end = item.get("start_frame"), item.get("end_frame")
        frames = ([(max(0, int(start)), min(n_frames - 1, int(end)))]
                  if start is not None and end is not None and int(start) <= int(end) else [])
        issues.append(Issue("relations_disagreement",
                            f"consistency: the relations pass reports a problem with the {item['kind']}"
                            f"{' near frames ' + format_ranges(frames) if frames else ''}: {item['description']}",
                            "consistency", frames=frames))
    return issues


def consistency_issues(spec: GraphSpec, annotation: EpisodeAnnotation, tolerance: int) -> List[Issue]:
    """Contradictions between events, outcome, target and the grasp/contain/support labels.

    Code only names the disagreement and the frames and facts involved; which
    side is right is left to Gemini, from the video. Contradictions among the
    events alone are :func:`event_consistency_issues`.
    """
    n = annotation.n_frames
    issues: List[Issue] = []

    def key(relation: str, a: str, b: str) -> Optional[Tuple[str, str, str]]:
        src, dst, _ = spec.canonical_order(a, b)
        index = spec.fact_index(src, dst, relation)
        return None if index is None else tuple(spec.facts[index].key)

    def window(frame: int) -> Tuple[int, int]:
        return (max(0, int(frame) - tolerance), min(n - 1, int(frame) + tolerance))

    def add(code: str, message: str, frames: Sequence[Tuple[int, int]], facts: Sequence[Optional[Tuple]]):
        issues.append(Issue(code, message, "consistency", frames=list(frames),
                            facts=[f for f in facts if f is not None]))

    events = annotation.events
    grasp: Dict[str, np.ndarray] = {}
    for obj in spec.targets:
        if key("grasp", EE_ID, obj) is not None:
            grasp[obj] = annotation.holds(spec, "grasp", EE_ID, obj)
    in_pot = annotation.held_by(spec, "contain", "pot", "banana") if key("contain", "banana", "pot") else None
    on_pot = annotation.held_by(spec, "support", "pot", "lid") if key("support", "lid", "pot") else None
    contain_key, support_key = key("contain", "banana", "pot"), key("support", "lid", "pot")

    def of(kind: str, obj: str) -> List[int]:
        return [e["frame"] for e in events if e["type"] == kind and e["object"] == obj]

    # Grasp and release events against the grasp labels, both ways.
    for obj, series in grasp.items():
        fact = key("grasp", EE_ID, obj)
        onsets, offsets = _changes(series)
        if series[0]:
            onsets = [0] + onsets
        for frame in of("grasp", obj):
            if not _near(onsets, frame, tolerance):
                add("grasp_event_label", f"consistency: grasp of {obj} at frame {frame}, but grasp(ee, {obj}) does "
                    f"not begin to hold within {tolerance} frames", [window(frame)], [fact])
        for kind in ("release", "drop"):
            for frame in of(kind, obj):
                if not _near(offsets, frame, tolerance):
                    add(f"{kind}_event_label", f"consistency: {kind} of {obj} at frame {frame}, but grasp(ee, {obj}) "
                        f"does not stop holding within {tolerance} frames", [window(frame)], [fact])
        for frame in onsets:
            if not _near(of("grasp", obj), frame, tolerance):
                add("grasp_label_event", f"consistency: grasp(ee, {obj}) begins to hold at frame {frame}, but no "
                    f"grasp event of {obj} is reported within {tolerance} frames", [window(frame)], [fact])
        for frame in offsets:
            if not _near(of("release", obj) + of("drop", obj), frame, tolerance):
                add("release_label_event", f"consistency: grasp(ee, {obj}) stops holding at frame {frame}, but no "
                    f"release or drop of {obj} is reported within {tolerance} frames", [window(frame)], [fact])

    # Placement against containment and support.
    if in_pot is not None:
        onsets, _ = _changes(in_pot)
        if in_pot[0]:
            onsets = [0] + onsets
        for frame in of("place", "banana"):
            after = in_pot[min(n - 1, frame + tolerance)]
            if not (_near(onsets, frame, tolerance) or after):
                add("place_label", f"consistency: the banana is placed at frame {frame}, but contain(pot holds "
                    "banana) does not hold there", [window(frame)], [contain_key, key("grasp", EE_ID, "banana")])
        for frame in onsets:
            if not _near(of("place", "banana") + of("drop", "banana"), frame, tolerance):
                add("contain_label_event", f"consistency: contain(pot holds banana) begins at frame {frame}, but no "
                    f"place or drop of the banana is reported within {tolerance} frames", [window(frame)],
                    [contain_key])
    if on_pot is not None:
        onsets, _ = _changes(on_pot)
        if on_pot[0]:
            onsets = [0] + onsets
        for kind in ("place", "lid_seated"):
            for frame in of(kind, "lid"):
                after = on_pot[min(n - 1, frame + tolerance)]
                if not (_near(onsets, frame, tolerance) or after):
                    add(f"{kind}_label", f"consistency: {kind.replace('_', ' ')} of the lid at frame {frame}, but "
                        "support(pot holds lid) does not hold there", [window(frame)], [support_key])
        for frame in onsets:
            if not _near(of("place", "lid") + of("lid_seated", "lid"), frame, tolerance):
                add("support_label_event", f"consistency: support(pot holds lid) begins at frame {frame}, but no "
                    f"place or lid_seated event is reported within {tolerance} frames", [window(frame)],
                    [support_key])

    # Completion and outcome against the labels.
    outcome = annotation.outcome
    completion = int(outcome.get("completion_frame", -1))
    if completion >= 0:
        facts = [contain_key, support_key, key("grasp", EE_ID, "lid")]
        if in_pot is not None and not in_pot[completion]:
            add("completion_contain", f"consistency: the outcome completes at frame {completion}, but "
                "contain(pot holds banana) does not hold there", [window(completion)], facts)
        if on_pot is not None and not on_pot[completion]:
            add("completion_support", f"consistency: the outcome completes at frame {completion}, but "
                "support(pot holds lid) does not hold there", [window(completion)], facts)
        lid = grasp.get("lid")
        lo, hi = window(completion)
        if lid is not None and bool(lid[completion:hi + 1].all()):
            add("completion_grasp", f"consistency: the outcome completes at frame {completion}, but grasp(ee, lid) "
                f"still holds through frame {hi}", [(lo, hi)], facts)
    last = n - 1
    if in_pot is not None and bool(outcome.get("banana_in_pot_at_end")) != bool(in_pot[last]):
        add("outcome_banana_end", f"consistency: the outcome says banana_in_pot_at_end="
            f"{bool(outcome.get('banana_in_pot_at_end'))}, but contain(pot holds banana) at the last frame is "
            f"{bool(in_pot[last])}", [window(last)], [contain_key])
    if on_pot is not None and bool(outcome.get("lid_closed_at_end")) != bool(on_pot[last]):
        add("outcome_lid_end", f"consistency: the outcome says lid_closed_at_end={bool(outcome.get('lid_closed_at_end'))}"
            f", but support(pot holds lid) at the last frame is {bool(on_pot[last])}", [window(last)], [support_key])

    # The active target: the banana until it rests in the pot released, then the lid; the banana again if it leaves.
    if in_pot is not None and "banana" in grasp and len(annotation.active_target) == n:
        expected = np.where(in_pot & ~grasp["banana"], "lid", "banana")
        actual = np.array([str(v) for v in annotation.active_target])
        boundaries = [t for t in range(1, n) if expected[t] != expected[t - 1] or actual[t] != actual[t - 1]]
        wrong = [t for t in range(n) if expected[t] != actual[t] and not _near(boundaries, t, tolerance)]
        if wrong:
            ranges = frame_ranges(wrong)
            add("target_labels", f"consistency: active_target disagrees with where the banana is on frames "
                f"{format_ranges(ranges)} (banana until it rests in the pot released, then the lid; the banana "
                "again if it leaves the pot)", ranges, [contain_key, key("grasp", EE_ID, "banana")])

    return issues


# --------------------------------------------------------------------------- #
# Whole episode
# --------------------------------------------------------------------------- #
def build_annotation(
    spec: GraphSpec,
    *,
    episode_index: int,
    n_frames: int,
    fps: float,
    mode: str,
    events_raw: Mapping[str, Any],
    provenance: Mapping[str, Any],
    spec_identity: Mapping[str, Any],
    relations_raw: Optional[Mapping[str, Any]] = None,
    absolute_raw: Optional[Mapping[str, Any]] = None,
    temporal_raw: Optional[Mapping[str, Any]] = None,
    settings: Optional[ValidationSettings] = None,
    extra_disagreements: Sequence[Mapping[str, Any]] = (),
    events_only: bool = False,
) -> EpisodeAnnotation:
    """Validate one episode's answers.

    Relations come either as a relations answer (``relations_raw``: per fact id)
    or as named interval lists (``absolute_raw``/``temporal_raw``).
    ``events_only`` validates the events pass alone -- events, target, outcome,
    anchors and the contradictions among them -- with no relation labels yet:
    what has to hold before relations are asked for.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    settings = settings or ValidationSettings(
        check_event_anchors=mode == "full_episode", check_anchor_gaps=mode == "full_episode")
    events, event_issues, event_warnings = validate_events(spec, events_raw.get("events"), n_frames)
    target, target_issues = validate_target(spec, events_raw.get("active_target"), n_frames)
    keyframes, key_issues, key_warnings = validate_keyframes(spec, events_raw.get("keyframes"), n_frames)
    outcome, outcome_issues = validate_outcome(events_raw.get("outcome"), n_frames)

    relation_issues: List[Issue] = []
    disagreements = [] if events_only else list(extra_disagreements)
    if events_only:
        absolute_list, temporal_list = [], []
    elif relations_raw is not None:
        absolute_list, temporal_list, relation_issues, found = relations_to_intervals(spec, relations_raw)
        disagreements += found
    else:
        absolute_list = (absolute_raw or {}).get("intervals") or []
        temporal_list = (temporal_raw or {}).get("intervals") or []
    absolute, abs_issues, abs_warnings = expand_intervals(spec, absolute_list, n_frames, "absolute")
    temporal, temp_issues, temp_warnings = expand_intervals(spec, temporal_list, n_frames, "temporal")
    if not events_only:
        abs_issues += coverage_issues(spec, absolute, n_frames, "absolute")
        temp_issues += coverage_issues(spec, temporal, n_frames, "temporal")

    structural = event_issues + target_issues + outcome_issues
    annotation = EpisodeAnnotation(
        episode_index=int(episode_index), n_frames=int(n_frames), fps=float(fps), mode=mode,
        temporal_window=spec.temporal_window, absolute=absolute, temporal=temporal,
        active_target=target, events=events, outcome=outcome, keyframes=keyframes,
        provenance=dict(provenance), spec_identity=dict(spec_identity),
        warnings=event_warnings + key_warnings + abs_warnings + temp_warnings,
        issues=structural + key_issues + relation_issues + abs_issues + temp_issues,
    )
    if not event_issues:
        annotation.issues += anchor_issues(spec, keyframes, events, n_frames, settings)
    if not (event_issues or outcome_issues):
        annotation.issues += event_consistency_issues(spec, events, outcome, n_frames, settings.consistency_tolerance)
    if events_only:
        return annotation
    annotation.issues += disagreement_issues(n_frames, disagreements)
    # Label contradictions are only meaningful once both answers are complete.
    if not structural and not (relation_issues or abs_issues or temp_issues):
        annotation.issues += consistency_issues(spec, annotation, settings.consistency_tolerance)
    return annotation


def repair_plan(annotation: EpisodeAnnotation) -> Dict[str, Dict[str, Any]]:
    """What to ask for again, by stage, in repair order.

    ``events``: the events pass, whole. ``anchors``: the listed keyframes only.
    ``consistency``: events, target and outcome together with the facts the
    contradictions involve. ``relations``: the listed facts only, over the
    listed frames when every issue names its frames.
    """
    plan: Dict[str, Dict[str, Any]] = {}
    for issue in annotation.issues:
        entry = plan.setdefault(issue.stage, {"issues": [], "facts": [], "anchors": [], "frames": [],
                                              "whole_episode": False})
        entry["issues"].append(issue.to_json())
        for fact in ([issue.fact] if issue.fact else []) + list(issue.facts):
            if list(fact) not in entry["facts"]:
                entry["facts"].append(list(fact))
        entry["anchors"].extend(issue.anchors)
        if issue.frames:
            entry["frames"].extend(list(r) for r in issue.frames)
        else:
            entry["whole_episode"] = True
    return {stage: plan[stage] for stage in STAGES if stage in plan}


def _interval_key(spec: GraphSpec, item: Mapping[str, Any]) -> Optional[Tuple[str, str, str]]:
    src, dst = normalize_entity(spec, item.get("src")), normalize_entity(spec, item.get("dst"))
    if src is None or dst is None or src == dst:
        return None
    relation = str(item.get("relation", ""))
    csrc, cdst, _ = spec.canonical_order(src, dst)
    return (csrc, cdst, relation)


def replace_fact_range(previous: Sequence[Mapping[str, Any]], patch: Sequence[Mapping[str, Any]], spec: GraphSpec,
                       facts: Sequence[Sequence[str]], frame_range: Optional[Tuple[int, int]] = None
                       ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Previous intervals with the repaired facts replaced over ``frame_range`` (default: everywhere).

    Returns ``(merged, rejected)``. Patch entries for facts that were not
    requested, or lying wholly outside the requested frames, are rejected
    rather than applied; entries straddling the range are cut to it.
    """
    repaired = {tuple(f) for f in facts}
    lo, hi = (frame_range if frame_range is not None else (0, 10 ** 9))
    kept: List[Dict[str, Any]] = []
    for item in previous:
        if _interval_key(spec, item) not in repaired:
            kept.append(dict(item))
            continue
        start = _as_int(item.get("start_frame", item.get("start")))
        end = _as_int(item.get("end_frame", item.get("end")))
        if start is None or end is None:
            continue
        if end < lo or start > hi:
            kept.append(dict(item))
            continue
        if start < lo:
            kept.append({**item, "start_frame": start, "end_frame": lo - 1})
        if end > hi:
            kept.append({**item, "start_frame": hi + 1, "end_frame": end})
    accepted, rejected = [], []
    for item in patch:
        start = _as_int(item.get("start_frame", item.get("start")))
        end = _as_int(item.get("end_frame", item.get("end")))
        if _interval_key(spec, item) not in repaired or start is None or end is None or end < lo or start > hi:
            rejected.append(dict(item))
            continue
        accepted.append({**item, "start_frame": max(start, lo), "end_frame": min(end, hi)})
    return kept + accepted, rejected


def merge_keyframes(previous: Sequence[Mapping[str, Any]], patch: Sequence[Mapping[str, Any]],
                    requested: Sequence[Mapping[str, Any]], spec: GraphSpec
                    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Previous raw keyframes with the requested slots added or replaced; everything else rejected."""
    allowed: Dict[Tuple[str, str], List[Tuple[int, int]]] = collections.defaultdict(list)
    for anchor in requested:
        for a, b in anchor.get("frames") or ():
            allowed[(anchor.get("object"), anchor.get("camera"))].append((int(a), int(b)))

    def slot(item: Mapping[str, Any]) -> Tuple[Optional[int], str, Optional[str]]:
        return (_as_int(item.get("frame")), str(item.get("camera", "")).strip(), normalize_entity(spec, item.get("object")))

    accepted, rejected = {}, []
    for item in patch:
        frame, camera, obj = slot(item)
        ranges = allowed.get((obj, camera), [])
        if frame is None or not any(a <= frame <= b for a, b in ranges):
            rejected.append(dict(item))
            continue
        accepted[(frame, camera, obj)] = dict(item)
    merged = [dict(item) for item in previous if slot(item) not in accepted]
    return merged + list(accepted.values()), rejected


# --------------------------------------------------------------------------- #
# Past-only updates
# --------------------------------------------------------------------------- #
def assemble_past_only(spec: GraphSpec, updates: Sequence[Mapping[str, Any]], n_frames: int
                       ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Causal per-update answers -> events payload and named interval lists, held between updates.

    Each update's labels apply from its own frame up to the frame before the
    next update. Nothing an update says is applied to an earlier frame, which
    is what keeps the assembled graph readable at deployment timing.
    """
    ordered = sorted(updates, key=lambda u: int(u["frame"]))
    if not ordered or int(ordered[0]["frame"]) != 0:
        raise ValueError("past-only annotation needs an update at frame 0")
    absolute, temporal, target, keyframes = [], [], [], []
    events: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    completion = -1
    for position, update in enumerate(ordered):
        start = int(update["frame"])
        end = (int(ordered[position + 1]["frame"]) - 1) if position + 1 < len(ordered) else n_frames - 1
        if end < start:
            continue
        for item in update.get("facts") or ():
            base = {"relation": item.get("relation"), "src": item.get("src"), "dst": item.get("dst"),
                    "start_frame": start, "end_frame": end}
            absolute.append({**base, "label": item.get("label")})
            change = item.get("temporal_label")
            if change not in (None, "", "none"):
                temporal.append({**base, "label": change})
        target.append({"object": update.get("active_target"), "start_frame": start, "end_frame": end})
        for obj in update.get("objects") or ():
            keyframes.append({**obj, "frame": start})
        for event in update.get("events_so_far") or ():
            frame = _as_int(event.get("frame"))
            if frame is None or frame > start:
                continue
            events[(str(event.get("type")), str(event.get("object")), frame)] = dict(event)
        if completion < 0 and bool(update.get("task_complete", False)):
            completion = start
    last = ordered[-1]
    outcome = {
        "success": completion >= 0,
        "banana_in_pot_at_end": bool(last.get("banana_in_pot", False)),
        "lid_closed_at_end": bool(last.get("lid_closed", False)),
        "completion_frame": completion,
        "failure_reason": "" if completion >= 0 else "not completed within the recording",
    }
    events_payload = {"events": list(events.values()), "active_target": target,
                      "keyframes": keyframes, "outcome": outcome}
    return events_payload, {"intervals": absolute}, {"intervals": temporal}
