"""Gemini's answer for one episode -> per-frame labels and box keyframes, or why not.

Gemini decides every label and box; nothing here fills one in. Every gap,
conflict, illegal label and malformed box becomes an :class:`Issue` naming the
part to ask for again: the active target, specific facts, or the boxes of
specific entities in specific cameras.

An answer is ``{"active_target": [...], "facts": [...], "boxes": [...],
"notes": ...}`` with facts addressed by their id (``F00``, ``F01``, ...) in the
task's configured order, already in canonical orientation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from scenegraph.core.relation_rules import CHANGE_LABELS

from .schema import EE_ID, GraphSpec

ANNOTATION_FORMAT = "real_robot/so101-scene-graph-v1"

_EE_ALIASES = ("ee", "gripper", "end_effector", "robot_gripper", "robot")

Slot = Tuple[str, str]


@dataclass
class Issue:
    code: str
    message: str
    part: str
    fact: Optional[str] = None
    slot: Optional[Slot] = None
    frames: List[Tuple[int, int]] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {"code": self.code, "part": self.part, "message": self.message, "fact": self.fact,
                "slot": list(self.slot) if self.slot else None, "frames": [list(r) for r in self.frames]}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Issue":
        return cls(code=data["code"], message=data["message"], part=data["part"], fact=data.get("fact"),
                   slot=tuple(data["slot"]) if data.get("slot") else None,
                   frames=[tuple(r) for r in data.get("frames", [])])


@dataclass
class ValidationSettings:
    """What a complete answer has to contain. Part of every annotation's input identity."""

    box_every: int = 15
    stride: int = 1

    @classmethod
    def from_config(cls, annotation_cfg: Mapping[str, Any], stride: int) -> "ValidationSettings":
        return cls(box_every=int(annotation_cfg["annotation"]["box_every_frames"]), stride=int(stride))

    @property
    def max_box_gap(self) -> int:
        return self.box_every + self.stride

    def identity(self) -> Dict[str, Any]:
        return {"box_every": self.box_every, "stride": self.stride}


@dataclass
class EpisodeAnnotation:
    """One episode. Per-frame lists are indexed ``[fact][frame]``; boxes by ``(entity, camera)``."""

    episode_index: int
    task: str
    n_frames: int
    fps: float
    temporal_window: int
    absolute: List[List[Optional[str]]]
    temporal: List[List[Optional[str]]]
    active_target: List[Optional[str]]
    boxes: Dict[Slot, List[Dict[str, Any]]]
    answer: Dict[str, Any]
    settings: ValidationSettings
    warnings: List[str] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)
    input_identity: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    repair_log: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.issues

    def to_json(self, spec: GraphSpec) -> Dict[str, Any]:
        from ..common import stable_hash

        return {
            "format": ANNOTATION_FORMAT,
            "episode_index": int(self.episode_index),
            "task": self.task,
            "n_frames": int(self.n_frames),
            "fps": float(self.fps),
            "temporal_window": int(self.temporal_window),
            "status": "valid" if self.valid else "invalid",
            "issues": [issue.to_json() for issue in self.issues],
            "warnings": list(self.warnings),
            "notes": str(self.answer.get("notes") or ""),
            "active_target": [{"object": value, "start": start, "end": end}
                              for start, end, value in runs(self.active_target) if value is not None],
            "absolute": _fact_runs(spec, self.absolute),
            "temporal": _fact_runs(spec, self.temporal),
            "boxes": [{"entity": entity, "camera": camera, "keyframes": keyframes}
                      for (entity, camera), keyframes in self.boxes.items()],
            "answer": self.answer,
            "validation": self.settings.identity(),
            "input_identity": self.input_identity,
            "input_hash": stable_hash(self.input_identity) if self.input_identity else None,
            "provenance": self.provenance,
            "repair_log": self.repair_log,
            "usage": self.usage,
        }

    @classmethod
    def from_json(cls, spec: GraphSpec, data: Mapping[str, Any]) -> "EpisodeAnnotation":
        """Rebuilt from the stored answer, so the per-frame labels are exactly what validation makes of it."""
        if data.get("format") != ANNOTATION_FORMAT:
            raise ValueError(f"not an SO-101 scene-graph annotation: format={data.get('format')!r}")
        if data.get("task") != spec.task:
            raise ValueError(f"annotation is for task {data.get('task')!r}, not {spec.task!r}")
        annotation = build_annotation(
            spec, episode_index=int(data["episode_index"]), n_frames=int(data["n_frames"]),
            fps=float(data["fps"]), answer=data["answer"], settings=ValidationSettings(**data["validation"]))
        if (data.get("status") == "valid") != annotation.valid:
            raise ValueError(f"episode {data['episode_index']}: the stored status {data.get('status')!r} does not "
                             "match what the stored answer validates to; annotate it again")
        annotation.input_identity = dict(data.get("input_identity") or {})
        annotation.provenance = dict(data.get("provenance") or {})
        annotation.repair_log = list(data.get("repair_log") or [])
        annotation.usage = dict(data.get("usage") or {})
        return annotation


# --------------------------------------------------------------------------- #
# Helpers
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
    for fid, fact, series in zip(fact_ids(spec), spec.facts, labels):
        for start, end, value in runs(series):
            if value is not None:
                out.append({"fact": fid, "src": fact.src, "dst": fact.dst, "relation": fact.relation,
                            "label": value, "start": start, "end": end})
    return out


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


def normalize_entity(spec: GraphSpec, raw: Any) -> Optional[str]:
    """Entity id for a name Gemini used, or None."""
    if raw is None:
        return None
    text = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    if text in _EE_ALIASES:
        return EE_ID
    for entity in spec.entities:
        names = {entity.id.lower(), entity.name.lower().replace(" ", "_"),
                 entity.key.split(":", 1)[-1].lower(), entity.key.lower()}
        if text in names:
            return entity.id
    return None


def fact_ids(spec: GraphSpec) -> List[str]:
    """Stable short ids in configuration order: ``F00``, ``F01``, ..."""
    return [f"F{index:02d}" for index in range(len(spec.facts))]


def slots(spec: GraphSpec) -> List[Slot]:
    return [(entity.id, camera) for entity in spec.entities for camera in spec.cameras]


# --------------------------------------------------------------------------- #
# Parts
# --------------------------------------------------------------------------- #
def expand_facts(spec: GraphSpec, raw: Any, n_frames: int
                 ) -> Tuple[List[List[Optional[str]]], List[List[Optional[str]]], List[Issue], List[str]]:
    """Per-frame absolute and temporal labels for every fact, plus problems."""
    ids = fact_ids(spec)
    index_of = {fid: index for index, fid in enumerate(ids)}
    absolute: List[List[Optional[str]]] = [[None] * n_frames for _ in spec.facts]
    temporal: List[List[Optional[str]]] = [[None] * n_frames for _ in spec.facts]
    covered = {kind: [[False] * n_frames for _ in spec.facts] for kind in ("absolute", "temporal")}
    issues: List[Issue] = []
    warnings: List[str] = []
    conflicts: Dict[Tuple[int, str, str, str], List[int]] = {}
    K = spec.temporal_window

    for number, item in enumerate(raw or ()):
        item = item or {}
        fid = str(item.get("fact", "")).strip()
        index = index_of.get(fid)
        if index is None:
            issues.append(Issue("unknown_fact", f"facts entry {number}: unknown fact id {fid!r}", "facts"))
            continue
        fact = spec.facts[index]
        for kind, store, legal in (("absolute", absolute, spec.legal_labels(fact.relation)),
                                   ("temporal", temporal, list(CHANGE_LABELS))):
            entries = item.get(kind) or ()
            if kind == "temporal" and not fact.temporal:
                if entries:
                    warnings.append(f"{fid} {fact.label()}: ignored temporal intervals; the relation has none")
                continue
            for j, interval in enumerate(entries):
                interval = interval or {}
                raw_label = interval.get("label", "")
                label = None if raw_label is None else str(raw_label).strip()
                start, end = _as_int(interval.get("start")), _as_int(interval.get("end"))
                where = f"{fid} {fact.label()} {kind} interval {j} [{label!r} {start}-{end}]"
                if label is not None and label not in legal:
                    issues.append(Issue("illegal_label", f"{where}: legal labels are {legal}", "facts", fact=fid))
                    continue
                if start is None or end is None or start > end:
                    issues.append(Issue("bad_interval", f"{where}: start and end must be integers with start <= end",
                                        "facts", fact=fid))
                    continue
                if end < 0 or start >= n_frames:
                    issues.append(Issue("out_of_range", f"{where}: the episode has frames 0-{n_frames - 1}",
                                        "facts", fact=fid))
                    continue
                if start < 0 or end >= n_frames:
                    warnings.append(f"{where}: clipped to frames 0-{n_frames - 1}")
                series = store[index]
                for t in range(max(start, 0), min(end, n_frames - 1) + 1):
                    if not covered[kind][index][t]:
                        series[t] = label
                        covered[kind][index][t] = True
                    elif series[t] != label:
                        conflicts.setdefault((index, kind, series[t], label), []).append(t)

    for (index, kind, first, second), frames in conflicts.items():
        ranges = frame_ranges(frames)
        issues.append(Issue("conflict", f"{ids[index]} {spec.facts[index].label()} {kind}: both {first!r} and "
                            f"{second!r} on frames {format_ranges(ranges)}", "facts", fact=ids[index], frames=ranges))

    dropped = 0
    for series in temporal:
        for t in range(min(K, n_frames)):
            if series[t] is not None:
                series[t] = None
                dropped += 1
    if dropped:
        warnings.append(f"temporal: removed {dropped} label(s) before frame {K}, where v_t - v_(t-{K}) has no value")

    for index, fact in enumerate(spec.facts):
        for kind, store, first in (("absolute", absolute, 0), ("temporal", temporal, K)):
            if kind == "temporal" and not fact.temporal:
                continue
            missing = [t for t in range(first, n_frames) if not covered[kind][index][t]]
            if missing:
                ranges = frame_ranges(missing)
                issues.append(Issue("missing_coverage", f"{ids[index]} {fact.label()} {kind}: no label on frames "
                                    f"{format_ranges(ranges)}", "facts", fact=ids[index], frames=ranges))
    return absolute, temporal, issues, warnings


def validate_target(spec: GraphSpec, raw: Any, n_frames: int) -> Tuple[List[Optional[str]], List[Issue]]:
    target: List[Optional[str]] = [None] * n_frames
    issues: List[Issue] = []
    conflicts: List[int] = []
    for number, item in enumerate(raw or ()):
        item = item or {}
        obj = normalize_entity(spec, item.get("object"))
        start, end = _as_int(item.get("start")), _as_int(item.get("end"))
        where = f"active_target interval {number} [{item.get('object')} {start}-{end}]"
        if obj not in spec.targets:
            issues.append(Issue("bad_target", f"{where}: the target must be one of {list(spec.targets)}", "target"))
            continue
        if start is None or end is None or start > end or end < 0 or start >= n_frames:
            issues.append(Issue("bad_target", f"{where}: frames must lie in 0-{n_frames - 1}", "target"))
            continue
        for t in range(max(start, 0), min(end, n_frames - 1) + 1):
            if target[t] is None:
                target[t] = obj
            elif target[t] != obj:
                conflicts.append(t)
    if conflicts:
        ranges = frame_ranges(conflicts)
        issues.append(Issue("target_conflict", f"active_target names two objects on frames {format_ranges(ranges)}",
                            "target", frames=ranges))
    missing = [t for t in range(n_frames) if target[t] is None]
    if missing:
        ranges = frame_ranges(missing)
        issues.append(Issue("target_coverage", f"active_target is missing on frames {format_ranges(ranges)}",
                            "target", frames=ranges))
    return target, issues


def gemini_box_to_normalized(box: Any) -> Optional[List[float]]:
    """Gemini ``[ymin, xmin, ymax, xmax]`` in 0-1000 -> ``[x0, x1, y0, y1]`` in [0, 1]."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    values = [_as_int(v) for v in box]
    if any(v is None or not 0 <= v <= 1000 for v in values):
        return None
    ymin, xmin, ymax, xmax = values
    return [xmin / 1000.0, xmax / 1000.0, ymin / 1000.0, ymax / 1000.0]


def validate_boxes(spec: GraphSpec, raw: Any, n_frames: int, settings: ValidationSettings
                   ) -> Tuple[Dict[Slot, List[Dict[str, Any]]], List[Issue], List[str]]:
    """Box keyframes per ``(entity, camera)``: well formed, from frame 0 to the end, never too far apart."""
    wanted = slots(spec)
    collected: Dict[Slot, Dict[int, Dict[str, Any]]] = {slot: {} for slot in wanted}
    issues: List[Issue] = []
    warnings: List[str] = []
    for number, item in enumerate(raw or ()):
        item = item or {}
        entity = normalize_entity(spec, item.get("entity"))
        camera = str(item.get("camera", "")).strip().lower()
        slot = (entity, camera)
        if slot not in collected:
            warnings.append(f"boxes entry {number}: ignored {item.get('entity')!r} in camera {item.get('camera')!r}")
            continue
        for j, keyframe in enumerate(item.get("keyframes") or ()):
            keyframe = keyframe or {}
            frame = _as_int(keyframe.get("frame"))
            where = f"boxes {entity}/{camera} keyframe {j} [frame {keyframe.get('frame')}]"
            if frame is None or not 0 <= frame < n_frames:
                issues.append(Issue("bad_keyframe", f"{where}: the frame must be in 0-{n_frames - 1}", "boxes",
                                    slot=slot))
                continue
            if not isinstance(keyframe.get("visible"), bool):
                issues.append(Issue("bad_keyframe", f"{where}: visible must be a boolean", "boxes", slot=slot))
                continue
            visible = keyframe["visible"]
            box = None
            if visible:
                box = gemini_box_to_normalized(keyframe.get("box_2d"))
                if box is None or not (box[1] > box[0] and box[3] > box[2]):
                    issues.append(Issue("bad_box", f"{where}: a visible entity needs box_2d [ymin, xmin, ymax, xmax] "
                                        "in 0-1000 with ymin < ymax and xmin < xmax", "boxes", slot=slot,
                                        frames=[(frame, frame)]))
                    continue
            if frame in collected[slot]:
                warnings.append(f"{where}: duplicate keyframe; the later one is kept")
            collected[slot][frame] = {"frame": frame, "visible": visible, "box": box}

    gap = settings.max_box_gap
    boxes: Dict[Slot, List[Dict[str, Any]]] = {}
    for slot in wanted:
        keyframes = [collected[slot][f] for f in sorted(collected[slot])]
        boxes[slot] = keyframes
        if not keyframes:
            issues.append(Issue("missing_boxes", f"boxes {slot[0]}/{slot[1]}: no keyframes", "boxes", slot=slot,
                                frames=[(0, n_frames - 1)]))
            continue
        frames = [k["frame"] for k in keyframes]
        holes: List[Tuple[int, int]] = []
        if frames[0] != 0:
            holes.append((0, frames[0]))
        holes += [(a, b) for a, b in zip(frames, frames[1:]) if b - a > gap]
        if frames[-1] != n_frames - 1:
            holes.append((frames[-1], n_frames - 1))
        if holes:
            issues.append(Issue("box_gap", f"boxes {slot[0]}/{slot[1]}: keyframes must start at frame 0 and be at most "
                                f"{settings.box_every} frames apart up to frame {n_frames - 1}; missing between "
                                f"{format_ranges(holes)}", "boxes", slot=slot, frames=holes))
    return boxes, issues, warnings


def build_annotation(spec: GraphSpec, *, episode_index: int, n_frames: int, fps: float,
                     answer: Mapping[str, Any], settings: ValidationSettings) -> EpisodeAnnotation:
    answer = dict(answer or {})
    target, target_issues = validate_target(spec, answer.get("active_target"), n_frames)
    absolute, temporal, fact_issues, fact_warnings = expand_facts(spec, answer.get("facts"), n_frames)
    boxes, box_issues, box_warnings = validate_boxes(spec, answer.get("boxes"), n_frames, settings)
    return EpisodeAnnotation(
        episode_index=int(episode_index), task=spec.task, n_frames=int(n_frames), fps=float(fps),
        temporal_window=spec.temporal_window, absolute=absolute, temporal=temporal, active_target=target,
        boxes=boxes, answer=answer, settings=settings, warnings=fact_warnings + box_warnings,
        issues=target_issues + fact_issues + box_issues)


# --------------------------------------------------------------------------- #
# Repairs
# --------------------------------------------------------------------------- #
@dataclass
class Scope:
    """Which parts of an answer to ask for."""

    target: bool = False
    facts: List[str] = field(default_factory=list)
    boxes: List[Slot] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.target or self.facts or self.boxes)

    def to_json(self) -> Dict[str, Any]:
        return {"target": self.target, "facts": list(self.facts), "boxes": [list(s) for s in self.boxes]}


def full_scope(spec: GraphSpec) -> Scope:
    return Scope(target=True, facts=fact_ids(spec), boxes=slots(spec))


def repair_scope(annotation: EpisodeAnnotation, spec: GraphSpec) -> Scope:
    """The parts the issues name. A facts issue that names no fact asks for every fact."""
    scope = Scope()
    for issue in annotation.issues:
        if issue.part == "target":
            scope.target = True
        elif issue.part == "facts":
            for fid in ([issue.fact] if issue.fact else fact_ids(spec)):
                if fid not in scope.facts:
                    scope.facts.append(fid)
        elif issue.part == "boxes" and issue.slot and issue.slot not in scope.boxes:
            scope.boxes.append(issue.slot)
    order = {fid: i for i, fid in enumerate(fact_ids(spec))}
    scope.facts.sort(key=order.get)
    return scope


def combine(answers: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Answers to disjoint parts of one request, as one answer."""
    out: Dict[str, Any] = {"facts": [], "boxes": []}
    notes = []
    for answer in answers:
        if "active_target" in answer and "active_target" not in out:
            out["active_target"] = list(answer["active_target"] or [])
        out["facts"] += list(answer.get("facts") or [])
        out["boxes"] += list(answer.get("boxes") or [])
        if answer.get("notes"):
            notes.append(str(answer["notes"]))
    out["notes"] = " ".join(notes)
    return out


def merge_answer(spec: GraphSpec, previous: Mapping[str, Any], patch: Mapping[str, Any], scope: Scope
                 ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """``previous`` with the scoped parts replaced by ``patch``; entries outside the scope are rejected."""
    rejected: List[Dict[str, Any]] = []
    merged: Dict[str, Any] = {"active_target": list(previous.get("active_target") or [])}
    if scope.target and patch.get("active_target"):
        merged["active_target"] = list(patch["active_target"])
    elif patch.get("active_target") and not scope.target:
        rejected.append({"active_target": patch["active_target"]})

    replaced: Dict[str, List[Dict[str, Any]]] = {}
    for item in patch.get("facts") or ():
        fid = str((item or {}).get("fact", "")).strip()
        if fid in scope.facts:
            replaced.setdefault(fid, []).append(dict(item))
        else:
            rejected.append({"fact": item})
    merged["facts"] = [dict(item) for item in previous.get("facts") or ()
                       if str((item or {}).get("fact", "")).strip() not in replaced]
    merged["facts"] += [item for items in replaced.values() for item in items]

    new_boxes: Dict[Slot, List[Dict[str, Any]]] = {}
    for item in patch.get("boxes") or ():
        slot = (normalize_entity(spec, (item or {}).get("entity")), str((item or {}).get("camera", "")).strip().lower())
        if slot in scope.boxes:
            new_boxes.setdefault(slot, []).append(dict(item))
        else:
            rejected.append({"boxes": item})

    def slot_of(item: Mapping[str, Any]) -> Slot:
        return (normalize_entity(spec, item.get("entity")), str(item.get("camera", "")).strip().lower())

    merged["boxes"] = [dict(item) for item in previous.get("boxes") or () if slot_of(item or {}) not in new_boxes]
    merged["boxes"] += [item for items in new_boxes.values() for item in items]
    merged["notes"] = " ".join(str(n) for n in (previous.get("notes"), patch.get("notes")) if n)
    return merged, rejected
