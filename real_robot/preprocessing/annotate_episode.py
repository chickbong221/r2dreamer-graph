"""Gemini annotation of whole episodes (or, for deployment, of the past only).

    python -m real_robot.preprocessing.annotate_episode --episodes pilot
    python -m real_robot.preprocessing.annotate_episode --episodes 12 --dry-run

``full_episode`` (retrospective) mode makes two main calls over the whole
episode, both cameras, sampled at the recording rate:

1. ``events`` -- events at the first frame their condition holds, the active
   target, the outcome, and tracking anchors: boxes and every named point,
   each point placed or explicitly marked hidden;
2. ``relations`` -- for every fact id, absolute intervals over every frame and,
   for facts that have them, temporal intervals from frame ``K``, as two
   independent lists. It sees the events, target and outcome, and reports
   disagreements with them rather than labelling around them.

The events pass is validated and repaired on its own before the relations pass
is requested -- its structure, its anchors and the contradictions among its own
events and outcome -- so relations are never asked for against events already
known to be wrong. If the events pass is still invalid after ``repair_rounds``
rounds, relations are not requested at all and the episode is saved invalid.

Every request puts what is shared first: the frozen specification (identical
for every episode), then the two videos (identical for both passes of an
episode), then the episode and pass text -- so implicit prefix caching can
reuse the specification and the videos.

The combined answers are then validated (:mod:`real_robot.graphs.validate`) and
repaired, again for at most ``repair_rounds`` rounds, in order, rebuilding and
re-validating after every correction so that each request sees the current
answers and nothing is patched from stale context:

* ``events`` -- structural problems: the events pass again, whole;
* ``anchors`` -- missing keyframes: only those, merged by slot;
* ``consistency`` -- contradictions between events, target, outcome and the
  grasp/contain/support labels, and disagreements the relations pass
  reported: events, target and outcome together with the facts involved,
  resolved from the video;
* ``relations`` -- label problems: only the listed facts over the listed frames.

Patch entries outside what was requested are rejected, never applied. Repairs
confined to a short stretch send a clip around it instead of the whole
episode. A response cut off at the output limit is asked for again in smaller
pieces, never identically. After ``repair_rounds`` rounds an episode that is
still invalid is saved with its issues, and the dataset build refuses it.

An existing annotation is reused only when it was made from the current inputs
-- graph, frozen bins, prompts and schemas, Gemini settings, prepared videos and
validation rules. Otherwise the episode is annotated again, and every request
that has not changed is answered from the response cache at no cost.

``past_only`` mode asks for the state at every ``update_stride_frames``-th
frame from a clip that ends at that frame, carrying only its own previous
answer forward. It is deferred for the first training version and kept here
unchanged in intent.

Whole-episode graphs are retrospective annotations. Every saved record says
which mode produced it, how many calls and attempts it took, and how many
tokens they used -- failed attempts included, since they are billed too.

A response that cannot be used even after the client's own retries (malformed,
cut off beyond ``truncation_splits``, empty) leaves that episode without an
annotation and the run moves on; a request the service refuses or never answers
(quota, network) stops the run, since every later episode would meet the same.
Either way the tokens spent so far are reported, and everything already
answered stays in the response cache.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from scenegraph.core.relation_rules import ABS_LABELS, CHANGE_LABELS

from ..common import (
    PROMPT_DIR,
    add_config_arguments,
    episode_name,
    load_configs,
    read_json,
    repo_path,
    stable_hash,
    utc_now,
    write_json,
)
from ..graphs.schema import GraphSpec
from ..graphs.validate import (
    ANNOTATION_FORMAT,
    DISAGREEMENT_KINDS,
    EVENT_TYPES,
    STAGES,
    EpisodeAnnotation,
    Issue,
    ValidationSettings,
    assemble_past_only,
    build_annotation,
    fact_id_of,
    fact_ids,
    merge_keyframes,
    relations_to_intervals,
    repair_plan,
    replace_fact_range,
    runs,
)
from .define_bins import (
    _obj,
    cameras_text,
    entities_text,
    facts_text,
    load_frozen_bins,
    load_sections,
    points_text,
    reference_points_text,
    render_bin_spec,
    render_template,
    unobserved_text,
)
from .gemini_client import (
    GeminiClient,
    GeminiError,
    GeminiRequestFailed,
    GeminiTruncated,
    TextPart,
    VideoPart,
    normalize_usage,
    sum_usage,
)
from .prepare_videos import prepare_episode, prepared_video_path

BOX = {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}
POINT = {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}
INTEGER, STRING, BOOLEAN = {"type": "integer"}, {"type": "string"}, {"type": "boolean"}


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
def _entities(spec: GraphSpec) -> Dict[str, Any]:
    return {"type": "string", "enum": list(spec.entity_ids)}


def _point_entries(spec: GraphSpec) -> Dict[str, Any]:
    names = sorted({name for names in spec.points.values() for name in names})
    return {"type": "array", "items": _obj({"name": {"type": "string", "enum": names},
                                            "visible": BOOLEAN, "point": POINT})}


def keyframes_schema(spec: GraphSpec) -> Dict[str, Any]:
    return {"type": "array", "items": _obj({
        "frame": INTEGER,
        "camera": {"type": "string", "enum": list(spec.cameras)},
        "object": _entities(spec), "visible": BOOLEAN,
        "box_2d": BOX, "points": _point_entries(spec)})}


def _events_properties(spec: GraphSpec) -> Dict[str, Any]:
    return {
        "events": {"type": "array", "items": _obj({
            "type": {"type": "string", "enum": list(EVENT_TYPES)},
            "object": {"type": "string", "enum": list(spec.object_ids)},
            "frame": INTEGER, "evidence": STRING})},
        "active_target": {"type": "array", "items": _obj({
            "object": {"type": "string", "enum": list(spec.targets)},
            "start_frame": INTEGER, "end_frame": INTEGER})},
        "outcome": _obj({"success": BOOLEAN, "banana_in_pot_at_end": BOOLEAN,
                         "lid_closed_at_end": BOOLEAN, "completion_frame": INTEGER,
                         "failure_reason": STRING}),
    }


def events_schema(spec: GraphSpec) -> Dict[str, Any]:
    return _obj({**_events_properties(spec), "keyframes": keyframes_schema(spec)})


def anchors_schema(spec: GraphSpec) -> Dict[str, Any]:
    return _obj({"keyframes": keyframes_schema(spec)})


def _intervals(labels: Sequence[str]) -> Dict[str, Any]:
    return {"type": "array", "items": _obj({"start": INTEGER, "end": INTEGER,
                                            "label": {"type": "string", "enum": list(labels)}})}


def _facts_property(spec: GraphSpec, ids: Sequence[str]) -> Dict[str, Any]:
    index = dict(zip(fact_ids(spec), spec.facts))
    absolute = sorted({label for fid in ids for label in ABS_LABELS[index[fid].relation]})
    return {"type": "array", "items": _obj({"fact": {"type": "string", "enum": list(ids)},
                                            "absolute": _intervals(absolute),
                                            "temporal": _intervals(CHANGE_LABELS)})}


def relations_schema(spec: GraphSpec, ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    ids = list(ids) if ids is not None else fact_ids(spec)
    return _obj({
        "facts": _facts_property(spec, ids),
        "pass1_disagreements": {"type": "array", "items": _obj({
            "kind": {"type": "string", "enum": list(DISAGREEMENT_KINDS)},
            "start_frame": INTEGER, "end_frame": INTEGER, "description": STRING})},
    })


def reconcile_schema(spec: GraphSpec, ids: Sequence[str]) -> Dict[str, Any]:
    properties = dict(_events_properties(spec))
    if ids:
        properties["facts"] = _facts_property(spec, ids)
    return _obj(properties)


def past_only_schema(spec: GraphSpec) -> Dict[str, Any]:
    relations = list(spec.relations_in_use)
    labels = sorted({label for r in relations for label in ABS_LABELS[r]})
    return _obj({
        "frame": INTEGER,
        "active_target": {"type": "string", "enum": list(spec.targets)},
        "facts": {"type": "array", "items": _obj({
            "relation": {"type": "string", "enum": relations},
            "src": _entities(spec), "dst": _entities(spec),
            "label": {"type": "string", "enum": labels},
            "temporal_label": {"type": "string", "enum": list(CHANGE_LABELS) + ["none"]}})},
        "objects": {"type": "array", "items": _obj({
            "camera": {"type": "string", "enum": list(spec.cameras)},
            "object": _entities(spec), "visible": BOOLEAN,
            "box_2d": BOX, "points": _point_entries(spec)})},
        "events_so_far": {"type": "array", "items": _obj({
            "type": {"type": "string", "enum": list(EVENT_TYPES)},
            "object": {"type": "string", "enum": list(spec.object_ids)},
            "frame": INTEGER})},
        "task_complete": BOOLEAN,
        "banana_in_pot": BOOLEAN,
        "lid_closed": BOOLEAN,
    })


# --------------------------------------------------------------------------- #
# Context rendering
# --------------------------------------------------------------------------- #
def events_context(events_raw: Mapping[str, Any]) -> str:
    """Events, active target and outcome. Keyframes stay out of relation prompts."""
    return "```json\n" + json.dumps({k: events_raw.get(k) for k in ("events", "active_target", "outcome")},
                                     indent=1) + "\n```"


def labels_context(spec: GraphSpec, annotation: EpisodeAnnotation, ids: Sequence[str], lo: int, hi: int) -> str:
    """Current labels of some facts over frames ``lo..hi``, as runs."""
    index = dict(zip(fact_ids(spec), range(len(spec.facts))))
    lines = []
    for fid in ids:
        position = index[fid]
        fact = spec.facts[position]

        def text(series: Sequence[Optional[str]]) -> str:
            parts = [f"{a + lo}-{b + lo} {v if v is not None else '(none)'}" for a, b, v in runs(list(series[lo:hi + 1]))]
            return "; ".join(parts) if parts else "(none)"

        line = f"- `{fid}` {fact.relation}({fact.src}, {fact.dst}): absolute {text(annotation.absolute[position])}"
        if fact.temporal:
            line += f" | temporal {text(annotation.temporal[position])}"
        lines.append(line)
    return "\n".join(lines) if lines else "(no facts)"


def issues_text(issues: Sequence[Mapping[str, Any]], limit: int = 60) -> str:
    lines = [f"- {issue['message']}" for issue in issues[:limit]]
    if len(issues) > limit:
        lines.append(f"- ... and {len(issues) - limit} more")
    return "\n".join(lines)


def merge_anchor_requests(anchors: Sequence[Mapping[str, Any]], n_frames: int, event_tolerance: int
                          ) -> List[Dict[str, Any]]:
    """Requested anchors grouped by entity and camera, with overlapping frame ranges merged."""
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for anchor in anchors:
        obj, camera = anchor.get("object"), anchor.get("camera")
        if not obj or not camera:
            continue
        entry = grouped.setdefault((obj, camera), {"ranges": [], "rules": set(), "points": set()})
        pad = event_tolerance if anchor.get("rule") == "event" else 0
        for a, b in anchor.get("frames") or ():
            entry["ranges"].append((max(0, int(a) - pad), min(n_frames - 1, int(b) + pad)))
        entry["rules"].add(str(anchor.get("rule", "")))
        entry["points"].update(anchor.get("points") or ())
    out = []
    for (obj, camera), entry in sorted(grouped.items()):
        merged: List[List[int]] = []
        for a, b in sorted(entry["ranges"]):
            if merged and a <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        out.append({"object": obj, "camera": camera, "frames": merged, "rules": sorted(entry["rules"]),
                    "points": sorted(entry["points"])})
    return out


def anchor_requests_text(requests: Sequence[Mapping[str, Any]]) -> str:
    notes = {"initial": "including frame 0", "points": "listing every named point, visible or hidden",
             "event": "at the event frames", "gap": "spaced as the rule requires",
             "table_plane": "visible, with at least three table surface points placed far apart and not in a line",
             "well-formed": "with a valid box and points"}
    lines = []
    for request in requests:
        frames = "; ".join(str(a) if a == b else f"{a}-{b}" for a, b in request["frames"])
        why = ", ".join(notes[r] for r in request["rules"] if r in notes)
        lines.append(f"- `{request['object']}` in `{request['camera']}`: frames {frames}" + (f" ({why})" if why else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Annotator
# --------------------------------------------------------------------------- #
class EpisodeAnnotator:
    def __init__(self, configs: Mapping[str, Mapping[str, Any]], mode: Optional[str] = None,
                 client: Optional[GeminiClient] = None, source=None, bins: Optional[Mapping[str, Any]] = None):
        from ..data.episode_dataset import RawEpisodeSource

        self.configs = configs
        self.annotation_cfg = configs["annotation"]
        self.mode = mode or self.annotation_cfg["annotation"]["mode"]
        self.source = source if source is not None else RawEpisodeSource(configs, mode=self.mode)
        self.spec = GraphSpec.from_config(configs["graph"])
        self.bins = bins if bins is not None else load_frozen_bins(configs["dataset"], self.spec)
        self.sections = load_sections(os.path.join(PROMPT_DIR, "episode_annotation.md"))
        self.client = client or GeminiClient(self.annotation_cfg["gemini"])
        self.settings = ValidationSettings.from_config(self.annotation_cfg, self.mode)
        self.requests: List[Dict[str, Any]] = []
        self._spec_text: Optional[str] = None

    @property
    def section_cfg(self) -> Mapping[str, Any]:
        return self.annotation_cfg["annotation"]

    # --------------------------------------------------------- identity
    def prompt_version(self) -> str:
        with open(os.path.join(PROMPT_DIR, "episode_annotation.md"), "r", encoding="utf-8") as handle:
            template = handle.read()
        ids = fact_ids(self.spec)
        schemas = [events_schema(self.spec), anchors_schema(self.spec), relations_schema(self.spec),
                   reconcile_schema(self.spec, ids), past_only_schema(self.spec)]
        return stable_hash([template, self.spec_text(), schemas])

    def input_identity(self, episode: int, prepared: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Everything an episode's annotation depends on. A change to any of it re-annotates the episode.

        ``prepared`` is a current prepared-video index to read the copies' digests from; without it the
        copies are made first, or confirmed current.
        """
        if prepared is None:
            prepared = prepare_episode(self.source, episode, self.annotation_cfg["videos"])
        return {
            "format": ANNOTATION_FORMAT,
            "mode": self.mode,
            "graph": stable_hash(self.spec.identity()),
            "bins": self.bins["bins_hash"],
            "prompt_version": self.prompt_version(),
            "gemini": self.client.settings(),
            "video_fps": float(self.annotation_cfg["gemini"]["video_fps"]),
            "videos": {camera: prepared["cameras"][camera]["sha256"] for camera in self.spec.cameras},
            "n_frames": int(self.source.lengths()[episode]),
            "validation": self.settings.identity(),
            "repairs": dict(self.section_cfg["repairs"]),
        }

    def provenance(self) -> Dict[str, Any]:
        return {
            "backend": self.client.backend,
            "model": self.client.model,
            "gemini_settings": self.client.settings(),
            "prompt_version": self.prompt_version(),
            "bins_hash": self.bins["bins_hash"],
            "video_fps": float(self.annotation_cfg["gemini"]["video_fps"]),
            "mode": self.mode,
            "repair_rounds": int(self.section_cfg["repair_rounds"]),
            "created": utc_now(),
            "requests": list(self.requests),
            "usage": usage_totals(self.requests),
        }

    # ----------------------------------------------------------- prompts
    def spec_text(self) -> str:
        """The part of every prompt that is identical for every episode and every call."""
        if self._spec_text is None:
            fps = self.source.fps()
            K = self.spec.temporal_window
            self._spec_text = render_template(self.sections["SPEC"], {
                "FPS": f"{fps:g}", "CAMERAS": cameras_text(self.spec), "ENTITIES": entities_text(self.spec),
                "REFERENCE_POINTS": reference_points_text(self.spec), "FACTS": facts_text(self.spec),
                "UNOBSERVED": unobserved_text(self.spec), "K": K, "K_MINUS_ONE": K - 1,
                "K_SECONDS": f"{K / fps:.2f}", "POINTS": points_text(self.spec),
                "BIN_SPEC": render_bin_spec(self.bins["bins"]),
            })
        return self._spec_text

    def episode_text(self, episode: int, start: Optional[int] = None, end: Optional[int] = None) -> str:
        n = self.source.lengths()[episode]
        return render_template(self.sections["EPISODE"], {
            "EPISODE": episode, "N_FRAMES": n, "LAST_FRAME": n - 1,
            "VIDEO_START": 0 if start is None else start, "VIDEO_END": n - 1 if end is None else end,
            "TASK": " ".join(self.configs["dataset"]["source"]["task"].split()),
        })

    def section(self, name: str, episode: int, **values: Any) -> str:
        n = self.source.lengths()[episode]
        temporal = [fid for fid, fact in zip(fact_ids(self.spec), self.spec.facts) if fact.temporal]
        base = {"LAST_FRAME": n - 1, "KEYFRAME_EVERY": self.settings.keyframe_every, "K": self.spec.temporal_window,
                "FACT_SCOPE": "every fact id in the specification",
                "TEMPORAL_FACTS": ", ".join(f"`{fid}`" for fid in temporal)}
        return render_template(self.sections[name], {**base, **values})

    def video_parts(self, episode: int, start: Optional[int] = None, end: Optional[int] = None) -> List[Any]:
        prepare_episode(self.source, episode, self.annotation_cfg["videos"])
        fps = self.source.fps()
        parts: List[Any] = []
        for camera in self.spec.cameras:
            label = f"Camera `{camera}`" + (f", frames {start} to {end}" if start is not None else "") + ":"
            parts.append(TextPart(label))
            parts.append(VideoPart(prepared_video_path(self.configs["dataset"], episode, camera),
                                   fps=float(self.annotation_cfg["gemini"]["video_fps"]), source_fps=fps,
                                   start_frame=start, end_frame=end))
        return parts

    def _call(self, episode: int, name: str, tail: str, schema: Mapping[str, Any],
              start: Optional[int] = None, end: Optional[int] = None) -> Any:
        parts = ([TextPart(self.spec_text())] + self.video_parts(episode, start, end)
                 + [TextPart(self.episode_text(episode, start, end) + "\n\n" + tail)])
        label = f"{episode_name(episode)}/{self.mode}/{name}"
        clip = None if start is None else [start, end]
        try:
            parsed, record = self.client.generate_json(parts, schema, label)
        except GeminiError as exc:
            # A failed call is billed for every response it received; the exception carries them.
            attempts = list(getattr(exc, "attempts", None) or ())
            self.requests.append({"label": label, "failed": type(exc).__name__,
                                  "truncated": isinstance(exc, GeminiTruncated), "clip": clip,
                                  "attempts": len(attempts),
                                  "usage": sum_usage(a.get("usage_normalized") for a in attempts)})
            raise
        usage = (record.get("usage_total") or record.get("usage_normalized")
                 or normalize_usage(record.get("usage")))
        self.requests.append({"label": label, "key": record.get("key"), "cached": bool(record.get("cached")),
                              "usage": usage, "clip": clip, "attempts": len(record.get("attempts") or ()) or 1})
        self._save_raw(episode, name, parsed)
        return parsed

    def _save_raw(self, episode: int, name: str, parsed: Any) -> None:
        directory = os.path.join(repo_path(self.configs["dataset"]["paths"]["annotations"]), self.mode, "raw",
                                 episode_name(episode))
        write_json(os.path.join(directory, f"{name.replace('/', '_')}.json"), parsed)

    def repair_window(self, frames: Sequence[Sequence[int]], whole_episode: bool, n: int
                      ) -> Tuple[int, int, Optional[int], Optional[int]]:
        """``(lo, hi, clip_start, clip_end)``: the frames to replace and the video to send."""
        repairs = self.section_cfg["repairs"]
        if whole_episode or not frames:
            return 0, n - 1, None, None
        margin = int(repairs["clip_margin_frames"])
        lo = max(0, min(int(a) for a, _ in frames) - margin)
        hi = min(n - 1, max(int(b) for _, b in frames) + margin)
        if hi - lo + 1 > float(repairs["full_video_fraction"]) * n:
            return 0, n - 1, None, None
        # Temporal labels at lo compare with lo - K, so the clip starts K frames earlier.
        return lo, hi, max(0, lo - self.spec.temporal_window), hi

    # ------------------------------------------------------------ passes
    def ask_events(self, episode: int, name: str, tail: str) -> Dict[str, Any]:
        """The events pass. If the answer is too long, events and anchors are asked for separately."""
        try:
            return dict(self._call(episode, name, tail, events_schema(self.spec)))
        except GeminiTruncated:
            pass
        n = self.source.lengths()[episode]
        core = dict(self._call(
            episode, f"{name}_core",
            tail + "\n\nIn this answer report only `events`, `active_target` and `outcome`; the keyframes are "
                   "requested separately, one camera at a time.", reconcile_schema(self.spec, [])))
        keyframes: List[Any] = []
        for camera in self.spec.cameras:
            requests = [{"object": e.id, "camera": camera, "frames": [[0, n - 1]], "rules": ["gap"], "points": []}
                        for e in self.spec.entities]
            keyframes += self.ask_anchors(episode, f"{name}_anchors_{camera}", tail, requests,
                                          "The keyframes did not fit in one answer, so they are requested one "
                                          "camera at a time.", None, None)
        core["keyframes"] = keyframes
        return core

    def ask_anchors(self, episode: int, name: str, tail: str, requests: Sequence[Mapping[str, Any]], issues: str,
                    start: Optional[int], end: Optional[int], depth: int = 0) -> List[Any]:
        text = tail + "\n\n" + self.section("REPAIR anchors", episode, ISSUES=issues,
                                            ANCHOR_REQUESTS=anchor_requests_text(requests))
        try:
            parsed = self._call(episode, name, text, anchors_schema(self.spec), start, end)
            return list(parsed.get("keyframes") or [])
        except GeminiTruncated:
            if len(requests) <= 1 or depth >= int(self.section_cfg["truncation_splits"]):
                raise
            half = len(requests) // 2
            return (self.ask_anchors(episode, f"{name}_a", tail, requests[:half], issues, start, end, depth + 1)
                    + self.ask_anchors(episode, f"{name}_b", tail, requests[half:], issues, start, end, depth + 1))

    def ask_facts(self, episode: int, name: str, make_tail: Callable[[Sequence[str]], str], ids: Sequence[str],
                  start: Optional[int], end: Optional[int], depth: int = 0) -> List[Mapping[str, Any]]:
        """Relations answers for ``ids``, halving the set whenever an answer is cut off."""
        try:
            return [self._call(episode, name, make_tail(ids), relations_schema(self.spec, ids), start, end)]
        except GeminiTruncated:
            if len(ids) <= 1 or depth >= int(self.section_cfg["truncation_splits"]):
                raise
            half = len(ids) // 2
            return (self.ask_facts(episode, f"{name}_a", make_tail, ids[:half], start, end, depth + 1)
                    + self.ask_facts(episode, f"{name}_b", make_tail, ids[half:], start, end, depth + 1))

    # ---------------------------------------------------------- state
    def _build(self, episode: int, state: Mapping[str, Any], events_only: bool = False) -> EpisodeAnnotation:
        return build_annotation(
            self.spec, episode_index=episode, n_frames=self.source.lengths()[episode], fps=self.source.fps(),
            mode=self.mode, events_raw=state["events"],
            absolute_raw={"intervals": state["absolute"]}, temporal_raw={"intervals": state["temporal"]},
            provenance=self.provenance(), settings=self.settings, extra_disagreements=state["disagreements"],
            spec_identity={"graph": stable_hash(self.spec.identity()), "bins": self.bins["bins_hash"]},
            events_only=events_only,
        )

    @staticmethod
    def _collect(parsed_list: Sequence[Mapping[str, Any]], spec: GraphSpec
                 ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], int]:
        absolute, temporal, disagreements, unknown = [], [], [], 0
        for parsed in parsed_list:
            a, t, issues, found = relations_to_intervals(spec, parsed)
            absolute += a
            temporal += t
            disagreements += found
            unknown += len(issues)
        return absolute, temporal, disagreements, unknown

    # ------------------------------------------------------------ full episode
    def annotate_full(self, episode: int) -> EpisodeAnnotation:
        self.requests = []
        spec = self.spec
        log: List[Dict[str, Any]] = []
        rounds = int(self.section_cfg["repair_rounds"])

        state: Dict[str, Any] = {"events": self.ask_events(episode, "events", self.section("PASS events", episode)),
                                 "absolute": [], "temporal": [], "disagreements": []}
        # The relations pass is shown the events, target and outcome, so they are
        # validated and repaired first: never ask for labels against known-wrong events.
        annotation, last_round = self._repair(episode, state, self._build(episode, state, events_only=True), log,
                                              rounds, first_round=0, events_only=True)
        if not annotation.valid:
            annotation.issues.append(Issue(
                "relations_not_requested",
                f"relations: not requested, because the events pass is still invalid after {rounds} repair "
                "round(s)", "relations"))
            annotation.repair_log = log
            annotation.provenance = self.provenance()
            return annotation

        all_ids = fact_ids(spec)

        def relations_tail(ids: Sequence[str]) -> str:
            scope = ("every fact id in the specification" if list(ids) == all_ids
                     else "these fact ids only: " + ", ".join(f"`{fid}`" for fid in ids))
            temporal = [fid for fid in ids if spec.facts[int(fid[1:])].temporal]
            return self.section("PASS relations", episode, CONTEXT=events_context(state["events"]),
                                FACT_SCOPE=scope,
                                TEMPORAL_FACTS=", ".join(f"`{fid}`" for fid in temporal) or "none of them")

        absolute, temporal, disagreements, unknown = self._collect(
            self.ask_facts(episode, "relations", relations_tail, all_ids, None, None), spec)
        if unknown:
            log.append({"round": last_round, "phase": "relations", "stage": "relations", "ignored_entries": unknown})
        state.update({"absolute": absolute, "temporal": temporal, "disagreements": disagreements})
        annotation, _ = self._repair(episode, state, self._build(episode, state), log, rounds,
                                     first_round=last_round)
        annotation.repair_log = log
        annotation.provenance = self.provenance()
        return annotation

    def _repair(self, episode: int, state: Dict[str, Any], annotation: EpisodeAnnotation, log: List[Dict[str, Any]],
                rounds: int, first_round: int, events_only: bool = False) -> Tuple[EpisodeAnnotation, int]:
        """At most ``rounds`` repair rounds in stage order, rebuilding after every request.

        Returns the annotation and the number of the last round used; rounds are
        numbered across both phases, so every request keeps a distinct name.
        """
        round_index = first_round
        for _ in range(rounds):
            if annotation.valid:
                break
            round_index += 1
            for stage in STAGES:
                plan = repair_plan(annotation)
                if stage not in plan:
                    continue
                entry = plan[stage]
                name = f"{stage}_repair{round_index}"
                before = len(annotation.issues)
                if stage == "events" or (stage == "anchors" and not entry["anchors"]):
                    note = self._repair_events(episode, name, state, entry)
                elif stage == "anchors":
                    note = self._repair_anchors(episode, name, state, entry)
                elif stage == "consistency":
                    note = self._reconcile(episode, name, state, entry, annotation)
                else:
                    note = self._repair_relations(episode, name, state, entry, annotation)
                annotation = self._build(episode, state, events_only=events_only)
                log.append({"round": round_index, "phase": "events" if events_only else "combined", "stage": stage,
                            "issues_before": before, "issues_after": len(annotation.issues), **note})
                if annotation.valid:
                    break
        return annotation, round_index

    def _repair_events(self, episode: int, name: str, state: Dict[str, Any], entry: Mapping[str, Any]
                       ) -> Dict[str, Any]:
        tail = self.section("PASS events", episode) + "\n\n" + self.section(
            "REPAIR events", episode, ISSUES=issues_text(entry["issues"]),
            PREVIOUS="```json\n" + json.dumps(state["events"], indent=1) + "\n```")
        state["events"] = self.ask_events(episode, name, tail)
        return {"asked": "events pass, whole"}

    def _repair_anchors(self, episode: int, name: str, state: Dict[str, Any], entry: Mapping[str, Any]
                        ) -> Dict[str, Any]:
        n = self.source.lengths()[episode]
        requests = merge_anchor_requests(entry["anchors"], n, self.settings.event_anchor_tolerance)
        frames = [tuple(r) for request in requests for r in request["frames"]]
        _, _, start, end = self.repair_window(frames, False, n)
        keyframes = self.ask_anchors(episode, name, self.section("PASS events", episode), requests,
                                     issues_text(entry["issues"]), start, end)
        merged, rejected = merge_keyframes(state["events"].get("keyframes") or [], keyframes, requests, self.spec)
        state["events"] = {**state["events"], "keyframes": merged}
        return {"asked": f"{len(requests)} anchor slot group(s)", "clip": [start, end] if start is not None else None,
                "accepted": len(keyframes) - len(rejected), "rejected_out_of_scope": len(rejected)}

    def _reconcile(self, episode: int, name: str, state: Dict[str, Any], entry: Mapping[str, Any],
                   annotation: EpisodeAnnotation) -> Dict[str, Any]:
        spec = self.spec
        n = self.source.lengths()[episode]
        facts = [tuple(f) for f in entry["facts"]]
        ids = [fact_id_of(spec, f) for f in facts]
        # Events, target and outcome are answered for the whole episode, so the whole video is sent.
        lo, hi, _, _ = self.repair_window(entry["frames"], entry["whole_episode"], n)
        previous = (events_context(state["events"]) + "\n\nCurrent labels:\n"
                    + labels_context(spec, annotation, ids, lo, hi))
        tail = self.section("PASS events", episode) + "\n\n" + self.section(
            "RECONCILE", episode, ISSUES=issues_text(entry["issues"]),
            FACT_LIST=", ".join(f"`{fid}`" for fid in ids) if ids else "no facts (return no `facts`)",
            RANGE_START=lo, RANGE_END=hi, TEMPORAL_START=max(spec.temporal_window, lo), PREVIOUS=previous)
        parsed = self._call(episode, name, tail, reconcile_schema(spec, ids))
        state["events"] = {**state["events"], **{key: parsed[key] for key in ("events", "active_target", "outcome")
                                                 if key in parsed}}
        rejected = 0
        if ids:
            absolute, temporal, _, _ = self._collect([{"facts": parsed.get("facts") or []}], spec)
            state["absolute"], rejected_a = replace_fact_range(state["absolute"], absolute, spec, facts, (lo, hi))
            state["temporal"], rejected_t = replace_fact_range(state["temporal"], temporal, spec, facts, (lo, hi))
            rejected = len(rejected_a) + len(rejected_t)
        # Every reported disagreement was part of this request and has been decided.
        state["disagreements"] = []
        return {"asked": f"events, target, outcome and {len(ids)} fact(s) over frames {lo}-{hi}",
                "rejected_out_of_scope": rejected}

    def _repair_relations(self, episode: int, name: str, state: Dict[str, Any], entry: Mapping[str, Any],
                          annotation: EpisodeAnnotation) -> Dict[str, Any]:
        spec = self.spec
        n = self.source.lengths()[episode]
        facts = [tuple(f) for f in entry["facts"]] or [tuple(f.key) for f in spec.facts]
        ids = [fact_id_of(spec, f) for f in facts]
        whole = entry["whole_episode"] or not entry["facts"]
        lo, hi, start, end = self.repair_window(entry["frames"], whole, n)

        def tail(subset: Sequence[str]) -> str:
            return self.section(
                "REPAIR relations", episode, CONTEXT=events_context(state["events"]),
                ISSUES=issues_text(entry["issues"]), FACT_LIST=", ".join(f"`{fid}`" for fid in subset),
                RANGE_START=lo, RANGE_END=hi, TEMPORAL_START=max(spec.temporal_window, lo),
                PREVIOUS=labels_context(spec, annotation, subset, lo, hi))

        absolute, temporal, disagreements, unknown = self._collect(
            self.ask_facts(episode, name, tail, ids, start, end), spec)
        state["absolute"], rejected_a = replace_fact_range(state["absolute"], absolute, spec, facts, (lo, hi))
        state["temporal"], rejected_t = replace_fact_range(state["temporal"], temporal, spec, facts, (lo, hi))
        state["disagreements"] = list(state["disagreements"]) + disagreements
        return {"asked": f"{len(ids)} fact(s) over frames {lo}-{hi}", "clip": [start, end] if start is not None else None,
                "rejected_out_of_scope": len(rejected_a) + len(rejected_t) + unknown}

    # --------------------------------------------------------- past only
    def annotate_past_only(self, episode: int) -> EpisodeAnnotation:
        self.requests = []
        cfg = self.section_cfg["past_only"]
        stride, window = int(cfg["update_stride_frames"]), int(cfg["window_frames"])
        n = self.source.lengths()[episode]
        K = self.spec.temporal_window
        schema = past_only_schema(self.spec)
        updates: Dict[int, Mapping[str, Any]] = {}
        frames = list(range(0, n, stride))

        def ask(u: int, previous_frame: int, previous: Any, repair: Optional[Mapping[str, Any]] = None) -> Any:
            text = self.section("PASS past_only", episode, FRAME=u, FRAME_MINUS_K=max(u - K, 0),
                                PREVIOUS_FRAME=previous_frame if previous is not None else "none",
                                PREVIOUS=("```json\n" + json.dumps(previous, indent=1) + "\n```")
                                if previous is not None else "(this is the first update)")
            if repair is not None:
                text += "\n\n" + self.section(
                    "REPAIR events", episode, ISSUES=issues_text(repair["issues"]),
                    PREVIOUS="```json\n" + json.dumps(updates.get(u), indent=1) + "\n```")
            answer = dict(self._call(episode, f"update_{u:04d}" + ("_repair" if repair else ""), text, schema,
                                     start=max(0, u - window + 1), end=u))
            answer["frame"] = u    # the clip ends at u; an answer about any other frame is not accepted
            return answer

        previous, previous_frame = None, -1
        for u in frames:
            updates[u] = ask(u, previous_frame, previous)
            previous, previous_frame = updates[u], u
        annotation = self._assemble(episode, updates)

        for round_index in range(1, int(self.section_cfg["repair_rounds"]) + 1):
            if annotation.valid:
                break
            implicated = self._implicated_updates(annotation, frames)
            print(f"[annotate] episode {episode}: {len(annotation.issues)} issue(s); re-asking "
                  f"{len(implicated)} update(s), round {round_index}", flush=True)
            for u in implicated:
                position = frames.index(u)
                before = frames[position - 1] if position else -1
                # The frames this update's answer is held over.
                last = frames[position + 1] - 1 if position + 1 < len(frames) else n - 1
                messages = [i.to_json() for i in annotation.issues
                            if not i.frames or any(a <= last and u <= b for a, b in i.frames)]
                updates[u] = ask(u, before, updates.get(before), repair={
                    "issues": messages or [i.to_json() for i in annotation.issues]})
            annotation = self._assemble(episode, updates)
        annotation.provenance = self.provenance()
        return annotation

    def _assemble(self, episode: int, updates: Mapping[int, Mapping[str, Any]]) -> EpisodeAnnotation:
        events, absolute, temporal = assemble_past_only(self.spec, list(updates.values()),
                                                        self.source.lengths()[episode])
        return build_annotation(
            self.spec, episode_index=episode, n_frames=self.source.lengths()[episode], fps=self.source.fps(),
            mode=self.mode, events_raw=events, absolute_raw=absolute, temporal_raw=temporal,
            provenance=self.provenance(), settings=self.settings,
            spec_identity={"graph": stable_hash(self.spec.identity()), "bins": self.bins["bins_hash"]})

    @staticmethod
    def _implicated_updates(annotation: EpisodeAnnotation, frames: Sequence[int]) -> List[int]:
        chosen = set()
        for issue in annotation.issues:
            if not issue.frames:
                chosen.update(frames)
                continue
            for a, b in issue.frames:
                for position, u in enumerate(frames):
                    end = frames[position + 1] - 1 if position + 1 < len(frames) else annotation.n_frames - 1
                    if u <= b and a <= end:
                        chosen.add(u)
        return sorted(chosen)

    # -------------------------------------------------------------- io
    def save(self, annotation: EpisodeAnnotation, identity: Mapping[str, Any]) -> str:
        annotation.input_identity = dict(identity)
        path = self.source.annotation_path(annotation.episode_index)
        write_json(path, annotation.to_json(self.spec))
        return path


def usage_totals(requests: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Calls, attempts and tokens.

    Tokens count every attempt that reached the API in this run, failed ones
    included -- a malformed or cut-off response is billed like a usable one.
    Answers read from the response cache cost nothing and add no tokens.
    """
    totals: Dict[str, Any] = collections.Counter()
    for request in requests:
        totals["calls"] += 1
        if request.get("cached"):
            totals["cached_calls"] += 1
            continue
        failed = bool(request.get("failed"))
        if request.get("truncated"):
            totals["truncated_calls"] += 1
        elif failed:
            totals["failed_calls"] += 1
        if request.get("clip"):
            totals["clip_calls"] += 1
        attempts = int(request.get("attempts") or (0 if failed else 1))
        totals["attempts"] += attempts
        totals["failed_attempts"] += attempts if failed else max(attempts - 1, 0)
        for key, value in (request.get("usage") or {}).items():
            if isinstance(value, int):
                totals[key] += value
    return dict(totals)


def usage_line(usage: Mapping[str, Any]) -> str:
    return (f"{usage.get('calls', 0)} call(s) ({usage.get('cached_calls', 0)} cached, {usage.get('attempts', 0)} "
            f"attempt(s) sent, {usage.get('failed_attempts', 0)} failed), input {usage.get('input_tokens', 0)} / "
            f"cached {usage.get('cached_tokens', 0)} / output {usage.get('output_tokens', 0)} / thinking "
            f"{usage.get('thought_tokens', 0)} tokens")


def reuse_decision(path: str, identity: Mapping[str, Any], repair_rounds: int) -> Tuple[bool, str]:
    """Whether a saved annotation can stand, and why."""
    from .artifacts import mismatched_fields

    if not os.path.isfile(path):
        return False, "no annotation yet"
    stored = read_json(path)
    if stored.get("format") != ANNOTATION_FORMAT:
        return False, "written by an earlier version of this package"
    if stored.get("input_hash") != stable_hash(dict(identity)):
        fields = mismatched_fields(dict(identity), stored.get("input_identity") or {})
        return False, "inputs changed: " + (", ".join(fields) or "unknown")
    if stored.get("status") == "valid":
        return True, "valid and made from the current inputs"
    if int((stored.get("provenance") or {}).get("repair_rounds", 0)) >= int(repair_rounds):
        return True, ("invalid; asking again with the same inputs and repair rounds would replay the same cached "
                      "answers (change an input or raise annotation.repair_rounds)")
    return False, "invalid, and more repair rounds are configured now"


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Annotate episodes with Gemini.")
    parser.add_argument("--episodes", default="pilot")
    parser.add_argument("--mode", choices=("full_episode", "past_only"), default=None)
    parser.add_argument("--force", action="store_true",
                        help="annotate again even when a current annotation exists (unchanged requests stay cached)")
    parser.add_argument("--dry-run", action="store_true", help="print the first pass prompt and call nothing")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    configs = load_configs(["dataset", "annotation", "graph"], args.overrides)
    annotator = EpisodeAnnotator(configs, mode=args.mode)
    episodes = annotator.source.select(args.episodes)

    if args.dry_run:
        episode = episodes[0]
        if annotator.mode == "full_episode":
            tail = annotator.section("PASS events", episode)
        else:
            tail = annotator.section("PASS past_only", episode, FRAME=0, FRAME_MINUS_K=0, PREVIOUS_FRAME="none",
                                     PREVIOUS="(this is the first update)")
        print(annotator.spec_text() + "\n\n[videos: " + ", ".join(annotator.spec.cameras) + "]\n\n"
              + annotator.episode_text(episode) + "\n\n" + tail)
        return

    rounds = int(annotator.section_cfg["repair_rounds"])
    summary: List[Tuple[int, str]] = []
    run_requests: List[Mapping[str, Any]] = []

    def report_run() -> None:
        usable = [e for e, s in summary if s == "valid"]
        totals = usage_totals(run_requests)
        if run_requests:
            per = max(len(usable), 1)
            print(f"[annotate] this run: {usage_line(totals)}; {len(usable)} usable episode(s) -> "
                  f"{totals.get('attempts', 0) / per:.1f} attempts and {totals.get('input_tokens', 0) / per:.0f} "
                  "input tokens per usable episode (repairs and failed attempts included)")

    for episode in episodes:
        path = annotator.source.annotation_path(episode)
        identity = annotator.input_identity(episode)
        if not args.force:
            reuse, reason = reuse_decision(path, identity, rounds)
            if reuse:
                status = read_json(path).get("status")
                summary.append((episode, "valid" if status == "valid" else "INVALID (kept)"))
                print(f"[annotate] episode {episode}: kept ({reason})")
                continue
            if os.path.isfile(path):
                print(f"[annotate] episode {episode}: annotating again ({reason})", flush=True)
        try:
            annotation = (annotator.annotate_full(episode) if annotator.mode == "full_episode"
                          else annotator.annotate_past_only(episode))
        except GeminiRequestFailed as exc:
            run_requests += annotator.requests
            print(f"[annotate] episode {episode}: {exc}", flush=True)
            report_run()
            raise SystemExit("[annotate] stopped: the service refused or did not answer (quota, network or a "
                             "rejected request). Every answer so far is cached; run the same command again later.")
        except GeminiError as exc:
            run_requests += annotator.requests
            summary.append((episode, f"FAILED ({type(exc).__name__})"))
            print(f"[annotate] episode {episode}: no annotation saved, {usage_line(usage_totals(annotator.requests))}"
                  f": {exc}", flush=True)
            continue
        saved = annotator.save(annotation, identity)
        run_requests += annotator.requests
        status = "valid" if annotation.valid else f"INVALID ({len(annotation.issues)} issues)"
        summary.append((episode, status))
        print(f"[annotate] episode {episode}: {status}, {usage_line(usage_totals(annotator.requests))} -> {saved}",
              flush=True)
        for issue in annotation.issues[:10]:
            print(f"    {issue.message}")

    report_run()
    unusable = [f"{e} {s}" for e, s in summary if s != "valid"]
    if unusable:
        raise SystemExit(f"[annotate] episodes without a valid annotation: {unusable}")


if __name__ == "__main__":
    main()
