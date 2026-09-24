"""Label each episode's scene graph with Gemini.

    python -m real_robot.preprocessing.annotate_episode --episodes pilot

One request per episode shows both cameras over the whole episode and asks for
the active target, every fact's label intervals and box keyframes. An answer
cut off at the output limit is asked for again in smaller pieces. Problems
found by validation are sent back for the parts they concern, at most
``repair_rounds`` times; the episode is saved either way, with its issues, and
packing refuses an invalid one. Every response is cached on disk, so a rerun
repeats no call that was already answered.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from scenegraph.core.relation_rules import AFFORDANCE_RELATIONS, CHANGE_LABELS, SPATIAL_RELATIONS

from ..common import (
    PROMPT_DIR,
    add_config_arguments,
    episode_name,
    file_sha256,
    load_configs,
    read_json,
    repo_path,
    stable_hash,
    utc_now,
    write_json,
)
from ..graphs.schema import EE_OBJECT, GraphSpec
from ..graphs.validate import (
    ANNOTATION_FORMAT,
    EpisodeAnnotation,
    Scope,
    ValidationSettings,
    build_annotation,
    combine,
    fact_ids,
    full_scope,
    merge_answer,
    normalize_entity,
    repair_scope,
)
from .gemini_client import GeminiClient, GeminiError, GeminiTruncated, TextPart, VideoPart, sum_usage
from .prepare_videos import prepared_status, stride_for

PROMPT_PATH = os.path.join(PROMPT_DIR, "scene_graph.md")
PHYSICAL = ("contact", "grasp", "support", "contain")


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
def load_sections(path: str = PROMPT_PATH) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    parts = re.split(r"^<!--\s*(.+?)\s*-->\s*$", text, flags=re.M)
    return {name.strip(): body.strip() for name, body in zip(parts[1::2], parts[2::2])}


def fill(template: str, values: Mapping[str, Any]) -> str:
    def replace(match: "re.Match[str]") -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"prompt placeholder {{{{{key}}}}} has no value")
        return str(values[key])

    return re.sub(r"\{\{([A-Z_]+)\}\}", replace, template)


def _range_text(lower: float, upper: float) -> str:
    if lower <= -1000:
        return f"below {upper:g}"
    if upper >= 1000:
        return f"{lower:g} or more"
    return f"{lower:g} to {upper:g}"


def labels_text(spec: GraphSpec, labels_cfg: Mapping[str, Any]) -> str:
    """The frozen label definitions for the relations this task uses."""
    used = set(spec.relations_in_use)
    lines: List[str] = []
    spatial = [r for r in SPATIAL_RELATIONS if r in used]
    if spatial:
        lines.append("### Spatial bins, in centimetres between reference points")
        for relation in spatial:
            scopes = sorted({spec.scope(f.src, f.dst) for f in spec.facts if f.relation == relation})
            for scope in scopes:
                bins = labels_cfg["spatial"][relation][scope]
                who = "gripper to object" if scope == EE_OBJECT else "object to object"
                text = ", ".join(f"`{label}` {_range_text(*bounds)}" for label, bounds in bins.items())
                lines.append(f"- `{relation}`, {who}: {text}.")
    if any(f.temporal for f in spec.facts):
        lines.append("\n### Temporal labels, change over the K-frame window")
        if spatial:
            bins = labels_cfg["temporal"]["spatial"]
            text = ", ".join(f"`{label}` {_range_text(*bounds)} cm" for label, bounds in bins.items())
            lines.append(f"- {' and '.join(f'`{r}`' for r in spatial)}: {text}.")
        if any(r in used for r in AFFORDANCE_RELATIONS):
            lines.append(f"- Compatibility relations: {labels_cfg['temporal']['compatibility']}")
    physical = [r for r in PHYSICAL if r in used]
    if physical:
        lines.append("\n### Physical relations")
        for relation in physical:
            holders = " (`src-holds` / `dst-holds`)" if relation in ("support", "contain") else ""
            lines.append(f"- `{relation}`{holders}: {labels_cfg['physical'][relation]}")
    compat = [r for r in AFFORDANCE_RELATIONS if r in used]
    if compat:
        lines.append("\n### Compatibility relations")
        for relation in compat:
            meanings = labels_cfg["compatibility"][relation]
            text = "; ".join(f"`{label}` when {meaning}" for label, meaning in meanings.items())
            lines.append(f"- `{relation}`: {text}.")
    lines.append("\n### General rules")
    lines += [f"- {rule}" for rule in labels_cfg["general"]]
    return "\n".join(lines)


def facts_text(spec: GraphSpec) -> str:
    lines = []
    for fid, fact in zip(fact_ids(spec), spec.facts):
        temporal = "; also temporal" if fact.temporal else ""
        lines.append(f"- `{fid}` {fact.label()}: {' | '.join(spec.legal_labels(fact.relation))}{temporal}")
    return "\n".join(lines)


def _interval_schema(labels: Sequence[str]) -> Dict[str, Any]:
    return {"type": "array", "items": {
        "type": "object",
        "properties": {"start": {"type": "integer"}, "end": {"type": "integer"},
                       "label": {"type": ["string", "null"], "enum": [*labels, None]}},
        "required": ["start", "end", "label"]}}


def answer_schema(spec: GraphSpec, scope: Scope) -> Dict[str, Any]:
    properties: Dict[str, Any] = {}
    if scope.target:
        properties["active_target"] = {"type": "array", "items": {
            "type": "object",
            "properties": {"start": {"type": "integer"}, "end": {"type": "integer"},
                           "object": {"type": "string", "enum": list(spec.targets)}},
            "required": ["start", "end", "object"]}}
    if scope.facts:
        index = dict(zip(fact_ids(spec), spec.facts))
        labels: List[str] = []
        for fid in scope.facts:
            for label in spec.legal_labels(index[fid].relation):
                if label not in labels:
                    labels.append(label)
        properties["facts"] = {"type": "array", "items": {
            "type": "object",
            "properties": {"fact": {"type": "string", "enum": list(scope.facts)},
                           "absolute": _interval_schema(labels),
                           "temporal": _interval_schema(list(CHANGE_LABELS))},
            "required": ["fact", "absolute", "temporal"]}}
    if scope.boxes:
        properties["boxes"] = {"type": "array", "items": {
            "type": "object",
            "properties": {
                "entity": {"type": "string", "enum": sorted({entity for entity, _ in scope.boxes})},
                "camera": {"type": "string", "enum": sorted({camera for _, camera in scope.boxes})},
                "keyframes": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"frame": {"type": "integer"}, "visible": {"type": "boolean"},
                                   "box_2d": {"type": "array", "items": {"type": "integer"},
                                              "minItems": 4, "maxItems": 4}},
                    "required": ["frame", "visible", "box_2d"]}}},
            "required": ["entity", "camera", "keyframes"]}}
    properties["notes"] = {"type": "string"}
    return {"type": "object", "properties": properties, "required": list(properties)}


def split_scope(scope: Scope) -> List[Scope]:
    """Smaller requests for an answer that did not fit the output limit."""
    if scope.facts and (scope.target or scope.boxes):
        return [Scope(target=scope.target, boxes=list(scope.boxes)), Scope(facts=list(scope.facts))]
    if len(scope.facts) > 1:
        half = len(scope.facts) // 2
        return [Scope(facts=scope.facts[:half]), Scope(facts=scope.facts[half:])]
    if scope.target and scope.boxes:
        return [Scope(target=True), Scope(boxes=list(scope.boxes))]
    if len(scope.boxes) > 1:
        half = len(scope.boxes) // 2
        return [Scope(boxes=scope.boxes[:half]), Scope(boxes=scope.boxes[half:])]
    return [scope]


def scoped_answer(spec: GraphSpec, answer: Mapping[str, Any], scope: Scope) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if scope.target:
        out["active_target"] = answer.get("active_target") or []
    if scope.facts:
        out["facts"] = [item for item in answer.get("facts") or ()
                        if str((item or {}).get("fact", "")).strip() in scope.facts]
    if scope.boxes:
        out["boxes"] = [item for item in answer.get("boxes") or ()
                        if (normalize_entity(spec, (item or {}).get("entity")),
                            str((item or {}).get("camera", "")).strip().lower()) in scope.boxes]
    return out


def issues_text(annotation: EpisodeAnnotation, limit: int = 60) -> str:
    lines = [f"- {issue.message}" for issue in annotation.issues[:limit]]
    if len(annotation.issues) > limit:
        lines.append(f"- ... and {len(annotation.issues) - limit} more")
    return "\n".join(lines)


def usage_totals(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    billed = [r for r in records if not r.get("cached")]
    return {"calls": len(billed), "cached_responses": len(records) - len(billed),
            **sum_usage(r.get("usage_total") or r.get("usage_normalized") for r in billed)}


# --------------------------------------------------------------------------- #
# Annotator
# --------------------------------------------------------------------------- #
class EpisodeAnnotator:
    def __init__(self, configs: Mapping[str, Mapping[str, Any]], source, client: Optional[GeminiClient] = None):
        self.configs = configs
        self.source = source
        self.dataset_cfg = configs["dataset"]
        self.annotation_cfg = configs["annotation"]
        self.labels_cfg = configs["labels"]
        self.client = client or GeminiClient(self.annotation_cfg["gemini"])
        self.sections = load_sections()
        self.prompt_hash = file_sha256(PROMPT_PATH)[:16]
        self.stride = stride_for(source.fps(), self.annotation_cfg["videos"])
        self.settings = ValidationSettings.from_config(self.annotation_cfg, self.stride)
        section = self.annotation_cfg["annotation"]
        self.repair_rounds = int(section["repair_rounds"])
        self.max_splits = int(section["truncation_splits"])

    def path(self, episode: int) -> str:
        return os.path.join(repo_path(self.dataset_cfg["paths"]["annotations"]), episode_name(episode) + ".json")

    def input_identity(self, spec: GraphSpec, prepared: Mapping[str, Any]) -> Dict[str, Any]:
        return {"format": ANNOTATION_FORMAT, "graph": spec.identity(), "labels": stable_hash(self.labels_cfg),
                "prompt": self.prompt_hash, "gemini": self.client.settings(),
                "videos": {camera: info["sha256"] for camera, info in sorted(prepared["cameras"].items())},
                "rendered_spec": stable_hash(self.spec_text(spec)),
                "instruction": spec.instruction, "target_rule": spec.target_rule,
                "rows": int(prepared["rows"]), "source_fps": self.source.fps(),
                "validation": self.settings.identity(), "repair_rounds": self.repair_rounds,
                "truncation_splits": self.max_splits}

    # ------------------------------------------------------------- text
    def spec_text(self, spec: GraphSpec) -> str:
        fps = self.source.fps()
        K = spec.temporal_window
        descriptions = self.dataset_cfg["source"].get("camera_descriptions") or {}
        return fill(self.sections["SPEC"], {
            "FPS": f"{fps:g}",
            "CAMERAS": "\n".join(f"  - `{c}`: {descriptions.get(c, c)}" for c in spec.cameras),
            "STRIDE": self.stride,
            "VIDEO_FPS": f"{fps / self.stride:g}",
            "ENTITIES": "\n".join(f"- `{e.id}`: {e.description}. Reference point: {e.reference}."
                                  for e in spec.entities),
            "SIZES": "\n".join(f"  - {size}" for size in self.labels_cfg["reference_sizes"]),
            "FACTS": facts_text(spec),
            "UNOBSERVED": self.labels_cfg["unobserved"],
            "K": K, "K_SECONDS": f"{K / fps:.2f}", "K_MINUS_ONE": K - 1,
            "LABELS": labels_text(spec, self.labels_cfg),
            "BOX_EVERY": self.settings.box_every,
        })

    def job_text(self, spec: GraphSpec, scope: Scope, n_frames: int) -> str:
        values = {"LAST_FRAME": n_frames - 1, "K": spec.temporal_window, "TARGET_RULE": spec.target_rule}
        full = full_scope(spec)
        items = []
        if scope.target:
            items.append(fill(self.sections["ITEM target"], values))
        if scope.facts:
            index = dict(zip(fact_ids(spec), spec.facts))
            temporal = [fid for fid in scope.facts if index[fid].temporal]
            values["FACT_SCOPE"] = ("every fact above" if scope.facts == full.facts
                                    else "facts " + ", ".join(f"`{f}`" for f in scope.facts))
            values["TEMPORAL_FACTS"] = ", ".join(f"`{f}`" for f in temporal) or "none of these"
            items.append(fill(self.sections["ITEM facts"], values))
        if scope.boxes:
            values["BOX_SCOPE"] = ("every entity in every camera" if scope.boxes == full.boxes
                                   else ", ".join(f"`{e}` in `{c}`" for e, c in scope.boxes))
            items.append(fill(self.sections["ITEM boxes"], values))
        items.append(self.sections["ITEM notes"])
        numbered = "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))
        return fill(self.sections["JOB"], {"ITEMS": numbered})

    def episode_text(self, spec: GraphSpec, episode: int, prepared: Mapping[str, Any]) -> str:
        n = int(prepared["rows"])
        return fill(self.sections["EPISODE"], {
            "EPISODE": episode, "N_FRAMES": n, "LAST_FRAME": n - 1, "SHOWN": prepared["shown"],
            "CAMERA_ORDER": ", then ".join(f"`{c}`" for c in spec.cameras),
            "TASK": spec.instruction,
        })

    def video_parts(self, spec: GraphSpec, prepared: Mapping[str, Any]) -> List[VideoPart]:
        fps = float(prepared["fps"])
        return [VideoPart(path=repo_path(prepared["cameras"][camera]["path"]), fps=fps, source_fps=fps)
                for camera in spec.cameras]

    # ------------------------------------------------------------- calls
    def ask(self, spec: GraphSpec, episode: int, prepared: Mapping[str, Any], scope: Scope,
            tail: Callable[[Scope], str], name: str, depth: int = 0
            ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Answers covering ``scope``, split into smaller requests when one is cut off."""
        text = "\n\n".join(t for t in (self.episode_text(spec, episode, prepared), tail(scope),
                                       self.job_text(spec, scope, int(prepared["rows"]))) if t)
        parts = [TextPart(self.spec_text(spec)), *self.video_parts(spec, prepared), TextPart(text)]
        label = f"{episode_name(episode)}/{name}"
        try:
            parsed, record = self.client.generate_json(parts, answer_schema(spec, scope), label)
        except GeminiTruncated as exc:
            pieces = split_scope(scope)
            if depth >= self.max_splits or len(pieces) < 2:
                raise
            print(f"[annotate] {label}: cut off at the output limit; asking in {len(pieces)} pieces", flush=True)
            answers: List[Dict[str, Any]] = []
            records: List[Dict[str, Any]] = [{"label": label, "outcome": "truncated", "cached": False,
                                              "usage_total": exc.usage}]
            for i, piece in enumerate(pieces):
                got, used = self.ask(spec, episode, prepared, piece, tail, f"{name}.{i}", depth + 1)
                answers += got
                records += used
            return answers, records
        return [parsed if isinstance(parsed, dict) else {}], [record]

    def annotate(self, episode: int, prepared: Mapping[str, Any]) -> EpisodeAnnotation:
        spec = self.source.spec(episode)
        n = int(prepared["rows"])
        records: List[Dict[str, Any]] = []

        def build(answer: Mapping[str, Any]) -> EpisodeAnnotation:
            return build_annotation(spec, episode_index=episode, n_frames=n, fps=self.source.fps(),
                                    answer=answer, settings=self.settings)

        answers, used = self.ask(spec, episode, prepared, full_scope(spec), lambda scope: "", "graph")
        records += used
        answer = combine(answers)
        annotation = build(answer)
        log: List[Dict[str, Any]] = []
        for round_number in range(1, self.repair_rounds + 1):
            if annotation.valid:
                break
            scope = repair_scope(annotation, spec)
            problems = issues_text(annotation)

            def tail(piece: Scope, problems=problems, previous=answer) -> str:
                return fill(self.sections["REPAIR"], {
                    "ISSUES": problems,
                    "PREVIOUS": json.dumps(scoped_answer(spec, previous, piece), separators=(",", ":"))})

            try:
                patches, used = self.ask(spec, episode, prepared, scope, tail, f"repair{round_number}")
            except GeminiError as exc:
                log.append({"round": round_number, "scope": scope.to_json(), "error": str(exc)})
                records.append({"label": f"repair{round_number}", "cached": False, "usage_total": exc.usage})
                break
            records += used
            answer, rejected = merge_answer(spec, answer, combine(patches), scope)
            before = len(annotation.issues)
            annotation = build(answer)
            log.append({"round": round_number, "scope": scope.to_json(), "issues_before": before,
                        "issues_after": len(annotation.issues), "rejected": rejected})
        annotation.repair_log = log
        annotation.usage = usage_totals(records)
        annotation.provenance = {"created": utc_now(), "model": self.client.model,
                                 "requests": [{k: r.get(k) for k in ("label", "key", "cached", "finish_reason",
                                                                        "usage_total")} for r in records]}
        return annotation

    def save(self, annotation: EpisodeAnnotation) -> str:
        spec = self.source.spec(annotation.episode_index)
        return write_json(self.path(annotation.episode_index), annotation.to_json(spec))


def reusable(path: str, identity: Mapping[str, Any]) -> bool:
    if not os.path.isfile(path):
        return False
    stored = read_json(path).get("input_identity") or {}
    return stable_hash(stored) == stable_hash(identity)


def main(argv: Optional[Sequence[str]] = None) -> None:
    from ..data.source import LeRobotSource

    parser = argparse.ArgumentParser(description="Annotate episodes' scene graphs with Gemini.")
    parser.add_argument("--episodes", default="pilot")
    parser.add_argument("--force", action="store_true", help="annotate again even when the inputs are unchanged")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    configs = load_configs(["dataset", "annotation", "graph", "labels"], args.overrides)
    source = LeRobotSource(configs)
    annotator = EpisodeAnnotator(configs, source)
    annotator.client.use_response_cache = not args.force
    minimum = int(configs["annotation"]["annotation"]["min_frames"])
    totals: List[Dict[str, Any]] = []
    counts = {"valid": 0, "invalid": 0, "reused": 0, "skipped": 0, "failed": 0}
    for episode in source.select(args.episodes):
        if source.lengths()[episode] < minimum:
            print(f"[annotate] episode {episode}: skipped, {source.lengths()[episode]} frames < {minimum}", flush=True)
            counts["skipped"] += 1
            continue
        prepared, reason = prepared_status(source, episode, configs["annotation"]["videos"])
        if prepared is None:
            print(f"[annotate] episode {episode}: skipped, videos {reason}; run prepare_videos", flush=True)
            counts["skipped"] += 1
            continue
        path = annotator.path(episode)
        identity = annotator.input_identity(source.spec(episode), prepared)
        if not args.force and reusable(path, identity):
            print(f"[annotate] episode {episode}: current, reused {path}", flush=True)
            counts["reused"] += 1
            continue
        try:
            annotation = annotator.annotate(episode, prepared)
        except GeminiError as exc:
            print(f"[annotate] episode {episode}: FAILED: {exc}", flush=True)
            counts["failed"] += 1
            continue
        annotation.input_identity = identity
        annotator.save(annotation)
        usage = annotation.usage
        totals.append(usage)
        status = "valid" if annotation.valid else f"INVALID, {len(annotation.issues)} issue(s)"
        counts["valid" if annotation.valid else "invalid"] += 1
        print(f"[annotate] episode {episode} ({annotation.task}): {status}; {usage['calls']} call(s), "
              f"tokens in {usage['input_tokens']} (cached {usage['cached_tokens']}) out {usage['output_tokens']} "
              f"thought {usage['thought_tokens']}", flush=True)
    run = sum_usage(totals)
    print(f"[annotate] {counts}; tokens in {run['input_tokens']} (cached {run['cached_tokens']}) "
          f"out {run['output_tokens']} thought {run['thought_tokens']}", flush=True)


if __name__ == "__main__":
    main()
