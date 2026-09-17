"""One shared label specification, proposed by Gemini and then frozen.

    python -m real_robot.preprocessing.define_bins propose
    python -m real_robot.preprocessing.define_bins freeze      # after reading the proposal
    python -m real_robot.preprocessing.define_bins show

``propose`` shows Gemini three representative episodes and asks for
physically grounded bins: reference object sizes, centimetre ranges for every
spatial label in each scope, evidence rules for the physical relations,
meanings for the compatibility labels, and change thresholds over the fixed
window ``K``. The sizes and thresholds are estimates read off RGB video; the
measurement anchors are stated explicitly so they mean the same thing as the
geometry, and ``evaluation/check_bins.py`` compares them with measured
distances once geometry exists. Nothing is normalised to an episode's own range
of motion: the same label has to mean the same physical situation everywhere.

The proposal needs motion, not frame-exact timing, so its videos are sampled at
``bins.video_fps`` rather than the annotation rate.

``freeze`` validates a proposal -- every label of every relation in use
present, ranges contiguous and ordered -- and writes ``graph_spec.json`` with
its hash. A hand-edited proposal can be frozen with ``--from``. Annotation
refuses to run against anything that is not frozen, and the hash becomes part
of every annotation's and every model's identity.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from scenegraph.core.relation_rules import (
    ABS_LABELS,
    AFFORDANCE_RELATIONS,
    CHANGE_LABELS,
    COMPAT_LABELS,
    PHYSICAL_RELATIONS,
    SPATIAL_LABELS,
    SPATIAL_RELATIONS,
    TEMPORAL_RELATIONS,
)

from ..common import (
    PROMPT_DIR,
    add_config_arguments,
    file_sha256,
    load_configs,
    read_json,
    repo_path,
    stable_hash,
    utc_now,
    write_json,
)
from ..graphs.schema import EE_OBJECT, OBJECT_OBJECT, GraphSpec

SPEC_FORMAT = "real_robot/bin-spec-v1"
CAMERA_DESCRIPTIONS = {
    "high": "fixed camera above and in front of the workspace, looking down at the table",
    "wrist_right": "camera mounted on the right wrist, moving with the gripper; the fingers are at the bottom of its image",
}
POINT_DESCRIPTIONS = {
    # Surface points on the fingers themselves: depth is read where the point
    # is, and the space between open fingertips is table or object, not gripper.
    ("ee", "fingertip_1"): "the tip of one gripper finger, on the finger's own visible surface",
    ("ee", "fingertip_2"): "the tip of the other gripper finger, on the finger's own visible surface",
    ("banana", "center"): "the centre of the banana",
    ("banana", "grasp_region"): "the middle of the banana, where the fingers should close around it",
    ("pot", "rim_left"): "the leftmost point of the pot's top rim in this image",
    ("pot", "rim_right"): "the rightmost point of the pot's top rim in this image",
    ("pot", "rim_near"): "the point of the pot's top rim nearest to the camera (lowest in the image)",
    ("pot", "rim_far"): "the point of the pot's top rim farthest from the camera (highest in the image)",
    ("lid", "center"): "the centre of the lid's top surface",
    ("lid", "handle"): "the top of the lid's handle or knob",
    ("table", "surface_1"): "a point on bare table surface, away from every object",
    ("table", "surface_2"): "a second bare table point, far from the first",
    ("table", "surface_3"): "a third bare table point, not in line with the first two",
    ("table", "surface_4"): "a fourth bare table point, spread away from the other three",
}
ENTITY_DESCRIPTIONS = {"ee": "the robot's right gripper (end effector)"}
# The point each entity's spatial facts are measured from. The geometry stage
# measures the same points, so a label and a measured distance mean one thing.
REFERENCE_POINTS = {
    "ee": "the gripper's closing point, midway between the two fingertips",
    "banana": "the centre of the banana",
    "pot": "the centre of the pot's opening, at the height of its rim",
    "lid": "the centre of the lid",
    "table": "the tabletop surface",
}


# --------------------------------------------------------------------------- #
# Prompt rendering shared with annotation
# --------------------------------------------------------------------------- #
def render_template(template: str, values: Mapping[str, Any]) -> str:
    text = template
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text


def load_sections(path: str) -> Dict[str, str]:
    """``<!-- NAME -->`` delimited sections of a prompt file."""
    import re

    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    sections: Dict[str, str] = {}
    matches = list(re.finditer(r"<!--\s*([A-Za-z_ ]+?)\s*-->", text))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group(1).strip()] = text[match.end():end].strip()
    return sections


def cameras_text(spec: GraphSpec) -> str:
    return "\n".join(f"  - `{c}`: {CAMERA_DESCRIPTIONS.get(c, c)}" for c in spec.cameras)


def entities_text(spec: GraphSpec) -> str:
    return "\n".join(f"- `{e.id}`: {ENTITY_DESCRIPTIONS.get(e.id, 'the ' + e.name)}" for e in spec.entities)


def scopes_in_use(spec: GraphSpec) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for fact in spec.facts:
        scope = spec.scope(fact.src, fact.dst)
        out.setdefault(fact.relation, [])
        if scope not in out[fact.relation]:
            out[fact.relation].append(scope)
    return out


def relations_text(spec: GraphSpec) -> str:
    lines = []
    for relation, scopes in scopes_in_use(spec).items():
        temporal = "has temporal-change labels" if relation in TEMPORAL_RELATIONS else "no temporal labels"
        lines.append(f"- `{relation}` ({', '.join(scopes)}; {temporal}): "
                     + " | ".join(f"`{label}`" for label in ABS_LABELS[relation]))
    return "\n".join(lines)


def facts_text(spec: GraphSpec) -> str:
    lines = []
    for index, fact in enumerate(spec.facts):
        temporal = "temporal: yes" if fact.temporal else "temporal: no"
        labels = " | ".join(ABS_LABELS[fact.relation])
        lines.append(f"- `F{index:02d}` `{fact.relation}({fact.src}, {fact.dst})`: {labels} ({temporal})")
    return "\n".join(lines)


def reference_points_text(spec: GraphSpec, indent: str = "  ") -> str:
    return "\n".join(f"{indent}- `{e.id}`: {REFERENCE_POINTS.get(e.id, 'the centre of the ' + e.name)}"
                     for e in spec.entities)


def unobserved_text(spec: GraphSpec) -> str:
    """Which physical relations may be `unobserved`: only those whose vocabulary has the label."""
    physical = [r for r in spec.relations_in_use if r in PHYSICAL_RELATIONS]
    allowed = [r for r in physical if "unobserved" in ABS_LABELS[r]]
    refused = [r for r in physical if "unobserved" not in ABS_LABELS[r]]
    lines = []
    if allowed:
        lines.append("- " + ", ".join(f"`{r}`" for r in allowed) + " may be `unobserved` only when neither camera "
                     "can show whether it holds.")
    if refused:
        lines.append("- " + ", ".join(f"`{r}`" for r in refused) + " has no `unobserved` label: always decide it "
                     "from the best evidence either camera gives.")
    lines.append("- A compatibility relation is `unobserved` whenever the two entities are not near enough to judge "
                 "the fit (their planar distance is neither `very-near` nor `near`).")
    return "\n".join(lines)


def points_text(spec: GraphSpec, indent: str = "   ") -> str:
    lines = []
    for owner, names in spec.points.items():
        for name in names:
            lines.append(f"{indent}- `{owner}` `{name}`: {POINT_DESCRIPTIONS.get((owner, name), name)}")
    return "\n".join(lines)


def render_bin_spec(bins: Mapping[str, Any]) -> str:
    """The frozen specification as the tables annotators read."""
    out = ["### Reference dimensions"]
    for item in bins["reference_objects"]:
        out.append(f"- {item['object']} {item['dimension']}: about {item['approx_cm']:g} cm ({item['visual_cue']})")
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for item in bins["spatial_bins"]:
        groups.setdefault((item["relation"], item["scope"]), []).append(item)
    for (relation, scope), items in groups.items():
        order = SPATIAL_LABELS[relation]
        items = sorted(items, key=lambda i: order.index(i["label"]))
        out += ["", f"### {relation} ({scope})", "| label | range (cm) | meaning |", "|---|---|---|"]
        out += [f"| `{i['label']}` | {_range(i['lower_cm'], i['upper_cm'])} | {i['meaning']} |" for i in items]
    out += ["", "### Physical relations"]
    for item in bins["physical_rules"]:
        out.append(f"- `{item['relation']}` = `{item['label']}`: {item['meaning']} Evidence: {item['evidence']}")
    out += ["", "### Compatibility"]
    for item in bins["compatibility_bins"]:
        out.append(f"- `{item['relation']}` = `{item['label']}`: {item['meaning']}")
    out += ["", "### Temporal change over the window",
            "| family | scope | label | change over the window (cm) | meaning |", "|---|---|---|---|---|"]
    for item in sorted(bins["temporal_bins"], key=lambda i: (i["family"], i["scope"], CHANGE_LABELS.index(i["label"]))):
        change = (_range(item["lower_cm_per_window"], item["upper_cm_per_window"])
                  if item["family"] != "compatibility" else "-")
        out.append(f"| {item['family']} | {item['scope']} | `{item['label']}` | {change} | {item['meaning']} |")
    out += ["", "### General rules"] + [f"- {rule}" for rule in bins["general_rules"]]
    return "\n".join(out)


def _range(lower: float, upper: float) -> str:
    lo = "-inf" if float(lower) <= -999 else f"{float(lower):g}"
    hi = "+inf" if float(upper) >= 999 else f"{float(upper):g}"
    return f"[{lo}, {hi})"


# --------------------------------------------------------------------------- #
# Schema and validation
# --------------------------------------------------------------------------- #
def _obj(properties: Mapping[str, Any]) -> Dict[str, Any]:
    return {"type": "object", "properties": dict(properties), "required": list(properties)}


def bin_schema(spec: GraphSpec) -> Dict[str, Any]:
    used = scopes_in_use(spec)
    spatial = [r for r in used if r in SPATIAL_RELATIONS]
    physical = [r for r in used if r in PHYSICAL_RELATIONS]
    compat = [r for r in used if r in AFFORDANCE_RELATIONS]
    spatial_labels = sorted({l for r in spatial for l in SPATIAL_LABELS[r]})
    physical_labels = sorted({l for r in physical for l in ABS_LABELS[r]})
    string, number = {"type": "string"}, {"type": "number"}
    return _obj({
        "reference_objects": {"type": "array", "items": _obj({
            "object": {"type": "string", "enum": list(spec.entity_ids)},
            "dimension": string, "approx_cm": number, "visual_cue": string})},
        "spatial_bins": {"type": "array", "items": _obj({
            "relation": {"type": "string", "enum": spatial},
            "scope": {"type": "string", "enum": [EE_OBJECT, OBJECT_OBJECT]},
            "label": {"type": "string", "enum": spatial_labels},
            "meaning": string, "lower_cm": number, "upper_cm": number})},
        "physical_rules": {"type": "array", "items": _obj({
            "relation": {"type": "string", "enum": physical},
            "label": {"type": "string", "enum": physical_labels},
            "meaning": string, "evidence": string})},
        "compatibility_bins": {"type": "array", "items": _obj({
            "relation": {"type": "string", "enum": compat},
            "label": {"type": "string", "enum": list(COMPAT_LABELS)},
            "meaning": string})},
        "temporal_bins": {"type": "array", "items": _obj({
            "family": {"type": "string", "enum": ["planar-distance", "height-offset", "compatibility"]},
            "scope": {"type": "string", "enum": [EE_OBJECT, OBJECT_OBJECT, "any"]},
            "label": {"type": "string", "enum": list(CHANGE_LABELS)},
            "meaning": string, "lower_cm_per_window": number, "upper_cm_per_window": number})},
        "general_rules": {"type": "array", "items": string},
    })


def _contiguous(items: Sequence[Mapping[str, Any]], order: Sequence[str], lower_key: str, upper_key: str,
                where: str, tolerance: float = 0.5) -> List[str]:
    problems = []
    by_label = {}
    for item in items:
        if item["label"] in by_label:
            problems.append(f"{where}: {item['label']!r} is defined twice")
        by_label[item["label"]] = item
    missing = [label for label in order if label not in by_label]
    if missing:
        return problems + [f"{where}: missing labels {missing}"]
    previous_upper = None
    for label in order:
        lower, upper = float(by_label[label][lower_key]), float(by_label[label][upper_key])
        if not lower < upper:
            problems.append(f"{where}: {label!r} has lower {lower} >= upper {upper}")
        if previous_upper is not None and abs(lower - previous_upper) > tolerance:
            problems.append(f"{where}: {label!r} starts at {lower} but the previous label ends at {previous_upper}")
        previous_upper = upper
    return problems


def validate_bins(bins: Mapping[str, Any], spec: GraphSpec) -> List[str]:
    problems: List[str] = []
    used = scopes_in_use(spec)
    for key in ("reference_objects", "spatial_bins", "physical_rules", "compatibility_bins",
                "temporal_bins", "general_rules"):
        if key not in bins:
            problems.append(f"missing section {key!r}")
    if problems:
        return problems
    named = {item["object"] for item in bins["reference_objects"]}
    for obj in spec.object_ids:
        if obj != "table" and obj not in named:
            problems.append(f"reference_objects: no dimension for {obj}")
    for relation in (r for r in used if r in SPATIAL_RELATIONS):
        for scope in used[relation]:
            items = [i for i in bins["spatial_bins"] if i["relation"] == relation and i["scope"] == scope]
            where = f"spatial_bins {relation} ({scope})"
            problems += _contiguous(items, SPATIAL_LABELS[relation], "lower_cm", "upper_cm", where)
            by_label = {i["label"]: i for i in items}
            if relation == "planar-distance" and "very-near" in by_label and abs(float(by_label["very-near"]["lower_cm"])) > 0.5:
                problems.append(f"{where}: very-near must start at 0 cm")
            if relation == "height-offset" and "level" in by_label:
                level = by_label["level"]
                if not float(level["lower_cm"]) < 0 < float(level["upper_cm"]):
                    problems.append(f"{where}: level must straddle 0 cm")
    for relation in (r for r in used if r in PHYSICAL_RELATIONS):
        labels = {i["label"] for i in bins["physical_rules"] if i["relation"] == relation}
        missing = [l for l in ABS_LABELS[relation] if l not in labels and l != "unobserved"]
        if missing:
            problems.append(f"physical_rules {relation}: no rule for {missing}")
    for relation in (r for r in used if r in AFFORDANCE_RELATIONS):
        labels = {i["label"] for i in bins["compatibility_bins"] if i["relation"] == relation}
        missing = [l for l in COMPAT_LABELS if l not in labels]
        if missing:
            problems.append(f"compatibility_bins {relation}: no meaning for {missing}")
    families = []
    for relation in (r for r in used if r in SPATIAL_RELATIONS):
        families += [(relation, scope) for scope in used[relation]]
    for family, scope in families:
        items = [i for i in bins["temporal_bins"] if i["family"] == family and i["scope"] == scope]
        where = f"temporal_bins {family} ({scope})"
        problems += _contiguous(items, CHANGE_LABELS, "lower_cm_per_window", "upper_cm_per_window", where)
        stable = next((i for i in items if i["label"] == "stable"), None)
        if stable and not float(stable["lower_cm_per_window"]) < 0 < float(stable["upper_cm_per_window"]):
            problems.append(f"{where}: stable must straddle 0 cm")
    if any(r in AFFORDANCE_RELATIONS for r in used):
        labels = {i["label"] for i in bins["temporal_bins"] if i["family"] == "compatibility"}
        missing = [l for l in CHANGE_LABELS if l not in labels]
        if missing:
            problems.append(f"temporal_bins compatibility: no meaning for {missing}")
    return problems


# --------------------------------------------------------------------------- #
# Frozen spec
# --------------------------------------------------------------------------- #
def spec_paths(dataset_cfg: Mapping[str, Any]) -> Dict[str, str]:
    root = repo_path(dataset_cfg["paths"]["spec"])
    return {"proposed": os.path.join(root, "graph_spec.proposed.json"),
            "frozen": os.path.join(root, "graph_spec.json")}


def load_frozen_bins(dataset_cfg: Mapping[str, Any], spec: GraphSpec) -> Dict[str, Any]:
    path = spec_paths(dataset_cfg)["frozen"]
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no frozen bin specification at {path}; run define_bins propose, read it, then freeze"
        )
    data = read_json(path)
    if data.get("format") != SPEC_FORMAT or not data.get("frozen"):
        raise ValueError(f"{path} is not a frozen bin specification")
    if data["graph_hash"] != stable_hash(spec.identity()):
        raise ValueError(
            f"{path} was frozen for a different graph configuration (graph.yaml changed since). "
            "Re-propose and re-freeze rather than annotating against a stale specification."
        )
    if stable_hash(data["bins"]) != data["bins_hash"]:
        raise ValueError(f"{path} was edited after freezing; its bins no longer match bins_hash")
    return data


def choose_episodes(source, bins_cfg: Mapping[str, Any]) -> List[int]:
    """Representative episodes to propose bins from. The frozen bins then apply to every episode."""
    from ..data.selection import extremes_and_median

    available = source.available()
    chosen = bins_cfg.get("episodes", "auto")
    if chosen == "auto":
        return extremes_and_median(available, source.lengths())[: int(bins_cfg.get("count", 3))]
    chosen = [int(e) for e in chosen]
    outside = [e for e in chosen if e not in available]
    if outside:
        raise SystemExit(f"bin-definition episodes {outside} are not in the dataset")
    return chosen


def main(argv: Optional[Sequence[str]] = None) -> None:
    from ..data.episode_dataset import RawEpisodeSource
    from .gemini_client import GeminiClient, TextPart, VideoPart
    from .prepare_videos import prepare_episode, prepared_video_path

    parser = argparse.ArgumentParser(description="Propose, validate and freeze the shared bins.")
    sub = parser.add_subparsers(dest="command", required=True)
    propose = sub.add_parser("propose")
    propose.add_argument("--dry-run", action="store_true", help="print the prompt, call nothing")
    add_config_arguments(propose)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--from", dest="source_path", default="")
    freeze.add_argument("--force", action="store_true")
    add_config_arguments(freeze)
    show = sub.add_parser("show")
    add_config_arguments(show)
    args = parser.parse_args(argv)

    configs = load_configs(["dataset", "annotation", "graph"], args.overrides)
    spec = GraphSpec.from_config(configs["graph"])
    paths = spec_paths(configs["dataset"])

    if args.command == "show":
        data = load_frozen_bins(configs["dataset"], spec)
        print(render_bin_spec(data["bins"]))
        return

    if args.command == "freeze":
        path = repo_path(args.source_path) if args.source_path else paths["proposed"]
        proposal = read_json(path)
        bins = proposal.get("bins", proposal)
        problems = validate_bins(bins, spec)
        if problems:
            raise SystemExit("[bins] cannot freeze:\n  " + "\n  ".join(problems))
        if os.path.isfile(paths["frozen"]) and not args.force:
            raise SystemExit(f"{paths['frozen']} exists; a frozen specification is not replaced without --force")
        write_json(paths["frozen"], {
            "format": SPEC_FORMAT, "frozen": True, "created": utc_now(),
            "bins": bins, "bins_hash": stable_hash(bins),
            "graph_identity": spec.identity(), "graph_hash": stable_hash(spec.identity()),
            "temporal_window": spec.temporal_window,
            "provenance": {"frozen_from": os.path.relpath(path, repo_path("")).replace(os.sep, "/"),
                           "proposal_sha256": file_sha256(path),
                           **{k: v for k, v in proposal.get("provenance", {}).items()}},
        })
        print(f"[bins] frozen -> {paths['frozen']} (bins_hash {stable_hash(bins)})")
        return

    source = RawEpisodeSource(configs)
    episodes = choose_episodes(source, configs["annotation"]["bins"])
    template_path = os.path.join(PROMPT_DIR, "bin_definition.md")
    with open(template_path, "r", encoding="utf-8") as handle:
        template = handle.read()
    fps = source.fps()
    text = render_template(template, {
        "FPS": f"{fps:g}", "CAMERAS": cameras_text(spec), "TASK": " ".join(configs["dataset"]["source"]["task"].split()),
        "N_EPISODES": len(episodes),
        "EPISODES": "\n".join(f"- episode {e}: {source.lengths()[e]} frames" for e in episodes),
        "ENTITIES": entities_text(spec), "RELATIONS": relations_text(spec),
        "REFERENCE_POINTS": reference_points_text(spec), "UNOBSERVED": unobserved_text(spec),
        "VIDEO_FPS": f"{float(configs['annotation']['bins']['video_fps']):g}",
        "K": spec.temporal_window, "K_SECONDS": f"{spec.temporal_window / fps:.2f}",
    })
    schema = bin_schema(spec)
    if args.dry_run:
        print(text)
        return
    gemini_cfg = configs["annotation"]["gemini"]
    parts = [TextPart(text)]
    for episode in episodes:
        prepare_episode(source, episode, configs["annotation"]["videos"])
        for camera in spec.cameras:
            parts.append(TextPart(f"Episode {episode}, camera `{camera}`:"))
            parts.append(VideoPart(prepared_video_path(configs["dataset"], episode, camera),
                                   fps=float(configs["annotation"]["bins"]["video_fps"]), source_fps=fps))
    client = GeminiClient(gemini_cfg)
    parsed, record = client.generate_json(parts, schema, label="bin_definition")
    problems = validate_bins(parsed, spec)
    write_json(paths["proposed"], {
        "bins": parsed,
        "problems": problems,
        "provenance": {"model": client.model, "backend": client.backend,
                       "prompt_sha256": stable_hash([template, schema]),
                       "episodes": episodes, "request_key": record["key"], "created": utc_now()},
    })
    if not any(p.startswith("missing section") for p in problems):
        print(render_bin_spec(parsed))
    if problems:
        print("[bins] the proposal has problems; edit it or re-propose before freezing:\n  " + "\n  ".join(problems))
    print(f"[bins] proposal -> {paths['proposed']}")


if __name__ == "__main__":
    main()
