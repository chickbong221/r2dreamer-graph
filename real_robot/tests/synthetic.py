"""The shipped configs, and complete answers built from them."""

from __future__ import annotations

from typing import Any, Dict, List

from ..common import load_configs
from ..graphs.schema import GraphConfig, GraphSpec
from ..graphs.validate import ValidationSettings, fact_ids, slots

TASKS = ("banana_pot_lid", "blue_on_red", "cubes_in_cup")


def configs() -> Dict[str, dict]:
    return load_configs(["dataset", "annotation", "graph", "labels"])


def graph_config() -> GraphConfig:
    return GraphConfig.from_config(configs()["graph"])


def spec(task: str = "banana_pot_lid") -> GraphSpec:
    return graph_config().spec(task)


def settings(stride: int = 3) -> ValidationSettings:
    return ValidationSettings(box_every=15, stride=stride)


def gemini_box(entity: int, camera: int) -> List[int]:
    return [100 + 20 * entity, 100 + 30 * camera, 300 + 20 * entity, 400 + 30 * camera]


def target_intervals(spec: GraphSpec, n: int) -> List[Dict[str, Any]]:
    if len(spec.targets) == 1:
        return [{"start": 0, "end": n - 1, "object": spec.targets[0]}]
    half = n // 2
    return [{"start": 0, "end": half - 1, "object": spec.targets[0]},
            {"start": half, "end": n - 1, "object": spec.targets[1]}]


def fact_entry(spec: GraphSpec, fid: str, n: int) -> Dict[str, Any]:
    fact = spec.facts[fact_ids(spec).index(fid)]
    labels = spec.legal_labels(fact.relation)
    third = n // 3
    absolute = [{"start": 0, "end": third - 1, "label": labels[0]},
                {"start": third, "end": n - 1, "label": labels[1 % len(labels)]}]
    temporal = ([{"start": spec.temporal_window, "end": n - 1, "label": "stable"}] if fact.temporal else [])
    return {"fact": fid, "absolute": absolute, "temporal": temporal}


def box_entry(spec: GraphSpec, entity: str, camera: str, n: int, every: int = 15) -> Dict[str, Any]:
    e = spec.entity_ids.index(entity)
    c = spec.cameras.index(camera)
    frames = list(range(0, n, every))
    if frames[-1] != n - 1:
        frames.append(n - 1)
    return {"entity": entity, "camera": camera,
            "keyframes": [{"frame": f, "visible": True, "box_2d": gemini_box(e, c)} for f in frames]}


def complete_answer(spec: GraphSpec, n: int, every: int = 15) -> Dict[str, Any]:
    return {"active_target": target_intervals(spec, n),
            "facts": [fact_entry(spec, fid, n) for fid in fact_ids(spec)],
            "boxes": [box_entry(spec, entity, camera, n, every) for entity, camera in slots(spec)],
            "notes": ""}
