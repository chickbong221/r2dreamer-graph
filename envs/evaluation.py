"""Fixed MS-HAB evaluation panels and per-sub-scene illumination.

B and C share one vector simulator and one policy batch. No extra normal
evaluation is constructed. The panel is also the source of metric grouping.
"""

from collections import defaultdict
from dataclasses import dataclass
import re

import numpy as np


@dataclass(frozen=True)
class EvalCase:
    scene: str
    object: str
    plan_index: int
    group: str
    condition: str = "nominal"
    intensity: float = 1.0
    repetition: int = 0


def lighting_conditions(config):
    lighting = getattr(config, "eval_lighting", None)
    if lighting is None or not lighting.enabled:
        return []
    conditions = dict(lighting.conditions)
    if not conditions or conditions.get("nominal") != 1.0:
        raise ValueError("lighting evaluation needs nominal: 1.0 and changed conditions")
    if not any(float(v) != 1.0 for v in conditions.values()):
        raise ValueError("lighting evaluation has no changed illumination")
    for name, value in conditions.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
            raise ValueError(f"invalid lighting condition name: {name!r}")
        if not np.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"invalid light intensity: {name}={value}")
    if int(lighting.envs_per_condition) <= 0:
        raise ValueError("lighting envs_per_condition must be positive")
    return [(str(k), float(v)) for k, v in conditions.items()]


def build_panel(by_object, config):
    """Indices refer to each scene's order in the flattened task plan list."""
    indexed, scene_count = defaultdict(list), defaultdict(int)
    for obj, plans in by_object.items():
        for plan in plans:
            scene = str(plan.build_config_name)
            indexed[(scene, obj)].append((scene_count[scene], plan))
            scene_count[scene] += 1
    for values in indexed.values():
        values.sort(key=lambda item: (str(item[1].init_config_name), item[0]))
    scenes, objects = sorted(scene_count), sorted(by_object)
    count = int(config.eval_episode_num)
    mode = str(getattr(config, "eval_panel", ""))
    cases = []
    if not scenes or not objects:
        raise ValueError("evaluation panel has no task plans")

    def add(scene, obj, k, group, condition="nominal", intensity=1.0):
        candidates = indexed.get((scene, obj), [])
        if not candidates:
            raise ValueError(f"no evaluation plan for {obj} in {scene}")
        index = candidates[k % len(candidates)][0]
        cases.append(EvalCase(scene, obj, index, group, condition, intensity, k))

    if mode == "objects":
        if len(scenes) != 1 or count <= 0 or count % len(objects):
            raise ValueError("object evaluation needs one scene and equal episodes per object")
        for k in range(count // len(objects)):
            for obj in objects:
                add(scenes[0], obj, k, "object")
    elif mode == "scenes":
        if len(objects) != 1 or count <= 0 or count % len(scenes):
            raise ValueError("scene evaluation needs one object and equal episodes per scene")
        for k in range(count // len(scenes)):
            for scene in scenes:
                add(scene, objects[0], k, "scene")
    else:
        raise ValueError(f"unknown fixed evaluation panel: {mode!r}")
    conditions = lighting_conditions(config)
    if conditions:
        if mode != "scenes":
            raise ValueError("lighting evaluation is attached to B's scene panel only")
        training_scenes = list(config.train_build_config_ids)
        if len(training_scenes) != 1:
            raise ValueError("lighting comparison needs one named training scene")
        for condition, intensity in conditions:
            for k in range(int(config.eval_lighting.envs_per_condition)):
                add(training_scenes[0], objects[0], k, "light", condition, intensity)
    return cases


def panel_metrics(cases, values, training_scenes):
    """Separate primary B/A scores from lighting; never pool them together."""
    values = {k: np.asarray(v, dtype=float) for k, v in values.items()}
    for key, arr in values.items():
        if arr.shape != (len(cases),) or not np.isfinite(arr).all():
            raise ValueError(f"invalid per-environment evaluation metric: {key}")
    outcomes = {"score", "length", "success_once", "success_at_end", "fail_once"}
    groups = defaultdict(list)
    for i, case in enumerate(cases):
        if case.group != "light":
            groups["eval"].append(i)
        if case.group == "object":
            groups[f"eval_object/{case.object}"].append(i)
        elif case.group == "scene":
            groups["eval_scene/all"].append(i)
            split = "training" if case.scene in training_scenes else "held_out"
            groups[f"eval_scene/{split}"].append(i)
        elif case.group == "light":
            groups[f"eval_light/{case.condition}"].append(i)
    result = {}
    for group, indices in groups.items():
        result[f"{group}/episodes"] = len(indices)
        for key, arr in values.items():
            if group != "eval" and key not in outcomes:
                continue
            result[f"{group}/{key}"] = float(arr[indices].mean())
    nominal = result.get("eval_light/nominal/success_once")
    if nominal is not None:
        for condition in sorted({c.condition for c in cases if c.group == "light"}):
            if condition == "nominal":
                continue
            result[f"eval_light/{condition}/success_delta_vs_nominal"] = (
                result[f"eval_light/{condition}/success_once"] - nominal)
    return result


class LightingController:
    """Scale native lights, not pixels/materials/renderers; never compound scales."""

    def __init__(self):
        self._signature = None
        self._original = []

    def apply(self, scene, cases):
        if not any(c.group == "light" for c in cases):
            return
        sub_scenes = list(scene.sub_scenes)
        if getattr(scene, "parallel_in_single_scene", False) or len(sub_scenes) != len(cases):
            raise RuntimeError("lighting evaluation requires isolated per-env render scenes")
        signature = tuple(id(s) for s in sub_scenes)
        if signature != self._signature:
            original = []
            for sub in sub_scenes:
                render = sub.render_system
                ambient = np.asarray(render.ambient_light, float).copy()
                entities = sub.entities
                lights = []
                for entity in entities:
                    for component in entity.components:
                        if type(component).__name__.endswith("LightComponent"):
                            lights.append((component, np.asarray(component.color, float).copy()))
                if not lights:
                    raise RuntimeError("no native lights found; refusing an inert lighting evaluation")
                original.append((render, ambient, lights))
            self._signature, self._original = signature, original
        for case, (render, ambient, lights) in zip(cases, self._original):
            render.ambient_light = ambient * case.intensity
            for light, color in lights:
                light.color = color * case.intensity


class SuccessMilestones:
    """First measured threshold crossings; absent is not zero steps."""

    def __init__(self):
        self.crossed = {}

    def update(self, metrics, step):
        for key, value in list(metrics.items()):
            if not key.endswith("/success_once"):
                continue
            for threshold in (50, 70, 80):
                name = key.rsplit("/", 1)[0] + f"/steps_to_{threshold}"
                if value >= threshold / 100 and name not in self.crossed:
                    self.crossed[name] = int(step)
        return dict(self.crossed)


def check_lighting_reset(base, cases):
    """Verify the matched robot/target states before a lighting rollout begins."""
    nominal = {c.repetition: i for i, c in enumerate(cases)
               if c.group == "light" and c.condition == "nominal"}
    if not nominal:
        return {}

    def cpu(value):
        return np.asarray(value.detach().cpu() if hasattr(value, "detach") else value)

    arrays = {
        "robot_pose": cpu(base.agent.robot.pose.raw_pose),
        "robot_qpos": cpu(base.agent.robot.qpos),
        "robot_qvel": cpu(base.agent.robot.qvel),
        "target_pose": cpu(base.subtask_objs[0].pose.raw_pose),
    }
    maximum = 0.0
    for i, case in enumerate(cases):
        if case.group != "light":
            continue
        ref = nominal[case.repetition]
        if case.scene != cases[ref].scene or case.plan_index != cases[ref].plan_index:
            raise RuntimeError("lighting cases do not name the same scene and task plan")
        for name, arr in arrays.items():
            delta = float(np.max(np.abs(arr[i] - arr[ref])))
            if not np.isfinite(delta) or delta > 1e-4:
                raise RuntimeError(f"lighting comparison has unmatched {name}: {case.condition}/{case.repetition}, delta={delta}")
            maximum = max(maximum, delta)
    return {"eval_light/reset_max_state_difference": maximum}


def lighting_pixel_metrics(obs, cases):
    nominal = {c.repetition: i for i, c in enumerate(cases)
               if c.group == "light" and c.condition == "nominal"}
    if not nominal:
        return {}
    images = [v for k, v in obs.items() if k.startswith("image_")]
    if not images:
        raise RuntimeError("lighting evaluation has no policy RGB images")
    differences = defaultdict(list)
    for i, case in enumerate(cases):
        if case.group != "light" or case.condition == "nominal":
            continue
        for image in images:
            ref = nominal[case.repetition]
            delta = (image[i].float() - image[ref].float()).abs().mean().item()
            differences[case.condition].append(delta)
    result = {}
    for condition, values in differences.items():
        change = float(np.mean(values))
        if not np.isfinite(change) or change <= 0:
            raise RuntimeError(f"lighting condition {condition!r} did not change policy RGB")
        result[f"eval_light/{condition}/reset_rgb_mae"] = change
    for case in cases:
        if case.group == "light":
            result[f"eval_light/{case.condition}/intensity_multiplier"] = case.intensity
    return result
