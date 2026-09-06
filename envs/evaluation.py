"""Fixed MS-HAB evaluation panels and per-sub-scene illumination.

B and C share one vector simulator and one policy batch. No extra normal
evaluation is constructed. The panel is also the source of metric grouping.
"""

from collections import defaultdict
import contextlib
from dataclasses import dataclass
import re

import numpy as np

# Two sub-scenes at a matched state, with no illumination difference at all,
# were measured to differ by ~0.008 MAE over ~2% of pixels; a working
# 0.4x/2.0x build moves ~40 MAE over ~100%. These sit far above the first and
# far below the second, so neither noise nor a real change is near them.
LIGHTING_MIN_RGB_MAE = 1.0
LIGHTING_MIN_CHANGED_PIXEL_FRACTION = 0.5


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


def evaluation_video_rows(cases, training_scenes):
    rows = {"eval/video": 0}
    if not cases:
        return rows
    training = set(training_scenes)
    nominal = [i for i, c in enumerate(cases)
               if c.group != "light" and c.intensity == 1.0]
    rows["eval/video"] = next(
        (i for i in nominal if cases[i].scene in training), 0)
    unseen = next((i for i in nominal if cases[i].scene not in training), None)
    if unseen is not None:
        rows["eval/unseen_scene"] = unseen
    dim = [i for i, c in enumerate(cases)
           if c.group == "light" and c.intensity < 1.0 and c.scene in training]
    if dim:
        rows["eval/dim_light"] = min(dim, key=lambda i: cases[i].intensity)
    return rows


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


def case_intensities(cases):
    """Per sub-scene intensity, or empty when no condition changes the light."""
    scales = [float(c.intensity) for c in cases]
    return [] if all(v == 1.0 for v in scales) else scales


@contextlib.contextmanager
def construction_lighting(intensities):
    """Scale each sub-scene's lights as the scene builder creates them.

    Not afterwards: the GPU render group binds lights when it is built and
    refreshes only poses from then on, so a colour written after the first
    render is never read. ReplicaCAD also creates its own lighting inside
    ``scene_builder.build``, so the env's ``_load_lighting`` hook never runs
    and is not the place either.

    Positions, shadows and every other argument pass through untouched.
    Scaling happens once per created light, from the dataset's own value, so
    a later rebuild rescales the original rather than compounding. The patch
    lives on the scene class for one construction and is always restored.
    """
    from mani_skill.envs.scene import ManiSkillScene

    scales = [float(v) for v in intensities]
    if not scales or any(not np.isfinite(v) or v <= 0 for v in scales):
        raise ValueError(f"invalid construction light intensities: {intensities}")
    created = {"point": 0, "directional": 0, "ambient": 0}
    originals = {name: getattr(ManiSkillScene, name)
                 for name in ("add_point_light", "add_directional_light",
                              "set_ambient_light")}

    def scale_for(index):
        if index >= len(scales):
            raise RuntimeError(
                f"construction lighting has {len(scales)} intensities but the "
                f"scene reached sub-scene {index}; the hook is not scoped to "
                "the environment it was opened for")
        return scales[index]

    def per_scene(add, kind):
        # Colour is a keyword in ReplicaCAD's calls and positional in
        # ManiSkill's own default lighting. Accept either.
        def patched(self, *args, **kwargs):
            chosen = kwargs.pop("scene_idxs", None)
            indices = (list(range(len(self.sub_scenes)))
                       if chosen is None else list(chosen))
            if "color" in kwargs:
                colour, slot = kwargs.pop("color"), None
            elif len(args) >= 2:
                colour, slot = args[1], 1
            else:
                raise TypeError(f"{kind} light created without a colour")
            result = None
            for index in indices:
                created[kind] += 1
                scaled = np.asarray(colour, float) * scale_for(index)
                call = list(args)
                if slot is None:
                    result = add(self, *call, color=scaled,
                                 scene_idxs=[index], **kwargs)
                else:
                    call[slot] = scaled
                    result = add(self, *call, scene_idxs=[index], **kwargs)
            return result
        return patched

    def patched_ambient(self, color):
        for index, sub in enumerate(self.sub_scenes):
            created["ambient"] += 1
            sub.render_system.ambient_light = (
                np.asarray(color, float) * scale_for(index))

    ManiSkillScene.add_point_light = per_scene(
        originals["add_point_light"], "point")
    ManiSkillScene.add_directional_light = per_scene(
        originals["add_directional_light"], "directional")
    ManiSkillScene.set_ambient_light = patched_ambient
    try:
        yield created
    finally:
        for name, function in originals.items():
            setattr(ManiSkillScene, name, function)


def _sub_scene_lights(sub):
    """This sub-scene's ambient colour and every light colour on it."""
    ambient = np.asarray(sub.render_system.ambient_light, float).reshape(-1)
    colours = []
    for entity in sub.entities:
        for component in entity.components:
            if type(component).__name__.endswith("LightComponent"):
                colours.append(np.asarray(component.color, float).reshape(-1))
    return ambient, colours


def verify_construction_lighting(scene, cases):
    """The built scene carries the intended intensities, relative to nominal.

    Ratios against a nominal sub-scene, so this needs no record of the
    dataset's original colours and cannot be fooled by a rebuild. It reads
    the scene rather than the pixels; ``lighting_pixel_metrics`` is what says
    the renderer used them.
    """
    scales = case_intensities(cases)
    if not scales:
        return {}
    sub_scenes = list(scene.sub_scenes)
    if getattr(scene, "parallel_in_single_scene", False) or len(sub_scenes) != len(cases):
        raise RuntimeError("lighting evaluation requires isolated per-env render scenes")
    reference = next((i for i, v in enumerate(scales) if v == 1.0), None)
    if reference is None:
        raise RuntimeError("lighting evaluation has no nominal sub-scene to measure against")
    ambient, colours = _sub_scene_lights(sub_scenes[reference])
    if not colours:
        raise RuntimeError("no native lights found; refusing an inert lighting evaluation")
    worst = 0.0
    for index, scale in enumerate(scales):
        built_ambient, built_colours = _sub_scene_lights(sub_scenes[index])
        if not built_colours or len(built_colours) != len(colours):
            raise RuntimeError(
                f"sub-scene {index} carries {len(built_colours)} light(s), "
                f"nominal carries {len(colours)}")
        for want, got in zip([ambient] + colours, [built_ambient] + built_colours):
            usable = np.abs(want) > 1e-6
            if not usable.any():
                continue
            ratio = got[usable] / want[usable]
            worst = max(worst, float(np.max(np.abs(ratio - scale))))
    if worst > 1e-3:
        raise RuntimeError(
            f"the built scene does not carry the intended light intensities "
            f"(worst relative error {worst:.4g}); construction-time lighting "
            "did not apply")
    return {"eval_light/construction_intensity_max_error": worst}


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
    differences, changed, brightness = (defaultdict(list) for _ in range(3))
    for i, case in enumerate(cases):
        if case.group != "light":
            continue
        ref = nominal[case.repetition]
        for image in images:
            frame = image[i].float()
            brightness[case.condition].append(frame.mean().item())
            if case.intensity == 1.0:
                continue
            delta = (frame - image[ref].float()).abs()
            differences[case.condition].append(delta.mean().item())
            changed[case.condition].append(
                (delta >= 1).any(dim=-1).float().mean().item())
    result = {}
    for condition, values in brightness.items():
        result[f"eval_light/{condition}/reset_mean_brightness"] = float(np.mean(values))
    for condition, values in differences.items():
        change = float(np.mean(values))
        fraction = float(np.mean(changed[condition]))
        result[f"eval_light/{condition}/reset_rgb_mae"] = change
        result[f"eval_light/{condition}/reset_changed_pixel_fraction"] = fraction
        # A positive MAE is too weak: two sub-scenes differ slightly whatever
        # the lighting does. Demand a change no cross-sub-scene artifact
        # reaches.
        if (not np.isfinite(change) or change < LIGHTING_MIN_RGB_MAE
                or fraction < LIGHTING_MIN_CHANGED_PIXEL_FRACTION):
            raise RuntimeError(
                f"lighting condition {condition!r} barely changed policy RGB: "
                f"mae={change:.4g} (needs >= {LIGHTING_MIN_RGB_MAE}), changed "
                f"pixels={fraction:.4g} (needs >= "
                f"{LIGHTING_MIN_CHANGED_PIXEL_FRACTION}). A difference this "
                "small is sub-scene variation, not illumination.")
    for case in cases:
        if case.group == "light":
            result[f"eval_light/{case.condition}/intensity_multiplier"] = case.intensity
    return result
