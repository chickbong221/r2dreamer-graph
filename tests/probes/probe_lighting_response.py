"""Does the renderer respond to the evaluation's light scaling?

Standalone diagnostic. Three matched sub-scenes -- one task plan, one spawn,
one reset state, no robot actions -- differing only in light intensity.

    # runtime: change the lights of a built scene, re-render, compare each
    # sub-scene against its OWN nominal baseline
    python tests/probes/probe_lighting_response.py --mode runtime --shader-pack minimal

    # construction: build the lights already scaled, against a separately
    # built all-nominal scene with the same seed, plan, spawn and cameras
    python tests/probes/probe_lighting_response.py --mode construction --shader-pack minimal

``shader_dir`` is deliberately left as None: ManiSkill folds it into both
sensor_configs and human_render_camera_configs, which would override the
per-camera shader_pack this probe varies. Camera poses, resolution, scene,
plan and spawn are identical across every run.

Nothing here trains, mines, or writes a config.
"""

import argparse
import contextlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs.evaluation import EvalCase, LightingController, check_lighting_reset

CONDITIONS = (("dim", 0.4), ("nominal", 1.0), ("bright", 2.0))
NOMINAL = [name for name, _ in CONDITIONS].index("nominal")
INTENSITIES = [value for _, value in CONDITIONS]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="runtime",
                        choices=["runtime", "construction"])
    parser.add_argument("--shader-pack", default="minimal",
                        choices=["minimal", "default"])
    parser.add_argument("--out", default="logdir/lighting_probe")
    parser.add_argument("--task-group", default="tidy_house")
    parser.add_argument("--object", default="004_sugar_box")
    parser.add_argument("--scene", default="v3_sc0_staging_00.scene_instance.json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--size", type=int, default=112)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def plans_for(args):
    from mani_skill import ASSET_DIR
    from mshab.envs.planner import plan_data_from_file

    rearrange = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
    plan_path = (rearrange / "task_plans" / args.task_group / "pick"
                 / args.split / f"{args.object}.json")
    plan_data = plan_data_from_file(plan_path)
    plans = [p for p in plan_data.plans
             if str(p.build_config_name) == args.scene]
    if not plans:
        available = sorted({str(p.build_config_name) for p in plan_data.plans})
        raise SystemExit(
            f"{args.object} has no plan in {args.scene!r} "
            f"({len(available)} scenes available, e.g. {available[:3]})")
    return plans, plan_data, rearrange


def build_env(args):
    """One sub-scene per condition, pinned to a single scene's plans."""
    import gymnasium as gym
    import mshab.envs  # noqa: F401 - register MS-HAB tasks

    plans, plan_data, rearrange = plans_for(args)
    return gym.make(
        "PickSubtaskTrain-v0",
        num_envs=len(CONDITIONS),
        obs_mode="rgb+segmentation",
        render_mode="rgb_array",
        reward_mode="normalized_dense",
        sim_backend="gpu",
        max_episode_steps=200,
        reconfiguration_freq=0,
        # None, so the per-camera shader_pack below is what actually applies.
        shader_dir=None,
        sensor_configs=dict(width=args.size, height=args.size,
                            shader_pack=args.shader_pack),
        task_plans=plans,
        scene_builder_cls=plan_data.dataset,
        spawn_data_fp=(rearrange / "spawn_data" / args.task_group / "pick"
                       / args.split / "spawn_data.pt"),
    )


def reset_fixed(env, args):
    """The same scene, plan, spawn and seed every time. Reconfigures."""
    import torch

    base = env.unwrapped
    count = len(CONDITIONS)
    scene_index = int(base.scene_builder.build_config_names_to_idxs[args.scene])
    env.reset(seed=args.seed, options=dict(
        reconfigure=True,
        build_config_idxs=[scene_index] * count,
        task_plan_idxs=torch.zeros(count, dtype=torch.int, device=base.device),
        spawn_selection_idxs=[0] * count,
    ))
    return base


@contextlib.contextmanager
def construction_time_intensities(intensities):
    """Scale the lights as the scene builder creates them, per sub-scene.

    ReplicaCAD builds its own lighting inside ``scene_builder.build``, so the
    env's ``_load_lighting`` hook returns early and patching it would do
    nothing. These two ``ManiSkillScene`` calls are the creation path.
    Positions, materials and every other setting are passed through unchanged.
    """
    from mani_skill.envs.scene import ManiSkillScene

    original_point = ManiSkillScene.add_point_light
    original_directional = ManiSkillScene.add_directional_light
    original_ambient = ManiSkillScene.set_ambient_light
    seen = {"point": 0, "directional": 0, "ambient": 0}

    def per_scene(add, kind):
        # Colour is positional in ManiSkill's own default lighting and keyword
        # in ReplicaCAD's; accept either and touch nothing else.
        def patched(self, *args, **kwargs):
            seen[kind] += 1
            scene_idxs = kwargs.pop("scene_idxs", None)
            indices = (list(range(len(self.sub_scenes)))
                       if scene_idxs is None else list(scene_idxs))
            if "color" in kwargs:
                colour, position = kwargs.pop("color"), None
            elif len(args) >= 2:
                colour, position = args[1], 1
            else:
                raise TypeError(f"{kind} light created without a colour")
            result = None
            for index in indices:
                scaled = np.asarray(colour, float) * intensities[index]
                call = list(args)
                if position is None:
                    result = add(self, *call, color=scaled, scene_idxs=[index],
                                 **kwargs)
                else:
                    call[position] = scaled
                    result = add(self, *call, scene_idxs=[index], **kwargs)
            return result
        return patched

    def patched_ambient(self, color):
        seen["ambient"] += 1
        for index, sub in enumerate(self.sub_scenes):
            sub.render_system.ambient_light = (
                np.asarray(color, float) * intensities[index])

    ManiSkillScene.add_point_light = per_scene(original_point, "point")
    ManiSkillScene.add_directional_light = per_scene(
        original_directional, "directional")
    ManiSkillScene.set_ambient_light = patched_ambient
    try:
        yield seen
    finally:
        ManiSkillScene.add_point_light = original_point
        ManiSkillScene.add_directional_light = original_directional
        ManiSkillScene.set_ambient_light = original_ambient


def read_lights(scene):
    """Every light the controller could touch, per sub-scene. No fixed count."""
    rows = []
    for sub in scene.sub_scenes:
        lights = []
        for entity in sub.entities:
            for component in entity.components:
                if type(component).__name__.endswith("LightComponent"):
                    lights.append({
                        "component": type(component).__name__,
                        "entity": str(entity.name),
                        "color": np.asarray(component.color, float).tolist(),
                    })
        rows.append({
            "ambient": np.asarray(sub.render_system.ambient_light, float).tolist(),
            "lights": lights,
        })
    return rows


def write_error(before, after, intensities):
    """How far the Python-side values are from original x intensity.

    This says the controller wrote what it meant to. It is not evidence that
    the GPU renderer consumed it -- only the pixels can say that.
    """
    worst = 0.0
    for scale, was, now in zip(intensities, before, after):
        worst = max(worst, float(np.max(np.abs(
            np.asarray(now["ambient"]) - np.asarray(was["ambient"]) * scale))))
        if len(now["lights"]) != len(was["lights"]):
            return float("inf")
        for old_light, new_light in zip(was["lights"], now["lights"]):
            worst = max(worst, float(np.max(np.abs(
                np.asarray(new_light["color"])
                - np.asarray(old_light["color"]) * scale))))
    return worst


def state_fingerprint(base):
    """Robot and target state, so two builds can be shown to match."""
    def cpu(value):
        return np.asarray(value.detach().cpu() if hasattr(value, "detach")
                          else value, dtype=float)

    return {
        "robot_pose": cpu(base.agent.robot.pose.raw_pose),
        "robot_qpos": cpu(base.agent.robot.qpos),
        "robot_qvel": cpu(base.agent.robot.qvel),
        "target_pose": cpu(base.subtask_objs[0].pose.raw_pose),
    }


def state_difference(first, second):
    if set(first) != set(second):
        return float("inf")
    return max(float(np.max(np.abs(first[key] - second[key])))
               for key in sorted(first))


def capture(base):
    """A fresh sensor render, not the observation the reset returned."""
    data = base._get_obs_sensor_data()
    return {name: value["rgb"].detach().cpu().numpy()
            for name, value in data.items() if "rgb" in value}


def compare_to_reference(images, reference, repeat):
    """Per camera, per sub-scene: change against that sub-scene's reference."""
    report = {}
    for camera in sorted(images):
        frames = images[camera].astype(np.float32)
        base = reference[camera].astype(np.float32)
        again = repeat[camera].astype(np.float32)
        conditions, noise = {}, {}
        for index, (name, intensity) in enumerate(CONDITIONS):
            diff = np.abs(frames[index] - base[index])
            conditions[name] = {
                "intensity": intensity,
                "mean_brightness": float(frames[index].mean()),
                "reference_mean_brightness": float(base[index].mean()),
                "mae_vs_reference": float(diff.mean()),
                "max_abs_diff_vs_reference": float(diff.max()),
                "changed_channel_fraction": float((diff >= 1).mean()),
                "changed_pixel_fraction": float((diff >= 1).any(axis=-1).mean()),
            }
            # Same state captured twice: whatever moves here is not lighting.
            spread = np.abs(frames[index] - again[index])
            noise[name] = {
                "mae": float(spread.mean()),
                "max_abs_diff": float(spread.max()),
                "changed_channel_fraction": float((spread >= 1).mean()),
            }
        report[camera] = {"conditions": conditions, "render_noise": noise,
                          "shape": list(images[camera].shape)}
    return report


def save_strips(images, out):
    from PIL import Image

    for camera, frames in sorted(images.items()):
        panels, gap = [], None
        for index, _ in enumerate(CONDITIONS):
            panel = np.asarray(frames[index]).astype(np.uint8)
            if gap is None:
                gap = np.full((panel.shape[0], 2, panel.shape[2]), 255, np.uint8)
            panels.extend([panel, gap])
        strip = np.concatenate(panels[:-1], axis=1)
        path = out / f"{camera}_dim_nominal_bright.png"
        Image.fromarray(strip).save(path)
        print(f"[probe] wrote {path}", flush=True)


def render_table(report):
    lines = []
    for camera, entry in report.items():
        floor = entry["render_noise"]["nominal"]["mae"]
        lines.append(f"  {camera}  (render noise MAE = {floor:.6g})")
        lines.append(f"    {'condition':<9}{'mean':>10}{'ref mean':>11}"
                     f"{'MAE':>11}{'max':>8}{'chan>=1':>10}{'px>=1':>10}")
        for name, _ in CONDITIONS:
            values = entry["conditions"][name]
            lines.append(
                f"    {name:<9}{values['mean_brightness']:>10.4g}"
                f"{values['reference_mean_brightness']:>11.4g}"
                f"{values['mae_vs_reference']:>11.4g}"
                f"{values['max_abs_diff_vs_reference']:>8.4g}"
                f"{values['changed_channel_fraction']:>10.4g}"
                f"{values['changed_pixel_fraction']:>10.4g}")
    return "\n".join(lines)


def run_runtime(args, out):
    """Each sub-scene against its own nominal baseline; nominal is the control."""
    env = build_env(args)
    try:
        base = reset_fixed(env, args)
        before_state = state_fingerprint(base)
        baseline = capture(base)
        baseline_repeat = capture(base)

        before = read_lights(base.scene)
        cases = [EvalCase(scene=args.scene, object=args.object, plan_index=0,
                          group="light", condition=name, intensity=value,
                          repetition=0)
                 for name, value in CONDITIONS]
        LightingController().apply(base.scene, cases)
        after = read_lights(base.scene)

        images = capture(base)
        after_state = state_fingerprint(base)
        report = compare_to_reference(images, baseline, baseline_repeat)
        save_strips(images, out)
        np.savez_compressed(
            out / "images.npz",
            **images,
            **{f"{name}_baseline": value for name, value in baseline.items()})

        summary = {
            "mode": "runtime",
            "reference": "each sub-scene's own capture before apply",
            "lights_per_sub_scene": [len(row["lights"]) for row in before],
            "light_components": sorted(
                {light["component"] for row in before for light in row["lights"]}),
            "ambient_before": [row["ambient"] for row in before],
            "ambient_after": [row["ambient"] for row in after],
            "controller_write_max_error": write_error(before, after, INTENSITIES),
            "reset_max_state_difference": check_lighting_reset(base, cases).get(
                "eval_light/reset_max_state_difference"),
            "state_drift_across_apply": state_difference(before_state, after_state),
            "cameras": report,
        }
        print(f"\n[probe] lights/sub-scene={summary['lights_per_sub_scene']}  "
              f"types={summary['light_components']}")
        print(f"[probe] controller wrote original*intensity to within "
              f"{summary['controller_write_max_error']:.3g} "
              f"(Python-side only; not proof the GPU used it)")
        print(f"[probe] physics untouched across apply, max state drift "
              f"{summary['state_drift_across_apply']:.3g}")
        return summary
    finally:
        env.close()


def build_and_capture(args, intensities, label):
    """One env whose lights are already scaled when the scene is built."""
    import gc
    import torch

    print(f"[probe] constructing {label} build with intensities {intensities}",
          flush=True)
    with construction_time_intensities(intensities) as seen:
        env = build_env(args)
        try:
            base = reset_fixed(env, args)
            lights = read_lights(base.scene)
            fingerprint = state_fingerprint(base)
            images = capture(base)
            repeat = capture(base)
        finally:
            env.close()
    del env
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[probe] {label}: intercepted {seen} creation call(s), "
          f"lights/sub-scene={[len(row['lights']) for row in lights]}", flush=True)
    return {"images": images, "repeat": repeat, "lights": lights,
            "state": fingerprint, "intercepted": dict(seen)}


def run_construction(args, out):
    """Scaled-at-build against a separately built, all-nominal scene."""
    scaled = build_and_capture(args, INTENSITIES, "scaled")
    save_strips(scaled["images"], out)
    np.savez_compressed(out / "images.npz", **scaled["images"])
    # Patched the same way with 1.0 everywhere, so the only difference between
    # the two builds is the numbers, not the code path.
    reference = build_and_capture(args, [1.0] * len(CONDITIONS), "nominal")
    np.savez_compressed(
        out / "images.npz", **scaled["images"],
        **{f"{name}_reference": value
           for name, value in reference["images"].items()})

    report = compare_to_reference(scaled["images"], reference["images"],
                                  scaled["repeat"])
    drift = state_difference(scaled["state"], reference["state"])
    summary = {
        "mode": "construction",
        "reference": "a separately built all-nominal scene, same seed and plan",
        "intercepted_creation_calls": {
            "scaled": scaled["intercepted"], "nominal": reference["intercepted"]},
        "lights_per_sub_scene": [len(row["lights"]) for row in scaled["lights"]],
        "light_components": sorted({light["component"]
                                    for row in scaled["lights"]
                                    for light in row["lights"]}),
        "ambient_scaled": [row["ambient"] for row in scaled["lights"]],
        "ambient_nominal": [row["ambient"] for row in reference["lights"]],
        "light_colors_scaled": [[light["color"] for light in row["lights"]]
                                for row in scaled["lights"]],
        "light_colors_nominal": [[light["color"] for light in row["lights"]]
                                 for row in reference["lights"]],
        "state_difference_between_builds": drift,
        "cameras": report,
    }
    print(f"\n[probe] lights/sub-scene={summary['lights_per_sub_scene']}  "
          f"types={summary['light_components']}")
    print(f"[probe] ambient built as {summary['ambient_scaled']} "
          f"vs nominal {summary['ambient_nominal']}")
    print(f"[probe] the two builds' reset states differ by {drift:.3g}")
    return summary


def main():
    args = parse_args()
    out = Path(args.out) / args.shader_pack / args.mode
    out.mkdir(parents=True, exist_ok=True)

    runner = run_runtime if args.mode == "runtime" else run_construction
    summary = runner(args, out)
    summary.update(shader_pack=args.shader_pack, scene=args.scene,
                   object=args.object, size=args.size, seed=args.seed,
                   conditions={name: value for name, value in CONDITIONS})
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(render_table(summary["cameras"]))
    print(f"[probe] wrote {out / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
