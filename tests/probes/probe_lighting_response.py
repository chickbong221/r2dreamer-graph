"""Does the renderer respond to the evaluation's light scaling?

Standalone diagnostic. Three matched sub-scenes -- one task plan, one spawn,
one reset state, no robot actions -- differing only in the intensity the
shipped ``LightingController`` applies. Policy-camera images are captured
fresh after the lights are changed, never read back from the reset.

Run once per shader pack and compare the two reports:

    python tests/probes/probe_lighting_response.py --shader-pack minimal
    python tests/probes/probe_lighting_response.py --shader-pack default

``shader_dir`` is deliberately left as None: ManiSkill folds it into both
sensor_configs and human_render_camera_configs, which would override the
per-camera shader_pack this probe is trying to vary. Camera poses, resolution,
scene, plan and spawn are identical across both runs.

Nothing here trains, mines, or writes a config.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs.evaluation import EvalCase, LightingController, check_lighting_reset

CONDITIONS = (("dim", 0.4), ("nominal", 1.0), ("bright", 2.0))
NOMINAL = [name for name, _ in CONDITIONS].index("nominal")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
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


def build_env(args):
    """One env per condition, pinned to a single scene's plans."""
    import gymnasium as gym
    import mshab.envs  # noqa: F401 - register MS-HAB tasks
    from mani_skill import ASSET_DIR
    from mshab.envs.planner import plan_data_from_file

    rearrange = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
    subtask = "pick"
    plan_path = (rearrange / "task_plans" / args.task_group / subtask
                 / args.split / f"{args.object}.json")
    plan_data = plan_data_from_file(plan_path)
    plans = [p for p in plan_data.plans
             if str(p.build_config_name) == args.scene]
    if not plans:
        available = sorted({str(p.build_config_name) for p in plan_data.plans})
        raise SystemExit(
            f"{args.object} has no plan in {args.scene!r} "
            f"({len(available)} scenes available, e.g. {available[:3]})")
    print(f"[probe] {args.object} in {args.scene}: {len(plans)} plan(s), "
          f"using index 0", flush=True)

    env = gym.make(
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
        spawn_data_fp=(rearrange / "spawn_data" / args.task_group / subtask
                       / args.split / "spawn_data.pt"),
    )
    return env


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


def write_error(before, after, cases):
    """How far the Python-side values are from original x intensity.

    This says the controller wrote what it meant to. It is not evidence that
    the GPU renderer consumed it -- only the pixels can say that.
    """
    worst = 0.0
    for case, was, now in zip(cases, before, after):
        scale = float(case.intensity)
        worst = max(worst, float(np.max(np.abs(
            np.asarray(now["ambient"]) - np.asarray(was["ambient"]) * scale))))
        if len(now["lights"]) != len(was["lights"]):
            return float("inf")
        for old_light, new_light in zip(was["lights"], now["lights"]):
            worst = max(worst, float(np.max(np.abs(
                np.asarray(new_light["color"])
                - np.asarray(old_light["color"]) * scale))))
    return worst


def capture(base):
    """A fresh sensor render, not the observation the reset returned."""
    data = base._get_obs_sensor_data()
    return {name: value["rgb"].detach().cpu().numpy()
            for name, value in data.items() if "rgb" in value}


def compare(images, repeat):
    """Per camera: brightness, change against nominal, and the noise floor."""
    report = {}
    for camera in sorted(images):
        frames = images[camera].astype(np.float32)
        again = repeat[camera].astype(np.float32)
        reference = frames[NOMINAL]
        conditions = {}
        for index, (name, intensity) in enumerate(CONDITIONS):
            frame = frames[index]
            diff = np.abs(frame - reference)
            conditions[name] = {
                "intensity": intensity,
                "mean_brightness": float(frame.mean()),
                "mae_vs_nominal": float(diff.mean()),
                "max_abs_diff_vs_nominal": float(diff.max()),
                "changed_channel_fraction": float((diff >= 1).mean()),
                "changed_pixel_fraction": float((diff >= 1).any(axis=-1).mean()),
            }
        # Same state captured twice: whatever moves here is not lighting.
        noise = {}
        for index, (name, _) in enumerate(CONDITIONS):
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


def render_table(report, noise_key="nominal"):
    lines = []
    for camera, entry in report.items():
        floor = entry["render_noise"][noise_key]["mae"]
        lines.append(f"  {camera}  (render noise MAE = {floor:.6g})")
        lines.append(f"    {'condition':<9}{'mean':>10}{'MAE':>11}"
                     f"{'max':>8}{'chan>=1':>10}{'px>=1':>9}")
        for name, _ in CONDITIONS:
            values = entry["conditions"][name]
            lines.append(
                f"    {name:<9}{values['mean_brightness']:>10.4g}"
                f"{values['mae_vs_nominal']:>11.4g}"
                f"{values['max_abs_diff_vs_nominal']:>8.4g}"
                f"{values['changed_channel_fraction']:>10.4g}"
                f"{values['changed_pixel_fraction']:>9.4g}")
    return "\n".join(lines)


def main():
    import torch

    args = parse_args()
    out = Path(args.out) / args.shader_pack
    out.mkdir(parents=True, exist_ok=True)

    env = build_env(args)
    try:
        base = env.unwrapped
        count = len(CONDITIONS)
        scene_index = int(base.scene_builder.build_config_names_to_idxs[args.scene])
        env.reset(seed=args.seed, options=dict(
            reconfigure=True,
            build_config_idxs=[scene_index] * count,
            task_plan_idxs=torch.zeros(count, dtype=torch.int,
                                       device=base.device),
            spawn_selection_idxs=[0] * count,
        ))

        cases = [EvalCase(scene=args.scene, object=args.object, plan_index=0,
                          group="light", condition=name, intensity=value,
                          repetition=0)
                 for name, value in CONDITIONS]

        before = read_lights(base.scene)
        LightingController().apply(base.scene, cases)
        after = read_lights(base.scene)
        # Raises unless the three sub-scenes share a robot and target state.
        matched = check_lighting_reset(base, cases)

        images = capture(base)
        repeat = capture(base)
        report = compare(images, repeat)
        save_strips(images, out)
        np.savez_compressed(out / "images.npz", **images)

        summary = {
            "shader_pack": args.shader_pack,
            "scene": args.scene,
            "object": args.object,
            "size": args.size,
            "seed": args.seed,
            "conditions": {name: value for name, value in CONDITIONS},
            "lights_per_sub_scene": [len(row["lights"]) for row in before],
            "light_components": sorted(
                {light["component"] for row in before for light in row["lights"]}),
            "ambient_before": [row["ambient"] for row in before],
            "ambient_after": [row["ambient"] for row in after],
            "controller_write_max_error": write_error(before, after, cases),
            "reset_max_state_difference": matched.get(
                "eval_light/reset_max_state_difference"),
            "cameras": report,
        }
        (out / "metrics.json").write_text(json.dumps(summary, indent=2))

        print(f"\n[probe] shader_pack={args.shader_pack}  "
              f"lights/sub-scene={summary['lights_per_sub_scene']}  "
              f"types={summary['light_components']}")
        print(f"[probe] controller wrote original*intensity to within "
              f"{summary['controller_write_max_error']:.3g} "
              f"(Python-side only; not proof the GPU used it)")
        difference = summary["reset_max_state_difference"]
        print("[probe] matched reset state, max difference "
              + ("unavailable" if difference is None else f"{difference:.3g}"))
        print(render_table(report))
        print(f"[probe] wrote {out / 'metrics.json'}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
