"""Draw an episode's annotation over its videos, to check it by eye.

    python -m real_robot.evaluation.render_annotations --episodes pilot

Writes ``outputs/renders/episode_XXXXXX.mp4``: the cameras side by side with
every entity's interpolated box, and below them the frame, the active target,
the physical facts that hold and the gripper-to-target distance and height.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Sequence

import numpy as np

from ..common import add_config_arguments, episode_name, load_configs, read_json, repo_path
from ..graphs.pack import episode_boxes
from ..graphs.validate import EpisodeAnnotation

PALETTE = [(255, 80, 80), (255, 210, 0), (80, 200, 255), (160, 255, 120), (255, 120, 255), (255, 160, 60),
           (120, 140, 255), (220, 220, 220)]
PHYSICAL = ("contact", "grasp", "support", "contain")


def panel_lines(spec, annotation: EpisodeAnnotation, t: int) -> List[str]:
    target = annotation.active_target[t]
    lines = [f"F {t:04d}   target: {target or '?'}"]
    held = []
    for index, fact in enumerate(spec.facts):
        label = annotation.absolute[index][t]
        if fact.relation in PHYSICAL and label not in ("not-holds", None):
            held.append(f"{fact.relation}({fact.src},{fact.dst})={label}")
        if target and (fact.src, fact.dst) == ("ee", target) and fact.relation in ("planar-distance", "height-offset"):
            change = annotation.temporal[index][t]
            lines.append(f"{fact.relation}(ee,{target}) = {label or '?'}" + (f", {change}" if change else ""))
    lines.append("holds: " + ("  ".join(held) if held else "-"))
    return lines


def render_episode(source, spec, annotation: EpisodeAnnotation, out_path: str, font_size: int = 16) -> int:
    from PIL import Image, ImageDraw

    from ..preprocessing.prepare_videos import _font, write_video

    font = _font(font_size)
    boxes, visible = episode_boxes(spec, annotation)
    streams = [source.frames(annotation.episode_index, camera) for camera in spec.cameras]

    def frames():
        for t, items in enumerate(zip(*streams)):
            images = []
            for c, (_, rgb) in enumerate(items):
                image = Image.fromarray(rgb)
                draw = ImageDraw.Draw(image)
                height, width = rgb.shape[:2]
                for e, entity in enumerate(spec.entities):
                    if not visible[t, e, c]:
                        continue
                    x0, x1, y0, y1 = boxes[t, e, c] * np.array([width, width, height, height])
                    color = PALETTE[e % len(PALETTE)]
                    draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
                    draw.text((x0 + 3, y0 + 2), entity.id, fill=color, font=font)
                images.append(np.asarray(image))
            row = np.concatenate(images, axis=1)
            lines = panel_lines(spec, annotation, t)
            panel = Image.new("RGB", (row.shape[1], (font_size + 6) * len(lines) + 8), (0, 0, 0))
            draw = ImageDraw.Draw(panel)
            for i, line in enumerate(lines):
                draw.text((6, 4 + i * (font_size + 6)), line, fill=(255, 255, 255), font=font)
            yield np.concatenate([row, np.asarray(panel)], axis=0)

    return write_video(frames(), out_path, source.fps())


def main(argv: Optional[Sequence[str]] = None) -> None:
    from ..data.source import LeRobotSource

    parser = argparse.ArgumentParser(description="Render annotations over the episode videos.")
    parser.add_argument("--episodes", default="pilot")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    configs = load_configs(["dataset", "annotation", "graph"], args.overrides)
    source = LeRobotSource(configs)
    paths = configs["dataset"]["paths"]
    for episode in source.select(args.episodes):
        path = os.path.join(repo_path(paths["annotations"]), episode_name(episode) + ".json")
        if not os.path.isfile(path):
            print(f"[render] episode {episode}: not annotated", flush=True)
            continue
        spec = source.spec(episode)
        annotation = EpisodeAnnotation.from_json(spec, read_json(path))
        out = os.path.join(repo_path(paths["renders"]), episode_name(episode) + ".mp4")
        count = render_episode(source, spec, annotation, out)
        status = "valid" if annotation.valid else f"invalid, {len(annotation.issues)} issue(s)"
        print(f"[render] episode {episode} ({status}): {count} frames -> {out}", flush=True)


if __name__ == "__main__":
    main()
