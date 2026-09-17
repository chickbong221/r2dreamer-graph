"""Graph overlays for annotated episodes.

    python -m real_robot.evaluation.render_annotations --episodes pilot

For each frame: both cameras with every tracked box and named point drawn in
its node's colour (the active target marked), the frame's positive physical
facts and nearby events as text, and the node-link graph drawn by the
repository's own renderer from exactly the graph that gets packed. Writes an
MP4 per episode and a contact sheet of the event frames, so a person can check
labels, boxes and graph against the video in one place.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from ..common import add_config_arguments, episode_name, load_configs, repo_path


def _rgb255(color) -> tuple:
    return tuple(int(round(255 * c)) for c in color)


def draw_camera(rgb: np.ndarray, camera: str, t: int, spec, tracks, annotation, colors) -> np.ndarray:
    from PIL import Image, ImageDraw
    from ..preprocessing.prepare_videos import _font

    image = Image.fromarray(np.asarray(rgb))
    draw = ImageDraw.Draw(image)
    width, height = image.size
    c = spec.cameras.index(camera)
    font = _font(14)
    target = annotation.active_target[t]
    for e, entity in enumerate(spec.entities):
        if not tracks["visible"][t, e, c]:
            continue
        x0, x1, y0, y1 = tracks["boxes"][t, e, c]
        color = _rgb255(colors[entity.node_id])
        thick = 4 if entity.id == target else 2
        draw.rectangle([x0 * width, y0 * height, x1 * width, y1 * height], outline=color, width=thick)
        label = entity.id + (" [target]" if entity.id == target else "")
        quality = float(tracks["quality"][t, e, c])
        if quality < 0.2:
            label += " (held)"
        draw.text((x0 * width + 3, max(0, y0 * height - 16)), label, fill=color, font=font)
    names = [str(v) for v in tracks["point_names"]]
    for p, name in enumerate(names):
        x, y = tracks["points"][t, p, c]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        owner = name.split(":", 1)[0]
        color = _rgb255(colors[spec.entity(owner).node_id])
        cx, cy = x * width, y * height
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=color, outline=(255, 255, 255))
        draw.text((cx + 5, cy - 6), name.split(":", 1)[1], fill=color, font=_font(11))
    draw.rectangle([0, 0, 190, 20], fill=(0, 0, 0))
    draw.text((4, 3), f"{camera}  frame {t}", fill=(255, 255, 0), font=font)
    return np.asarray(image)


def facts_panel(spec, annotation, t: int, width: int, height: int, extra_lines: Sequence[str] = ()) -> np.ndarray:
    from PIL import Image, ImageDraw
    from ..preprocessing.prepare_videos import _font

    image = Image.new("RGB", (width, height), (250, 246, 240))
    draw = ImageDraw.Draw(image)
    font, small = _font(15), _font(13)
    lines = [f"frame {t}   target: {annotation.active_target[t]}"]
    lines += list(extra_lines)
    lines.append("")
    positives = {"holds", "src-holds", "dst-holds"}
    for fact, series, temporal in zip(spec.facts, annotation.absolute, annotation.temporal):
        label = series[t]
        if label is None:
            continue
        if fact.relation in ("contact", "grasp", "support", "contain"):
            if label in positives:
                lines.append(f"{fact.relation}({fact.src}, {fact.dst}) = {label}")
        elif fact.src == "ee" and fact.dst == annotation.active_target[t]:
            change = temporal[t] or "-"
            lines.append(f"{fact.relation}(ee, {fact.dst}) = {label}  [{change}]")
    nearby = [e for e in annotation.events if abs(e["frame"] - t) <= 7]
    if nearby:
        lines.append("")
        lines += [f"event @{e['frame']}: {e['type']} {e['object']}" for e in nearby]
    y = 8
    for line in lines:
        draw.text((10, y), line, fill=(40, 30, 20), font=font if y == 8 else small)
        y += 18
        if y > height - 18:
            break
    return np.asarray(image)


def _resize_height(image: np.ndarray, height: int) -> np.ndarray:
    from PIL import Image

    h, w = image.shape[:2]
    return np.asarray(Image.fromarray(image).resize((max(1, int(round(w * height / h))), height), Image.BILINEAR))


def render_episode(configs, source, episode: int, out_dir: str, graph_every: int = 3,
                   reward: Optional[Mapping[str, np.ndarray]] = None) -> Dict[str, str]:
    from scenegraph.viz.graph_draw import render_graph
    from scenegraph.viz.palette import ColorMap
    from ..graphs.pack import build_frame_graph
    from ..preprocessing.prepare_videos import read_frames, write_video

    spec = source.spec
    annotation = source.annotation(episode)
    tracks = source.tracks(episode)
    geometry = source.geometry(episode)
    frames = {camera: read_frames(source.video_path(episode, camera)) for camera in spec.cameras}
    n = annotation.n_frames
    colors = ColorMap()
    colors.assign_all([e.node_id for e in spec.entities])
    colors_dict = colors.as_dict()

    key_frames = {0, n - 1, *[int(e["frame"]) for e in annotation.events]}
    if annotation.outcome.get("completion_frame", -1) >= 0:
        key_frames.add(int(annotation.outcome["completion_frame"]))
    key_frames = sorted(key_frames)[:12]
    kept: Dict[int, np.ndarray] = {}

    def composites():
        panel = None
        for t in range(n):
            views = [draw_camera(frames[camera][t], camera, t, spec, tracks, annotation, colors_dict)
                     for camera in spec.cameras]
            height = max(v.shape[0] for v in views)
            strip = np.concatenate([_resize_height(v, height) for v in views], axis=1)
            if panel is None or t % max(1, graph_every) == 0 or t in key_frames:
                graph = build_frame_graph(spec, annotation, t, tracks["boxes"][t], tracks["visible"][t],
                                          geometry["centroids"][t], geometry["centroid_known"][t])
                panel = _resize_height(render_graph(graph, None, colormap=colors), height)
            extra = []
            if reward is not None:
                extra.append(f"stage {int(reward['stage'][t])}  S={float(reward['S'][t]):.3f}")
                if t < len(reward["reward"]) and np.isfinite(reward["reward"][t]):
                    extra.append(f"r(t->t+1) = {float(reward['reward'][t]):+.3f}")
            text = facts_panel(spec, annotation, t, 420, height, extra)
            composite = np.concatenate([strip, panel, text], axis=1)
            if composite.shape[1] % 2:
                composite = np.concatenate([composite, np.zeros_like(composite[:, :1])], axis=1)
            if composite.shape[0] % 2:
                composite = np.concatenate([composite, np.zeros_like(composite[:1])], axis=0)
            if t in key_frames:
                kept[t] = composite
            yield composite

    os.makedirs(out_dir, exist_ok=True)
    video = os.path.join(out_dir, f"{episode_name(episode)}_{annotation.mode}_graph.mp4")
    write_video(composites(), video, source.fps())
    tiles = [kept[t] for t in key_frames]
    columns = 2
    rows = [np.concatenate(tiles[i:i + columns] + [np.zeros_like(tiles[0])] * (columns - len(tiles[i:i + columns])),
                           axis=1) for i in range(0, len(tiles), columns)]
    sheet = np.concatenate(rows, axis=0)
    from PIL import Image

    sheet_path = os.path.join(out_dir, f"{episode_name(episode)}_{annotation.mode}_events.png")
    scale = min(1.0, 3000 / sheet.shape[1])
    Image.fromarray(sheet).resize((int(sheet.shape[1] * scale), int(sheet.shape[0] * scale))).save(sheet_path)
    return {"video": video, "sheet": sheet_path}


def main(argv: Optional[Sequence[str]] = None) -> None:
    from ..data.episode_dataset import RawEpisodeSource
    from ..preprocessing.freshness import ArtifactChain, warn_stale
    from ..rewards.kitchen import compute_rewards, load_scales

    parser = argparse.ArgumentParser(description="Render graph overlays.")
    parser.add_argument("--episodes", default="pilot")
    parser.add_argument("--mode", choices=("full_episode", "past_only"), default=None)
    parser.add_argument("--graph-every", type=int, default=3, help="redraw the graph panel every N frames")
    parser.add_argument("--no-reward", action="store_true")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    configs = load_configs(["dataset", "annotation", "graph", "reward"], args.overrides)
    source = RawEpisodeSource(configs, mode=args.mode)
    out_dir = os.path.join(repo_path(configs["dataset"]["paths"]["renders"]), "annotations")
    episodes = source.select(args.episodes)
    warn_stale(ArtifactChain(configs, source), episodes, "geometry", "[render]")
    for episode in episodes:
        reward = None
        if not args.no_reward:
            try:
                scales = load_scales(configs["reward"], required=False)
                result = compute_rewards(source.reward_inputs(episode), scales, configs["reward"])
                reward = {"stage": result.stage, "S": result.S, "reward": result.reward}
            except FileNotFoundError as exc:
                print(f"[render] episode {episode}: no reward overlay ({exc})")
        paths = render_episode(configs, source, episode, out_dir, args.graph_every, reward)
        print(f"[render] episode {episode}: {paths['video']}\n                 {paths['sheet']}", flush=True)


if __name__ == "__main__":
    main()
