"""Decode the source videos and write the copies Gemini reads.

    python -m real_robot.preprocessing.prepare_videos --episodes pilot

The LeRobot videos are AV1. Each camera is decoded with PyAV, checked
frame-for-frame against the episode's recorded rows, and re-encoded as H.264
with the camera name, the frame number and the timestamp burned into every
frame. Gemini reads the frame number off the image; that, not its own notion
of time, is what makes a frame-exact interval boundary possible.

The decoding helpers here are also what tracking, geometry and the dataset
build use to read source frames, so every stage agrees on frame indexing.

``index.json`` records, per camera, the digest of the source video a copy was
made from and the digest of the copy itself. A copy is reused only while the
video settings, the episode's rows and frame rate, the source video and the
copy on disk all still match that record; otherwise it is made again.
:func:`prepared_status` answers the same question without making anything,
for stages that only need to know whether an annotation's videos are current.
"""

from __future__ import annotations

import argparse
import os
from fractions import Fraction
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..common import (
    add_config_arguments,
    episode_name,
    load_configs,
    read_json,
    repo_path,
    stable_hash,
    utc_now,
    write_json,
)


# --------------------------------------------------------------------------- #
# Video I/O
# --------------------------------------------------------------------------- #
def video_metadata(path: str) -> Dict[str, object]:
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        return {
            "codec": stream.codec_context.name,
            "width": int(stream.codec_context.width),
            "height": int(stream.codec_context.height),
            "frames_declared": int(stream.frames or 0),
            "average_rate": float(stream.average_rate) if stream.average_rate else None,
        }


def iter_frames(path: str) -> Iterator[Tuple[int, Optional[float], np.ndarray]]:
    """``(index, pts_seconds, rgb uint8 HxWx3)`` in decode order."""
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for index, frame in enumerate(container.decode(stream)):
            seconds = float(frame.pts * stream.time_base) if frame.pts is not None else None
            yield index, seconds, frame.to_ndarray(format="rgb24")


def read_frames(path: str, indices: Optional[Sequence[int]] = None) -> np.ndarray:
    """All frames, or the requested subset in the requested order."""
    if indices is None:
        return np.stack([rgb for _, _, rgb in iter_frames(path)])
    wanted = {int(i) for i in indices}
    found: Dict[int, np.ndarray] = {}
    last = max(wanted) if wanted else -1
    for index, _, rgb in iter_frames(path):
        if index in wanted:
            found[index] = rgb
        if index >= last:
            break
    missing = sorted(wanted - set(found))
    if missing:
        raise IndexError(f"{path}: frames {missing[:10]} are beyond the end of the video")
    return np.stack([found[int(i)] for i in indices])


def frame_times(path: str) -> List[Optional[float]]:
    return [seconds for _, seconds, _ in iter_frames(path)]


def write_video(frames: Iterable[np.ndarray], path: str, fps: float, codec: str = "libx264",
                crf: int = 18) -> int:
    """Encode RGB frames; returns the number written. Falls back to mpeg4."""
    import av

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".partial.mp4"
    iterator = iter(frames)
    first = next(iterator, None)
    if first is None:
        raise ValueError(f"no frames to write to {path}")
    height, width = first.shape[:2]
    rate = Fraction(fps).limit_denominator(1000)
    count = 0
    container = av.open(tmp, mode="w")
    try:
        try:
            stream = container.add_stream(codec, rate=rate)
        except Exception:
            stream = container.add_stream("mpeg4", rate=rate)
        stream.width, stream.height = int(width) - int(width) % 2, int(height) - int(height) % 2
        stream.pix_fmt = "yuv420p"
        if stream.codec_context.name in ("libx264", "h264"):
            stream.options = {"crf": str(int(crf))}
        for rgb in _chain(first, iterator):
            rgb = np.ascontiguousarray(rgb[: stream.height, : stream.width])
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)
            count += 1
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    os.replace(tmp, path)
    return count


def _chain(first, rest):
    yield first
    yield from rest


def _font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans-Bold.ttf", "DejaVuSansMono-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def overlay_text(rgb: np.ndarray, text: str, font_size: int = 18) -> np.ndarray:
    """Burn one line of text into the top-left corner on a dark backing box."""
    from PIL import Image, ImageDraw

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font = _font(font_size)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    pad = max(3, font_size // 4)
    draw.rectangle([0, 0, right - left + 2 * pad, bottom - top + 2 * pad], fill=(0, 0, 0))
    draw.text((pad - left, pad - top), text, fill=(255, 255, 0), font=font)
    return np.asarray(image)


def frame_label(camera: str, index: int, total: int, fps: float) -> str:
    return f"{camera.upper()}  F {index:04d}/{total - 1:04d}  t {index / fps:06.2f}s"


def trim_video(source: str, out: str, start_frame: int, end_frame: int, fps: float,
               codec: str = "libx264", crf: int = 18) -> str:
    """Frames ``start_frame..end_frame`` (inclusive) of an already-labelled video."""
    def frames():
        for index, _, rgb in iter_frames(source):
            if index > end_frame:
                break
            if index >= start_frame:
                yield rgb

    write_video(frames(), out, fps, codec, crf)
    return out


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #
def prepared_video_path(dataset_cfg, episode: int, camera: str) -> str:
    return os.path.join(repo_path(dataset_cfg["paths"]["videos"]), episode_name(episode), f"{camera}.mp4")


def video_settings(video_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    return {"overlay": bool(video_cfg["overlay"]), "codec": video_cfg["codec"],
            "crf": int(video_cfg["crf"]), "font_size": int(video_cfg["font_size"])}


def index_path(dataset_cfg, episode: int) -> str:
    return os.path.join(repo_path(dataset_cfg["paths"]["videos"]), episode_name(episode), "index.json")


def prepared_status(source, episode: int, video_cfg: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """``(index, "current")`` when every prepared copy is current, else ``(None, why not)``. Makes nothing."""
    from .artifacts import file_digest, source_video_digest

    dataset_cfg = source.dataset_cfg
    path = index_path(dataset_cfg, episode)
    if not os.path.isfile(path):
        return None, "not prepared yet"
    record = read_json(path)
    if record.get("settings_hash") != stable_hash(video_settings(video_cfg)):
        return None, "the video settings changed"
    if int(record.get("rows", -1)) != int(source.lengths()[episode]) or float(record.get("fps", -1)) != source.fps():
        return None, "the episode's rows or frame rate changed"
    cameras = record.get("cameras") or {}
    for camera in dataset_cfg["source"]["cameras"]:
        entry = cameras.get(camera)
        if entry is None:
            return None, f"no {camera} copy was made"
        if entry.get("source_sha256") is None:
            return None, f"the {camera} copy does not record its source video"
        if entry["source_sha256"] != source_video_digest(source, episode, camera):
            return None, f"the {camera} source video changed"
        digest = file_digest(prepared_video_path(dataset_cfg, episode, camera))
        if digest is None:
            return None, f"the {camera} copy is missing"
        if digest != entry.get("sha256"):
            return None, f"the {camera} copy changed on disk"
    return record, "current"


def prepare_episode(source, episode: int, video_cfg, force: bool = False) -> Dict[str, object]:
    from .artifacts import file_digest, source_video_digest

    dataset_cfg = source.dataset_cfg
    fps = source.fps()
    rows = source.lengths()[episode]
    settings = video_settings(video_cfg)
    if not force:
        existing, reason = prepared_status(source, episode, video_cfg)
        if existing is not None:
            return existing
        if os.path.isfile(index_path(dataset_cfg, episode)):
            print(f"[videos] episode {episode}: making the copies again ({reason})", flush=True)
    record: Dict[str, object] = {"episode_index": episode, "rows": rows, "fps": fps,
                                 "settings": settings, "settings_hash": stable_hash(settings),
                                 "created": utc_now(), "cameras": {}}
    for camera in dataset_cfg["source"]["cameras"]:
        src = source.video_path(episode, camera)
        out = prepared_video_path(dataset_cfg, episode, camera)
        total = rows

        def labelled():
            for index, _, rgb in iter_frames(src):
                yield overlay_text(rgb, frame_label(camera, index, total, fps), settings["font_size"]) \
                    if settings["overlay"] else rgb

        written = write_video(labelled(), out, fps, settings["codec"], settings["crf"])
        if written != rows:
            raise RuntimeError(
                f"episode {episode} {camera}: decoded {written} frames but the episode has {rows} "
                "rows; video and table do not align"
            )
        record["cameras"][camera] = {"path": os.path.relpath(out, repo_path("")).replace(os.sep, "/"),
                                     "frames": written, "source": os.path.basename(src),
                                     "source_sha256": source_video_digest(source, episode, camera),
                                     "sha256": file_digest(out)}
    write_json(index_path(dataset_cfg, episode), record)
    return record


def main(argv: Optional[Sequence[str]] = None) -> None:
    from ..data.episode_dataset import RawEpisodeSource

    parser = argparse.ArgumentParser(description="Write frame-labelled H.264 copies for Gemini.")
    parser.add_argument("--episodes", default="pilot")
    parser.add_argument("--force", action="store_true")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    configs = load_configs(["dataset", "annotation", "graph"], args.overrides)
    source = RawEpisodeSource(configs)
    episodes = source.select(args.episodes)
    for episode in episodes:
        record = prepare_episode(source, episode, configs["annotation"]["videos"], force=args.force)
        print(f"[videos] episode {episode}: " + ", ".join(
            f"{cam}={info['frames']}" for cam, info in record["cameras"].items()), flush=True)


if __name__ == "__main__":
    main()
