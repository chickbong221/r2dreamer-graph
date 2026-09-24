"""Decode each episode's frames and write the copies Gemini reads.

    python -m real_robot.preprocessing.prepare_videos --episodes pilot

The LeRobot videos are AV1, many episodes per file. Each camera's segment is
decoded with PyAV, checked frame for frame against the episode's length, and
every ``stride``-th frame is re-encoded as H.264 at ``videos.fps`` with the
camera name, the recorded frame number and the time burned in. Gemini reads
the frame number off the image, so its intervals come back in recorded frames.

``index.json`` records, per camera, the source file's digest and span and the
copy's digest. A copy is reused only while all of those and the video settings
still match; otherwise it is made again.
"""

from __future__ import annotations

import argparse
import os
from fractions import Fraction
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

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
def iter_frames(path: str) -> Iterator[Tuple[int, Optional[float], np.ndarray]]:
    """``(index, pts_seconds, rgb uint8 HxWx3)`` in decode order."""
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for index, frame in enumerate(container.decode(stream)):
            seconds = float(frame.pts * stream.time_base) if frame.pts is not None else None
            yield index, seconds, frame.to_ndarray(format="rgb24")


def iter_segment(path: str, start_s: float, end_s: float, count: int, fps: float
                 ) -> Iterator[Tuple[int, np.ndarray]]:
    """``(frame_index, rgb)`` for the ``count`` frames in ``[start_s, end_s)`` of a concatenated video."""
    import av

    half = 0.5 / float(fps)
    expected = 0
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        container.seek(max(0, int((start_s - 1.0) / stream.time_base)), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            seconds = float(frame.pts * stream.time_base)
            if seconds < start_s - half:
                continue
            if seconds >= end_s - half or expected >= count:
                break
            index = int(round((seconds - start_s) * fps))
            if index != expected:
                raise ValueError(f"{path}: expected frame {expected} of the segment at {start_s:.3f}s, "
                                 f"decoded one at {seconds:.3f}s (frame {index})")
            yield expected, frame.to_ndarray(format="rgb24")
            expected += 1
    if expected != count:
        raise ValueError(f"{path}: {expected} frames in [{start_s:.3f}s, {end_s:.3f}s), the episode has {count}")


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
def stride_for(source_fps: float, video_cfg: Mapping[str, Any]) -> int:
    return max(1, int(round(float(source_fps) / float(video_cfg["fps"]))))


def prepared_video_path(dataset_cfg, episode: int, camera: str) -> str:
    return os.path.join(repo_path(dataset_cfg["paths"]["videos"]), episode_name(episode), f"{camera}.mp4")


def index_path(dataset_cfg, episode: int) -> str:
    return os.path.join(repo_path(dataset_cfg["paths"]["videos"]), episode_name(episode), "index.json")


def video_settings(video_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    return {"sampling_version": 2, "fps": float(video_cfg["fps"]), "overlay": bool(video_cfg["overlay"]), "codec": video_cfg["codec"],
            "crf": int(video_cfg["crf"]), "font_size": int(video_cfg["font_size"])}


def source_identity(source, episode: int, camera: str) -> Dict[str, Any]:
    from .artifacts import source_video_digest

    path = source.video_path(episode, camera)
    return {"file": os.path.relpath(path, source.root).replace(os.sep, "/"),
            "sha256": source_video_digest(source, episode, camera),
            "span": list(source.video_span(episode, camera))}


def prepared_status(source, episode: int, video_cfg: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """``(index, "current")`` when every prepared copy is current, else ``(None, why not)``. Makes nothing."""
    from .artifacts import file_digest

    dataset_cfg = source.dataset_cfg
    path = index_path(dataset_cfg, episode)
    if not os.path.isfile(path):
        return None, "not prepared yet"
    record = read_json(path)
    if record.get("settings_hash") != stable_hash(video_settings(video_cfg)):
        return None, "the video settings changed"
    if int(record.get("rows", -1)) != int(source.lengths()[episode]) or \
            float(record.get("source_fps", -1)) != source.fps():
        return None, "the episode's length or frame rate changed"
    cameras = record.get("cameras") or {}
    for camera in source.cameras():
        entry = cameras.get(camera)
        if entry is None:
            return None, f"no {camera} copy was made"
        if entry.get("source") != source_identity(source, episode, camera):
            return None, f"the {camera} source video changed"
        digest = file_digest(prepared_video_path(dataset_cfg, episode, camera))
        if digest is None:
            return None, f"the {camera} copy is missing"
        if digest != entry.get("sha256"):
            return None, f"the {camera} copy changed on disk"
    return record, "current"


def prepare_episode(source, episode: int, video_cfg: Mapping[str, Any], force: bool = False) -> Dict[str, Any]:
    from .artifacts import file_digest

    dataset_cfg = source.dataset_cfg
    fps = source.fps()
    rows = source.lengths()[episode]
    settings = video_settings(video_cfg)
    stride = stride_for(fps, video_cfg)
    if not force:
        existing, reason = prepared_status(source, episode, video_cfg)
        if existing is not None:
            return existing
        if os.path.isfile(index_path(dataset_cfg, episode)):
            print(f"[videos] episode {episode}: making the copies again ({reason})", flush=True)
    record: Dict[str, Any] = {"episode_index": episode, "rows": rows, "source_fps": fps, "stride": stride,
                              "fps": fps / stride, "shown": len(range(0, rows, stride)) + int((rows - 1) % stride != 0), "settings": settings,
                              "settings_hash": stable_hash(settings), "created": utc_now(), "cameras": {}}
    for camera in source.cameras():
        out = prepared_video_path(dataset_cfg, episode, camera)

        def shown():
            for index, rgb in source.frames(episode, camera):
                if index % stride and index != rows - 1:
                    continue
                yield overlay_text(rgb, frame_label(camera, index, rows, fps), settings["font_size"]) \
                    if settings["overlay"] else rgb

        written = write_video(shown(), out, fps / stride, settings["codec"], settings["crf"])
        if written != record["shown"]:
            raise RuntimeError(f"episode {episode} {camera}: wrote {written} frames, expected {record['shown']}")
        try:
            stored_path = os.path.relpath(out, repo_path(""))
        except ValueError:  # Windows output and repository on different drives.
            stored_path = os.path.abspath(out)
        record["cameras"][camera] = {"path": stored_path.replace(os.sep, "/"),
                                     "frames": written, "source": source_identity(source, episode, camera),
                                     "sha256": file_digest(out)}
    write_json(index_path(dataset_cfg, episode), record)
    return record


def main(argv: Optional[Sequence[str]] = None) -> None:
    from ..data.source import LeRobotSource

    parser = argparse.ArgumentParser(description="Write frame-labelled H.264 copies for Gemini.")
    parser.add_argument("--episodes", default="pilot")
    parser.add_argument("--force", action="store_true")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    configs = load_configs(["dataset", "annotation", "graph"], args.overrides)
    source = LeRobotSource(configs)
    minimum = int(configs["annotation"]["annotation"]["min_frames"])
    for episode in source.select(args.episodes):
        if source.lengths()[episode] < minimum:
            print(f"[videos] episode {episode}: skipped, {source.lengths()[episode]} frames < {minimum}", flush=True)
            continue
        record = prepare_episode(source, episode, configs["annotation"]["videos"], force=args.force)
        print(f"[videos] episode {episode}: {record['shown']} of {record['rows']} frames per camera at "
              f"{record['fps']:g} fps", flush=True)


if __name__ == "__main__":
    main()
