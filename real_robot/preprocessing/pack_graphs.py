"""Pack validated annotations into per-episode graph arrays.

    python -m real_robot.preprocessing.pack_graphs --episodes all --name so101_v1

Writes ``outputs/graphs/<name>/episode_XXXXXX.npz`` -- the repository's packed
graph arrays stacked over the episode's frames, ``graph_valid``, and the
LeRobot ``episode_index``, ``frame_index`` and ``index`` of every row -- and
``manifest.json`` with the vocabularies and what each episode was packed from.
Join them to the LeRobot data on ``index`` (or episode and frame index).
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..common import (
    add_config_arguments,
    canonical_json,
    episode_name,
    load_configs,
    read_json,
    repo_path,
    stable_hash,
    utc_now,
    write_json,
)
from ..graphs.pack import pack_episode
from ..graphs.validate import EpisodeAnnotation
from ..graphs.vocabulary import build_vocab, vocab_sizes, vocab_tables
from .annotate_episode import EpisodeAnnotator
from .prepare_videos import prepared_status

MANIFEST_FORMAT = "real_robot/so101-graphs-v1"


def contract(source, vocab) -> Dict[str, Any]:
    """What every episode in one named set must share."""
    try:
        record = source.source_record()
        origin = {"repo_id": record.get("repo_id"), "revision": record.get("resolved_revision")}
    except FileNotFoundError:
        origin = {"repo_id": source.configs["dataset"]["source"]["repo_id"], "revision": None}
    return {"format": MANIFEST_FORMAT, "graph": source.graph.identity(), "vocab": vocab_tables(vocab),
            "source": origin, "labels": stable_hash(source.configs["labels"]),
            "source_config": stable_hash(source.configs["dataset"]["source"]),
            "annotation_config": stable_hash(source.configs["annotation"]),
            "graph_config": stable_hash(source.configs["graph"]),
            "prompt": EpisodeAnnotator(source.configs, source).prompt_hash}


def current_annotation(annotator, episode, data):
    """Reject a stale or misplaced answer before joining it to training rows."""
    source = annotator.source
    if data.get("episode_index") != episode:
        raise ValueError("annotation belongs to a different episode")
    if data.get("n_frames") != source.lengths()[episode] or data.get("fps") != source.fps():
        raise ValueError("annotation length or frame rate differs from the source")
    prepared, reason = prepared_status(source, episode, annotator.annotation_cfg["videos"])
    if prepared is None:
        raise ValueError(f"prepared videos are not current: {reason}")
    spec = source.spec(episode)
    wanted = annotator.input_identity(spec, prepared)
    if canonical_json(data.get("input_identity")) != canonical_json(wanted):
        raise ValueError("annotation inputs changed; run annotate_episode again")
    annotation = EpisodeAnnotation.from_json(spec, data)
    if not annotation.valid:
        raise ValueError(f"invalid annotation, {len(annotation.issues)} issue(s)")
    return annotation


def main(argv: Optional[Sequence[str]] = None) -> None:
    from ..data.source import LeRobotSource

    parser = argparse.ArgumentParser(description="Pack annotated episodes into graph arrays.")
    parser.add_argument("--episodes", default="all")
    parser.add_argument("--name", required=True)
    parser.add_argument("--force", action="store_true",
                        help="replace a set packed under a different graph contract")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    if args.name in (".", "..") or any(c in args.name for c in '/\\:'):
        parser.error("--name must be a single directory name")
    configs = load_configs(["dataset", "annotation", "graph", "labels"], args.overrides)
    source = LeRobotSource(configs)
    annotator = EpisodeAnnotator(configs, source)
    vocab = build_vocab(source.graph)
    out_dir = os.path.join(repo_path(configs["dataset"]["paths"]["graphs"]), args.name)
    manifest_path = os.path.join(out_dir, "manifest.json")
    wanted = contract(source, vocab)

    manifest: Dict[str, Any] = {}
    if os.path.isfile(manifest_path):
        manifest = read_json(manifest_path)
        stored = {key: manifest.get(key) for key in wanted}
        if canonical_json(stored) != canonical_json(wanted):
            if not args.force:
                raise SystemExit(f"{out_dir} was packed under a different graph contract; pass --force to replace "
                                 "it or choose another --name")
            # Replace the manifest membership; old unlisted artifacts are not training data.
            manifest = {}

    episodes: Dict[str, Any] = dict(manifest.get("episodes") or {})
    skipped: Dict[str, Any] = dict(manifest.get("skipped") or {})
    annotations = repo_path(configs["dataset"]["paths"]["annotations"])
    for episode in source.select(args.episodes):
        key = str(episode)
        episodes.pop(key, None)
        path = os.path.join(annotations, episode_name(episode) + ".json")
        if not os.path.isfile(path):
            skipped[key] = "not annotated"
            print(f"[pack] episode {episode}: skipped, not annotated", flush=True)
            continue
        spec = source.spec(episode)
        try:
            data = read_json(path)
            annotation = current_annotation(annotator, episode, data)
        except (ValueError, KeyError, TypeError) as exc:
            skipped[key] = str(exc)
            print(f"[pack] episode {episode}: skipped, {skipped[key]}", flush=True)
            continue
        arrays, valid = pack_episode(spec, vocab, annotation)
        n = annotation.n_frames
        index = source.global_index(episode)
        if len(index) != n:
            raise SystemExit(f"episode {episode}: {n} annotated frames but {len(index)} dataset rows")
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, episode_name(episode) + ".npz"), **arrays, graph_valid=valid,
                            episode_index=np.full(n, episode, dtype=np.int64),
                            frame_index=np.arange(n, dtype=np.int64), index=index)
        episodes[key] = {"task": spec.task, "n_frames": n, "valid_frames": int(valid.sum()),
                         "file": episode_name(episode) + ".npz", "annotation": data.get("input_hash"),
                         "answer_hash": stable_hash(data["answer"])}
        skipped.pop(key, None)
        print(f"[pack] episode {episode} ({spec.task}): {n} frames, {int(valid.sum())} complete", flush=True)

    first = next(iter(episodes.values()), None)
    shapes = {}
    if first is not None:
        with np.load(os.path.join(out_dir, first["file"])) as data:
            shapes = {name: {"dtype": str(data[name].dtype), "per_frame": list(data[name].shape[1:])}
                      for name in data.files}
    write_json(manifest_path, {
        **wanted,
        "name": args.name,
        "updated": utc_now(),
        "vocab_sizes": vocab_sizes(vocab),
        "cameras": list(source.graph.cameras),
        "box_format": "graph_node_bbox is [x0, x1, y0, y1] per camera, normalised to the image; zero when not visible",
        "centroids": "unknown; graph_node_centroid is zero",
        "arrays": shapes,
        "episodes": dict(sorted(episodes.items(), key=lambda item: int(item[0]))),
        "skipped": dict(sorted(skipped.items(), key=lambda item: int(item[0]))),
    })
    print(f"[pack] {len(episodes)} episode(s) in {out_dir}; {len(skipped)} skipped", flush=True)


if __name__ == "__main__":
    main()
