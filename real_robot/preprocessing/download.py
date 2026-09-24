"""Download the LeRobot snapshot at a pinned revision.

    python -m real_robot.preprocessing.download

The configured revision (``main`` by default) is resolved to a commit sha once,
downloaded at that sha, and recorded with a checksum for every file in
``source.json``. A later run that resolves to a different sha refuses to
overwrite the snapshot: annotations and packed graphs refer to the frames of
one revision.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence

from ..common import (
    add_config_arguments,
    file_sha256,
    load_configs,
    read_json,
    repo_path,
    stable_hash,
    utc_now,
    write_json,
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Download the dataset at a pinned revision.")
    parser.add_argument("--force", action="store_true",
                        help="allow replacing a snapshot pinned at a different revision")
    parser.add_argument("--no-checksums", action="store_true",
                        help="skip sha256 of every file (sizes are still recorded)")
    add_config_arguments(parser)
    args = parser.parse_args(argv)

    from huggingface_hub import HfApi, snapshot_download

    cfg = load_configs(["dataset"], args.overrides)["dataset"]
    source = cfg["source"]
    root = repo_path(cfg["paths"]["source"])
    record_path = os.path.join(root, "source.json")

    info = HfApi().dataset_info(source["repo_id"], revision=source["revision"])
    sha = info.sha
    if os.path.isfile(record_path):
        pinned = read_json(record_path)["resolved_revision"]
        if pinned != sha and not args.force:
            raise SystemExit(
                f"{root} is pinned at {pinned}, but {source['revision']!r} now resolves to {sha}. "
                "Everything built so far refers to the pinned frames; pass --force to replace it."
            )
    print(f"[download] {source['repo_id']} @ {source['revision']} -> {sha}", flush=True)
    snapshot_download(repo_id=source["repo_id"], repo_type="dataset", revision=sha, local_dir=root)

    meta = read_json(os.path.join(root, "meta", "info.json"))
    problems = []
    if int(meta["fps"]) != int(source["fps"]):
        problems.append(f"fps is {meta['fps']}, config says {source['fps']}")
    features = meta.get("features", {})
    for name, column in source["fields"].items():
        if column not in features:
            problems.append(f"field {name!r} -> {column!r} is not a dataset feature")
    for camera, key in source["cameras"].items():
        if key not in features or features[key].get("dtype") != "video":
            problems.append(f"camera {camera!r} -> {key!r} is not a video feature")
    if problems:
        raise SystemExit("[download] the snapshot does not match dataset.yaml:\n  " + "\n  ".join(problems))

    files = {}
    for directory, _, names in os.walk(root):
        if ".cache" in directory.split(os.sep) or ".huggingface" in directory.split(os.sep):
            continue
        for name in sorted(names):
            path = os.path.join(directory, name)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            if rel == "source.json":
                continue
            entry = {"size": os.path.getsize(path)}
            if not args.no_checksums:
                entry["sha256"] = file_sha256(path)
            files[rel] = entry
    write_json(record_path, {
        "repo_id": source["repo_id"],
        "requested_revision": source["revision"],
        "resolved_revision": sha,
        "downloaded": utc_now(),
        "codebase_version": meta.get("codebase_version"),
        "robot_type": meta.get("robot_type"),
        "total_episodes": int(meta["total_episodes"]),
        "total_frames": int(meta["total_frames"]),
        "fps": meta["fps"],
        "files": files,
        "files_digest": stable_hash(files),
    })
    print(f"[download] {len(files)} files recorded in {record_path}", flush=True)


if __name__ == "__main__":
    main()
