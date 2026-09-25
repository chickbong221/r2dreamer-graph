"""Everything a --data real task needs before training, doing only what is missing.

    python -m sim_vla.data.prepare_real --task stackcube --lerobot /data/so101-multitask

1. The LeRobot snapshot, downloaded at the revision the committed graphs were
   annotated on. Needs internet; skipped once its ``source.json`` exists.
2. The sim_vla dataset (``task.dataset``), converted again whenever the graphs
   or the image size changed, and left alone otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .convert_real import IMAGE_SIZE, ConversionError, Graphs, convert, stale_reason


def ensure_snapshot(root: Path, repo_id: str, revision: str) -> None:
    record = root / "source.json"
    if record.is_file():
        pinned = json.loads(record.read_text(encoding="utf-8")).get("resolved_revision")
        if pinned != revision:
            raise SystemExit(f"{root} holds revision {pinned}, but the graphs were annotated on "
                             f"{revision}; pass a different --lerobot directory")
        print(f"[prepare_real] snapshot {root} @ {revision[:12]}: present", flush=True)
        return
    from real_robot.preprocessing.download import main as download

    print(f"[prepare_real] downloading {repo_id} @ {revision[:12]} -> {root}", flush=True)
    download(["--set", f"dataset.source.repo_id='{repo_id}'",
              "--set", f"dataset.source.revision='{revision}'",
              "--set", f"dataset.paths.source='{root.resolve().as_posix()}'"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Download and convert what a --data real task needs, skipping what is current")
    parser.add_argument("--task", required=True, help="a --data real task, e.g. stackcube or cubes_in_cup")
    parser.add_argument("--lerobot", default=None, help="where the LeRobot snapshot lives; default task.source.lerobot")
    parser.add_argument("--graphs", default=None, help="the packed graphs; default task.source.graphs")
    parser.add_argument("--out", default=None, help="the .h5 to write; default task.dataset")
    parser.add_argument("--image-size", type=int, nargs=2, default=list(IMAGE_SIZE), metavar=("H", "W"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    from ..config import load_config

    args = parse_args(argv)
    task = load_config(args.task, "dreamer", data="real")["task"]
    source = dict(task.get("source") or {})
    graphs_dir = Path(args.graphs or source["graphs"])
    lerobot = Path(args.lerobot or source["lerobot"])
    out = Path(args.out or task["dataset"])
    try:
        origin = Graphs(graphs_dir).manifest.get("source") or {}
        ensure_snapshot(lerobot, str(origin["repo_id"]), str(origin["revision"]))
        reason = stale_reason(out, graphs_dir, args.image_size)
        if reason is None:
            print(f"[prepare_real] {out} is up to date", flush=True)
            return 0
        print(f"[prepare_real] converting: {reason}", flush=True)
        convert(env_id=str(task["env_id"]), lerobot=lerobot, graphs_dir=graphs_dir, out=out,
                size=args.image_size, overwrite=True)
    except ConversionError as exc:
        raise SystemExit(f"[prepare_real] {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
