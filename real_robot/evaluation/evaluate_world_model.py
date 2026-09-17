"""World-model diagnostics for one checkpoint, on the diagnostic episodes.

    python -m real_robot.evaluation.evaluate_world_model --world-model wm_base
    python -m real_robot.evaluation.evaluate_world_model --world-model wm_base --checkpoint best_diagnostic

The same report training logs periodically (see ``world_model_diagnostics``):
losses over fixed windows, reward and continuation heads, one-step and short
open-loop prediction, and burn-in sensitivity. The diagnostic episodes are
training episodes: the report describes fitting and pipeline behaviour, not
generalisation, and says so in its ``note``.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence

import torch

from ..common import add_config_arguments, load_configs, repo_path, utc_now, write_json
from ..data.episode_dataset import BuiltEpisodeStore
from ..models.world_model import load_world_model
from ..training.encode_dataset import resolve_checkpoint
from .world_model_diagnostics import DiagnosticWindows, run_diagnostics


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Diagnostics for a pretrained world model.")
    parser.add_argument("--world-model", required=True, help="run name under runs/world_model, or a checkpoint path")
    parser.add_argument("--checkpoint", default="final", help="final | latest | best_diagnostic | step_XXXXXXXX")
    parser.add_argument("--episodes", default="diagnostic", help="diagnostic, or explicit ids such as 3,17,42")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    configs = load_configs(["dataset", "world_model"], args.overrides)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = resolve_checkpoint(configs["dataset"], args.world_model, args.checkpoint)
    model, payload, manifest = load_world_model(checkpoint, device)
    store = BuiltEpisodeStore(manifest.root)
    cfg = payload["meta"]["world_model_config"]
    if args.episodes == "diagnostic":
        episodes = manifest.diagnostic_episodes() or list(payload["meta"].get("diagnostic_episodes", []))
    else:
        built = set(manifest.all_episodes())
        episodes = [int(e) for e in args.episodes.split(",") if e.strip()]
        missing = [e for e in episodes if e not in built]
        if missing:
            raise SystemExit(f"episodes {missing} are not packed in {manifest.root}")
    if not episodes:
        raise SystemExit("no diagnostic episodes are packed in this dataset")

    sequence, diagnostics = cfg["sequence"], cfg["diagnostics"]
    windows = DiagnosticWindows(store, episodes, int(sequence["burn_in"]), int(sequence["length"]),
                                int(sequence["batch_size"]), int(diagnostics["windows"]), int(diagnostics["seed"]))
    report = run_diagnostics(model, store, windows, diagnostics, device)
    report.update({
        "created": utc_now(),
        "world_model": {"checkpoint": checkpoint, "kind": payload.get("kind"), "step": int(payload["step"]),
                        "selection": payload.get("selection")},
        "dataset": {"root": manifest.root, "identity": manifest.identity_hash},
    })
    run = args.world_model if not os.path.isfile(repo_path(args.world_model)) else "checkpoints"
    out = os.path.join(repo_path(configs["dataset"]["paths"]["runs"]), "world_model", run,
                       f"diagnostics_{payload.get('kind', 'checkpoint')}_step{int(payload['step']):08d}.json")
    write_json(out, report)
    print(f"[eval] {os.path.basename(checkpoint)} at step {payload['step']}: {len(episodes)} diagnostic episode(s), "
          f"{len(windows.draws)} windows")
    print(f"  note: {report['note']}")
    for key in sorted(report["scalars"]):
        print(f"  {key:60s} {report['scalars'][key]:.4f}")
    print(f"[eval] -> {out}")


if __name__ == "__main__":
    main()
