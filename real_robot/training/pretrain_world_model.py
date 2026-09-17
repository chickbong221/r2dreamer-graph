"""Pretrain the offline world model on every recorded episode.

    python -m real_robot.training.pretrain_world_model --run-name wm_base
    python -m real_robot.training.pretrain_world_model --run-name wm_base --resume

Burn-in plus learning windows are drawn from every training episode -- all of
them -- each window contiguous frames of one episode. Periodically, without
gradient updates, the model is measured on fixed windows and episodes of the
diagnostic selection: losses, reward and continuation heads, one-step and
short open-loop prediction, and burn-in sensitivity, logged as
``diagnostic/...``. Those episodes are training episodes, so the numbers show
fitting and pipeline behaviour, not generalisation; the run's config and every
diagnostic report say so.

Checkpoints: ``latest.pt`` for resumption, ``step_XXXXXXXX.pt`` snapshots,
``final.pt`` (the default for everything downstream), and
``best_diagnostic.pt``, the lowest diagnostic model loss, labelled as such.
Only world-model parameters exist in this process, so nothing here can update a
policy.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence

import torch
from torch.amp import GradScaler, autocast

import tools

from ..common import RunLogger, add_config_arguments, load_configs, repo_path, utc_now, write_json
from ..data.episode_dataset import BuiltEpisodeStore
from ..data.selection import DIAGNOSTIC_NOTE
from ..data.sequence_dataset import SequenceSampler
from ..evaluation.world_model_diagnostics import DiagnosticWindows, run_diagnostics
from ..models.world_model import (
    OfflineWorldModel,
    compose_model_config,
    make_optimizer,
    to_torch,
    world_model_identity,
)
from .checkpoints import CheckpointManager, CheckpointSelection, restore_rng

CONSOLE = ("loss/model", "loss/rew", "loss/dyn", "loss/relabs", "reward/mae", "cont/accuracy",
           "diagnostic/loss/model", "diagnostic/reward/mae", "diagnostic/one_step/reward_abs", "opt/grad_skipped")


def main(argv: Optional[Sequence[str]] = None) -> None:
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Offline world-model pretraining.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-partial-dataset", action="store_true",
                        help="train although some training episodes are not packed (smoke tests only)")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    configs = load_configs(["dataset", "world_model"], args.overrides)
    cfg = configs["world_model"]
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    tools.set_seed_everywhere(int(cfg["seed"]))
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    store = BuiltEpisodeStore(cfg["dataset"])
    manifest = store.manifest
    coverage = manifest.require_complete(args.allow_partial_dataset, "pretrain_world_model")
    training_episodes = manifest.training_episodes()
    diagnostic_episodes = manifest.diagnostic_episodes()
    diagnostic_source = "selection"
    if not diagnostic_episodes:
        if not args.allow_partial_dataset:
            raise SystemExit(f"no diagnostic episode of selection {manifest.selection_version} is packed")
        diagnostic_episodes = training_episodes
        diagnostic_source = "every packed episode (partial dataset: no diagnostic episode is packed)"
        print(f"[wm] {diagnostic_source}", flush=True)

    model_config = compose_model_config(cfg, manifest, str(device))
    saved_config = OmegaConf.to_container(model_config, resolve=True)
    identity = world_model_identity(manifest, model_config, cfg, partial_dataset=not coverage["complete"])
    model = OfflineWorldModel(model_config, manifest, cfg["observation_keys"]).to(device)
    optimizer, scheduler, clip = make_optimizer(model, model_config)
    scaler = GradScaler(enabled=device.type == "cuda" and model.amp_dtype == torch.float16)

    run_dir = os.path.join(repo_path(configs["dataset"]["paths"]["runs"]), "world_model", args.run_name)
    manager = CheckpointManager(run_dir, identity)
    diag_cfg = cfg["diagnostics"]
    select_cfg = diag_cfg["select_checkpoint"]
    selection = CheckpointSelection(manager, select_cfg["label"], select_cfg["metric"], select_cfg["mode"],
                                    note=DIAGNOSTIC_NOTE)
    meta = {"dataset_root": manifest.root, "model_config": saved_config,
            "observation_keys": dict(cfg["observation_keys"]), "world_model_config": cfg,
            "training_episodes": training_episodes, "diagnostic_episodes": diagnostic_episodes,
            "diagnostic_episodes_in_training": True, "diagnostic_source": diagnostic_source}
    start = 0
    if args.resume:
        payload = manager.resume()
        if payload is None:
            raise SystemExit(f"--resume: no latest.pt in {run_dir}")
        model.load_state_dict(payload["state"]["model"])
        optimizer.load_state_dict(payload["state"]["optimizer"])
        scheduler.load_state_dict(payload["state"]["scheduler"])
        scaler.load_state_dict(payload["state"]["scaler"])
        restore_rng(payload["rng"])
        start = int(payload["step"])
        print(f"[wm] resumed from step {start}", flush=True)
    elif os.path.isfile(manager.path("latest")) or os.path.isfile(manager.path("final")):
        raise SystemExit(f"{run_dir} already has a run; pass --resume or choose another --run-name")
    write_json(os.path.join(run_dir, "config.json"), {
        "world_model": cfg, "model": saved_config, "identity": identity, "created": utc_now(),
        "coverage": coverage, "training_episodes": training_episodes,
        "diagnostic_episodes": diagnostic_episodes, "diagnostic_episodes_in_training": True,
        "diagnostic_source": diagnostic_source, "note": DIAGNOSTIC_NOTE,
    })

    sequence = cfg["sequence"]
    burn_in, length = int(sequence["burn_in"]), int(sequence["length"])
    train = SequenceSampler(store, training_episodes, burn_in, length, sequence["batch_size"],
                            seed=int(cfg["seed"]) + start,
                            begin_fraction=float(sequence["start_at_episode_begin_fraction"]))
    windows = DiagnosticWindows(store, diagnostic_episodes, burn_in, length, int(sequence["batch_size"]),
                                int(diag_cfg["windows"]), int(diag_cfg["seed"]))
    train_cfg = cfg["train"]
    logger = RunLogger(run_dir, CONSOLE)
    report_dir = os.path.join(run_dir, "diagnostics")

    def state():
        return {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict()}

    steps = int(train_cfg["steps"])
    snapshot_every = int(train_cfg.get("snapshot_every") or 0)
    model.train()
    skipped = 0
    for step in range(start, steps):
        batch = to_torch(train.sample(), device)
        with autocast(device_type=device.type, dtype=model.amp_dtype, enabled=device.type == "cuda"):
            total, metrics, _ = model.compute_losses(batch, burn_in)
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        clip()
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= before:
            scheduler.step()
        else:
            skipped += 1
        optimizer.zero_grad(set_to_none=True)
        done = step + 1

        if done % int(train_cfg["log_every"]) == 0:
            values = {key: float(value) for key, value in metrics.items()}
            values["opt/lr"] = scheduler.get_last_lr()[0]
            values["opt/grad_skipped"] = float(skipped)
            logger.write(done, values)
        if done % int(diag_cfg["every"]) == 0 or done == steps:
            report = run_diagnostics(model, store, windows, diag_cfg, device)
            model.train()
            write_json(os.path.join(report_dir, f"step_{done:08d}.json"), {**report, "step": done})
            logger.write(done, report["scalars"])
            if selection.update(done, report["scalars"], state, meta):
                print(f"[wm] step {done}: {selection.label}.pt <- {selection.metric}={selection.value:.4f} "
                      "(diagnostic episodes are trained on)", flush=True)
        if done % int(train_cfg["checkpoint_every"]) == 0:
            manager.save("latest", done, state(), meta)
        if snapshot_every and done % snapshot_every == 0:
            manager.snapshot(done, state(), meta)
    manager.save("latest", steps, state(), meta)
    manager.save("final", steps, state(), meta)
    logger.close()
    print(f"[wm] finished {steps} updates -> {run_dir} (default checkpoint: final.pt)", flush=True)


if __name__ == "__main__":
    main()
