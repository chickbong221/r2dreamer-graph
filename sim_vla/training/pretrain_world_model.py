"""Stage 1A: train one arm's world model on the demonstrations.

Full recurrent dynamics, not an image autoencoder: the reward and continuation
heads are trained alongside reconstruction, the recurrent state is rebuilt with
burn-in, and the graph arm additionally trains the semantic prior that
imagination depends on.

The two arms get separate world models and are never initialised from each
other. ``sim_vla/runtime/checkpoint.py`` enforces that for the checkpoints that
do get written: a graph-trained world model reloaded as a baseline is not a
baseline, because the graph has already reached ``h`` and ``z``.

**Nothing is written to disk by default.** :func:`run` returns the trained
model as an object, and ``sim_vla/training/pipeline.py`` hands that same object
to Stage 1B in the same process. Pass ``--save-checkpoints`` when a run is long
enough that losing it would matter.

    python -m sim_vla.training.pretrain_world_model
        --task pickcube --experiment graph --steps 50000

This is training, not testing. ``sim_vla/run_tests.sh`` never calls it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..config import check_dataset_compatibility, load_config
from ..data.batch import to_model_batch
from ..data.dataset import DemoDataset
from ..data.normalization import fit_normalizer
from ..data.sequences import SequenceSampler
from ..models.model_config import DEFAULT_MODEL, load_model_config
from ..models.world_model import build_world_model
from ..runtime.checkpoint import CheckpointMeta, save


@dataclass
class Stage1A:
    """What Stage 1A produces, in memory.

    ``model`` is the object Stage 1B trains against -- not a reload of a file,
    because by default there is no file. ``data`` is still open: the caller
    owns it, since the next stage samples from the same dataset and reopening
    it would mean a second read of the same episodes to refit statistics that
    are already here.
    """

    model: Any
    data: Any
    sampler: Any
    model_cfg: Any
    normalizer: Any
    meta: CheckpointMeta
    coords: Any = None
    losses: Dict[str, float] = field(default_factory=dict)
    path: Optional[Path] = None      # where it was written, or None


def observation_shapes(batch: Dict[str, torch.Tensor]) -> Dict[str, tuple]:
    """Per-key shapes without the batch and time axes."""
    return {key: tuple(value.shape[2:]) for key, value in batch.items()
            if hasattr(value, "dim") and value.dim() >= 2}


def seed_everything(seed: int) -> None:
    """One seed for model initialisation and every torch sampler.

    The data sampler was seeded and nothing else was, so two arms of the same
    experiment started from different weights and drew different noise -- a
    difference the comparison would have attributed to the graph.
    """
    import random

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build(cfg: Dict[str, Any], *, device: str,
          model_yaml: Path | None = None):
    """The dataset, the sampler and the arm's world model, wired together.

    On any failure after the dataset is opened, the dataset is closed here.
    Ownership transfers to the caller only on success.
    """
    graph_enabled = bool(cfg["model"]["graph"]["enabled"])
    seed_everything(int(cfg["data"]["seed"]))
    data = DemoDataset(
        cfg["task"]["dataset"], graph_enabled=graph_enabled,
        ignore_terminations=bool(cfg["data"]["ignore_terminations"]))
    try:
        check_dataset_compatibility(cfg, data.metadata)

        # The camera count comes from the recording, not from the model preset:
        # the packed bbox is (n_max, n_cams, 4) and the encoder is 5*n_cams+3
        # wide.
        recorded = dict(data.metadata.get("graph") or {})
        if graph_enabled and recorded.get("n_cams"):
            cfg["model"]["graph"]["n_cams"] = int(recorded["n_cams"])
        cfg["device"] = device

        sampler = SequenceSampler(
            data, length=int(cfg["data"]["sequence_length"]),
            burn_in=int(cfg["data"]["burn_in"]), seed=int(cfg["data"]["seed"]))
        model_cfg = load_model_config(
            cfg, model_yaml=Path(model_yaml) if model_yaml is not None
            else DEFAULT_MODEL)

        probe = to_model_batch(sampler.batch(2), device)
        model = build_world_model(
            model_cfg, observation_shapes(probe),
            int(probe["action"].shape[-1]),
            graph_enabled=graph_enabled).to(device)
    except BaseException:
        data.close()
        raise
    return data, sampler, model, model_cfg


def run(cfg: Dict[str, Any], *, steps: int, device: str,
        out: Optional[Path] = None, save_checkpoint: bool = False,
        model_yaml: Path | None = None, log_every: int = 100) -> Stage1A:
    """Train one arm's world model and hand it back live.

    ``save_checkpoint`` is off by default. The stages run in one process and
    the next one takes ``Stage1A.model`` directly, so a checkpoint here buys
    the ability to resume -- not the ability to continue.
    """
    from ..models.action_space import ActionBounds, ActionCoordinates

    data, sampler, model, model_cfg = build(
        cfg, device=device, model_yaml=model_yaml)
    try:
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=float(model_cfg.lr))

        from ..models.action_space import FieldScaler

        mode = str(cfg["data"].get("normalization") or "mean_std")
        if mode not in FieldScaler.MODES + ("none",):
            raise SystemExit(
                f"data.normalization={mode!r} is not implemented; this "
                f"pipeline supports {list(FieldScaler.MODES)} or 'none'. A "
                "mode that is read and ignored would train on different units "
                "than it claims.")

        # Fitted once from the demonstrations both arms share. It travels to
        # the later stages as an object; it is written beside the weights only
        # when there are weights on disk for it to sit beside.
        normalizer = None
        if mode != "none":
            normalizer = fit_normalizer(data)
            # The mode belongs to the normalizer, so the numpy path and the
            # tensor path read it from the same place.
            normalizer.mode = mode
        if normalizer is not None and save_checkpoint and out is not None:
            normalizer.save(Path(out).with_name("normalization.json"))

        action_dim = int(sampler.batch(1)["action_target"].shape[-1])
        coords = ActionCoordinates(
            normalizer,
            ActionBounds.from_metadata(data.metadata, action_dim),
            device=device, action_dim=action_dim)

        last: Dict[str, float] = {}
        for step in range(int(steps)):
            batch = to_model_batch(
                sampler.batch(int(cfg["data"]["batch_size"])), device,
                normalizer=normalizer, coords=coords)
            total, losses, _aux = model.loss(batch)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optimizer.step()
            last = {name: float(value.detach())
                    for name, value in losses.items()}
            if log_every and step % int(log_every) == 0:
                print(f"[world_model] step {step} total {float(total):.4f} "
                      + " ".join(f"{k}={v:.3f}"
                                 for k, v in sorted(last.items())[:5]),
                      flush=True)

        meta = CheckpointMeta(
            graph_enabled=bool(cfg["model"]["graph"]["enabled"]),
            stage="world_model",
            env_id=str(cfg["task"]["env_id"]),
            feature_dim=int(model.feature_dim),
            dataset_identity={"dataset": str(cfg["task"]["dataset"]),
                              "episodes": len(data)},
            # The mode and a fingerprint of the fitted numbers, not just which
            # dataset they came from: two fits over the same episodes can
            # produce weights that cannot be read against each other.
            normalization_identity=(normalizer.descriptor()
                                    if normalizer is not None
                                    else {"mode": "none"}),
            config=cfg, step=int(steps),
        )

        path: Optional[Path] = None
        if save_checkpoint:
            if out is None:
                raise ValueError(
                    "save_checkpoint=True needs an output path; pass out=...")
            path = save(Path(out), meta, {"world_model": model},
                        {"world_model": optimizer})
    except BaseException:
        # Ownership transfers to the caller only when this returns.
        data.close()
        raise

    # data stays open on purpose -- see Stage1A.
    return Stage1A(model=model, data=data, sampler=sampler,
                   model_cfg=model_cfg, normalizer=normalizer, meta=meta,
                   coords=coords, losses=last, path=path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage 1A: pretrain one arm's world model")
    parser.add_argument("--task", default="pickcube")
    parser.add_argument("--experiment", default="dreamer",
                        choices=("dreamer", "graph", "graph_progress"))
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-config", default=str(DEFAULT_MODEL),
        help="Dreamer model preset, for example configs/model/size12M.yaml")
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="override data.batch_size (useful for a memory-light smoke run)")
    parser.add_argument(
        "--sequence-length", type=int, default=None,
        help="override data.sequence_length")
    parser.add_argument(
        "--burn-in", type=int, default=None,
        help="override data.burn_in; must be smaller than sequence-length")
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--save-checkpoints", action="store_true",
        help="write the world model to --out. Off by default: the pipeline "
             "passes the trained model to Stage 1B in the same process, so "
             "this exists for resuming a long run, not for continuing one.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    data_overrides = {}
    for key, value in (("batch_size", args.batch_size),
                       ("sequence_length", args.sequence_length),
                       ("burn_in", args.burn_in)):
        if value is not None:
            data_overrides[key] = value
    cfg = load_config(
        args.task, args.experiment,
        overrides={"data": data_overrides} if data_overrides else None)
    if int(cfg["data"]["batch_size"]) < 1:
        raise SystemExit("--batch-size must be at least 1")
    if int(cfg["data"]["sequence_length"]) < 2:
        raise SystemExit("--sequence-length must be at least 2")
    if not 0 <= int(cfg["data"]["burn_in"]) < int(
            cfg["data"]["sequence_length"]):
        raise SystemExit(
            "--burn-in must be non-negative and smaller than "
            "--sequence-length")
    model_yaml = Path(args.model_config)
    if not model_yaml.is_file():
        raise SystemExit(f"model config does not exist: {model_yaml}")
    cfg.setdefault("runtime", {})["model_config"] = str(model_yaml)

    out = Path(args.out or
               f"runs/sim_vla/{args.task}/{args.experiment}/world_model.pt")
    if args.save_checkpoints and out.exists() and not args.overwrite:
        raise SystemExit(
            f"{out} already exists; pass --overwrite to replace it. An arm's "
            "world model is trained once and its checkpoint is what a resumed "
            "run loads.")
    destination = str(out) if args.save_checkpoints else "nothing (checkpoints off)"
    print(f"[world_model] {args.task} / {args.experiment} -> {destination}",
          flush=True)
    print(f"[world_model] model={model_yaml} "
          f"batch={cfg['data']['batch_size']} "
          f"sequence={cfg['data']['sequence_length']} "
          f"burn_in={cfg['data']['burn_in']}", flush=True)
    if not args.save_checkpoints:
        # Said plainly rather than discovered afterwards: run on its own with
        # checkpoints off, this stage trains a model and then drops it.
        print("[world_model] checkpoints are off, so this run keeps nothing. "
              "Use `python -m sim_vla.training.pipeline` to train the world "
              "model and go straight into imitation in one process, or pass "
              "--save-checkpoints to keep this one.", flush=True)

    stage = run(cfg, steps=args.steps, device=args.device, out=out,
                save_checkpoint=args.save_checkpoints, model_yaml=model_yaml)
    stage.data.close()
    if stage.path is not None:
        print(f"[world_model] wrote {stage.path}")
    print(f"[world_model] final losses: {json.dumps(stage.losses, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
