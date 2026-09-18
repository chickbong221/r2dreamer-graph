"""Stage 1A: train one arm's world model on the demonstrations.

Full recurrent dynamics, not an image autoencoder: the reward and continuation
heads are trained alongside reconstruction, the recurrent state is rebuilt with
burn-in, and the graph arm additionally trains the semantic prior that
imagination depends on.

The two arms get separate checkpoints and are never initialised from each
other. ``sim_vla/runtime/checkpoint.py`` enforces that rather than advising it:
a graph-trained world model reloaded as a baseline is not a baseline, because
the graph has already reached ``h`` and ``z``.

    python -m sim_vla.training.pretrain_world_model \
        --task pickcube --experiment graph --steps 50000

This is training, not testing. ``sim_vla/run_tests.sh`` never calls it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from ..config import check_dataset_compatibility, load_config
from ..data.batch import to_model_batch
from ..data.dataset import DemoDataset
from ..data.normalization import fit_normalizer
from ..data.sequences import SequenceSampler
from ..models.model_config import DEFAULT_MODEL, load_model_config
from ..models.world_model import build_world_model
from ..runtime.checkpoint import CheckpointMeta, save


def observation_shapes(batch: Dict[str, torch.Tensor]) -> Dict[str, tuple]:
    """Per-key shapes without the batch and time axes."""
    return {key: tuple(value.shape[2:]) for key, value in batch.items()
            if hasattr(value, "dim") and value.dim() >= 2}


def build(cfg: Dict[str, Any], *, device: str,
          model_yaml: Path | None = None):
    """The dataset, the sampler and the arm's world model, wired together."""
    graph_enabled = bool(cfg["model"]["graph"]["enabled"])
    data = DemoDataset(
        cfg["task"]["dataset"], graph_enabled=graph_enabled,
        ignore_terminations=bool(cfg["data"]["ignore_terminations"]))
    check_dataset_compatibility(cfg, data.metadata)

    # The camera count comes from the recording, not from the model preset:
    # the packed bbox is (n_max, n_cams, 4) and the encoder is 5*n_cams+3 wide.
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
        model_cfg, observation_shapes(probe), int(probe["action"].shape[-1]),
        graph_enabled=graph_enabled).to(device)
    return data, sampler, model, model_cfg


def run(cfg: Dict[str, Any], *, steps: int, device: str, out: Path,
        overwrite: bool = False,
        model_yaml: Path | None = None) -> Dict[str, float]:
    data, sampler, model, model_cfg = build(
        cfg, device=device, model_yaml=model_yaml)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_cfg.lr))

    # Fitted once from the demonstrations both arms share, and written beside
    # the checkpoint so the second arm loads it rather than refitting.
    normalizer = fit_normalizer(data)
    normalizer.save(out.with_name("normalization.json"))

    last: Dict[str, float] = {}
    for step in range(int(steps)):
        batch = to_model_batch(sampler.batch(int(cfg["data"]["batch_size"])),
                               device)
        total, losses, _aux = model.loss(batch)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        last = {name: float(value.detach()) for name, value in losses.items()}
        if step % 100 == 0:
            print(f"[world_model] step {step} total {float(total):.4f} "
                  + " ".join(f"{k}={v:.3f}" for k, v in sorted(last.items())[:5]),
                  flush=True)

    save(out, CheckpointMeta(
        graph_enabled=bool(cfg["model"]["graph"]["enabled"]),
        stage="world_model",
        env_id=str(cfg["task"]["env_id"]),
        feature_dim=int(model.feature_dim),
        dataset_identity={"dataset": str(cfg["task"]["dataset"]),
                          "episodes": len(data)},
        normalization_identity=dict(normalizer.identity),
        config=cfg, step=int(steps),
    ), {"world_model": model}, {"world_model": optimizer})
    data.close()
    return last


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
    if out.exists() and not args.overwrite:
        raise SystemExit(
            f"{out} already exists; pass --overwrite to replace it. An arm's "
            "world model is trained once and its checkpoint is what every "
            "later stage loads.")
    print(f"[world_model] {args.task} / {args.experiment} -> {out}\n"
          f"[world_model] model={model_yaml} "
          f"batch={cfg['data']['batch_size']} "
          f"sequence={cfg['data']['sequence_length']} "
          f"burn_in={cfg['data']['burn_in']}", flush=True)
    last = run(cfg, steps=args.steps, device=args.device, out=out,
               overwrite=args.overwrite, model_yaml=model_yaml)
    print(f"[world_model] wrote {out}")
    print(f"[world_model] final losses: {json.dumps(last, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
