"""One process, three stages, one world model object passed between them.

    python -m sim_vla.training.pipeline --task pickcube --experiment graph \
        --world-steps 50000 --imitation-steps 20000 --online-steps 200000

Stage 1A trains the world model, Stage 1B trains the adapter and action expert
against *that* model, and Stage 2 continues with both. Nothing is written to
disk and nothing is reloaded from it: each stage receives the previous stage's
Python objects.

**Checkpoints are off by default.** ``--save-checkpoints`` turns them on for a
run long enough that losing it would matter, and it is the only thing that
writes anything here. It changes what is persisted, never what is trained --
with it off, the same weights reach Stage 2 by a shorter route.

The arm is one switch. ``--experiment`` picks it, and it decides whether the
graph encoder, the semantic latent, the graph losses and the progress
supervision exist at all -- not merely whether the actor sees them. The arms
are never initialised from each other, which is why there is no flag here to
start one from the other's weights.

This is training, not testing. ``sim_vla/run_tests.sh`` never calls it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import load_config
from ..models.model_config import DEFAULT_MODEL
from . import pretrain_world_model
from . import progress as progress_module
from . import train_imitation
from .actor_critic import ActorCriticConfig
from .online import OnlineConfig, run_online
from .progress import ProgressConfig, build_progress


def stage_paths(root: Path) -> Dict[str, Path]:
    """Where each stage writes, when writing is on at all."""
    return {"world_model": root / "world_model.pt",
            "imitation": root / "imitation.pt",
            "online": root}


def run(cfg: Dict[str, Any], *, world_steps: int, imitation_steps: int,
        online_steps: int, device: str, root: Path,
        save_checkpoints: bool = False,
        model_yaml: Optional[Path] = None) -> Dict[str, Any]:
    """Stage 1A -> 1B -> 2, handing the live objects forward."""
    paths = stage_paths(root)
    report: Dict[str, Any] = {}

    # Refused before anything expensive: the progress arm has no supervision
    # contract in this package, and discovering that after Stage 1A costs a
    # world model's worth of compute to learn nothing.
    from ..envs.maniskill import load_metadata

    try:
        metadata = load_metadata(cfg["task"]["dataset"])
    except Exception:                                      # noqa: BLE001
        metadata = {}
    progress_module.preflight(cfg, metadata)

    # One termination convention across the dataset, the replay, the env and
    # the continuation head. SimVlaEnv reports is_terminal=False always, which
    # is what ignore_terminations means; honouring recorded terminals in the
    # loader while the env never produces one would train the continuation
    # head on a signal the rollout cannot contain.
    if online_steps and not bool(cfg["data"]["ignore_terminations"]):
        raise SystemExit(
            "data.ignore_terminations=false is not supported with online "
            "training: sim_vla/envs/maniskill.py runs the env under "
            "ignore_terminations and never reports a terminal, so the "
            "continuation head would be trained on demonstrations that carry "
            "terminals and rollouts that cannot. Set it true, or give the env "
            "a termination path first.")

    # ------------------------------------------------------------- stage 1A
    print(f"[pipeline] stage 1A: world model, {world_steps} steps", flush=True)
    stage_a = pretrain_world_model.run(
        cfg, steps=int(world_steps), device=device,
        out=paths["world_model"], save_checkpoint=save_checkpoints,
        model_yaml=model_yaml)
    report["world_model"] = {"losses": stage_a.losses,
                             "feature_dim": int(stage_a.model.feature_dim),
                             "path": str(stage_a.path or "")}

    try:
        # ---------------------------------------------------------- stage 1B
        if int(imitation_steps) > 0:
            print(f"[pipeline] stage 1B: imitation, {imitation_steps} steps "
                  "(same world model object, frozen)", flush=True)
            stage_b = train_imitation.run(
                cfg, stage_a.model, stage_a.sampler, steps=int(imitation_steps),
                device=device, normalizer=stage_a.normalizer,
                coords=stage_a.coords,
                out=paths["imitation"], save_checkpoint=save_checkpoints,
                meta=stage_a.meta)
            report["imitation"] = {"losses": stage_b.losses,
                                   "path": str(stage_b.path or "")}
            # After any save, before online training. Stage 2 builds its own
            # actor optimizer; keeping this one alive holds a second set of
            # Adam moments for every trainable parameter, on the device, for
            # the whole run. empty_cache() would not have helped -- these are
            # live tensors, not cached blocks.
            if stage_b.trainer is not None:
                stage_b.trainer.release()
                stage_b = replace(stage_b, trainer=None)
        else:
            print("[pipeline] stage 1B skipped (--imitation-steps 0)",
                  flush=True)
            return report

        # ----------------------------------------------------------- stage 2
        if int(online_steps) <= 0:
            print("[pipeline] stage 2 skipped (--online-steps 0)", flush=True)
            return report

        # The world model is trainable again here. Stage 1B froze it to train
        # the policy against a fixed state; Stage 2 trains both.
        for parameter in stage_a.model.parameters():
            parameter.requires_grad_(True)
        stage_a.model.train()

        from ..envs.maniskill import SimVlaEnv
        from ..models.critics import ValueCritic

        critic = ValueCritic(stage_a.model_cfg,
                             int(stage_a.model.feature_dim)).to(device)
        progress_cfg = ProgressConfig(
            enabled=bool(cfg["model"]["progress"]["enabled"]),
            beta=float(cfg["model"]["progress"]["beta"]),
            warmup_start=int(cfg["model"]["progress"]["beta_warmup_start"]),
            warmup_end=int(cfg["model"]["progress"]["beta_warmup_end"]))
        progress_head = build_progress(
            stage_a.model_cfg, int(stage_a.model.feature_dim),
            graph_enabled=bool(cfg["model"]["graph"]["enabled"]),
            progress_enabled=progress_cfg.enabled)
        if progress_head is not None:
            progress_head = progress_head.to(device)

        env = SimVlaEnv(
            stage_a.data.metadata,
            graph_enabled=bool(cfg["model"]["graph"]["enabled"]),
            max_steps=int(cfg["eval"]["max_steps"]),
            seed=int(cfg["data"]["seed"]),
            record_graphs=bool(cfg["diagnostics"]["record_graphs"])).build()

        online_cfg = OnlineConfig(
            total_steps=int(online_steps),
            batch_size=int(cfg["data"]["batch_size"]),
            sequence_length=int(cfg["data"]["sequence_length"]),
            burn_in=int(cfg["data"]["burn_in"]),
            max_episode_steps=int(cfg["eval"]["max_steps"]),
            # One seed for the run: collection used to default to zero on its
            # own while the sampler took the configured one.
            seed=int(cfg["data"]["seed"]),
            ignore_terminations=bool(cfg["data"]["ignore_terminations"]),
            save_checkpoints=bool(save_checkpoints),
            eval_episodes=int(cfg["eval"]["episodes"]))
        ac_cfg = ActorCriticConfig(
            flow_steps=int(stage_b.actor.flow_steps),
            progress_beta=float(progress_cfg.beta) if progress_cfg.enabled
            else 0.0)

        print(f"[pipeline] stage 2: online, {online_steps} env steps",
              flush=True)
        try:
            trainer = run_online(
                cfg, stage_a.model, stage_b.actor, critic, stage_a.sampler,
                env, config=online_cfg, ac_config=ac_cfg, device=device,
                normalizer=stage_a.normalizer, coords=stage_a.coords,
                progress_head=progress_head,
                checkpoint_dir=paths["online"], meta=stage_a.meta)
            report["online"] = {"env_steps": trainer.env_steps,
                                "updates": trainer.updates}
        finally:
            env.close()
        return report
    finally:
        stage_a.data.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Stage 1A, 1B and 2 in one process")
    parser.add_argument("--task", default="pickcube")
    parser.add_argument("--experiment", default="dreamer",
                        choices=("dreamer", "graph", "graph_progress"))
    parser.add_argument("--world-steps", type=int, default=50_000)
    parser.add_argument("--imitation-steps", type=int, default=20_000,
                        help="0 stops after the world model")
    parser.add_argument("--online-steps", type=int, default=0,
                        help="environment steps; 0 stops after imitation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--burn-in", type=int, default=None)
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--save-checkpoints", action="store_true",
        help="write each stage's weights under --out. Off by default: the "
             "stages hand their models to each other in memory, so this is "
             "for resuming a long run, not for reaching the next stage.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    overrides = {}
    for key, value in (("batch_size", args.batch_size),
                       ("sequence_length", args.sequence_length),
                       ("burn_in", args.burn_in)):
        if value is not None:
            overrides[key] = value
    cfg = load_config(args.task, args.experiment,
                      overrides={"data": overrides} if overrides else None)
    model_yaml = Path(args.model_config)
    if not model_yaml.is_file():
        raise SystemExit(f"model config does not exist: {model_yaml}")
    cfg.setdefault("runtime", {})["model_config"] = str(model_yaml)

    root = Path(args.out or f"runs/sim_vla/{args.task}/{args.experiment}")
    print(f"[pipeline] {args.task} / {args.experiment} on {args.device}",
          flush=True)
    print("[pipeline] checkpoints: "
          + (f"on -> {root}" if args.save_checkpoints
             else "off (stages pass their models in memory)"), flush=True)

    report = run(cfg, world_steps=args.world_steps,
                 imitation_steps=args.imitation_steps,
                 online_steps=args.online_steps, device=args.device,
                 root=root, save_checkpoints=args.save_checkpoints,
                 model_yaml=model_yaml)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
