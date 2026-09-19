"""Staged entry points for SOLD + SmolVLA.

    python -m sim_vla.integrations.sold.run params
    python -m sim_vla.integrations.sold.run stage1a --steps 20000 --save
    python -m sim_vla.integrations.sold.run stage1b --steps 50000 --save \
        --world-checkpoint runs/sold/sold_autoencoder.pt
    python -m sim_vla.integrations.sold.run stage2 --steps 20000 --save \
        --world-checkpoint runs/sold/sold_world_model.pt
    python -m sim_vla.integrations.sold.run stage3 --steps 1000000 \
        --imitation-checkpoint runs/sold/sold_imitation.pt
    python -m sim_vla.integrations.sold.run pipeline \
        --autoencoder-steps 20000 --world-steps 50000 \
        --imitation-steps 20000 --online-steps 1000000

``stage1a`` and ``stage1b`` are the two halves of SOLD's own recipe: SAVi
first, then the slot dynamics and the reward head on top of it. They are
separate commands because SAVi pretraining is what the vendored
``checkpoints/`` would have saved you -- for a different task.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..checkpoint import StageMeta, load as load_checkpoint
from ..params import render
from . import data as demo_data
from . import model as build
from . import stages
from .config import context_bounds, load as load_cfg


def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    import yaml

    out: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def configuration(args) -> Dict[str, Any]:
    cfg = load_cfg(args.task, parse_overrides(args.set))
    if args.dataset:
        cfg.setdefault("data", {})["dataset"] = args.dataset
    if args.device:
        cfg["device"] = args.device
    if args.no_smolvla:
        cfg["smolvla"]["enabled"] = False
    return cfg


def out_dir(args, cfg: Mapping[str, Any]) -> Path:
    path = Path(args.out or Path("runs") / "sold_smolvla" /
                str(cfg["task"]["env_id"]))
    path.mkdir(parents=True, exist_ok=True)
    return path


class Run:
    """Everything a stage needs, built once from the config."""

    def __init__(self, cfg: Mapping[str, Any], *, device: str,
                 out: Optional[Path] = None, log=print):
        self.cfg = dict(cfg)
        self.out_dir = Path(out or Path("runs") / "sold_smolvla" /
                            str(cfg["task"]["env_id"]))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.world = dict(cfg["world_model"])
        self.smolvla = dict(cfg["smolvla"])
        self.data_cfg = dict(cfg.get("data") or {})
        self.device = device
        self.log = log

        env_cfg = dict(self.world.get("env") or {})
        self.image_size = tuple(env_cfg.get("image_size") or (64, 64))
        self.max_episode_steps = int(env_cfg.get("max_episode_steps") or 150)
        camera = str(self.data_cfg.get("camera") or "")
        self.source = demo_data.open_demos(
            str(self.data_cfg.get("dataset") or cfg["task"]["dataset"]),
            image_size=self.image_size, camera=camera or None)
        if not camera and log:
            log(f"[sold] no camera named; using {self.source.images.cameras[0]!r} "
                f"of {sorted(str(v) for v in (self.source.metadata.get('camera_keys') or {}).values())}")

        from ..action_space import converter_for

        self.normalizer = None
        mode = str(self.smolvla.get("action_normalization") or "identity")
        if mode != "identity":
            from ...data.normalization import fit_normalizer

            self.normalizer = fit_normalizer(self.source.dataset,
                                             fields=("actions", "proprio"))
            self.normalizer.mode = mode
        self.converter = converter_for(
            self.source.metadata, action_dim=self.source.action_dim, mode=mode,
            normalizer=self.normalizer, device=device)

        self.parts = build.components(
            self.world, image_size=self.image_size,
            action_dim=self.source.action_dim,
            max_episode_steps=self.max_episode_steps,
            action_low=self.converter.low_native,
            action_high=self.converter.high_native)
        for key in ("autoencoder", "dynamics", "reward", "actor", "critic",
                    "critic_target"):
            self.parts[key] = self.parts[key].to(device)

        low, _high = context_bounds(self.world)
        wanted = int((self.smolvla.get("adapter") or {}).get("context") or 0)
        self.context = wanted or low
        self.actor = None

    def build_actor(self):
        if self.actor is None:
            self.actor = build.build_actor(
                self.smolvla, num_slots=int(self.parts["num_slots"]),
                slot_dim=int(self.parts["slot_dim"]),
                action_dim=int(self.source.action_dim),
                context=self.context, device=self.device)
        return self.actor

    def meta(self, stage: str, *, smolvla: bool, step: int = 0,
             policy=None) -> StageMeta:
        return stages.stage_meta(
            self.cfg, stage=stage, parts=self.parts, source=self.source,
            smolvla=smolvla, converter=self.converter, actor=self.actor,
            policy=policy, parameters=self.parameters(),
            max_episode_steps=self.max_episode_steps, step=step)

    def parameters(self) -> Dict[str, Any]:
        return build.parameters(self.parts, actor=self.actor)

    def modules(self) -> Dict[str, Any]:
        out = {"autoencoder": self.parts["autoencoder"],
               "dynamics": self.parts["dynamics"],
               "reward": self.parts["reward"],
               "actor_gaussian": self.parts["actor"],
               "critic": self.parts["critic"],
               "critic_target": self.parts["critic_target"]}
        if self.actor is not None:
            out["adapter"] = self.actor.adapter
            out["smolvla_actor"] = self.actor
        return out

    def restore(self, path: Path, stage: str, *, smolvla: bool,
                strict: bool = False) -> None:
        load_checkpoint(path, self.meta(stage, smolvla=smolvla),
                        self.modules(), strict=strict)


# ------------------------------------------------------------------- commands
def command_params(args) -> int:
    """Measure what would be built, and decide sizing from the measurement.

    ``--no-dataset`` sizes from ``--action-dim`` instead of from the
    demonstrations, so the audit runs on a machine without the dataset or
    HDF5. The image size, slot count and every width still come from SOLD's
    own config files.
    """
    cfg = configuration(args)
    device = args.device or "cpu"
    if args.no_dataset:
        from .config import context_bounds

        world = dict(cfg["world_model"])
        env_cfg = dict(world.get("env") or {})
        image_size = tuple(env_cfg.get("image_size") or (64, 64))
        max_steps = int(env_cfg.get("max_episode_steps") or 150)
        parts = build.components(world, image_size=image_size,
                                 action_dim=int(args.action_dim),
                                 max_episode_steps=max_steps)
        actor = None
        if cfg["smolvla"].get("enabled", True) and not args.world_only:
            low, _high = context_bounds(world)
            context = int((cfg["smolvla"].get("adapter") or {}).get("context") or 0) or low
            try:
                actor = build.build_actor(
                    cfg["smolvla"], num_slots=int(parts["num_slots"]),
                    slot_dim=int(parts["slot_dim"]),
                    action_dim=int(args.action_dim), context=context,
                    device=device)
            except Exception as exc:                       # noqa: BLE001
                print(f"[params] SmolVLA not measured: {exc}")
        report = build.parameters(parts, actor=actor)
    else:
        run = Run(cfg, device=device, out=out_dir(args, cfg))
        if cfg["smolvla"].get("enabled", True) and not args.world_only:
            try:
                run.build_actor()
            except Exception as exc:                       # noqa: BLE001
                print(f"[params] SmolVLA not measured: {exc}")
        report = run.parameters()
    print(render(report, title="SOLD (sold/configs/train_sold.yaml)"))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


def command_stage1a(args) -> int:
    cfg = configuration(args)
    run = Run(cfg, device=args.device or cfg.get("device", "cuda"),
              out=out_dir(args, cfg))
    stage_cfg = dict(cfg["stages"]["autoencoder"])
    metrics = stages.pretrain_autoencoder(
        run.parts["autoencoder"], run.source,
        stages.AutoencoderConfig(steps=int(args.steps),
                                 batch_size=int(stage_cfg["batch_size"]),
                                 sequence_length=int(stage_cfg["sequence_length"]),
                                 lr=float(stage_cfg["lr"]),
                                 grad_clip=float(stage_cfg["grad_clip"]),
                                 log_every=int(stage_cfg["log_every"])),
        device=run.device, seed=int(run.data_cfg.get("seed", 0)),
        converter=run.converter)
    print(f"[sold:1a] done: {metrics}")
    if args.save:
        path = stages.save_stage(
            out_dir(args, cfg) / "sold_autoencoder.pt",
            run.meta("world_model", smolvla=False, step=int(args.steps)),
            run.parts)
        print(f"[sold:1a] wrote {path}")
    return 0


def command_stage1b(args) -> int:
    cfg = configuration(args)
    run = Run(cfg, device=args.device or cfg.get("device", "cuda"),
              out=out_dir(args, cfg))
    if args.world_checkpoint:
        run.restore(Path(args.world_checkpoint), "world_model", smolvla=False)
        print(f"[sold:1b] restored SAVi from {args.world_checkpoint}")
    else:
        print("[sold:1b] no --world-checkpoint: SAVi is untrained. SOLD's "
              "recipe pretrains it first; run stage1a, or use `pipeline`.")
    stage_cfg = dict(cfg["stages"]["world_model"])
    trainer = stages.WorldModelTrainer(
        run.parts, run.world, run.source,
        stages.WorldModelConfig(steps=int(args.steps),
                                batch_size=int(stage_cfg["batch_size"]),
                                log_every=int(stage_cfg["log_every"])),
        device=run.device, seed=int(run.data_cfg.get("seed", 0)),
        converter=run.converter)
    print(f"[sold:1b] finetune_autoencoder={trainer.finetune_autoencoder} "
          "(upstream's setting, honoured)")
    metrics = trainer.fit(int(args.steps))
    print(f"[sold:1b] done: {metrics}")
    if args.save:
        path = stages.save_stage(
            out_dir(args, cfg) / "sold_world_model.pt",
            run.meta("world_model", smolvla=False, step=int(args.steps)),
            run.parts,
            optimizers={"dynamics": trainer.dynamics_optimizer,
                        "reward": trainer.reward_optimizer})
        print(f"[sold:1b] wrote {path}")
    return 0


def command_stage2(args) -> int:
    cfg = configuration(args)
    run = Run(cfg, device=args.device or cfg.get("device", "cuda"),
              out=out_dir(args, cfg))
    if not args.world_checkpoint:
        raise SystemExit(
            "stage 2 trains against a pretrained world model; pass "
            "--world-checkpoint, or run `pipeline` to hand it over in memory.")
    run.restore(Path(args.world_checkpoint), "world_model", smolvla=False)
    actor = run.build_actor()
    stage_cfg = dict(cfg["stages"]["imitation"])
    trainer = stages.ImitationTrainer(
        run.parts, actor, run.source, run.converter,
        config=stages.ImitationConfig(
            steps=int(args.steps), batch_size=int(stage_cfg["batch_size"]),
            lr=float(stage_cfg["lr"]), grad_clip=float(stage_cfg["grad_clip"]),
            log_every=int(stage_cfg["log_every"]),
            sequence_length=int(stage_cfg["sequence_length"])),
        device=run.device, seed=int(run.data_cfg.get("seed", 0)))
    print(render(run.parameters(), title="SOLD + SmolVLA (stage 2)"))
    print(f"[sold:2] world model frozen: {trainer.frozen} tensors; "
          f"burn-in {trainer.burn_in} rows for a full {trainer.context}-frame "
          "context")
    metrics = trainer.fit(int(args.steps))
    print(f"[sold:2] done: {metrics}")
    if args.save:
        path = stages.save_stage(
            out_dir(args, cfg) / "sold_imitation.pt",
            run.meta("imitation", smolvla=True, step=int(args.steps)),
            run.parts, actor=actor, optimizers={"actor": trainer.optimizer})
        print(f"[sold:2] wrote {path}")
    return 0


def command_stage3(args) -> int:
    from .online import run_online

    cfg = configuration(args)
    run = Run(cfg, device=args.device or cfg.get("device", "cuda"),
              out=out_dir(args, cfg))
    if cfg["smolvla"].get("enabled", True):
        if not args.imitation_checkpoint:
            raise SystemExit(
                "stage 3 continues from an imitation-trained policy; pass "
                "--imitation-checkpoint, or run `pipeline`.")
        run.build_actor()
        run.restore(Path(args.imitation_checkpoint), "imitation", smolvla=True)
    elif args.world_checkpoint:
        run.restore(Path(args.world_checkpoint), "world_model", smolvla=False)
    run_online(run, steps=int(args.steps) if args.steps else None)
    return 0


def command_pipeline(args) -> int:
    from .online import run_online

    cfg = configuration(args)
    run = Run(cfg, device=args.device or cfg.get("device", "cuda"),
              out=out_dir(args, cfg))
    out = out_dir(args, cfg)
    save = bool(args.save)
    if not save:
        print("[pipeline] checkpoints are off: the stages hand objects on in "
              "memory and nothing is written. Pass --save to keep them.")

    stage_cfg = dict(cfg["stages"]["autoencoder"])
    stages.pretrain_autoencoder(
        run.parts["autoencoder"], run.source,
        stages.AutoencoderConfig(steps=int(args.autoencoder_steps),
                                 batch_size=int(stage_cfg["batch_size"]),
                                 sequence_length=int(stage_cfg["sequence_length"]),
                                 lr=float(stage_cfg["lr"]),
                                 grad_clip=float(stage_cfg["grad_clip"]),
                                 log_every=int(stage_cfg["log_every"])),
        device=run.device, seed=int(run.data_cfg.get("seed", 0)),
        converter=run.converter)
    if save:
        stages.save_stage(out / "sold_autoencoder.pt",
                          run.meta("world_model", smolvla=False,
                                   step=int(args.autoencoder_steps)),
                          run.parts)

    world_cfg = dict(cfg["stages"]["world_model"])
    trainer = stages.WorldModelTrainer(
        run.parts, run.world, run.source,
        stages.WorldModelConfig(steps=int(args.world_steps),
                                batch_size=int(world_cfg["batch_size"]),
                                log_every=int(world_cfg["log_every"])),
        device=run.device, seed=int(run.data_cfg.get("seed", 0)),
        converter=run.converter)
    trainer.fit(int(args.world_steps))
    if save:
        stages.save_stage(out / "sold_world_model.pt",
                          run.meta("world_model", smolvla=False,
                                   step=int(args.world_steps)),
                          run.parts)

    if not cfg["smolvla"].get("enabled", True):
        print("[pipeline] SmolVLA disabled; going straight to the upstream "
              "online run")
        run_online(run, steps=int(args.online_steps))
        return 0

    actor = run.build_actor()
    imitation_cfg = dict(cfg["stages"]["imitation"])
    imitation = stages.ImitationTrainer(
        run.parts, actor, run.source, run.converter,
        config=stages.ImitationConfig(
            steps=int(args.imitation_steps),
            batch_size=int(imitation_cfg["batch_size"]),
            lr=float(imitation_cfg["lr"]),
            grad_clip=float(imitation_cfg["grad_clip"]),
            log_every=int(imitation_cfg["log_every"]),
            sequence_length=int(imitation_cfg["sequence_length"])),
        device=run.device, seed=int(run.data_cfg.get("seed", 0)))
    print(render(run.parameters(), title="SOLD + SmolVLA"))
    imitation.fit(int(args.imitation_steps))
    imitation.release()
    if save:
        stages.save_stage(out / "sold_imitation.pt",
                          run.meta("imitation", smolvla=True,
                                   step=int(args.imitation_steps)),
                          run.parts, actor=actor)

    # Stage 2 froze the world model; Stage 3 is where it goes back to
    # learning. Architecture preservation is not weight freezing.
    thawed = stages.thaw(run.parts["autoencoder"], run.parts["dynamics"],
                         run.parts["reward"], run.parts["critic"],
                         run.parts["actor"])
    print(f"[pipeline] world model thawed for stage 3: {thawed} tensors")
    run_online(run, steps=int(args.online_steps))
    return 0


# ---------------------------------------------------------------------- entry
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="SOLD with a latent-conditioned SmolVLA actor")
    parser.add_argument("command",
                        choices=["params", "stage1a", "stage1b", "stage2",
                                 "stage3", "pipeline"])
    parser.add_argument("--task", default="pickcube")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--set", action="append", default=[],
                        metavar="KEY=VALUE")
    parser.add_argument("--no-smolvla", action="store_true",
                        help="run the upstream module, for a native baseline")

    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--autoencoder-steps", type=int, default=20_000)
    parser.add_argument("--world-steps", type=int, default=50_000)
    parser.add_argument("--imitation-steps", type=int, default=20_000)
    parser.add_argument("--online-steps", type=int, default=1_000_000)
    parser.add_argument("--world-checkpoint", default="")
    parser.add_argument("--imitation-checkpoint", default="")

    parser.add_argument("--json", default="")
    parser.add_argument("--world-only", action="store_true",
                        help="params: do not load SmolVLA")
    parser.add_argument("--no-dataset", action="store_true",
                        help="params: size from --action-dim instead")
    parser.add_argument("--action-dim", type=int, default=8)

    args = parser.parse_args(argv)
    defaults = {"stage1a": args.autoencoder_steps,
                "stage1b": args.world_steps,
                "stage2": args.imitation_steps,
                "stage3": args.online_steps}
    if not args.steps and args.command in defaults:
        args.steps = defaults[args.command]
    return {
        "params": command_params,
        "stage1a": command_stage1a,
        "stage1b": command_stage1b,
        "stage2": command_stage2,
        "stage3": command_stage3,
        "pipeline": command_pipeline,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
