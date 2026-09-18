"""Staged entry points for TD-MPC2 + SmolVLA.

    python -m sim_vla.integrations.tdmpc2.run params
    python -m sim_vla.integrations.tdmpc2.run stage1 --steps 50000 --save
    python -m sim_vla.integrations.tdmpc2.run stage2 --steps 20000 --save \
        --world-checkpoint runs/tdmpc2/tdmpc2_world_model.pt
    python -m sim_vla.integrations.tdmpc2.run stage3 --steps 1000000 \
        --imitation-checkpoint runs/tdmpc2/tdmpc2_imitation.pt
    python -m sim_vla.integrations.tdmpc2.run pipeline \
        --world-steps 50000 --imitation-steps 20000 --online-steps 1000000

``pipeline`` runs all three in one process and hands the objects on, so Stage 2
trains against the very world model Stage 1 produced and Stage 3 continues with
both. ``--save`` writes a checkpoint at the end of each stage; without it
nothing is written, which the run says on startup.

``--set a.b=c`` overrides any config key, so a sweep does not need a new file:

    --set world_model.model_size=19 --set smolvla.sites='[plan_proposals,update_pi]'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ...config import deep_merge
from ..checkpoint import StageMeta, load as load_checkpoint
from ..params import render
from . import agent as build
from . import data as demo_data
from . import stages
from .config import build_cfg, load as load_cfg


def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    """``a.b=c`` into a nested dict, with YAML scalars on the right."""
    import yaml

    out: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        value = yaml.safe_load(raw)
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
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
    path = Path(args.out or Path("runs") / "tdmpc2_smolvla" /
                str(cfg["task"]["env_id"]))
    path.mkdir(parents=True, exist_ok=True)
    return path


# ------------------------------------------------------------------- commands
def command_params(args) -> int:
    """Measure what would be built, and decide sizing from the measurement."""
    cfg = configuration(args)
    world = dict(cfg["world_model"])
    source = demo_data.open_demos(
        str((cfg.get("data") or {}).get("dataset") or cfg["task"]["dataset"]),
        render_size=int(world.get("render_size", 64)),
        include_state=bool(world.get("include_state", False)),
        cameras=(cfg.get("data") or {}).get("cameras")) \
        if not args.no_dataset else None

    shape = (source.obs_shape() if source is not None else
             {"rgb": (3 * int(args.cameras), int(world.get("render_size", 64)),
                      int(world.get("render_size", 64)))})
    action_dim = source.action_dim if source is not None else int(args.action_dim)
    node = build_cfg({**world, "device": args.device or "cpu",
                      "num_cameras": (len(source.images.cameras)
                                      if source is not None else int(args.cameras)),
                      "proprio_dim": (source.proprio_dim if source is not None
                                      and source.include_state else 0)},
                     obs_shape=shape, action_dim=action_dim,
                     episode_length=(source.episode_length if source is not None
                                     else 150))
    agent = build.build_agent(node)

    actor = None
    if cfg["smolvla"].get("enabled", True) and not args.world_only:
        try:
            actor = build.build_actor(cfg["smolvla"],
                                      feature_dim=build.latent_dim(agent),
                                      action_dim=action_dim,
                                      device=args.device or "cpu")
        except Exception as exc:                           # noqa: BLE001
            print(f"[params] SmolVLA not measured: {exc}")
    report = build.parameters(agent, actor=actor)
    print(render(report, title=f"TD-MPC2 (model_size={node.get('model_size')})"))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


def command_stage1(args) -> int:
    cfg = configuration(args)
    result = stages.pretrain_world_model(
        cfg, steps=int(args.steps), device=args.device or cfg.get("device", "cuda"),
        out=out_dir(args, cfg), save=bool(args.save))
    print(f"[stage1] done: {result.metrics}")
    if result.path:
        print(f"[stage1] wrote {result.path}")
    elif not args.save:
        print("[stage1] checkpoints are off; nothing was written. Pass --save "
              "to keep this world model.")
    return 0


def _restore_world(cfg: Mapping[str, Any], path: Path, *, device: str):
    """Rebuild Stage 1's agent and load its weights back into it."""
    world = dict(cfg["world_model"])
    data_cfg = dict(cfg.get("data") or {})
    source = demo_data.open_demos(
        str(data_cfg.get("dataset") or cfg["task"]["dataset"]),
        render_size=int(world.get("render_size", 64)),
        include_state=bool(world.get("include_state", False)),
        cameras=data_cfg.get("cameras"))
    node = build_cfg({**world, "device": device,
                      "num_cameras": len(source.images.cameras),
                      "proprio_dim": source.proprio_dim
                      if source.include_state else 0},
                     obs_shape=source.obs_shape(),
                     action_dim=source.action_dim,
                     episode_length=source.episode_length)
    normalizer = stages.normalizer_for(cfg, source)
    converter = build.build_converter(source.metadata,
                                      action_dim=source.action_dim,
                                      smolvla=cfg["smolvla"],
                                      normalizer=normalizer, device=device)
    agent = build.build_agent(node)
    wanted = stages.stage_meta(cfg, stage="world_model", node=node,
                               source=source, smolvla=False, converter=converter)
    load_checkpoint(path, wanted, {"model": agent.model})
    return stages.WorldModelResult(agent=agent, node=node, source=source,
                                   converter=converter, normalizer=normalizer,
                                   parameters=build.parameters(agent))


def command_stage2(args) -> int:
    cfg = configuration(args)
    device = args.device or cfg.get("device", "cuda")
    if not args.world_checkpoint:
        raise SystemExit(
            "stage 2 trains against a pretrained world model; pass "
            "--world-checkpoint, or run `pipeline` to hand it over in memory.")
    world = _restore_world(cfg, Path(args.world_checkpoint), device=device)
    result = stages.train_imitation(cfg, world, steps=int(args.steps),
                                    device=device, out=out_dir(args, cfg),
                                    save=bool(args.save))
    print(f"[stage2] done: {result.metrics}")
    if result.path:
        print(f"[stage2] wrote {result.path}")
    return 0


def command_stage3(args) -> int:
    from .online import run_online

    cfg = configuration(args)
    device = args.device or cfg.get("device", "cuda")
    actor = None
    world = None
    if cfg["smolvla"].get("enabled", True):
        if not args.imitation_checkpoint:
            raise SystemExit(
                "stage 3 continues from an imitation-trained policy; pass "
                "--imitation-checkpoint, or run `pipeline`.")
        world = _restore_world(cfg, Path(args.imitation_checkpoint), device=device)
        actor = build.build_actor(cfg["smolvla"],
                                  feature_dim=build.latent_dim(world.agent),
                                  action_dim=world.source.action_dim,
                                  device=device)
        wanted = stages.stage_meta(cfg, stage="imitation", node=world.node,
                                   source=world.source, smolvla=True,
                                   converter=world.converter, actor=actor)
        load_checkpoint(Path(args.imitation_checkpoint), wanted,
                        {"model": world.agent.model, "adapter": actor.adapter,
                         "actor": actor})
    elif args.world_checkpoint:
        world = _restore_world(cfg, Path(args.world_checkpoint), device=device)

    run_online(cfg, steps=int(args.steps) if args.steps else None,
               device=device,
               agent=world.agent if world is not None else None,
               actor=actor,
               source=world.source if world is not None else None,
               converter=world.converter if world is not None else None,
               node=world.node if world is not None else None)
    return 0


def command_pipeline(args) -> int:
    from .online import run_online

    cfg = configuration(args)
    device = args.device or cfg.get("device", "cuda")
    out = out_dir(args, cfg)
    save = bool(args.save)
    if not save:
        print("[pipeline] checkpoints are off: the three stages hand objects "
              "on in memory and nothing is written. Pass --save to keep them.")

    world = stages.pretrain_world_model(cfg, steps=int(args.world_steps),
                                        device=device, out=out, save=save)
    if not cfg["smolvla"].get("enabled", True):
        print("[pipeline] SmolVLA disabled; going straight to the upstream "
              "online run")
        run_online(cfg, steps=int(args.online_steps), device=device,
                   agent=world.agent, source=world.source,
                   converter=world.converter, node=world.node)
        return 0

    imitation = stages.train_imitation(cfg, world, steps=int(args.imitation_steps),
                                       device=device, out=out, save=save)
    imitation.trainer.release()
    # Stage 1 froze nothing; Stage 2 did, and Stage 3 is where the world model
    # goes back to learning. Architecture preservation is not weight freezing.
    thawed = 0
    for parameter in world.agent.model.parameters():
        if not parameter.requires_grad:
            parameter.requires_grad_(True)
            thawed += 1
    # ...except the target critics, which upstream keeps out of autograd.
    world.agent.model._target_Qs.requires_grad_(False)
    print(f"[pipeline] world model thawed for stage 3: {thawed} tensors "
          "(target critics stay out of autograd, as upstream has them)")

    run_online(cfg, steps=int(args.online_steps), device=device,
               agent=world.agent, actor=imitation.actor, source=world.source,
               converter=world.converter, node=world.node)
    return 0


# ---------------------------------------------------------------------- entry
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="TD-MPC2 with a latent-conditioned SmolVLA policy")
    parser.add_argument("command",
                        choices=["params", "stage1", "stage2", "stage3", "pipeline"])
    parser.add_argument("--task", default="pickcube")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--save", action="store_true",
                        help="write a checkpoint at the end of each stage")
    parser.add_argument("--set", action="append", default=[],
                        metavar="KEY=VALUE")
    parser.add_argument("--no-smolvla", action="store_true",
                        help="run the upstream agent, for a native baseline")

    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--world-steps", type=int, default=50_000)
    parser.add_argument("--imitation-steps", type=int, default=20_000)
    parser.add_argument("--online-steps", type=int, default=1_000_000)
    parser.add_argument("--world-checkpoint", default="")
    parser.add_argument("--imitation-checkpoint", default="")

    parser.add_argument("--json", default="", help="params: write the report")
    parser.add_argument("--no-dataset", action="store_true",
                        help="params: size from --cameras/--action-dim instead")
    parser.add_argument("--cameras", type=int, default=2)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--world-only", action="store_true",
                        help="params: do not load SmolVLA")

    args = parser.parse_args(argv)
    if args.command in ("stage1", "stage2") and not args.steps:
        args.steps = (args.world_steps if args.command == "stage1"
                      else args.imitation_steps)
    return {
        "params": command_params,
        "stage1": command_stage1,
        "stage2": command_stage2,
        "stage3": command_stage3,
        "pipeline": command_pipeline,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
