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
from .pretrain_world_model import seed_everything
from . import progress as progress_module
from . import train_imitation
from . import actor_critic
from .actor_critic import ActorCriticConfig
from .online import OnlineConfig, run_online
from .progress import ProgressConfig
from .wandb_logger import RunLogger, start_run


def stage_paths(root: Path) -> Dict[str, Path]:
    """Where each stage writes, when writing is on at all."""
    return {"world_model": root / "world_model.pt",
            "imitation": root / "imitation.pt",
            "online": root}


def online_configs(cfg, model_cfg, *, total_steps, flow_steps,
                   save_checkpoints=False):
    """Resolve shared Dreamer settings and the VLA-specific memory controls."""
    online_settings = cfg.get("online", {})
    online_cfg = OnlineConfig(
        total_steps=int(total_steps),
        train_ratio=float(online_settings.get("train_ratio", 64)),
        precision=str(online_settings.get("precision", "bfloat16")),
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
    ac_kwargs = dict(
        # The number of actions one generated chunk contributes, in the
        # environment and in imagination alike. Taken from actor.execute so
        # there is exactly one setting for it.
        execute=int((cfg.get("actor") or {}).get("execute") or 1),
        # The root model's horizon decides the discount, as it does for the
        # simulator's own trainer. It is unrelated to how long an imagined
        # rollout is, which is actor.execute.
        discount=1.0 - 1.0 / float(model_cfg.horizon),
        imagination_microbatch=int(
            online_settings.get("imagination_microbatch", 16)),
        precision=online_cfg.precision,
        flow_steps=int(flow_steps),
        critic_warmup=int(online_settings.get("critic_warmup", 150)),
        profile=bool(online_settings.get("profile", False)),
        # Starts at zero and is set per update from the warm-up; the
        # configured beta is the value it warms up *to*.
        progress_beta=0.0)
    if online_settings.get("actor_lr") is not None:
        ac_kwargs["actor_lr"] = float(online_settings["actor_lr"])
    ac_cfg = ActorCriticConfig(**ac_kwargs)

    return online_cfg, ac_cfg


def run(cfg: Dict[str, Any], *, world_steps: int, imitation_steps: int,
        online_steps: int, device: str, root: Path,
        save_checkpoints: bool = False,
        model_yaml: Optional[Path] = None,
        logger: Optional[RunLogger] = None,
        resume_from: Optional[Path] = None) -> Dict[str, Any]:
    """Stage 1A -> 1B -> 2, handing the live objects forward.

    ``logger`` is the one wandb run all three stages log into. The default is
    an inert one, so calling this directly costs nothing; ``main`` opens the
    real one and owns closing it.
    """
    paths = stage_paths(root)
    report: Dict[str, Any] = {}
    logger = logger if logger is not None else RunLogger()

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
    resume_paths = stage_paths(Path(resume_from)) if resume_from else {}
    reuse_world = resume_paths.get("world_model")
    reuse_imitation = resume_paths.get("imitation")
    if reuse_world is not None and not Path(reuse_world).is_file():
        reuse_world = None
    if reuse_imitation is not None and not Path(reuse_imitation).is_file():
        reuse_imitation = None
    if reuse_imitation is not None and reuse_world is None:
        # The actor was trained against one particular world model. Restoring
        # it beside a freshly initialised one pairs a trained policy with a
        # state representation it has never seen.
        raise SystemExit(
            f"{reuse_imitation} exists but its world model does not. Stage 1B "
            "was trained against a specific Stage 1A model and cannot be "
            "restored beside a new one; resume both or neither.")

    if reuse_world is not None:
        print(f"[pipeline] stage 1A: skipped, restoring {reuse_world}",
              flush=True)
        stage_a = pretrain_world_model.resume(
            cfg, device=device, path=Path(reuse_world), model_yaml=model_yaml)
    else:
        stage_a = pretrain_world_model.run(
            cfg, steps=int(world_steps), device=device,
            out=paths["world_model"], save_checkpoint=save_checkpoints,
            model_yaml=model_yaml,
            on_metrics=lambda m: logger.log(m, stage="world"))
    report["world_model"] = {"losses": stage_a.losses,
                             "feature_dim": int(stage_a.model.feature_dim),
                             "path": str(stage_a.path or "")}
    logger.summary({"feature_dim": int(stage_a.model.feature_dim),
                    "world_steps": int(world_steps)})
    if stage_a.lr is not None:
        logger.summary({"world_lr": float(stage_a.lr)})

    try:
        # ---------------------------------------------------------- stage 1B
        if reuse_imitation is not None:
            print(f"[pipeline] stage 1B: skipped, restoring {reuse_imitation}",
                  flush=True)
            stage_b = train_imitation.resume(
                cfg, stage_a.model, stage_a.sampler,
                path=Path(reuse_imitation), device=device, meta=stage_a.meta)
            report["imitation"] = {"losses": stage_b.losses,
                                   "path": str(stage_b.path or "")}
        elif int(imitation_steps) > 0:
            print(f"[pipeline] stage 1B: imitation, {imitation_steps} steps "
                  "(same world model object, frozen)", flush=True)
            stage_b = train_imitation.run(
                cfg, stage_a.model, stage_a.sampler, steps=int(imitation_steps),
                device=device, normalizer=stage_a.normalizer,
                coords=stage_a.coords,
                out=paths["imitation"], save_checkpoint=save_checkpoints,
                meta=stage_a.meta,
                on_metrics=lambda m: logger.log(m, stage="imitation"))
            report["imitation"] = {"losses": stage_b.losses,
                                   "path": str(stage_b.path or "")}
            logger.summary({"imitation_lr": train_imitation.imitation_lr(cfg)})
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
        enabled = bool(cfg["model"]["progress"]["enabled"])
        # The warm-up is scaled to this run's budget. The repository's absolute
        # defaults (400k -> 700k env steps) never turn shaping on inside a
        # shorter run, which would make the arm identical to plain `graph`
        # while being reported as a different method.
        warmup_start, warmup_end = progress_module.warmup_for(int(online_steps))
        progress_cfg = ProgressConfig(
            enabled=enabled,
            beta=float(cfg["model"]["progress"]["beta"]),
            warmup_start=warmup_start, warmup_end=warmup_end)
        # The exact object Stage 1A trained jointly with this world model on
        # the demonstrations, or restored from the world model's file beside
        # the weights it was trained with -- never built fresh here, where its
        # first shaping rewards would be noise.
        progress_head, potential = stage_a.progress_head, stage_a.potential
        if enabled and (progress_head is None or potential is None):
            raise SystemExit(
                "the graph_progress arm reached Stage 2 without the progress "
                "head Stage 1A fits; shaping from an untrained head is what "
                "pretraining it exists to prevent")
        if progress_head is not None:
            print(f"[pipeline] progress: head from stage 1A, "
                  f"{potential.describe()} beta={progress_cfg.beta} warmup="
                  f"{warmup_start}->{warmup_end} env steps", flush=True)
            report["progress"] = potential.describe() | {
                "beta": progress_cfg.beta,
                "warmup": [warmup_start, warmup_end],
                "head": "trained jointly with the world model in stage 1A"}
            # In the summary rather than only the log: which schedule the arm
            # was shaped against, and how strongly, is what distinguishes this
            # run from the plain graph arm.
            logger.summary({f"progress_{key}": value
                            for key, value in report["progress"].items()})

        online_env = cfg.get("online") or {}
        reconfiguration = online_env.get("reconfiguration_freq")
        env = SimVlaEnv(
            stage_a.data.metadata,
            graph_enabled=bool(cfg["model"]["graph"]["enabled"]),
            max_steps=int(cfg["eval"]["max_steps"]),
            seed=int(cfg["data"]["seed"]),
            record_graphs=bool(cfg["diagnostics"]["record_graphs"]),
            num_envs=int(online_env.get("num_envs", 1) or 1),
            reconfiguration_freq=(None if reconfiguration is None
                                  else int(reconfiguration))).build()

        online_cfg, ac_cfg = online_configs(
            cfg, stage_a.model_cfg, total_steps=online_steps,
            flow_steps=stage_b.actor.flow_steps, save_checkpoints=save_checkpoints)

        print(f"[pipeline] stage 2: online, {online_steps} env steps",
              flush=True)
        settings = {
            # What the env resolved to, not what was asked: ManiSkill picks the
            # reconfiguration frequency itself when none is given, and for
            # PegInsertionSide that choice depends on the env count.
            "num_envs": env.num_envs,
            "sim_backend": env.sim_backend,
            "reconfiguration_freq": env.live_reconfiguration_freq,
            "replay_batch": online_cfg.batch_size,
            "sequence_length": online_cfg.sequence_length,
            "train_ratio": online_cfg.train_ratio,
            "precision": online_cfg.precision,
            "imagination_microbatch": ac_cfg.imagination_microbatch,
            "discount": ac_cfg.discount,
            "flow_steps": ac_cfg.flow_steps,
            # What the actor update is, resolved rather than as written: this
            # is what a later comparison has to match on.
            "actor_objective": actor_critic.OBJECTIVE,
            "return": actor_critic.RETURN,
            "start_selection": actor_critic.START_SELECTION,
            "execute": ac_cfg.execute,
            "actor_lr": ac_cfg.actor_lr,
            "critic_lr": ac_cfg.critic_lr,
            "critic_warmup": ac_cfg.critic_warmup,
            "seed": int(cfg["data"]["seed"]),
            "profile": ac_cfg.profile,
        }
        print(f"[pipeline] online settings: {json.dumps(settings)}", flush=True)
        report["online_settings"] = settings
        logger.summary({f"online_{key}": value for key, value in settings.items()})
        try:
            trainer = run_online(
                cfg, stage_a.model, stage_b.actor, critic, stage_a.sampler,
                env, config=online_cfg, ac_config=ac_cfg, device=device,
                normalizer=stage_a.normalizer, coords=stage_a.coords,
                progress_head=progress_head, potential=potential,
                progress_config=progress_cfg,
                checkpoint_dir=paths["online"], meta=stage_a.meta,
                on_metrics=lambda m: logger.log(m, stage="online"))
            report["online"] = {"env_steps": trainer.env_steps,
                                "updates": trainer.updates}
            logger.summary({"online_env_steps": int(trainer.env_steps),
                            "online_updates": int(trainer.updates)})
        finally:
            env.close()
        return report
    finally:
        stage_a.data.close()


# Flags the online redesign removed, and what to do instead. argparse would
# reject them as unknown, which says they are gone but not why -- and a script
# that silently lost a flag it thought it was setting would be running a
# different experiment.
REMOVED_FLAGS = {
    "--actor-objective":
        "there is one online objective: the pathwise return of an executed "
        "chunk. Drop the flag.",
    "--flow-noise-std":
        "the stochastic flow sampler went with flow_reinforce. Drop the flag.",
    "--actor-transition-microbatch":
        "no flow transition is scored any more; --imagination-microbatch is "
        "what bounds the actor's memory.",
    "--imagination-batch":
        "imagination starts from every eligible replay state, so there is no "
        "cap to set; --imagination-microbatch is the memory control.",
    "--imag-horizon":
        "a rollout is exactly actor.execute transitions of one generated "
        "chunk. Set actor.execute in the config.",
    "--demo-anchor": "online imitation was removed; Stage 1B imitates.",
    "--anchor-rows": "online imitation was removed.",
    "--anchor-microbatch": "online imitation was removed.",
    "--anchor-windows": "online imitation was removed.",
    "--anchor-window-microbatch": "online imitation was removed.",
    "--grad-report-every":
        "it measured the RL gradient against the imitation anchor's, and "
        "there is no anchor.",
    "--eval-sampler":
        "there is one sampler, so the evaluation already runs the policy "
        "being trained.",
}


def refuse_removed_flags(argv) -> None:
    """Fail on a flag this pipeline no longer has, saying what replaced it."""
    used = [flag for flag in REMOVED_FLAGS
            if any(str(item) == flag or str(item).startswith(flag + "=")
                   for item in (argv or []))]
    if not used:
        return
    detail = "\n".join(f"  {flag}: {REMOVED_FLAGS[flag]}" for flag in used)
    raise SystemExit(
        "these flags were removed with the online actor redesign:\n" + detail
        + "\nThe online update now imagines one generated chunk from every "
        "eligible replay state, executes actor.execute of its actions, and "
        "maximises that chunk's bootstrapped return.")


def parse_args(argv=None):
    import sys

    refuse_removed_flags(sys.argv[1:] if argv is None else argv)
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
    parser.add_argument("--world-lr", type=float, default=None,
                        help="Stage 1A learning rate; unset keeps the model "
                             "preset's (4e-5)")
    parser.add_argument("--imitation-lr", type=float, default=None,
                        help="Stage 1B learning rate; unset keeps 1e-4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--burn-in", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="one seed for the whole run: model, adapter and "
                             "critic initialisation, the demonstration "
                             "sampler, the online replay, environment "
                             "collection and the flow sampler. Overrides "
                             "data.seed.")
    parser.add_argument("--train-ratio", type=float, default=None,
                        help="online replay timesteps per environment step; "
                             "0 uses the legacy 8 updates per collection")
    parser.add_argument("--online-precision", choices=("float32", "bfloat16"),
                        default=None)
    parser.add_argument("--imagination-microbatch", type=int, default=None,
                        help="start states imagined at once, with gradient "
                             "accumulation across groups; 0 imagines every "
                             "eligible replay state together")
    parser.add_argument("--critic-warmup", type=int, default=None,
                        help="actor-critic updates before the actor steps; "
                             "the actor has no signal through a fresh value "
                             "head, so this is a requirement not a margin")
    parser.add_argument("--profile-online", action="store_true",
                        help="measure per-phase wall time and CUDA peak "
                             "memory for each actor-critic update")
    parser.add_argument("--actor-lr", type=float, default=None,
                        help="actor learning rate")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="parallel online envs, stepped in lockstep with "
                             "updates between steps; more than 1 runs "
                             "ManiSkill's GPU backend")
    parser.add_argument("--reconfiguration-freq", type=int, default=None,
                        help="resets between scene rebuilds; unset keeps "
                             "ManiSkill's default, which for PegInsertionSide "
                             "is every reset at 1 env and never at more")
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--resume-from", default="",
        help="a directory holding world_model.pt (and optionally "
             "imitation.pt) from an earlier --save-checkpoints run. Those "
             "stages are restored instead of trained, so a rerun goes "
             "straight to online training. The arm, the env, the feature "
             "width, the pretrained revision and the normalization "
             "statistics all have to match; a mismatch is refused rather "
             "than coerced.")
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
                       ("burn_in", args.burn_in),
                       ("seed", args.seed)):
        if value is not None:
            overrides[key] = value
    online_overrides = {}
    for key, value in (("train_ratio", args.train_ratio),
                       ("precision", args.online_precision),
                       ("imagination_microbatch", args.imagination_microbatch),
                       ("critic_warmup", args.critic_warmup),
                       ("profile", True if args.profile_online else None),
                       ("actor_lr", args.actor_lr),
                       ("num_envs", args.num_envs),
                       ("reconfiguration_freq", args.reconfiguration_freq)):
        if value is not None:
            online_overrides[key] = value
    pretrain_overrides = {key: value for key, value in (
        ("world_lr", args.world_lr), ("imitation_lr", args.imitation_lr))
        if value is not None}
    cfg = load_config(args.task, args.experiment,
                      overrides={"data": overrides, "online": online_overrides,
                                 "pretrain": pretrain_overrides})
    model_yaml = Path(args.model_config)
    if not model_yaml.is_file():
        raise SystemExit(f"model config does not exist: {model_yaml}")
    cfg.setdefault("runtime", {})["model_config"] = str(model_yaml)

    # Seeded here, before anything is constructed. build() seeds again on its
    # own, but that is Stage 1A's call: the critic and the adapter are built
    # afterwards, and on the --resume-from path Stage 1A is skipped entirely.
    # Doing it once at the top covers fresh initialisation and restoration
    # alike, and makes the ordering a property of this function rather than of
    # which stages happen to run.
    seed = int(cfg["data"]["seed"])
    seed_everything(seed)

    root = Path(args.out or f"runs/sim_vla/{args.task}/{args.experiment}")
    print(f"[pipeline] {args.task} / {args.experiment} on {args.device} "
          f"seed {seed}", flush=True)
    print("[pipeline] checkpoints: "
          + (f"on -> {root}" if args.save_checkpoints
             else "off (stages pass their models in memory)"), flush=True)

    if args.resume_from:
        # Checked here rather than shrugged off inside run(): a typo would
        # otherwise retrain a stage this flag was passed precisely to skip,
        # and the run would look like it worked.
        resume_root = Path(args.resume_from)
        found = {name: path for name, path in stage_paths(resume_root).items()
                 if name != "online" and Path(path).is_file()}
        if not found:
            raise SystemExit(
                f"--resume-from {resume_root} holds no world_model.pt or "
                "imitation.pt. It wants the --out directory of an earlier "
                "--save-checkpoints run.")
        print("[pipeline] resuming: "
              + ", ".join(f"{name} <- {path}"
                          for name, path in sorted(found.items())), flush=True)

    logger = start_run(cfg, extra_config={
        "world_steps": int(args.world_steps),
        "imitation_steps": int(args.imitation_steps),
        "online_steps": int(args.online_steps),
        "device": str(args.device),
        "seed": int(cfg["data"]["seed"]),
        "model_config": str(model_yaml),
        "save_checkpoints": bool(args.save_checkpoints)})
    try:
        report = run(cfg, world_steps=args.world_steps,
                     imitation_steps=args.imitation_steps,
                     online_steps=args.online_steps, device=args.device,
                     root=root, save_checkpoints=args.save_checkpoints,
                     model_yaml=model_yaml, logger=logger,
                     resume_from=(Path(args.resume_from)
                                  if args.resume_from else None))
    except BaseException:
        # Marked failed rather than left running: a crashed 48-hour job that
        # shows as still-running in the dashboard is worse than no run at all.
        logger.finish(exit_code=1)
        raise
    logger.finish()
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
