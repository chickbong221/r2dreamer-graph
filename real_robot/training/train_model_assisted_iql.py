"""Model-assisted IQL -- the policy-training path -- and the progress-aware branch.

    python -m real_robot.training.train_model_assisted_iql --latents wm_base_final --rollouts h1 \\
        --run-name maiql_base
    python -m real_robot.training.train_model_assisted_iql --latents wm_base_final --rollouts h1 \\
        --run-name maiql_progress --progress

Each update follows IQL's reference order (``models/iql.py``): the value learns
from recorded transitions, values are recomputed with the updated network, the
actor learns from recorded actions weighted by IQL's clipped advantage weights
with those updated values, the critics learn from recorded transitions plus
the configured share of imagined ones, and the target critic follows. No
behaviour-cloning loss is added to the actor and no recorded-only policy is
trained first.

With ``--progress`` a second critic and value learn the discounted
potential-difference reward. For recorded transitions the potential is the
observed-graph potential cached with the latents; for imagined ones it is a
head fitted to those potentials on every recorded transition and measured on
the diagnostic rows. Both arms share one world model, one latent cache, one
task reward and one set of imagined transitions, and differ only in the actor's
objective. Networks, recorded batches, imagined batches and the progress head
each draw from their own seed, so the branch does not move the base run's
initialisation or batches. Expectile regression is non-linear, so this is not
IQL on a summed reward; it is the progress-aware method adapted to IQL.

Diagnostics -- action error, TD error, value statistics, advantage weights,
and recorded and imagined critic errors reported separately -- are logged as
``diagnostic/...`` on the diagnostic rows, which are trained on. The policy a
run produces is ``final.pt``; ``step_XXXXXXXX.pt`` snapshots are kept; no
checkpoint is selected by action error.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from ..common import RunLogger, add_config_arguments, load_configs, repo_path, stable_hash, utc_now, write_json
from ..data.latent_dataset import LatentTransitions, SyntheticTransitions, resolve_latents
from ..data.selection import DIAGNOSTIC_NOTE
from ..models.iql import IQL, potential_difference
from .checkpoints import CheckpointManager, restore_rng

CONSOLE = ("loss/critic", "loss/value", "loss/actor", "critic/td_abs_recorded", "critic/td_abs_synthetic",
           "weight/clipped_fraction", "progress/influence_raw", "diagnostic/action_mse",
           "diagnostic/td_abs_recorded", "diagnostic/td_abs_synthetic")


def concat_batches(first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = set(first) & set(second)
    return {key: torch.cat([first[key], second[key]]) for key in keys}


def head_rows(batch: Mapping[str, torch.Tensor], count: int) -> Dict[str, torch.Tensor]:
    return {key: value[:count] for key, value in batch.items()}


def with_progress_terms(batch: Dict[str, torch.Tensor], gamma: float, head=None) -> Dict[str, torch.Tensor]:
    """Add ``progress_reward`` and its validity: cached potentials for recorded rows, the head for imagined ones."""
    if head is None:
        batch["progress_reward"] = potential_difference(batch["phi"], batch["phi_next"], batch["cont"], gamma)
        batch["progress_valid"] = batch["phi_valid"]
    else:
        with torch.no_grad():
            phi, phi_next = head(batch["z"]), head(batch["z_next"])
        batch["progress_reward"] = potential_difference(phi, phi_next, batch["cont"], gamma)
        batch["progress_valid"] = torch.ones_like(batch["reward"])
    return batch


def resolve_rollouts(dataset_cfg: Mapping[str, Any], reference: str) -> str:
    path = repo_path(reference)
    if os.path.isdir(path):
        return path
    return os.path.join(repo_path(dataset_cfg["paths"]["rollouts"]), reference)


def run_identity(source: LatentTransitions, synthetic: SyntheticTransitions, iql_cfg: Mapping[str, Any],
                 run_cfg: Mapping[str, Any], use_progress: bool) -> Dict[str, Any]:
    progress = run_cfg["progress"]
    return {
        "kind": "model_assisted_iql_progress" if use_progress else "model_assisted_iql",
        "latents": source.compatibility,
        "rollouts": synthetic.identity(),
        "iql": stable_hash(dict(iql_cfg)),
        "critic": dict(run_cfg["critic"]),
        "seeds": dict(run_cfg["seeds"]),
        "steps": int(run_cfg["train"]["steps"]),
        "batch_size": int(run_cfg["train"]["batch_size"]),
        "progress_beta": float(progress["beta"]) if use_progress else 0.0,
        "progress": ({"schedule": source.identity.get("progress"), "head": stable_hash(progress["head"])}
                     if use_progress else None),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Model-assisted IQL, optionally progress-aware.")
    parser.add_argument("--latents", required=True, help="latent cache name under paths.latents, or a path")
    parser.add_argument("--rollouts", required=True, help="imagined transitions under paths.rollouts, or a path")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--resume", action="store_true")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    configs = load_configs(["dataset", "iql", "model_assisted_iql"], args.overrides)
    iql_cfg, run_cfg = configs["iql"], configs["model_assisted_iql"]
    device = torch.device(run_cfg["device"] if torch.cuda.is_available() else "cpu")
    seeds = {key: int(value) for key, value in run_cfg["seeds"].items()}
    train_cfg, progress_cfg = run_cfg["train"], run_cfg["progress"]

    source = LatentTransitions(resolve_latents(configs["dataset"], args.latents))
    use_progress = bool(args.progress or progress_cfg["enabled"])
    if use_progress and source.progress() is None:
        raise SystemExit("this latent cache has no progress potentials; encode it again with --progress "
                         "under a new --name")
    recorded = source.to_torch(device, include_progress=use_progress)
    diagnostic = recorded.diagnostic_rows()
    if diagnostic.size == 0:
        raise SystemExit("the latent cache has no diagnostic rows")
    rollouts = resolve_rollouts(configs["dataset"], args.rollouts)
    synthetic_source = SyntheticTransitions(rollouts, source.compatibility)
    synthetic = synthetic_source.to_torch(device)

    identity = run_identity(source, synthetic_source, iql_cfg, run_cfg, use_progress)
    run_dir = os.path.join(repo_path(configs["dataset"]["paths"]["runs"]), "iql", args.run_name)
    manager = CheckpointManager(run_dir, identity)
    logger = RunLogger(run_dir, CONSOLE)
    meta = {"latents": source.root, "rollouts": rollouts, "iql": dict(iql_cfg), "run": dict(run_cfg),
            "progress": use_progress, "progress_beta": identity["progress_beta"]}

    steps, batch_size = int(train_cfg["steps"]), int(train_cfg["batch_size"])
    # Built first and from its own seed: identical with and without the progress branch.
    agent = IQL(source.feat_dim, source.action_dim, iql_cfg, total_steps=steps,
                progress_beta=identity["progress_beta"], init_seed=seeds["networks"]).to(device)
    recorded_generator = torch.Generator(device=device).manual_seed(seeds["recorded_sampling"])
    synthetic_generator = torch.Generator(device=device).manual_seed(seeds["synthetic_sampling"])

    head = None
    if use_progress:
        from ..models.progress_adapter import ProgressHead

        head = ProgressHead(source.feat_dim, progress_cfg["head"]["hidden"], device, seed=seeds["progress_head"])

    def state():
        payload = {"model": agent.state_dict(), "optimizer": agent.optimizer_state(),
                   "generators": {"recorded": recorded_generator.get_state(),
                                  "synthetic": synthetic_generator.get_state()}}
        if head is not None:
            payload["progress_head"] = head.state_dict()
        return payload

    start = 0
    if args.resume:
        payload = manager.resume()
        if payload is None:
            raise SystemExit(f"--resume: no latest.pt in {run_dir}")
        agent.load_state_dict(payload["state"]["model"])
        agent.load_optimizer_state(payload["state"]["optimizer"])
        recorded_generator.set_state(payload["state"]["generators"]["recorded"])
        synthetic_generator.set_state(payload["state"]["generators"]["synthetic"])
        if head is not None:
            head.load_state_dict(payload["state"]["progress_head"])
        restore_rng(payload["rng"])
        start = int(payload["step"])
        print(f"[maiql] resumed from step {start}", flush=True)
    else:
        if os.path.isfile(manager.path("latest")) or os.path.isfile(manager.path("final")):
            raise SystemExit(f"{run_dir} already has a run; pass --resume or choose another --run-name")
        if head is not None:
            head_cfg = progress_cfg["head"]
            report = head.fit(recorded, diagnostic, int(head_cfg["steps"]), int(head_cfg["batch_size"]),
                              float(head_cfg["lr"]), float(head_cfg["huber_delta"]),
                              log=lambda step, values: logger.write(step, values))
            write_json(os.path.join(run_dir, "progress_head.json"), {**report, "created": utc_now()})
            diagnostic_mae = report["diagnostic"]["mae"] if report["diagnostic"] else float("nan")
            print(f"[maiql] progress head: MAE {report['recorded']['mae']:.4f} on all recorded transitions, "
                  f"{diagnostic_mae:.4f} on diagnostic rows (also fitted on)", flush=True)
    write_json(os.path.join(run_dir, "config.json"), {
        "iql": iql_cfg, "run": run_cfg, "identity": identity, "latents": source.root, "rollouts": rollouts,
        "progress": use_progress, "created": utc_now(), "note": DIAGNOSTIC_NOTE,
        "update_order": ["value (recorded)", "recompute values with the updated value network",
                         "actor (recorded, clipped advantage weights, no behaviour-cloning term)",
                         "critics (recorded + imagined)", "target critics"],
        "policy_output": "final.pt",
    })

    fraction = float(run_cfg["critic"]["synthetic_fraction"])
    n_synthetic = int(round(batch_size * fraction))
    gamma = float(iql_cfg["gamma"])
    synthetic_rows = int(run_cfg["diagnostics"]["synthetic_rows"])
    print(f"[maiql] value/actor batches: {batch_size} recorded; critic batches: {batch_size - n_synthetic} "
          f"recorded + {n_synthetic} imagined", flush=True)

    def diagnostics() -> Dict[str, float]:
        recorded_batches = diagnostic.iterate(4096)
        imagined = [synthetic.head(synthetic_rows)]
        values = agent.diagnostics(recorded_batches, imagined)
        return {f"diagnostic/{key}": value for key, value in values.items()}

    snapshot_every = int(train_cfg.get("snapshot_every") or 0)
    for step in range(start, steps):
        real = recorded.sample(batch_size, generator=recorded_generator)
        if use_progress:
            real = with_progress_terms(real, gamma)
        if n_synthetic > 0:
            imagined = synthetic.sample(n_synthetic, generator=synthetic_generator)
            if use_progress:
                imagined = with_progress_terms(imagined, gamma, head)
            critic_batch = concat_batches(head_rows(real, batch_size - n_synthetic), imagined)
        else:
            critic_batch = real
        metrics = agent.update(real, critic_batch)
        done = step + 1
        if done % int(train_cfg["log_every"]) == 0:
            logger.write(done, metrics)
        if done % int(train_cfg["diagnostics_every"]) == 0 or done == steps:
            logger.write(done, diagnostics())
        if done % int(train_cfg["checkpoint_every"]) == 0:
            manager.save("latest", done, state(), meta)
        if snapshot_every and done % snapshot_every == 0:
            manager.snapshot(done, state(), meta)
    manager.save("latest", steps, state(), meta)
    manager.save("final", steps, state(), meta)
    logger.close()
    print(f"[maiql] finished {steps} updates -> {run_dir} (policy: final.pt)", flush=True)


if __name__ == "__main__":
    main()
