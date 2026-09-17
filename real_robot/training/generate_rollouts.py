"""Short imagined transitions from recorded latent states, for the critics only.

    python -m real_robot.training.generate_rollouts --latents wm_base_final --name h1

1. **Behaviour policy.** An auxiliary Gaussian is fitted to every recorded
   latent/action pair of the cache (``behavior_policy`` in ``rollouts.yaml``)
   and checked on the diagnostic rows. It proposes plausible actions; it is
   not the IQL actor and adds nothing to its loss.
2. **Imagination.** Starting states are recorded latent states. The behaviour
   policy samples an action, the frozen world model advances the latent state,
   and its reward and continuation heads predict what the arrival is worth.
   Next states, rewards and continuation all come from the model; no recorded
   future graph, geometry or observation is attached to an imagined action.
   The graph decoder is not run at all.

The world model is the exact checkpoint the cache was encoded with: its file
SHA-256 and weight digest are checked before use, the model is frozen, and its
weights are checked again after generation. ``rollouts.json`` carries the
cache's compatibility record, which policy training compares field by field.
An output directory is never rewritten under a different contract.

A one-step horizon and a small critic share are starting restrictions. They
limit how much the critic relies on predictions; they do not make predictions
accurate.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from ..common import (
    add_config_arguments,
    file_sha256,
    load_configs,
    read_json,
    repo_path,
    stable_hash,
    utc_now,
    write_json,
)
from ..data.latent_dataset import ROLLOUT_FORMAT, LatentTransitions, resolve_latents
from ..models.behavior_policy import fit_behavior_policy, load_behavior_policy, save_behavior_policy
from ..models.world_model import load_world_model, weights_digest


def rollout_settings(cfg) -> Dict[str, Any]:
    """Everything that shapes the transitions; the device does not."""
    return {key: value for key, value in cfg.items() if key != "device"}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate imagined latent transitions for the critics.")
    parser.add_argument("--latents", required=True, help="latent cache name under paths.latents, or a path")
    parser.add_argument("--name", required=True, help="output directory name under paths.rollouts")
    parser.add_argument("--behavior-policy", default=None,
                        help="reuse a behaviour policy fitted on this same cache instead of fitting one")
    parser.add_argument("--force", action="store_true", help="regenerate a set with the same contract")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    configs = load_configs(["dataset", "rollouts"], args.overrides)
    cfg = configs["rollouts"]
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    if str(cfg["start_states"]) != "recorded":
        raise SystemExit("rollouts.start_states must be `recorded`: imagined rollouts start from recorded latents")

    source = LatentTransitions(resolve_latents(configs["dataset"], args.latents))
    compat = source.compatibility
    settings = rollout_settings(cfg)
    settings_hash = stable_hash(settings)
    out_dir = os.path.join(repo_path(configs["dataset"]["paths"]["rollouts"]), args.name)
    meta_path = os.path.join(out_dir, "rollouts.json")
    if os.path.isfile(meta_path):
        existing = read_json(meta_path)
        same = (existing.get("format") == ROLLOUT_FORMAT and existing.get("latent_compatibility") == compat
                and existing.get("settings_hash") == settings_hash)
        if not same:
            raise SystemExit(f"{out_dir} holds imagined transitions from a different latent cache or settings; "
                             "choose a new --name")
        if not args.force:
            print(f"[rollouts] {out_dir} already holds this set; pass --force to regenerate it")
            return
        os.remove(meta_path)
    os.makedirs(out_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(out_dir, "shard_*.npz")):
        os.remove(stale)

    world_model = source.identity["world_model"]
    checkpoint = repo_path(world_model["checkpoint"])
    model, payload, _ = load_world_model(checkpoint, device, expected_dataset=source.identity["dataset"],
                                         expected_weights={"sha256": world_model["sha256"],
                                                           "weights": world_model["weights"]})

    transitions = source.to_torch(device)
    diagnostic = transitions.diagnostic_rows()
    policy_identity = {"latent_compatibility": compat, "behavior_policy": stable_hash(cfg["behavior_policy"])}
    if args.behavior_policy:
        policy_path = repo_path(args.behavior_policy)
        policy, policy_payload = load_behavior_policy(policy_path, device, policy_identity)
        policy_sha = file_sha256(policy_path)
        report = policy_payload.get("report", {})
    else:
        print(f"[rollouts] fitting the behaviour policy on {transitions.size} recorded transitions", flush=True)
        policy, report = fit_behavior_policy(transitions, diagnostic, source.feat_dim, source.action_dim,
                                             cfg["behavior_policy"], device,
                                             log_dir=os.path.join(out_dir, "behavior_policy_log"))
        policy_path = os.path.join(out_dir, "behavior_policy.pt")
        policy_sha = save_behavior_policy(policy, policy_path, policy_identity, cfg["behavior_policy"]["network"],
                                          report)
        policy.requires_grad_(False)
        diagnostic_mse = report["diagnostic"]["action_mse"] if report["diagnostic"] else float("nan")
        print(f"[rollouts] behaviour policy: action MSE {report['training']['action_mse']:.5f} on all recorded "
              f"transitions, {diagnostic_mse:.5f} on diagnostic rows (also trained on)", flush=True)

    horizon, count, shard_size = int(cfg["horizon"]), int(cfg["count"]), int(cfg["shard_size"])
    noise = float(cfg["action_noise_std"])
    generator = torch.Generator(device=device).manual_seed(int(cfg["seed"]))
    rows: Dict[str, List[np.ndarray]] = {key: [] for key in ("z", "action", "reward", "z_next", "cont", "horizon",
                                                             "start_row")}
    written, shards, produced = 0, 0, 0
    reward_sum, cont_sum, generated = 0.0, 0.0, 0
    chunk = 4096

    def flush() -> None:
        nonlocal written, shards
        data = {key: np.concatenate(value) for key, value in rows.items()}
        np.savez(os.path.join(out_dir, f"shard_{shards:04d}.npz"), **data)
        written += int(data["z"].shape[0])
        shards += 1
        for value in rows.values():
            value.clear()
        print(f"[rollouts] {written} imagined transitions written", flush=True)

    while produced < count:
        starts = min(chunk, max(1, (count - produced + horizon - 1) // horizon))
        index = torch.randint(0, transitions.size, (starts,), device=device, generator=generator)
        z = transitions.feat[transitions.cur[index]].float()
        for step in range(horizon):
            if produced >= count:
                break
            action = policy.act(z, deterministic=False, noise_std=noise, generator=generator)
            out = model.imagine(z, action)
            take = min(starts, count - produced)
            rows["z"].append(z[:take].to(torch.float16).cpu().numpy())
            rows["action"].append(action[:take].cpu().numpy().astype(np.float32))
            rows["reward"].append(out["reward"][:take].float().cpu().numpy())
            rows["z_next"].append(out["feat"][:take].to(torch.float16).cpu().numpy())
            rows["cont"].append(out["cont"][:take].float().cpu().numpy())
            rows["horizon"].append(np.full(take, step + 1, dtype=np.int16))
            rows["start_row"].append(index[:take].cpu().numpy().astype(np.int64))
            reward_sum += float(out["reward"][:take].sum())
            cont_sum += float(out["cont"][:take].sum())
            generated += take
            produced += take
            z = out["feat"].float()
            if sum(len(a) for a in rows["z"]) >= shard_size:
                flush()
    if rows["z"]:
        flush()

    if weights_digest(model.state_dict()) != world_model["weights"]:
        raise RuntimeError("the world model's weights changed during generation; rollouts.json is not written")
    write_json(meta_path, {
        "format": ROLLOUT_FORMAT,
        "created": utc_now(),
        "latent_cache": source.root,
        "latent_compatibility": compat,
        "world_model": {"checkpoint": world_model["checkpoint"], "sha256": world_model["sha256"],
                        "weights": world_model["weights"], "step": int(payload["step"]), "frozen": True},
        "behavior_policy": {"path": os.path.relpath(policy_path, repo_path("")).replace(os.sep, "/"),
                            "sha256": policy_sha, "identity": policy_identity, "report": report},
        "settings": settings,
        "settings_hash": settings_hash,
        "shards": shards,
        "transitions": written,
        "predicted_reward_mean": reward_sum / max(generated, 1),
        "predicted_continuation_mean": cont_sum / max(generated, 1),
        "note": ("Next states, rewards and continuation are world-model predictions. No recorded future graph, "
                 "geometry or observation is attached to an imagined action."),
    })
    print(f"[rollouts] horizon {horizon}: {written} imagined transitions in {shards} shard(s) -> {out_dir}")


if __name__ == "__main__":
    main()
