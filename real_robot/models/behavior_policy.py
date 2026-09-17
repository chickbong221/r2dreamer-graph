"""The behaviour policy that proposes actions for imagined rollouts.

An auxiliary imitation model: a Gaussian over normalised commands, fitted by
maximum likelihood to the recorded latent/action pairs of one latent cache.
``generate_rollouts`` uses it to take plausible actions from recorded latent
states, so the world model is asked about states near the data. It is not the
IQL actor: it adds no loss to the actor, does not initialise it, and is not a
benchmark the actor is compared against. Its settings live under
``behavior_policy`` in ``rollouts.yaml``.

Its actions are samples from its distribution; ``action_noise_std`` adds noise
on top of that sampling, it does not create it.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from ..common import RunLogger, file_sha256, require_identity
from ..data.selection import DIAGNOSTIC_NOTE
from .iql import GaussianPolicy


@torch.no_grad()
def action_error(policy: GaussianPolicy, transitions) -> Dict[str, float]:
    squared, log_prob, rows = 0.0, 0.0, 0
    for batch in transitions.iterate(4096):
        squared += float((policy.act(batch["z"]) - batch["action"]).square().mean(-1).sum())
        log_prob += float(policy.log_prob(batch["z"], batch["action"]).sum())
        rows += int(batch["z"].shape[0])
    return {"action_mse": squared / max(rows, 1), "log_prob": log_prob / max(rows, 1), "rows": float(rows)}


def fit_behavior_policy(transitions, diagnostic, z_dim: int, a_dim: int, cfg: Mapping[str, Any], device,
                        log_dir: Optional[str] = None) -> Tuple[GaussianPolicy, Dict[str, Any]]:
    """Fit on every recorded transition; report the fit on all of them and on the diagnostic rows."""
    net = cfg["network"]
    seed = int(cfg["seed"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        policy = GaussianPolicy(z_dim, a_dim, net)
    policy = policy.to(device)
    generator = torch.Generator(device=torch.device(device)).manual_seed(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(cfg["lr"]))
    logger = RunLogger(log_dir, ("behavior/loss",)) if log_dir else None
    steps, batch_size, log_every = int(cfg["steps"]), int(cfg["batch_size"]), int(cfg["log_every"])
    for step in range(1, steps + 1):
        batch = transitions.sample(batch_size, generator=generator)
        loss = -policy.log_prob(batch["z"], batch["action"]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if logger is not None and step % log_every == 0:
            logger.write(step, {"behavior/loss": float(loss)})
    policy.eval()
    report = {
        "steps": steps,
        "training": action_error(policy, transitions),
        "diagnostic": action_error(policy, diagnostic) if diagnostic.size else None,
        "log_std_mean": float(policy.log_std.mean()),
        "note": DIAGNOSTIC_NOTE,
    }
    if logger is not None:
        logger.close()
    return policy, report


def save_behavior_policy(policy: GaussianPolicy, path: str, identity: Mapping[str, Any], net: Mapping[str, Any],
                         report: Mapping[str, Any]) -> str:
    from checkpointing import atomic_save

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    atomic_save({"state": policy.state_dict(), "identity": dict(identity), "network": dict(net),
                 "report": dict(report), "z_dim": int(policy.mean.net[0].in_features),
                 "a_dim": int(policy.log_std.numel())}, path)
    return file_sha256(path)


def load_behavior_policy(path: str, device, expected_identity: Mapping[str, Any]
                         ) -> Tuple[GaussianPolicy, Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require_identity(expected_identity, payload["identity"], f"behaviour policy {path}")
    policy = GaussianPolicy(payload["z_dim"], payload["a_dim"], payload["network"])
    policy.load_state_dict(payload["state"])
    return policy.to(device).eval().requires_grad_(False), payload
