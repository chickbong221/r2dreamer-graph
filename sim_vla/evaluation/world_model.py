"""World-model diagnostics, for both arms and a little more for the graph arm.

The open-loop rollout is the one that matters for imagination. A model whose
one-step reconstruction is good and whose fifteen-step rollout is not will
produce imagined returns that mean nothing, and the actor will happily optimise
them anyway -- so this is measured before the online stage rather than inferred
from it.
"""

from __future__ import annotations

from typing import Dict

import torch


@torch.no_grad()
def evaluate_world_model(model, batch: Dict[str, torch.Tensor], *,
                         horizon: int = 15) -> Dict[str, float]:
    """Reconstruction, reward prediction and an open-loop rollout."""
    out = model.observe(batch)
    post = out["post"]
    stoch, deter, _logit, sem = model.unpack(post, model.graph_enabled)
    feat = model.features(post)
    mask = batch["loss_mask"]
    denominator = mask.sum().clamp(min=1)

    metrics: Dict[str, float] = {}
    recon = model.decoder(stoch, deter, sem)   # stoch first, then deter
    for key, dist in recon.items():
        if key in batch:
            error = ((dist.mode() - batch[key]) ** 2).flatten(2).mean(-1)
            metrics[f"recon_mse_{key}"] = float((error * mask).sum() / denominator)

    reward_pred = model.reward_head(feat).mode()
    metrics["reward_mse"] = float(
        (((reward_pred - batch["reward"]) ** 2).squeeze(-1) * mask).sum()
        / denominator)

    # Open loop: condition on the first half, then predict forward with the
    # prior alone, which is what imagination will do.
    split = max(batch["action"].shape[1] // 2, 1)
    stoch, deter = stoch[:, split - 1], deter[:, split - 1]
    sem = sem[:, split - 1] if model.graph_enabled else None
    steps = max(min(int(horizon), batch["action"].shape[1] - split), 0)
    for offset in range(steps):
        action = batch["action"][:, split + offset]
        stoch, deter = model.rssm.img_step(stoch, deter, action, sem=sem)
        if model.graph_enabled:
            sem, _ = model.rssm.semantic_prior(deter, sem)
    if steps:
        open_feat = (model.rssm.get_feat(stoch, deter, sem)
                     if model.graph_enabled
                     else model.rssm.get_feat(stoch, deter))
        target = batch["reward"][:, split + steps - 1]
        metrics["open_loop_reward_mse"] = float(
            ((model.reward_head(open_feat).mode() - target) ** 2).mean())
    metrics["open_loop_steps"] = float(steps)

    if model.graph_enabled:
        # How far the prior that drives imagination is from the posterior the
        # graph produced. A large gap here is a graph arm whose imagined g is
        # not the g it was trained on.
        _, post_deter, _, post_sem = model.unpack(post, True)
        prior = model.rssm.semantic_prior_seq(post_deter)
        metrics["semantic_prior_mse"] = float(
            (((prior - post_sem) ** 2).mean(-1) * mask).sum() / denominator)
    return metrics
