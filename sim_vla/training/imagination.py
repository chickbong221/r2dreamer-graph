"""Imagined rollouts in latent space, for both arms.

The loop is the same for both; what differs is the width of the state and
whether a semantic prior runs::

    baseline  : (h, z)    -> adapter -> action -> (h', z')
    graph arm : (h, z, g) -> adapter -> action -> (h', z', g')

**No graph is extracted inside imagination.** There is no scene to extract one
from -- the states are predicted, not simulated -- so the graph arm's next
``g`` comes from the RSSM's semantic prior, which is what that prior was
trained for. A rollout that called the graph builder here would be conditioning
on a scene the model did not predict.

Gradients are kept throughout. The actor update differentiates the imagined
return with respect to the actions that produced it, so every transition, the
reward head and the flow sampler stay in the graph; the world model's
*parameters* are frozen for that update, which is not the same thing as
detaching its outputs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch


def flatten_start(post, graph_enabled: bool) -> tuple:
    """Fold (batch, time) into one batch of imagination start states.

    ``observe`` returns ``(stoch, deter, post_logit)`` and then ``sem``, so the
    semantic state is at index 3 and index 2 is a logit. Unpacked through the
    world model's own helper rather than by position.
    """
    from ..models.world_model import WorldModel

    stoch, deter, _logit, sem = WorldModel.unpack(post, graph_enabled)
    flat_stoch = stoch.reshape(-1, *stoch.shape[2:])
    flat_deter = deter.reshape(-1, deter.shape[-1])
    if not graph_enabled:
        return flat_stoch, flat_deter
    return flat_stoch, flat_deter, sem.reshape(-1, sem.shape[-1])


def imagine(world_model, actor, start, horizon: int, *, flow_steps: int = 10,
            instruction: Optional[torch.Tensor] = None,
            action_fn: Optional[Callable] = None) -> Dict[str, Any]:
    """Roll the latent dynamics forward under the actor.

    ``action_fn`` overrides how an action is produced from a feature, which is
    what lets a test drive this with a known policy rather than a flow sampler.
    """
    from ..models.flow_sampler import sample_actions

    graph_enabled = bool(world_model.graph_enabled)
    if graph_enabled:
        stoch, deter, sem = start
    else:
        stoch, deter = start
        sem = None

    feats: List[torch.Tensor] = []
    actions: List[torch.Tensor] = []
    batch = stoch.shape[0]

    for _ in range(int(horizon)):
        feat = (world_model.rssm.get_feat(stoch, deter, sem) if graph_enabled
                else world_model.rssm.get_feat(stoch, deter))
        if action_fn is not None:
            action = action_fn(feat)
        else:
            cond = actor.condition(feat, instruction)
            chunk = sample_actions(
                actor.velocity_fn(), cond, batch=batch,
                chunk=actor.chunk_size, dim=actor.action_dim,
                steps=int(flow_steps), device=feat.device, dtype=feat.dtype,
                differentiable=True)
            # The first action of the chunk is the one this transition uses,
            # which matches how the policy is executed online.
            action = chunk[:, 0]
        feats.append(feat)
        actions.append(action)
        stoch, deter = world_model.rssm.img_step(stoch, deter, action, sem=sem)
        if graph_enabled:
            # The prior, not an extractor: there is no scene here.
            sem, _ = world_model.rssm.semantic_prior(deter, sem)

    final = (world_model.rssm.get_feat(stoch, deter, sem) if graph_enabled
             else world_model.rssm.get_feat(stoch, deter))
    feats.append(final)
    return {
        "feat": torch.stack(feats, 0),            # (horizon + 1, B, D)
        "action": torch.stack(actions, 0),        # (horizon, B, A)
    }


def imagined_rewards(world_model, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Reward and continuation over an imagined rollout."""
    reward = world_model.reward_head(feat).mode()
    cont = world_model.cont_head(feat).mode()
    return {"reward": reward.squeeze(-1), "cont": cont.squeeze(-1)}


def lambda_return(reward: torch.Tensor, value: torch.Tensor,
                  cont: torch.Tensor, discount: float, lam: float
                  ) -> torch.Tensor:
    """Discounted lambda-return, computed backwards over the horizon.

    ``cont`` carries the bootstrap: under ``ignore_terminations`` it is one
    throughout, so the return bootstraps off the value at the horizon rather
    than being cut short by a terminal the online env never produces.
    """
    horizon = reward.shape[0]
    out = torch.empty_like(reward)
    carry = value[-1]
    for step in reversed(range(horizon)):
        carry = reward[step] + discount * cont[step] * (
            (1.0 - lam) * value[step + 1] + lam * carry)
        out[step] = carry
    return out
