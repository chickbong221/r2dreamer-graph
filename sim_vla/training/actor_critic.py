"""The online actor and critic objectives, shared by both arms.

The actor update differentiates the imagined return with respect to the actions
that produced it. Concretely: sample a chunk from the flow policy with the
graph kept, step the frozen world model with it, read the reward head and the
critic, and push that return's gradient back through every integration step of
the sampler into the adapter.

What this deliberately is not: ``log pi(a) * advantage``. A flow policy has no
tractable log-probability, and reusing the flow-matching regression as if it
were one would weight a reconstruction error by an advantage. The existing
simulator actor keeps its own loss; this is a different objective and it is the
same for both arms, so a difference between them is a difference in the state
and not in the update rule.

During the actor update the world-model and critic *parameters* are frozen and
their outputs are not detached -- the gradient has to pass through them to
reach the action. :func:`freeze_parameters` is the context manager that makes
that distinction operational, and a test asserts the frozen parameters are
unchanged afterwards.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch

from .imagination import imagine, imagined_rewards, lambda_return


@contextlib.contextmanager
def freeze_parameters(*modules):
    """Stop these modules' parameters training, without cutting the graph."""
    saved: List[tuple] = []
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            saved.append((parameter, parameter.requires_grad))
            parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, previous in saved:
            parameter.requires_grad_(previous)


@dataclass
class ActorCriticConfig:
    horizon: int = 15
    discount: float = 0.997
    lam: float = 0.95
    actor_lr: float = 3e-5
    critic_lr: float = 3e-4
    flow_steps: int = 10
    grad_clip: float = 1.0
    critic_warmup: int = 500
    demo_anchor: float = 0.0        # optional flow loss on demonstrations
    progress_beta: float = 0.0


def critic_loss(critic, feat: torch.Tensor, returns: torch.Tensor
                ) -> torch.Tensor:
    """Detached targets, over every imagined step but the bootstrap."""
    return critic.loss(feat[:-1].detach(), returns)


def actor_loss(world_model, actor, critic, start, config: ActorCriticConfig,
               *, instruction=None, progress_reward=None) -> Dict[str, Any]:
    """Imagine under the policy and maximise the return it earns."""
    with freeze_parameters(world_model, critic):
        rollout = imagine(world_model, actor, start, config.horizon,
                          flow_steps=config.flow_steps, instruction=instruction)
        feat = rollout["feat"]
        heads = imagined_rewards(world_model, feat)
        reward = heads["reward"][:-1]
        if progress_reward is not None and config.progress_beta:
            # Kept as a separate addend, and logged separately: the evaluation
            # reports environment return, and a shaping term folded in here
            # would not be visible in it.
            reward = reward + config.progress_beta * progress_reward
        # The live head for the values the actor differentiates through, and
        # the slow target for the bootstrap at the horizon -- which is what the
        # slow copy exists for.
        value = critic.value(feat)
        value = torch.cat([value[:-1], critic.target_value(feat[-1:])], dim=0)
        returns = lambda_return(reward, value, heads["cont"][:-1],
                                config.discount, config.lam)
        objective = -returns.mean()
    return {
        "loss": objective,
        "returns": returns.detach(),
        "feat": feat,
        "action": rollout["action"],
        # Forwarded so a diagnostic can probe the real graph nodes. The stacked
        # "action" is built after the rollout and is not on the path to the
        # objective, so autograd.grad against it always returns None.
        "action_steps": rollout.get("action_steps", []),
        "reward": reward.detach(),
    }


class ActorCriticTrainer:
    """One optimizer for the policy, one for the critic, one warm-up."""

    def __init__(self, world_model, actor, critic, config: ActorCriticConfig):
        self.world_model = world_model
        self.actor = actor
        self.critic = critic
        self.config = config
        self.actor_opt = torch.optim.AdamW(
            [p for p in actor.parameters() if p.requires_grad],
            lr=config.actor_lr)
        self.critic_opt = torch.optim.AdamW(critic.net.parameters(),
                                            lr=config.critic_lr)
        self.step = 0

    def update(self, start, *, instruction=None) -> Dict[str, float]:
        """Train the critic, then recompute and train the actor.

        The critic cannot be stepped while an actor graph that used its
        parameters is still waiting for backward; doing so triggers PyTorch's
        in-place version check. Conversely, training the actor before the very
        first critic update can produce an exactly zero gradient because the
        distribution heads start flat. The safe order is therefore:

        1. imagine once and update the critic from detached features/returns;
        2. update the slow target;
        3. if warm-up is over, imagine again through the updated, frozen critic
           and immediately consume that fresh actor graph.

        Recomputing is intentional. Retaining the first graph across the
        critic optimizer step would be invalid even if it happened not to
        raise on a particular PyTorch version.
        """
        self.step += 1
        # This rollout supplies detached critic targets only. Avoid retaining
        # a full SmolVLA/world-model graph that can never be used.
        with torch.no_grad():
            critic_out = actor_loss(
                self.world_model, self.actor, self.critic, start, self.config,
                instruction=instruction)

        # Detached inputs and detached targets: this graph belongs only to the
        # critic and can be consumed before a new actor graph is constructed.
        closs = critic_loss(
            self.critic, critic_out["feat"].detach(), critic_out["returns"])
        self.critic_opt.zero_grad(set_to_none=True)
        closs.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.net.parameters(),
                                       self.config.grad_clip)
        self.critic_opt.step()
        self.critic.update_target()

        metrics = {"return": float(critic_out["returns"].mean()),
                   "reward": float(critic_out["reward"].mean()),
                   "critic_loss": float(closs.detach())}

        warming = self.step <= self.config.critic_warmup
        if warming:
            # A random critic gives the actor a gradient toward noise, so the
            # policy is left alone until the value head means something.
            metrics["actor_loss"] = float("nan")
        else:
            # The critic and its slow target changed above, so build a fresh
            # objective. This graph is consumed before either is modified
            # again.
            out = actor_loss(
                self.world_model, self.actor, self.critic, start, self.config,
                instruction=instruction)
            trainable = [p for p in self.actor.parameters() if p.requires_grad]
            if not trainable:
                raise RuntimeError(
                    "the actor has no trainable parameters; the adapter and "
                    "the action expert are supposed to be")
            self.actor_opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            # Measured from .grad rather than taken from clip_grad_norm_'s
            # return, and reported alongside how many parameters actually
            # received one. A single number cannot distinguish "the gradient
            # is zero" from "nothing was measured".
            with torch.no_grad():
                populated = [p.grad for p in trainable if p.grad is not None]
                raw_norm = (
                    float(torch.sqrt(sum((g.detach() ** 2).sum()
                                         for g in populated)))
                    if populated else 0.0)
            torch.nn.utils.clip_grad_norm_(trainable, self.config.grad_clip)
            self.actor_opt.step()
            metrics |= {"actor_loss": float(out["loss"].detach()),
                        "actor_grad_norm": raw_norm,
                        "actor_params_with_grad": float(len(populated)),
                        "actor_params_trainable": float(len(trainable))}
        return metrics
