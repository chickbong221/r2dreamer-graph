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

The timeline
------------

An imagined rollout of horizon ``H`` produces ``H + 1`` features
``f_0 .. f_H`` and ``H`` actions ``a_0 .. a_(H-1)``, where ``a_t`` takes
``s_t`` to ``s_(t+1)``.

The reward head is trained -- in ``world_model.loss`` -- against
``batch["reward"]``, which the window layout defines as ``r_(t-1)``: the reward
earned by the transition that *arrived* at ``o_t``. So the head at a state
predicts the reward that got there, not the reward that leaves it. The reward
belonging to transition ``t`` is therefore read at the **successor**::

    reward[t]  ==  heads["reward"][t + 1]        for t in 0 .. H-1

which is ``heads["reward"][1:]``. Reading ``[:-1]`` instead took the reward
that preceded the rollout -- a value no imagined action can influence -- and
dropped the reward earned by the final action, which is the only reward that
action produces. The same shift applies to continuation: the flag at ``f_t``
describes the transition that arrived at ``s_t``, so whether transition ``t``
permits a bootstrap is ``heads["cont"][t + 1]``.

Values line up with states rather than transitions, so they are read at every
one of the ``H + 1`` features. The bootstrap at the horizon comes from the slow
target head, whose *parameters* are frozen but whose output stays
differentiable with respect to its input -- see
:meth:`~sim_vla.models.critics.ValueCritic.target_value`. The critic regresses
on ``feat[:-1]``, detached, against those same returns.

Frozen means "parameters do not move", not "outputs are constants": during the
actor update the world-model and critic parameters are excluded from the
gradient, but the gradient has to pass *through* their outputs to reach the
action. :func:`freeze_parameters` is the context manager that makes that
distinction operational, and a test asserts the frozen parameters are unchanged
afterwards and accumulate no gradients.
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
    critic_warmup: int = 150
    # Optional flow-matching anchor on demonstrations, mixed into the actor
    # update. Zero disables it; a nonzero value requires a demonstration
    # sampler to be supplied, and ActorCriticTrainer refuses the combination
    # rather than silently ignoring the weight.
    demo_anchor: float = 0.0
    progress_beta: float = 0.0


def critic_loss(critic, feat: torch.Tensor, returns: torch.Tensor
                ) -> torch.Tensor:
    """Detached targets, over every imagined step but the bootstrap."""
    return critic.loss(feat[:-1].detach(), returns)


def actor_loss(world_model, actor, critic, start, config: ActorCriticConfig,
               *, instruction=None, progress_reward=None, progress_head=None,
               coords=None, differentiable: bool = True) -> Dict[str, Any]:
    """Imagine under the policy and maximise the return it earns.

    ``differentiable=False`` builds no actor graph at all: the flow sampler
    runs under ``no_grad`` and the rollout is only good for detached critic
    targets. The previous code wrapped this call in ``torch.no_grad()`` and
    relied on that, but ``sample_actions`` re-enables grad internally, so a
    full expert graph was built anyway and then kept alive by the returned
    tensors.
    """
    with freeze_parameters(world_model, critic, progress_head):
        rollout = imagine(world_model, actor, start, config.horizon,
                          flow_steps=config.flow_steps, instruction=instruction,
                          coords=coords, differentiable=differentiable)
        feat = rollout["feat"]
        heads = imagined_rewards(world_model, feat)
        # Successor-indexed: see the module docstring. reward[t] is the reward
        # earned by the action that left s_t.
        reward = heads["reward"][1:]
        cont = heads["cont"][1:]
        shaping = progress_reward
        if progress_head is not None and config.progress_beta:
            from .progress import shaping_reward

            # Computed here rather than handed in, because it is a function of
            # *this* rollout's features: gamma * phi(s') - phi(s) over the
            # imagined trajectory. A precomputed tensor could silently belong
            # to a different rollout and would still have the right shape.
            # Potential-based, so it cannot change which policy is optimal.
            shaping = shaping_reward(progress_head, feat, config.discount)
        if shaping is not None and config.progress_beta:
            # Kept as a separate addend, and logged separately: the evaluation
            # reports environment return, and a shaping term folded in here
            # would not be visible in it.
            reward = reward + config.progress_beta * shaping
        # The live head for the values the actor differentiates through, and
        # the slow target for the bootstrap at the horizon -- which is what the
        # slow copy exists for. Both stay differentiable with respect to the
        # feature; only their parameters are frozen.
        value = critic.value(feat)
        value = torch.cat([value[:-1], critic.target_value(feat[-1:])], dim=0)
        returns = lambda_return(reward, value, cont, config.discount,
                                config.lam)
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
        # The *environment* reward stream, before shaping, so a run that scored
        # only on its own shaping is visible as exactly that.
        "reward": heads["reward"][1:].detach(),
        "shaped_reward": reward.detach(),
        "shaping": None if shaping is None else shaping.detach(),
        "cont": cont.detach(),
    }


class ActorCriticTrainer:
    """One optimizer for the policy, one for the critic, one warm-up."""

    def __init__(self, world_model, actor, critic, config: ActorCriticConfig,
                 *, coords=None, progress_head=None):
        self.world_model = world_model
        self.actor = actor
        self.critic = critic
        self.config = config
        self.coords = coords
        # Read by the actor objective and frozen there: imagination consults
        # the head, and the actor update must not train the head through the
        # shaping term.
        self.progress_head = progress_head
        if float(config.demo_anchor):
            raise NotImplementedError(
                "ActorCriticConfig.demo_anchor is declared but the anchored "
                "objective is not implemented: there is no demonstration "
                "sampler threaded into the actor update. Leave it at 0.0, or "
                "implement the anchor before switching it on -- a weight that "
                "is read and never applied is worse than one that is absent.")
        self.actor_opt = torch.optim.AdamW(
            [p for p in actor.parameters() if p.requires_grad],
            lr=config.actor_lr)
        self.critic_opt = torch.optim.AdamW(critic.net.parameters(),
                                            lr=config.critic_lr)
        self.step = 0

    def update(self, start, *, instruction=None, progress_reward=None,
               progress_beta: Optional[float] = None) -> Dict[str, float]:
        """Train the critic, then recompute and train the actor.

        The critic cannot be stepped while an actor graph that used its
        parameters is still waiting for backward; doing so triggers PyTorch's
        in-place version check. Conversely, training the actor before the very
        first critic update can produce an exactly zero gradient because the
        distribution heads start flat. The safe order is therefore:

        1. imagine once, with no actor graph at all, and update the critic from
           detached features/returns;
        2. update the slow target;
        3. if warm-up is over, imagine again -- this time differentiably --
           through the updated, frozen critic, and immediately consume that
           fresh actor graph.

        Recomputing is intentional. Retaining the first graph across the
        critic optimizer step would be invalid even if it happened not to
        raise on a particular PyTorch version.
        """
        self.step += 1
        if progress_beta is not None:
            # The warm-up is a function of environment steps, which the caller
            # counts, so the schedule arrives per update rather than being
            # recomputed from this trainer's own step counter.
            self.config.progress_beta = float(progress_beta)
        # This rollout supplies detached critic targets only, so it builds no
        # actor graph. torch.no_grad() alone would not have been enough:
        # sample_actions re-enables grad inside it.
        with torch.no_grad():
            critic_out = actor_loss(
                self.world_model, self.actor, self.critic, start, self.config,
                instruction=instruction, coords=self.coords,
                differentiable=False)

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
        # Dropped before a fresh graph is built, so the detached rollout's
        # tensors are not kept alive alongside it.
        del critic_out

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
                instruction=instruction, progress_reward=progress_reward,
                progress_head=self.progress_head,
                coords=self.coords, differentiable=True)
            if out.get("shaping") is not None:
                # Logged apart from the environment reward, always. An arm that
                # improved only on its own shaping must be visible as that.
                metrics |= {"shaping_reward": float(out["shaping"].mean()),
                            "progress_beta": float(self.config.progress_beta)}
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
