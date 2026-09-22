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

That prior runs inside ``img_step``: with the branch on it returns
``(stoch, deter, sem, sem_logit)`` and has already advanced ``g``. This loop
therefore unpacks four values and advances nothing itself, matching
``dreamer.py:_imagine``.

There are two rollout functions here, one per actor objective.

:func:`imagine` keeps gradients throughout. The ``pathwise`` actor update
differentiates the imagined return with respect to the actions that produced
it, so every transition, the reward head and the flow sampler stay in the
graph; the world model's *parameters* are frozen for that update, which is not
the same thing as detaching its outputs.

:func:`imagine_flow_reinforce` keeps none. The ``flow_reinforce`` objective
never differentiates through the rollout at all -- its gradient comes from
recomputing individual flow transitions against fixed recorded samples -- so
the rollout is collected as data and every tensor in it is detached. That is
the whole memory difference between the two: ``horizon * flow_steps``
transformer passes held live, versus none.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch


def start_states(world_model, batch, *, limit: int = 0, generator=None
                 ) -> tuple:
    """Fresh, detached, filtered imagination starts.

    Three things, and the first is the one that was wrong. The posterior must
    be re-encoded with the *current* world model: reusing the one computed
    before the update conditions the policy on states the model no longer
    produces, and wrapping a reshape in ``no_grad`` does not recompute
    anything.

    Then padding and burn-in positions are dropped -- a repeated final row is
    not a state the agent was ever in -- and the remainder is subsampled to
    ``limit``, because imagining from every position of every sequence is a
    batch the flow sampler cannot afford.
    """
    import torch

    with torch.no_grad():
        post = world_model.observe(batch)["post"]
        flat = flatten_start(post, world_model.graph_enabled)
        mask = batch.get("loss_mask")
        if mask is None:
            keep = torch.arange(flat[0].shape[0], device=flat[0].device)
        else:
            keep = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(-1)
        if limit and keep.numel() > int(limit):
            pick = torch.randperm(keep.numel(), device=keep.device,
                                  generator=generator)[: int(limit)]
            keep = keep[pick]
        # Detached explicitly: these seed a rollout the actor differentiates
        # through, and a gradient reaching back into the posterior would train
        # the world model from the actor's objective.
        return tuple(tensor[keep].detach() for tensor in flat)


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
            action_fn: Optional[Callable] = None, coords=None,
            differentiable: bool = True) -> Dict[str, Any]:
    """Roll the latent dynamics forward under the actor.

    ``action_fn`` overrides how an action is produced from a feature, which is
    what lets a test drive this with a known policy rather than a flow sampler.

    ``differentiable`` is the gradient policy for the whole rollout, threaded
    into the flow sampler. It has to be explicit: ``sample_actions`` re-enables
    grad internally, so an enclosing ``torch.no_grad()`` does not stop an
    expert graph being built, and the returned tensors then keep it alive.
    Critic targets and inference pass False; the actor update passes True.

    ``coords`` applies the same transformation the online policy applies: the
    sampled action is in normalized coordinates, it is clipped to the
    environment's bounds there (straight-through, so a saturated dimension
    still receives a gradient), and it is mapped into dynamics coordinates
    before the RSSM consumes it. Without this, imagination steps the dynamics
    with a value the environment would never accept, in units the world model
    was not trained on.
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
                steps=int(flow_steps),
                # The actor's device, not the feature's: condition() moves the
                # feature to where the pretrained weights are, and the noise
                # has to start there too.
                device=getattr(actor, "device", feat.device), dtype=feat.dtype,
                differentiable=differentiable)
            # Back to the world model's device for the next img_step.
            chunk = chunk.to(feat.device)
            # The first action of the chunk is the one this transition uses,
            # which matches how the policy is executed online: LatentPolicy
            # replans every ``execute`` steps and this replans every step, so
            # the two agree exactly at the default execute=1. Anything larger
            # is rejected before training rather than approximated here.
            action = chunk[:, 0]
        # What the environment would actually run, in the actor's coordinates.
        executed = coords.executed(action) if coords is not None else action
        feats.append(feat)
        actions.append(executed)
        # The RSSM reads dynamics coordinates. Converting here rather than
        # inside the RSSM keeps rssm.py exactly as the simulator has it.
        stepped = (coords.to_dynamics(executed) if coords is not None
                   else executed)
        # img_step advances the semantic state itself -- it calls
        # semantic_prior internally and returns (stoch, deter, sem, sem_logit)
        # when the branch is on. Unpacking two values and then calling
        # semantic_prior again would advance g twice per transition, which is
        # a different rollout than the one the prior was trained for.
        result = world_model.rssm.img_step(stoch, deter, stepped, sem)
        if graph_enabled:
            stoch, deter, sem, _sem_logit = result
        else:
            stoch, deter = result

    final = (world_model.rssm.get_feat(stoch, deter, sem) if graph_enabled
             else world_model.rssm.get_feat(stoch, deter))
    feats.append(final)
    return {
        "feat": torch.stack(feats, 0),            # (horizon + 1, B, D)
        "action": torch.stack(actions, 0),        # (horizon, B, A)
        # The stack is a *new* node; the objective's graph runs through these.
        # Probing the stack with autograd.grad returns None and says nothing.
        "action_steps": actions,
    }


def imagine_flow_reinforce(world_model, actor, start, horizon: int, *,
                           flow_steps: int, sigmas: torch.Tensor,
                           instruction: Optional[Any] = None, coords=None,
                           generator=None, store_device=None,
                           record_means: bool = False) -> Dict[str, Any]:
    """An imagined rollout recorded as data, with no autograd graph at all.

    This is the collection half of the ``flow_reinforce`` objective and the
    reason that objective is cheaper than the pathwise one. :func:`imagine`
    keeps the whole sequential expert chain alive so the return's gradient can
    be pushed back through it; here nothing is retained. The rollout is
    generated once under ``no_grad``, every tensor is stored detached, and the
    actor's gradient comes later from recomputing individual transition means
    against these fixed samples.

    Two consequences worth stating plainly. The flow states are recorded
    *before* ``coords.executed`` and ``coords.to_dynamics``: clipping and the
    coordinate change are deterministic downstream mappings, not Gaussian
    draws, and scoring them as if they were would attribute the policy a
    density it does not have. And the prefix cache is dropped at the end of
    every environment step -- it is the largest thing in the loop and nothing
    later needs it, because scoring rebuilds the prefix *with* gradients.

    ``record_means`` keeps the collection-time transition means so a later
    scoring pass can be checked against them. It doubles the record, so it is
    off unless a consistency check is being run.
    """
    from ..models.flow_sampler import sample_flow_path

    graph_enabled = bool(world_model.graph_enabled)
    if graph_enabled:
        stoch, deter, sem = start
    else:
        stoch, deter = start
        sem = None

    flow_steps = int(flow_steps)
    if tuple(sigmas.shape) != (flow_steps,):
        raise ValueError(
            f"expected one sigma per flow transition ({flow_steps},), got "
            f"{tuple(sigmas.shape)}")

    batch = stoch.shape[0]
    keep = (lambda tensor: tensor.detach().to(store_device).float()
            if store_device is not None else tensor.detach().float())

    feats: List[torch.Tensor] = []
    flow_states: List[torch.Tensor] = []
    flow_means: List[torch.Tensor] = []
    executed_actions: List[torch.Tensor] = []
    times: Optional[torch.Tensor] = None

    with torch.no_grad():
        for _ in range(int(horizon)):
            feat = (world_model.rssm.get_feat(stoch, deter, sem) if graph_enabled
                    else world_model.rssm.get_feat(stoch, deter))
            cond = actor.condition(feat, instruction)
            path = sample_flow_path(
                actor.velocity_fn(), cond, batch=batch,
                chunk=actor.chunk_size, dim=actor.action_dim,
                steps=flow_steps, sigmas=sigmas,
                # The actor's device, as in imagine(): condition() moves the
                # feature to where the pretrained weights live.
                device=getattr(actor, "device", feat.device),
                dtype=torch.float32, generator=generator)
            times = path["times"].detach()
            flow_states.append(keep(path["states"]))
            if record_means:
                flow_means.append(keep(path["means"]))
            action = path["chunk"][:, 0].to(feat.device)
            executed = coords.executed(action) if coords is not None else action
            feats.append(keep(feat))
            executed_actions.append(keep(executed))
            stepped = (coords.to_dynamics(executed) if coords is not None
                       else executed)
            result = world_model.rssm.img_step(stoch, deter, stepped, sem)
            if graph_enabled:
                stoch, deter, sem, _sem_logit = result
            else:
                stoch, deter = result
            # The prefix cache is the biggest object in this loop and scoring
            # rebuilds its own, with gradients. Holding this one would keep a
            # no-grad cache per imagined step for no purpose.
            del cond, path

        final = (world_model.rssm.get_feat(stoch, deter, sem) if graph_enabled
                 else world_model.rssm.get_feat(stoch, deter))
        feats.append(keep(final))

    record = {
        "features": torch.stack(feats, 0),                    # (H+1, B, F)
        "flow_states": torch.stack(flow_states, 0),           # (H, B, K+1, C, D)
        "executed_actions": torch.stack(executed_actions, 0),  # (H, B, D)
        "flow_times": times,                                  # (K,)
        "flow_sigmas": sigmas.detach().float(),               # (K,)
        "instruction": instruction,
        "horizon": int(horizon),
        "batch": int(batch),
        "flow_steps": flow_steps,
    }
    if record_means:
        record["flow_means"] = torch.stack(flow_means, 0)
    for name in ("features", "flow_states", "executed_actions"):
        if record[name].grad_fn is not None:
            raise RuntimeError(
                f"the imagined record's {name!r} carries a graph; collection "
                "must run entirely under no_grad or the memory this objective "
                "saves is spent anyway")
    return record


def gradient_chain(objective: torch.Tensor, rollout: Dict[str, Any],
                   actor) -> Dict[str, Any]:
    """Where an actor gradient dies, link by link.

    The actor objective runs objective -> return -> reward/value -> feature ->
    transition -> action -> flow sampler -> conditioning token -> adapter. A
    zero at the end says nothing about which link broke, and the chain is long
    enough that guessing is expensive. This reports each link separately.
    """
    report: Dict[str, Any] = {"objective_requires_grad": bool(objective.requires_grad)}
    if not objective.requires_grad:
        report["dead_at"] = "objective"
        return report

    # The per-step actions, not the stacked copy: the stack is built after the
    # fact and is not on the path from action to objective, so probing it
    # returns None whatever the truth is.
    steps = [a for a in (rollout.get("action_steps") or []) if a.requires_grad]
    if steps:
        grads = torch.autograd.grad(objective, steps, retain_graph=True,
                                    allow_unused=True)
        report["grad_to_action"] = [
            None if g is None else float(g.abs().sum()) for g in grads]
    else:
        report["grad_to_action"] = "no imagined action requires grad"

    params = [p for p in getattr(actor, "adapter", actor).parameters()
              if p.requires_grad]
    if params:
        grads = torch.autograd.grad(objective, params, retain_graph=True,
                                    allow_unused=True)
        report["grad_to_adapter"] = [
            None if g is None else float(g.abs().sum()) for g in grads]
    to_action = report.get("grad_to_action")
    to_adapter = report.get("grad_to_adapter", [1.0])
    live = lambda values: any(
        isinstance(g, float) and g > 0.0 for g in values)
    if isinstance(to_adapter, list) and live(to_adapter):
        # The parameters do receive gradient. If a caller still measured zero,
        # the loss is in how the norm was taken, not in the graph.
        report["dead_at"] = None
        report["note"] = ("gradient reaches the adapter; a zero norm means "
                          ".grad was not populated or a different parameter "
                          "list was measured")
    elif isinstance(to_action, list) and not live(to_action):
        report["dead_at"] = "return -> action"
    else:
        report["dead_at"] = "action -> adapter"
    return report


def imagined_rewards(world_model, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Reward and continuation over an imagined rollout.

    The two heads are read differently, and not by preference. ``reward`` is a
    ``symexp_twohot`` distribution whose ``mode()`` is a method; ``cont`` is a
    ``binary`` one whose ``mode`` is a *property*, so ``.mode()`` there calls a
    Tensor. ``dreamer.py:1280`` reads continuation as ``.mean`` -- the
    probability of continuing -- and that is what the lambda-return wants
    anyway, since a hard 0/1 mode would make the bootstrap discontinuous.
    """
    reward = world_model.reward_head(feat).mode()
    cont = world_model.cont_head(feat).mean
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
    # Accumulated into a list and stacked, as dreamer.py:_lambda_return does.
    # Writing into a preallocated tensor works, but this objective is
    # differentiated -- unlike dreamer's, which detaches the advantage -- so
    # the form that leaves no doubt about the graph is the one to use.
    carry = value[-1]
    collected = []
    for step in reversed(range(horizon)):
        carry = reward[step] + discount * cont[step] * (
            (1.0 - lam) * value[step + 1] + lam * carry)
        collected.append(carry)
    return torch.stack(list(reversed(collected)), 0)
