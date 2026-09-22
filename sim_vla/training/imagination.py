"""Imagined rollouts in latent space, for both arms.

The loop is the same for both; what differs is the width of the state and
whether a semantic prior runs::

    baseline  : (h, z)    -> adapter -> chunk -> (h', z') -> ...
    graph arm : (h, z, g) -> adapter -> chunk -> (h', z', g') -> ...

**No graph is extracted inside imagination.** There is no scene to extract one
from -- the states are predicted, not simulated -- so the graph arm's next
``g`` comes from the RSSM's semantic prior, which is what that prior was
trained for. A rollout that called the graph builder here would be conditioning
on a scene the model did not predict.

That prior runs inside ``img_step``: with the branch on it returns
``(stoch, deter, sem, sem_logit)`` and has already advanced ``g``. This loop
therefore unpacks four values and advances nothing itself, matching
``dreamer.py:_imagine``.

One rollout, one chunk
----------------------

An imagined rollout is what the online policy does between two replans. From
each start state the actor is conditioned once and generates one action chunk,
and the chunk's first ``execute`` actions are stepped through the dynamics in
order::

    feat[0]      the start state s_0
    action[t]    chunk[:, t], clipped and mapped as the environment would run it
    feat[t + 1]  img_step(s_t, action[t])                        t = 0 .. E-1

That is ``E = execute`` imagined transitions from one sampler call. The online
policy replans after the same ``E`` actions (see
:class:`~sim_vla.training.online.LatentPolicy`), so the policy optimised here is
the policy that is executed. A start need not coincide with a replanning
boundary of the policy that collected it: every eligible replay state is
treated as the start of a fresh hypothetical chunk.

The actor update keeps gradients through the sampler, every transition and the
heads read afterwards. The world model's *parameters* are frozen for that
update, which is not the same thing as detaching its outputs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional

import torch


def eligible_rows(batch: Mapping[str, Any], rows: int, device) -> torch.Tensor:
    """Flattened ``(batch * time)`` indices an imagined rollout may start from.

    Scored rows only: ``loss_mask`` drops burn-in and padding -- a repeated
    final row is not a state the agent was ever in -- and ``valid``, padding
    alone, is ANDed in so that holds even for a loader that loosened
    ``loss_mask``. A terminal state is never a start either: nothing continues
    from it. Under ``ignore_terminations`` the windows carry no terminal flag,
    so that excludes nothing further; with terminations honoured it drops
    exactly the states the environment would not continue from.
    """
    mask = batch.get("loss_mask")
    if mask is None:
        return torch.arange(int(rows), device=device)
    eligible = mask.bool()
    if batch.get("valid") is not None:
        eligible = eligible & batch["valid"].bool()
    terminal = batch.get("is_terminal")
    if terminal is not None:
        eligible = eligible & ~terminal.bool()
    eligible = eligible.reshape(-1).to(device)
    if int(eligible.numel()) != int(rows):
        raise ValueError(
            f"{int(eligible.numel())} mask entries for {int(rows)} posterior "
            "rows; the masks and the encoded window disagree about its shape")
    return torch.nonzero(eligible, as_tuple=False).squeeze(-1)


def start_states(world_model, batch, *, post=None) -> tuple:
    """Every eligible replay position, as detached imagination starts.

    The posterior must be the *current* world model's: reusing one computed
    before the world-model step conditions the policy on states the model no
    longer produces. ``post`` is for a caller that has just encoded this batch
    with the updated model -- the online trainer shares one encoding between
    the progress head and imagination -- and is computed here otherwise.

    There is no cap. Every eligible position is a start, so their number is a
    property of the batch (windows x scored rows) rather than a setting, and
    ``imagination_microbatch`` is what bounds the memory an update uses.
    """
    with torch.no_grad():
        if post is None:
            post = world_model.observe(batch)["post"]
        flat = flatten_start(post, world_model.graph_enabled)
        keep = eligible_rows(batch, int(flat[0].shape[0]), flat[0].device)
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


def imagine_chunk(world_model, actor, start, execute: int, *,
                  flow_steps: int = 10,
                  instruction: Optional[torch.Tensor] = None,
                  action_fn: Optional[Callable] = None, coords=None,
                  differentiable: bool = True) -> Dict[str, Any]:
    """One generated chunk per start, its first ``execute`` actions imagined.

    ``action_fn`` maps the start feature to a whole ``(batch, chunk, action)``
    chunk in place of the flow sampler, which is what lets a test drive this
    with a known policy.

    ``differentiable`` is threaded into the flow sampler and has to be
    explicit: ``sample_actions`` re-enables grad internally, so an enclosing
    ``torch.no_grad()`` does not stop an expert graph being built. The caller
    that wants no graph at all -- conditioning included -- runs this under
    ``no_grad`` *and* passes False.

    ``coords`` applies the transformation the online policy applies: the
    sampled action is in normalized coordinates, it is clipped to the
    environment's bounds there (straight-through, so a saturated dimension
    still receives a gradient), and it is mapped into dynamics coordinates
    before the RSSM consumes it.
    """
    from ..models.flow_sampler import sample_actions

    execute = int(execute)
    if execute < 1:
        raise ValueError(f"execute={execute} must be at least 1")
    graph_enabled = bool(world_model.graph_enabled)
    if graph_enabled:
        stoch, deter, sem = start
    else:
        stoch, deter = start
        sem = None

    feat = (world_model.rssm.get_feat(stoch, deter, sem) if graph_enabled
            else world_model.rssm.get_feat(stoch, deter))
    batch = stoch.shape[0]
    if action_fn is not None:
        chunk = action_fn(feat)
    else:
        cond = actor.condition(feat, instruction)
        chunk = sample_actions(
            actor.velocity_fn(), cond, batch=batch,
            chunk=actor.chunk_size, dim=actor.action_dim,
            steps=int(flow_steps),
            # The actor's device, not the feature's: condition() moves the
            # feature to where the pretrained weights are, and the noise has
            # to start there too.
            device=getattr(actor, "device", feat.device), dtype=feat.dtype,
            differentiable=differentiable)
        # Back to the world model's device for img_step.
        chunk = chunk.to(feat.device)
    if chunk.dim() != 3 or int(chunk.shape[1]) < execute:
        raise ValueError(
            f"a chunk of shape {tuple(chunk.shape)} cannot supply "
            f"execute={execute} actions per start")

    feats: List[torch.Tensor] = [feat]
    actions: List[torch.Tensor] = []
    for step in range(execute):
        # What the environment would actually run, in the actor's coordinates.
        executed = (coords.executed(chunk[:, step]) if coords is not None
                    else chunk[:, step])
        actions.append(executed)
        # The RSSM reads dynamics coordinates. Converting here rather than
        # inside the RSSM keeps rssm.py exactly as the simulator has it.
        stepped = (coords.to_dynamics(executed) if coords is not None
                   else executed)
        # img_step advances the semantic state itself; unpacking two values
        # and calling semantic_prior again would advance g twice.
        result = world_model.rssm.img_step(stoch, deter, stepped, sem)
        if graph_enabled:
            stoch, deter, sem, _sem_logit = result
        else:
            stoch, deter = result
        feats.append(world_model.rssm.get_feat(stoch, deter, sem)
                     if graph_enabled
                     else world_model.rssm.get_feat(stoch, deter))
    return {
        "feat": torch.stack(feats, 0),            # (execute + 1, B, D)
        "action": torch.stack(actions, 0),        # (execute, B, A)
        # The stack is a *new* node; the objective's graph runs through these.
        # Probing the stack with autograd.grad returns None and says nothing.
        "action_steps": actions,
        "chunk": chunk,                           # (B, C, A), as sampled
    }


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
    probability of continuing -- and that is what the return wants anyway,
    since a hard 0/1 mode would make the bootstrap discontinuous.
    """
    reward = world_model.reward_head(feat).mode()
    cont = world_model.cont_head(feat).mean
    return {"reward": reward.squeeze(-1), "cont": cont.squeeze(-1)}


def chunk_return(reward: torch.Tensor, cont: torch.Tensor,
                 bootstrap: torch.Tensor, discount: float) -> torch.Tensor:
    """The discounted return of an executed chunk, bootstrapped at its end::

        G = r_0 + g c_0 (r_1 + g c_1 ( ... (r_(E-1) + g c_(E-1) V(s_E))))

    ``reward[t]`` and ``cont[t]`` belong to transition ``t`` (read at the
    successor -- see :mod:`sim_vla.training.actor_critic`) and ``bootstrap``
    is the value at the chunk's final state. This is the lambda-return at
    ``lambda = 1``. A continuation of zero at transition ``t`` keeps that
    transition's own reward -- the action earned it -- and removes every later
    reward and the bootstrap; continuation products do that on their own.
    Under ``ignore_terminations`` continuation is one throughout, so the return
    always bootstraps.
    """
    if tuple(reward.shape) != tuple(cont.shape):
        raise ValueError(
            f"reward {tuple(reward.shape)} and continuation "
            f"{tuple(cont.shape)} must share (transition, batch)")
    if tuple(bootstrap.shape) != tuple(reward.shape[1:]):
        raise ValueError(
            f"bootstrap {tuple(bootstrap.shape)} must be one value per start, "
            f"{tuple(reward.shape[1:])}")
    carry = bootstrap
    for step in reversed(range(int(reward.shape[0]))):
        carry = reward[step] + discount * cont[step] * carry
    return carry
