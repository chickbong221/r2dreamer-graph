"""The online actor and critic objectives, shared by both arms.

There are two actor objectives, selected by ``ActorCriticConfig.actor_objective``
and identical across the arms: whichever is chosen, both arms use it, so a
difference between them stays a difference in the state and not in the update
rule.

``pathwise`` (the default, and the original) differentiates the imagined return
with respect to the actions that produced it. Sample a chunk from the flow
policy with the graph kept, step the frozen world model with it, read the
reward head and the critic, and push that return's gradient back through every
integration step of the sampler into the adapter.

``flow_reinforce`` is a Dreamer-style score-function update. The deterministic
Euler sampler has no density, so this objective samples a *different*,
explicitly stochastic policy -- the same field with fixed Gaussian noise at
every transition -- collects an imagined rollout of it under ``no_grad``, and
then differentiates ``sum_k log pi(u_(k+1) | u_k, s)`` against a detached,
return-EMA-normalized advantage. Nothing is differentiated through the imagined
dynamics or through the sequential denoising chain.

Two things ``flow_reinforce`` is deliberately **not**. It is not PPO: there is
no importance ratio, no ratio clipping, no old-policy copy and no second epoch
over an imagined batch -- exactly one optimizer step per freshly collected
batch, so every actor parameter still equals the one that drew the samples when
they are scored. And it does not reuse the flow-matching regression as a
log-probability: that is a squared error, and weighting it by an advantage
would be weighting a reconstruction error. The imitation anchor uses the
flow-matching loss as what it is -- an imitation gradient -- and is summed in
separately.

One consequence worth knowing before reading a flat training curve: **neither
objective has a learning signal through identically zero reward and value
heads**, and both start that way. The heads' output layers are initialised to
zero, so their outputs are constant in the feature and their gradient with
respect to it is zero too. ``flow_reinforce`` then multiplies ``log pi`` by a
zero advantage; ``pathwise`` differentiates a return that does not depend on
the action. Measured on a fresh toy model, both give an actor gradient norm of
exactly 0.0.

So ``critic_warmup`` is load-bearing for both, and a zero actor gradient early
in a run is expected rather than a bug. What distinguishes "warming up" from
"broken" is whether the advantage and the RL gradient become nonzero *after*
warm-up, which is why ``advantage_abs``, ``advantage_std`` and
``rl_grad_norm`` are logged rather than left to be inferred from the loss.

A caution for anyone reading the older tests: asserting that ``p.grad is not
None`` does not show a live gradient. Both objectives populate ``.grad`` with
zeros in that state.

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
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch

import networks

from .imagination import (imagine, imagine_flow_reinforce, imagined_rewards,
                          lambda_return)
from ..models.flow_sampler import check_sigmas
from .precision import autocast
from .profiling import Phases


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


PATHWISE = "pathwise"
FLOW_REINFORCE = "flow_reinforce"
ACTOR_OBJECTIVES = (PATHWISE, FLOW_REINFORCE)


@contextlib.contextmanager
def deterministic_modules(*modules):
    """``eval()`` for the duration, restoring each module's previous mode.

    Collection and scoring have to see the *same* network. If dropout were
    active, the transition mean recomputed while scoring would not be the mean
    that was actually sampled from, so the Gaussian being differentiated would
    belong to a different distribution than the one that produced ``u_(k+1)``
    -- a wrong density that still trains, still descends, and reports nothing.

    ``eval()`` is not ``no_grad()``: this changes dropout and batch-norm
    behaviour only, and scoring still builds its graph inside it.
    """
    saved = [(module, bool(module.training)) for module in modules
             if module is not None and hasattr(module, "train")]
    try:
        for module, _previous in saved:
            module.eval()
        yield
    finally:
        for module, previous in saved:
            module.train(previous)


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
    # Accumulate the same mean objective over smaller groups of start states.
    # Zero runs the whole imagination batch at once (the old behavior).
    imagination_microbatch: int = 0
    precision: str = "float32"
    # Optional flow-matching anchor on demonstrations, mixed into the actor
    # update. Zero disables it; a nonzero value requires a demonstration
    # sampler to be supplied, and ActorCriticTrainer refuses the combination
    # rather than silently ignoring the weight.
    demo_anchor: float = 0.0
    progress_beta: float = 0.0

    # ------------------------------------------------------- actor objective
    # "pathwise" is the original: differentiate the imagined return through
    # every flow step. "flow_reinforce" collects the rollout without a graph
    # and differentiates log pi of the recorded denoising path against a
    # detached advantage. The default is unchanged, so an existing run is
    # unaffected by this setting existing.
    actor_objective: str = PATHWISE
    # Injected Gaussian noise for flow_reinforce's stochastic sampler. Required
    # and strictly positive for that objective; unused by pathwise.
    flow_noise_std: float = 0.0
    flow_noise_schedule: str = "constant_per_step_scaled_by_sqrt_k"
    # Scored (environment-step, start-state, flow-step) tuples per backward.
    # This, not imagination_microbatch, is what bounds the actor's backward
    # memory under flow_reinforce.
    actor_transition_microbatch: int = 16
    # The anchor has two budgets and they cost different things.
    #
    # anchor_windows is how many demonstration *windows* are drawn and encoded;
    # each costs a world-model forward over the whole window. anchor_rows is
    # how many eligible *positions* survive to be conditioned; each costs one
    # actor forward. One window yields many eligible rows, so tying the two
    # together -- drawing anchor_rows windows to keep anchor_rows rows -- paid
    # for the expensive half roughly `sequence_length` times over.
    anchor_windows: int = 8
    anchor_window_microbatch: int = 4
    anchor_rows: int = 64
    anchor_microbatch: int = 16
    # Bounded retries when a sampled window has no eligible anchor row before
    # the run is failed rather than quietly unanchored.
    anchor_retries: int = 4
    # How often to pay for a separate RL-versus-anchor gradient measurement.
    # Exact measurement needs a snapshot of every gradient, so this is a
    # cadence: 0 disables it. Without it, "the actor moved" says nothing about
    # which of the two objectives moved it, and an anchor that dominates looks
    # identical to one that is balanced.
    grad_report_every: int = 50
    # Per-phase wall time and CUDA peak memory. Off by default because
    # measuring GPU time requires a synchronize at every phase boundary, which
    # is itself a cost; on, it is the only way to see which part of an update
    # the time and the memory actually go to.
    profile: bool = False
    # Kept at zero deliberately. A fixed-sigma Gaussian transition's entropy
    # does not depend on the velocity mean, so an "entropy bonus" here would
    # have exactly zero gradient, and the sampled negative log probability is
    # not Dreamer's action entropy either. Learnable exploration is a separate
    # experiment, not a constant added to this loss.
    actor_entropy: float = 0.0


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


def flow_reinforce_targets(world_model, critic, record: Dict[str, Any],
                           config: ActorCriticConfig, *, progress_head=None,
                           return_ema=None) -> Dict[str, Any]:
    """Detached returns, advantages and survival weights for one record.

    Computed once, before the critic optimizer step, and then held fixed for
    the whole update: an advantage recomputed against a critic that has since
    moved no longer matches the samples it weights.

    The timeline is the one the module docstring describes and is unchanged
    from the pathwise objective -- reward and continuation are read at the
    successor, values at every feature, the bootstrap from the slow head.

    The survival weight is ``w[0] = 1`` and ``w[t] = prod_{j<t} gamma*cont[j]``,
    an *exclusive* cumulative product. ``dreamer.py`` writes the inclusive one
    and then slices, which is the same thing said differently for its
    indexing; here the exclusive form is what pairs with successor-indexed
    continuation. The property it has to have: at a transition that terminates,
    that transition's own reward still counts -- it is the reward the action
    earned -- while everything after it is weighted zero.
    """
    device = next(critic.net.parameters()).device
    feat = record["features"].to(device)
    with torch.no_grad():
        heads = imagined_rewards(world_model, feat)
        reward = heads["reward"][1:]
        cont = heads["cont"][1:]
        environment_reward = reward
        shaping = None
        if progress_head is not None and config.progress_beta:
            from .progress import shaping_reward

            shaping = shaping_reward(progress_head, feat, config.discount)
            reward = reward + config.progress_beta * shaping
        value = critic.value(feat)
        value = torch.cat(
            [value[:-1], critic.target_value(feat[-1:], detach=True)], dim=0)
        returns = lambda_return(reward, value, cont, config.discount,
                                config.lam)
        raw_advantage = returns - value[:-1]
        if return_ema is not None:
            # Once per newly collected batch, never per microbatch: this both
            # reads and advances a running statistic, so calling it per group
            # would advance it several times per update and scale each group
            # by a different number.
            _offset, scale = return_ema(returns.float())
        else:
            scale = torch.ones((), device=device, dtype=torch.float32)
        advantage = (raw_advantage.float() / scale).detach()
        survive = config.discount * cont
        weights = torch.cat(
            [torch.ones_like(survive[:1]),
             torch.cumprod(survive[:-1], dim=0)], dim=0).float().detach()
    return {
        "returns": returns.detach(),
        "value": value.detach(),
        "advantage": advantage,
        "raw_advantage": raw_advantage.detach(),
        "advantage_scale": scale.detach(),
        "weights": weights,
        "reward": environment_reward.detach(),
        "shaped_reward": reward.detach(),
        "shaping": None if shaping is None else shaping.detach(),
        "cont": cont.detach(),
    }


class ActorCriticTrainer:
    """One optimizer for the policy, one for the critic, one warm-up."""

    def __init__(self, world_model, actor, critic, config: ActorCriticConfig,
                 *, coords=None, progress_head=None, demo_sampler=None,
                 to_model_batch=None, device=None, seed: int = 0):
        self.world_model = world_model
        self.actor = actor
        self.critic = critic
        self.config = config
        self.coords = coords
        # Read by the actor objective and frozen there: imagination consults
        # the head, and the actor update must not train the head through the
        # shaping term.
        self.progress_head = progress_head
        if config.actor_objective not in ACTOR_OBJECTIVES:
            raise ValueError(
                f"unknown actor_objective {config.actor_objective!r}; "
                f"expected one of {ACTOR_OBJECTIVES}")
        # The anchor needs demonstrations, and a weight that is read and never
        # applied is worse than one that is absent. Both halves are checked
        # here rather than at the first update, which is minutes of training
        # later.
        self.demo_sampler = demo_sampler
        self.to_model_batch = to_model_batch
        if float(config.demo_anchor) and (demo_sampler is None
                                          or to_model_batch is None):
            raise ValueError(
                f"demo_anchor={config.demo_anchor} was requested but no "
                "demonstration sampler (or batch converter) was supplied to "
                "ActorCriticTrainer. The anchor is a real flow-matching loss "
                "on demonstrations; it cannot be applied without them. Pass "
                "both, or set demo_anchor to 0.0.")
        if config.actor_objective == FLOW_REINFORCE:
            from ..models.flow_sampler import flow_sigmas

            # Validated now: flow_sigmas refuses a zero or non-finite scale,
            # and discovering that after the critic warm-up would waste the
            # warm-up.
            self.flow_sigmas = flow_sigmas(
                config.flow_noise_std, int(config.flow_steps),
                schedule=config.flow_noise_schedule, device=device)
        else:
            self.flow_sigmas = None
        self.actor_opt = torch.optim.AdamW(
            [p for p in actor.parameters() if p.requires_grad],
            lr=config.actor_lr)
        self.critic_opt = torch.optim.AdamW(critic.net.parameters(),
                                            lr=config.critic_lr)
        # One scale for the combined return, as dreamer.py has it. Held on the
        # trainer so its running statistics are checkpointed with the run.
        self.return_ema = networks.ReturnEMA(
            device=device if device is not None
            else next(critic.net.parameters()).device)
        # One generator per device, created on demand. A single CPU generator
        # cannot serve both draws: the flow sampler runs on the actor's device
        # and ``torch.randn(device=cuda, generator=<cpu generator>)`` raises,
        # so a nonzero seed used to fail on GPU while passing every CPU test.
        #
        # Seed 0 is a seed, not "unseeded". Treating it as falsy left exactly
        # one configuration -- the default one -- irreproducible.
        self.seed = int(seed)
        self._generators: Dict[str, torch.Generator] = {}
        self.step = 0
        self.actor_steps = 0

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

        ``flow_reinforce`` takes a different route entirely and is dispatched
        here: it collects one rollout with no graph and reuses it for both the
        critic and the actor, because nothing in it was differentiated.
        """
        if self.config.actor_objective == FLOW_REINFORCE:
            return self._update_flow_reinforce(
                start, instruction=instruction, progress_beta=progress_beta)
        count = int(start[0].shape[0])
        if count == 0 or any(s.shape[0] != count for s in start):
            raise ValueError("imagination starts must have a common nonzero batch")
        microbatch = int(self.config.imagination_microbatch)
        if microbatch < 0:
            raise ValueError("imagination_microbatch must be nonnegative")
        microbatch = microbatch or count
        device = start[0].device
        self.step += 1
        if progress_beta is not None:
            # The warm-up is a function of environment steps, which the caller
            # counts, so the schedule arrives per update rather than being
            # recomputed from this trainer's own step counter.
            self.config.progress_beta = float(progress_beta)
        # Release the previous update's gradients before allocating rollouts.
        self.actor_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        metrics = {"return": 0.0, "reward": 0.0, "critic_loss": 0.0}
        for offset in range(0, count, microbatch):
            stop = min(offset + microbatch, count)
            small_start = tuple(s[offset:stop] for s in start)
            weight = (stop - offset) / count
            # Explicit False is required: the sampler otherwise enables grad.
            with torch.no_grad(), autocast(device, self.config.precision):
                critic_out = actor_loss(
                    self.world_model, self.actor, self.critic, small_start,
                    self.config, instruction=instruction, coords=self.coords,
                    differentiable=False)
            with autocast(device, self.config.precision):
                closs = critic_loss(
                    self.critic, critic_out["feat"].detach(),
                    critic_out["returns"])
            (closs * weight).backward()
            metrics["return"] += weight * float(critic_out["returns"].float().mean())
            metrics["reward"] += weight * float(critic_out["reward"].float().mean())
            metrics["critic_loss"] += weight * float(closs.detach())
            del critic_out, closs
        # One optimizer/target step per full batch, never per microbatch.
        torch.nn.utils.clip_grad_norm_(self.critic.net.parameters(),
                                       self.config.grad_clip)
        self.critic_opt.step()
        self.critic.update_target()
        self.critic_opt.zero_grad(set_to_none=True)

        warming = self.step <= self.config.critic_warmup
        if warming:
            # A random critic gives the actor a gradient toward noise, so the
            # policy is left alone until the value head means something.
            metrics["actor_loss"] = float("nan")
        else:
            # The critic and its slow target changed above, so build a fresh
            # objective. This graph is consumed before either is modified
            # again.
            trainable = [p for p in self.actor.parameters() if p.requires_grad]
            if not trainable:
                raise RuntimeError(
                    "the actor has no trainable parameters; the adapter and "
                    "the action expert are supposed to be")
            metrics["actor_loss"] = 0.0
            for offset in range(0, count, microbatch):
                stop = min(offset + microbatch, count)
                small_start = tuple(s[offset:stop] for s in start)
                weight = (stop - offset) / count
                shaping = progress_reward
                if (torch.is_tensor(shaping) and shaping.ndim >= 2
                        and shaping.shape[1] == count):
                    shaping = shaping[:, offset:stop]
                with autocast(device, self.config.precision):
                    out = actor_loss(
                        self.world_model, self.actor, self.critic, small_start,
                        self.config, instruction=instruction,
                        progress_reward=shaping, progress_head=self.progress_head,
                        coords=self.coords, differentiable=True)
                # Backward now frees this rollout before constructing the next.
                # Weight by its size, including a possibly shorter last group.
                (out["loss"] * weight).backward()
                metrics["actor_loss"] += weight * float(out["loss"].detach())
                if out.get("shaping") is not None:
                    metrics["shaping_reward"] = (
                        metrics.get("shaping_reward", 0.0)
                        + weight * float(out["shaping"].float().mean()))
                    metrics["progress_beta"] = float(self.config.progress_beta)
                del out
            if float(self.config.demo_anchor):
                # The anchor belongs to *both* objectives: it is an imitation
                # gradient on demonstrations, not a property of how the RL
                # gradient is estimated. Accumulated into the same .grad and
                # consumed by the same single optimizer step below.
                metrics |= self._anchored_backward(trainable, device)
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
            self.actor_steps += 1
            metrics |= {"actor_grad_norm": raw_norm,
                        "actor_params_with_grad": float(len(populated)),
                        "actor_params_trainable": float(len(trainable))}
        return metrics

    # --------------------------------------------------------- reproducibility

    def generator_for(self, device) -> torch.Tensor:
        """A generator living on the device the draw actually happens on.

        Two different devices are involved in one update: the flow sampler
        draws on the actor's device, and row selection draws on the world
        model's. Each gets its own generator, both derived from the run's one
        seed, so a run reproduces on either and neither draw can raise a
        device-mismatch from a generator that was made somewhere else.

        The per-device offset uses ``crc32`` rather than ``hash``: Python
        randomizes string hashing per process, which would make the streams
        differ between runs of the same seed.
        """
        device = torch.device(device)
        index = 0 if device.index is None else int(device.index)
        key = f"{device.type}:{index}"
        generator = self._generators.get(key)
        if generator is None:
            generator = torch.Generator(device=device)
            offset = zlib.crc32(key.encode("utf-8"))
            generator.manual_seed((self.seed + offset) % (2 ** 63 - 1))
            self._generators[key] = generator
        return generator

    @property
    def actor_device(self) -> torch.Tensor:
        """Where the actor's own tensors live, as ``imagine`` resolves it."""
        declared = getattr(self.actor, "device", None)
        if declared is not None:
            return torch.device(declared)
        return next(self.critic.net.parameters()).device

    @property
    def model_device(self) -> torch.Tensor:
        """Where the world model's features and row indices live."""
        try:
            return next(self.world_model.parameters()).device
        except StopIteration:                              # pragma: no cover
            return self.actor_device

    # ------------------------------------------------------ flow_reinforce

    def collect(self, start, *, instruction=None, record_means: bool = False
                ) -> Dict[str, Any]:
        """One ephemeral imagined batch, gathered with no graph anywhere.

        Collected in groups of start states so the peak is bounded by
        ``imagination_microbatch`` rather than by the whole imagination batch:
        the prefix cache of one environment step is the largest live object,
        and it is released as soon as that step is sampled.
        """
        count = int(start[0].shape[0])
        microbatch = int(self.config.imagination_microbatch) or count
        parts: List[Dict[str, Any]] = []
        # The same mode scoring will use. See deterministic_modules: a dropout
        # mask that differs between sampling and scoring silently changes which
        # distribution the recorded sample came from.
        with deterministic_modules(self.actor, self.world_model):
            for offset in range(0, count, microbatch):
                stop = min(offset + microbatch, count)
                parts.append(imagine_flow_reinforce(
                    self.world_model, self.actor,
                    tuple(s[offset:stop] for s in start),
                    self.config.horizon,
                    flow_steps=int(self.config.flow_steps),
                    sigmas=self.flow_sigmas, instruction=instruction,
                    coords=self.coords,
                    generator=self.generator_for(self.actor_device),
                    record_means=record_means))
        record = dict(parts[0])
        if len(parts) > 1:
            # Concatenated on the start-state axis, which is axis 1 for every
            # stacked field: axis 0 is the imagined horizon.
            for key in ("features", "flow_states", "executed_actions"):
                record[key] = torch.cat([p[key] for p in parts], dim=1)
            if record_means:
                record["flow_means"] = torch.cat(
                    [p["flow_means"] for p in parts], dim=1)
        record["batch"] = count
        return record

    def score_path(self, record: Dict[str, Any], index: torch.Tensor,
                   instruction=None, *, validate: bool = True
                   ) -> torch.Tensor:
        """``sum_k ell`` for the flow transitions named by flattened ``index``.

        ``index`` addresses ``(environment step, start state, flow step)``
        tuples of the recorded rollout. Each one is scored by rebuilding the
        actor's conditioning **with gradients** from the detached feature and
        running one expert velocity evaluation on the detached ``u_k``.

        Two things this must not do, and both were easy to write by accident.
        It must not reuse the collection-time prefix: that cache was built
        under ``no_grad``, so reusing it would sever the adapter from the loss
        and train nothing while reporting a descending number. And it must not
        resample ``u_(k+1)``: both ends of the transition are fixed samples
        here, and a freshly drawn target would turn a score-function estimator
        into a reparameterized one against a different distribution.
        """
        from ..models.flow_sampler import (check_sigmas,
                                           transition_log_prob,
                                           transition_mean)

        horizon = int(record["horizon"])
        batch = int(record["batch"])
        steps = int(record["flow_steps"])
        device = getattr(self.actor, "device", record["features"].device)

        flat_step = torch.div(index, batch * steps, rounding_mode="floor")
        remainder = index - flat_step * (batch * steps)
        flat_batch = torch.div(remainder, steps, rounding_mode="floor")
        flat_flow = remainder - flat_batch * steps
        if validate:
            # A device-tensor reduction read as a Python bool blocks until the
            # queue drains. The bound is a property of the whole index range,
            # so _update_flow_reinforce checks it once before the loop and
            # passes validate=False here; a caller that builds its own index
            # still gets the check by default.
            if bool((flat_step >= horizon).any()):
                raise IndexError("transition index past the imagined horizon")

        feat = record["features"][flat_step, flat_batch].to(device)
        states = record["flow_states"][flat_step, flat_batch]
        u_k = states[torch.arange(states.shape[0]), flat_flow].to(device)
        u_next = states[torch.arange(states.shape[0]), flat_flow + 1].to(device)
        times = record["flow_times"].to(device)[flat_flow]
        sigmas = record["flow_sigmas"].to(device)[flat_flow]

        # Detaching the *feature* is not detaching the adapter: the feature is
        # a fixed input, and condition() runs the adapter and the frozen VLM
        # prefix on it with autograd live. That is the whole gradient path.
        # eval() here, not no_grad(): the mode must match collection's so the
        # recomputed mean is the mean that was sampled from, while the graph
        # this builds is exactly what the actor's gradient comes from.
        with deterministic_modules(self.actor, self.world_model):
            cond = self.actor.condition(feat.detach(), instruction)
            mean = transition_mean(self.actor.velocity_fn(), u_k.detach(),
                                   times, cond, 1.0 / float(steps))
            return transition_log_prob(u_next.detach(), mean,
                                       sigmas.reshape(-1, 1, 1),
                                       validate=validate)

    def _update_flow_reinforce(self, start, *, instruction=None,
                               progress_beta: Optional[float] = None
                               ) -> Dict[str, float]:
        """Collect once, fix the targets, then score the recorded path.

        One actor optimizer step per newly collected imagined batch. There is
        no importance ratio and none is needed: every actor parameter is still
        exactly the one that produced these samples when they are scored.
        """
        count = int(start[0].shape[0])
        if count == 0 or any(s.shape[0] != count for s in start):
            raise ValueError(
                "imagination starts must have a common nonzero batch")
        device = start[0].device
        self.step += 1
        if progress_beta is not None:
            self.config.progress_beta = float(progress_beta)
        self.actor_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)

        phases = Phases(device, bool(self.config.profile))
        self.phases = phases
        with phases("collect"), autocast(device, self.config.precision):
            record = self.collect(start, instruction=instruction)
        with phases("targets"), autocast(device, self.config.precision):
            targets = flow_reinforce_targets(
                self.world_model, self.critic, record, self.config,
                progress_head=self.progress_head,
                return_ema=self.return_ema)

        metrics: Dict[str, float] = {
            "return": float(targets["returns"].float().mean()),
            "reward": float(targets["reward"].float().mean()),
            "advantage": float(targets["advantage"].mean()),
            "advantage_raw": float(targets["raw_advantage"].float().mean()),
            "advantage_std": float(targets["advantage"].std()),
            # Magnitude, not the signed mean. A mean advantage near zero is
            # normal and says nothing; an *absolute* mean near zero says the
            # update has no signal at all, which is the state a flat critic
            # leaves both objectives in.
            "advantage_abs": float(targets["advantage"].abs().mean()),
            "advantage_scale": float(targets["advantage_scale"]),
            "imagination_starts": float(count),
        }
        if targets["shaping"] is not None:
            metrics["shaping_reward"] = float(targets["shaping"].float().mean())
            metrics["progress_beta"] = float(self.config.progress_beta)

        # Critic: detached features, detached returns, the same survival
        # weights the actor uses. Trained after the targets were computed, so
        # the advantages belong to the critic that produced them.
        feat = record["features"].to(device)
        with phases("critic"):
            with autocast(device, self.config.precision):
                closs = self.critic.loss(feat[:-1].detach(),
                                         targets["returns"],
                                         mask=targets["weights"])
            closs.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.net.parameters(),
                                           self.config.grad_clip)
            self.critic_opt.step()
            self.critic.update_target()
        self.critic_opt.zero_grad(set_to_none=True)
        metrics["critic_loss"] = float(closs.detach())
        del closs

        if self.step <= self.config.critic_warmup:
            # Identical to the pathwise warm-up: no actor step, and no anchor
            # either, so the policy that leaves warm-up is the one that
            # entered it.
            metrics["actor_loss"] = float("nan")
            return metrics

        trainable = [p for p in self.actor.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                "the actor has no trainable parameters; the adapter and the "
                "action expert are supposed to be")
        horizon = int(record["horizon"])
        steps = int(record["flow_steps"])
        total = horizon * count * steps
        # The global normalizer from the plan: 1/(B*H), with the sum over k
        # *inside*. Dividing by B*H*K instead would silently rescale the RL
        # term against the anchor by a factor of K.
        norm = 1.0 / float(count * horizon)
        weights = targets["weights"].reshape(-1)
        advantage = targets["advantage"].reshape(-1)
        order = torch.arange(total, device=advantage.device)
        # Validated once, here, for the whole update: the sigma schedule is
        # fixed and the index range is known, so neither needs re-checking per
        # microbatch. Both checks are device-tensor reductions read as Python
        # bools, i.e. host synchronizations, and the scoring loop runs
        # thousands of times per update.
        check_sigmas(record["flow_sigmas"])
        if total and int(order[-1]) >= horizon * count * steps:
            raise IndexError("transition index past the imagined horizon")
        # (t, b) pair for each (t, b, k) tuple, so each tuple picks up its
        # environment transition's weight and advantage.
        pair = torch.div(order, steps, rounding_mode="floor")
        scale = (weights[pair] * advantage[pair]).detach()

        microbatch = max(int(self.config.actor_transition_microbatch), 1)
        # Accumulated on the device and read once at the end. Calling float()
        # per microbatch forces a host synchronization per backward pass,
        # which at 38k transitions and microbatch 16 is thousands of stalls an
        # update -- pure wall-clock cost for a number nobody reads until the
        # update is over.
        loss_total = torch.zeros((), device=device, dtype=torch.float32)
        logp_total = torch.zeros((), device=device, dtype=torch.float32)
        with phases("score"):
            for offset in range(0, total, microbatch):
                index = order[offset:offset + microbatch]
                with autocast(device, self.config.precision):
                    logp = self.score_path(record, index,
                                           instruction=instruction,
                                           validate=False)
                group = -(norm * scale[index].to(logp.device) * logp).sum()
                group.backward()
                loss_total += group.detach().float().to(device)
                logp_total += logp.detach().float().sum().to(device)
                del logp, group
        metrics["actor_loss"] = float(loss_total)
        metrics["actor_logp_mean"] = float(logp_total) / max(total, 1)
        metrics["scored_transitions"] = float(total)
        # The RL term's gradient on its own, before any anchor is added. This
        # is the number that distinguishes "the critic is still flat" from
        # "the score-function path is broken", and it is only meaningful after
        # warm-up, which is where this line runs.
        metrics["rl_grad_norm"] = self._gradient_norm(trainable)

        if float(self.config.demo_anchor):
            with phases("anchor"):
                metrics |= self._anchored_backward(trainable, device)

        with torch.no_grad():
            populated = [p.grad for p in trainable if p.grad is not None]
            raw_norm = (float(torch.sqrt(sum((g.detach() ** 2).sum()
                                             for g in populated)))
                        if populated else 0.0)
        torch.nn.utils.clip_grad_norm_(trainable, self.config.grad_clip)
        self.actor_opt.step()
        self.actor_steps += 1
        # The batch is ephemeral: it belonged to the parameters that have just
        # been replaced, and nothing may score it again.
        del record, targets
        metrics |= {"actor_grad_norm": raw_norm,
                    "actor_params_with_grad": float(len(populated)),
                    "actor_params_trainable": float(len(trainable)),
                    "actor_steps": float(self.actor_steps)}
        metrics |= phases.metrics()
        return metrics

    def _gradient_norm(self, params) -> float:
        with torch.no_grad():
            populated = [p.grad for p in params if p.grad is not None]
            if not populated:
                return 0.0
            return float(torch.sqrt(
                sum((g.detach().float() ** 2).sum() for g in populated)))

    def _anchored_backward(self, trainable, device) -> Dict[str, float]:
        """Add the weighted anchor gradient, occasionally measuring it alone.

        The two terms have to end up summed in one ``.grad``, clipped once,
        and consumed by one optimizer step -- that is the training behaviour
        and diagnostics may not change it.

        But they must be *measured* separately, and measuring the anchor by
        subtracting the RL gradient from the accumulated sum does not work:
        when the RL term is orders of magnitude larger, the anchor is already
        gone to float32 rounding before the subtraction sees it, and the
        result reads as an anchor of zero for an anchor that is present and
        nonzero. So on a diagnostic update the RL gradient is moved aside, the
        anchor is accumulated into an empty ``.grad`` and measured there, and
        the RL gradient is added back. The parameters end up with exactly the
        sum they would have had; only the order of two additions changed.

        The subtraction is still reported, as ``retained_anchor_grad_norm``:
        the difference between it and ``anchor_grad_norm`` is precisely how
        much of the anchor survived being added to the RL term, which is the
        thing worth knowing when choosing ``demo_anchor``.
        """
        every = int(self.config.grad_report_every)
        report = every > 0 and self.actor_steps % every == 0
        if not report:
            return self._anchor_backward(device)

        rl_norm = self._gradient_norm(trainable)
        with torch.no_grad():
            stashed = [(p, None if p.grad is None else p.grad.detach().clone())
                       for p in trainable]
            for parameter, _previous in stashed:
                parameter.grad = None

        metrics = self._anchor_backward(device)

        with torch.no_grad():
            # .grad now holds the weighted anchor gradient and nothing else.
            anchor_norm = self._gradient_norm(trainable)
            retained = 0.0
            for parameter, previous in stashed:
                if previous is None:
                    continue
                if parameter.grad is None:
                    parameter.grad = previous
                    continue
                combined = parameter.grad.detach() + previous
                retained += float(
                    ((combined - previous).float() ** 2).sum())
                parameter.grad = combined
            del stashed
        metrics |= {
            "rl_grad_norm": rl_norm,
            "anchor_grad_norm": anchor_norm,
            # How much of that anchor is still visible once it has been added
            # to the RL term. Equal to anchor_grad_norm when the two are
            # comparable; near zero when the anchor was rounded away.
            "retained_anchor_grad_norm": float(retained ** 0.5),
            "anchor_grad_ratio": (anchor_norm / rl_norm
                                  if rl_norm > 0 else float("inf"))}
        return metrics

    def _anchor_backward(self, device) -> Dict[str, float]:
        """Accumulate the demonstration anchor into the same actor gradient."""
        from ..models.flow_sampler import flow_matching_loss
        from .train_imitation import prepare_anchor_rows

        rows = None
        for _ in range(max(int(self.config.anchor_retries), 1)):
            # Windows, not rows: one window yields many eligible positions, so
            # the sampler is asked for the cheap budget and the expensive one
            # is applied afterwards.
            batch = self.to_model_batch(
                self.demo_sampler.batch(int(self.config.anchor_windows)))
            rows = prepare_anchor_rows(
                self.world_model, batch, int(self.actor.chunk_size),
                max_rows=int(self.config.anchor_rows),
                window_microbatch=int(self.config.anchor_window_microbatch),
                precision=self.config.precision,
                generator=self.generator_for(self.model_device))
            if rows is not None:
                break
        if rows is None:
            raise RuntimeError(
                "demo_anchor is nonzero but no sampled demonstration window "
                f"held an eligible row in {self.config.anchor_retries} tries. "
                "Silently dropping the anchor would change the objective for "
                "the rest of the run; fix the sampler or set demo_anchor to 0.")
        feat, chunk_targets_, mask = rows
        # One denominator for the whole selection, computed before it is cut
        # into groups, so the anchor's value does not depend on the partition.
        valid = float(mask.sum()) * float(chunk_targets_.shape[-1])
        group_size = max(int(self.config.anchor_microbatch), 1)
        weight = float(self.config.demo_anchor)
        # One noise/time draw per selected row, made once before the groups so
        # the anchor's value and gradient do not depend on how the rows are
        # partitioned. Drawing inside the loop would make each partition score
        # a different sample, which is a different loss, not a different
        # arrangement of the same one.
        generator = self.generator_for(feat.device)
        noise = torch.randn(chunk_targets_.shape, device=feat.device,
                            dtype=chunk_targets_.dtype, generator=generator)
        times = torch.rand((chunk_targets_.shape[0],), device=feat.device,
                           dtype=chunk_targets_.dtype, generator=generator)
        total = torch.zeros((), device=device, dtype=torch.float32)
        for offset in range(0, feat.shape[0], group_size):
            stop = offset + group_size
            with autocast(device, self.config.precision):
                cond = self.actor.condition(feat[offset:stop].detach(), None)
                loss, _metrics = flow_matching_loss(
                    self.actor.velocity_fn(), chunk_targets_[offset:stop],
                    cond, mask=mask[offset:stop], denominator=valid,
                    noise=noise[offset:stop], times=times[offset:stop])
            (weight * loss).backward()
            total += loss.detach().float().to(device)
            del loss, cond
        return {"anchor_loss": float(total),
                "anchor_rows": float(feat.shape[0]),
                "anchor_valid_targets": valid,
                "demo_anchor": weight}
