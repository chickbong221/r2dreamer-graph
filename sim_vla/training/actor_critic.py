"""The online actor and critic: one executed chunk per imagination start.

Both arms use this same update, so a difference between them stays a
difference in the state and not in the update rule.

For every eligible replay latent state the actor is conditioned once and
generates one action chunk, and the chunk's first ``execute`` actions are
stepped through the frozen world model -- what the online policy does between
two replans (see :mod:`sim_vla.training.imagination`). One rollout per start
gives ``execute`` rewards, ``execute`` continuations and a final state, and one
return::

    G = V_slow(s_E)
    for t = E-1 .. 0:
        G = r_t + gamma * c_t * G

the lambda-return at ``lambda = 1`` over the executed chunk, bootstrapped by
the slow critic at its end.

**Actor.** Maximise ``mean(G)``, pathwise: the gradient runs back through the
bootstrap value, the reward and continuation heads, every imagined transition
and every flow integration step, into the adapter and the action expert. With
``demo_anchor`` set, Stage 1B's flow-matching loss on fresh demonstration rows
is weighted into the same gradient before the one actor step.

**Critic.** Regress ``V(s_0)`` onto the detached ``G``, with the critic's own
distributional loss. The start state is the only training point: the states
inside a chunk already have their next actions committed, which is not what a
state-only critic estimates, and none of them is evaluated just to form this
return either.

**One rollout, both optimizers.** Each microbatch of starts is imagined once.
After warm-up its actor loss is backpropagated, then its critic loss from the
detached start features and the detached return, and the graph is released
before the next microbatch. No parameter -- actor, critic or slow target --
moves until every microbatch has been through backward, and each microbatch is
weighted by its share of the starts, so the step is the full-batch mean however
the starts were grouped.

The timeline
------------

A rollout of ``E`` executed actions produces ``E + 1`` features ``f_0 .. f_E``
and ``E`` actions ``a_0 .. a_(E-1)``, where ``a_t`` takes ``s_t`` to
``s_(t+1)``.

The reward head is trained -- in ``world_model.loss`` -- against
``batch["reward"]``, which the window layout defines as ``r_(t-1)``: the reward
earned by the transition that *arrived* at ``o_t``. So the head at a state
predicts the reward that got there, not the reward that leaves it, and the
reward belonging to transition ``t`` is read at the **successor**::

    reward[t]  ==  heads["reward"][t + 1]        for t in 0 .. E-1

Reading ``[:-1]`` instead would take the reward that preceded the rollout -- a
value no imagined action can influence -- and drop the reward earned by the
final action. Continuation shifts the same way: whether transition ``t``
permits the rest of the return is ``heads["cont"][t + 1]``.

Flat heads, and why the warm-up is load-bearing
-----------------------------------------------

The value and reward heads' output layers start at zero, so their outputs are
constant in the feature and their gradient with respect to it is zero. Through
them the imagined return does not depend on the action, and the actor gradient
is exactly zero. ``critic_warmup`` is therefore not a safety margin: it is the
number of updates it takes for the critic to mean something. During it the
rollout is built with no actor graph at all -- conditioning included -- and
only the critic trains.

Frozen means "parameters do not move", not "outputs are constants": during the
actor update the world-model, critic and progress-head parameters are excluded
from the gradient, but the gradient has to pass *through* their outputs to
reach the action. :func:`freeze_parameters` is the context manager that makes
that distinction operational.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch

from .imagination import chunk_return, imagine_chunk, imagined_rewards
from .precision import autocast
from .profiling import Phases

# What this update optimises, as recorded in checkpoints and run summaries.
OBJECTIVE = "pathwise"
RETURN = "bootstrapped_executed_chunk"
START_SELECTION = "all_eligible_replay_states"


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
    # Actions executed from one generated chunk before replanning, in
    # imagination and in the environment alike. Required, because it is
    # actor.execute: a second default here could quietly disagree with it.
    execute: int
    discount: float = 0.997
    actor_lr: float = 3e-5
    critic_lr: float = 3e-4
    flow_steps: int = 10
    grad_clip: float = 1.0
    critic_warmup: int = 150
    # Starts imagined together, with gradients accumulated across groups.
    # Zero imagines every start at once.
    imagination_microbatch: int = 0
    precision: str = "float32"
    # Set per update from the warm-up schedule; the configured beta is the
    # value it warms up *to*.
    progress_beta: float = 0.0
    # Per-phase wall time and CUDA peak memory for each update. Measuring GPU
    # time synchronizes at every phase boundary, so this is for a profiling
    # run, not for training.
    profile: bool = False
    # Flow-matching imitation on demonstrations, weighted into the actor's
    # gradient after warm-up. Zero disables it; nonzero needs a demonstration
    # sampler. Windows are encoded by the world model (the expensive half);
    # rows are the eligible positions kept from them, conditioned in groups.
    demo_anchor: float = 0.0
    anchor_windows: int = 8
    anchor_rows: int = 64
    anchor_microbatch: int = 16
    # Actor steps between separate RL and anchor gradient measurements; 0
    # never measures. Without it a dominating anchor looks like a balanced one.
    grad_report_every: int = 50

    def __post_init__(self):
        if int(self.execute) < 1:
            raise ValueError(f"execute={self.execute} must be at least 1")
        if int(self.flow_steps) < 1:
            raise ValueError(f"flow_steps={self.flow_steps} must be at least 1")
        if int(self.imagination_microbatch) < 0:
            raise ValueError("imagination_microbatch must be nonnegative")
        if int(self.critic_warmup) < 0:
            raise ValueError("critic_warmup must be nonnegative")
        if not float(self.demo_anchor) >= 0:
            raise ValueError(f"demo_anchor={self.demo_anchor} must be >= 0")
        if int(self.anchor_windows) < 1 or int(self.anchor_rows) < 1:
            raise ValueError("anchor_windows and anchor_rows must be >= 1")


def executed_chunk_objective(world_model, actor, critic, start,
                             config: ActorCriticConfig, *, instruction=None,
                             progress_head=None, coords=None,
                             differentiable: bool = True) -> Dict[str, Any]:
    """Imagine one executed chunk per start and form its bootstrapped return.

    ``differentiable=False`` builds no graph anywhere -- not in the adapter's
    conditioning, the sampler, the dynamics or the heads -- which is what the
    critic warm-up needs. ``differentiable=True`` keeps the actor's whole path
    live while the world model, the critic and the progress head stay frozen.

    The progress arm's shaping belongs to the reward, not to the actor: both
    losses are formed from the same training reward, so the critic predicts
    the return the actor maximises. Per imagined transition::

        shaping[t] = gamma * cont[t] * phi(s_(t+1)) - phi(s_t)
        reward[t] += beta * shaping[t]

    and ``reward`` (unshaped) and ``shaping`` are reported apart.
    """
    context = contextlib.nullcontext() if differentiable else torch.no_grad()
    with context, freeze_parameters(world_model, critic, progress_head):
        rollout = imagine_chunk(world_model, actor, start, config.execute,
                                flow_steps=config.flow_steps,
                                instruction=instruction, coords=coords,
                                differentiable=differentiable)
        feat = rollout["feat"]
        heads = imagined_rewards(world_model, feat)
        # Successor-indexed: see the module docstring.
        reward = heads["reward"][1:]
        cont = heads["cont"][1:]
        training_reward = reward
        shaping = None
        if progress_head is not None and config.progress_beta:
            from .progress import shaping_reward

            # Computed from *this* rollout's features. A precomputed tensor
            # could silently belong to another rollout and still fit.
            shaping = shaping_reward(progress_head, feat, config.discount,
                                     cont=cont)
            training_reward = reward + config.progress_beta * shaping
        # The slow target's parameters are frozen; its output is not, and the
        # last executed action reaches the return through it.
        bootstrap = critic.target_value(feat[-1])
        returns = chunk_return(training_reward, cont, bootstrap,
                               config.discount)
    return {
        "loss": -returns.mean(),
        "returns": returns,
        "feat": feat,
        "action": rollout["action"],
        "action_steps": rollout["action_steps"],
        # The environment stream before shaping, so a run that scored only on
        # its own shaping is visible as exactly that.
        "reward": reward.detach(),
        "shaped_reward": training_reward.detach(),
        "shaping": None if shaping is None else shaping.detach(),
        "cont": cont.detach(),
        "bootstrap": bootstrap.detach(),
    }


class ActorCriticTrainer:
    """One optimizer for the policy, one for the critic, one warm-up."""

    def __init__(self, world_model, actor, critic, config: ActorCriticConfig,
                 *, coords=None, progress_head=None, device=None,
                 demo_sampler=None, to_model_batch=None):
        self.world_model = world_model
        self.actor = actor
        self.critic = critic
        self.config = config
        self.coords = coords
        # Read by the objective and frozen there: the actor update must not
        # train the head through the shaping term.
        self.progress_head = progress_head
        chunk = getattr(actor, "chunk_size", None)
        if chunk is not None and int(config.execute) > int(chunk):
            raise ValueError(
                f"execute={config.execute} but the actor generates {chunk} "
                "actions per chunk; executing more would invent commands")
        # Checked here rather than at the first actor step, which comes after
        # the whole critic warm-up: a weight that is never applied is worse
        # than an absent one.
        self.demo_sampler = demo_sampler
        self.to_model_batch = to_model_batch
        if float(config.demo_anchor) and (demo_sampler is None
                                          or to_model_batch is None):
            raise ValueError(
                f"demo_anchor={config.demo_anchor} needs a demonstration "
                "sampler and a batch converter; pass both or set it to 0")
        self.actor_opt = torch.optim.AdamW(
            [p for p in actor.parameters() if p.requires_grad],
            lr=config.actor_lr)
        self.critic_opt = torch.optim.AdamW(critic.net.parameters(),
                                            lr=config.critic_lr)
        self.device = device
        self.step = 0
        self.actor_steps = 0
        self.phases: Optional[Phases] = None

    def update(self, start, *, instruction=None,
               progress_beta: Optional[float] = None,
               on_progress: Optional[Callable] = None) -> Dict[str, float]:
        """One critic step and -- after warm-up -- one actor step, from one
        rollout per microbatch of starts."""
        count = int(start[0].shape[0])
        if count == 0 or any(int(s.shape[0]) != count for s in start):
            raise ValueError(
                "imagination starts must have a common nonzero batch")
        microbatch = int(self.config.imagination_microbatch) or count
        device = start[0].device
        self.step += 1
        if progress_beta is not None:
            # A function of environment steps, which the caller counts.
            self.config.progress_beta = float(progress_beta)
        warming = self.step <= int(self.config.critic_warmup)
        trainable = [p for p in self.actor.parameters() if p.requires_grad]
        if not warming and not trainable:
            raise RuntimeError(
                "the actor has no trainable parameters; the adapter and the "
                "action expert are supposed to be")

        self.actor_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        phases = Phases(device, bool(self.config.profile))
        self.phases = phases
        # Accumulated on the device and read once at the end: a float() per
        # microbatch is a host synchronization per microbatch.
        zero = lambda: torch.zeros((), device=device, dtype=torch.float32)
        totals = {name: zero() for name in (
            "return", "reward", "cont", "bootstrap", "critic_loss",
            "actor_loss")}
        shaping_total = zero()
        shaped = False
        for offset in range(0, count, microbatch):
            stop = min(offset + microbatch, count)
            small_start = tuple(s[offset:stop] for s in start)
            weight = (stop - offset) / count
            details = dict(microbatch=offset // microbatch + 1,
                           microbatches=(count + microbatch - 1) // microbatch,
                           imagination_starts=count, actor_training=not warming)
            if on_progress is not None:
                on_progress("imagine", **details)
            with phases("imagine"), autocast(device, self.config.precision):
                out = executed_chunk_objective(
                    self.world_model, self.actor, self.critic, small_start,
                    self.config, instruction=instruction,
                    progress_head=self.progress_head, coords=self.coords,
                    differentiable=not warming)
            if not warming:
                if on_progress is not None:
                    on_progress("actor_backward", **details)
                with phases("actor_backward"):
                    (out["loss"] * weight).backward()
            if on_progress is not None:
                on_progress("critic_backward", **details)
            with phases("critic_backward"):
                with autocast(device, self.config.precision):
                    closs = self.critic.loss(out["feat"][0].detach(),
                                             out["returns"].detach())
                (closs * weight).backward()
            totals["return"] += weight * out["returns"].detach().float().mean()
            totals["reward"] += weight * out["reward"].float().mean()
            totals["cont"] += weight * out["cont"].float().mean()
            totals["bootstrap"] += weight * out["bootstrap"].float().mean()
            totals["critic_loss"] += weight * closs.detach().float()
            totals["actor_loss"] += weight * out["loss"].detach().float()
            if out["shaping"] is not None:
                shaped = True
                shaping_total += weight * out["shaping"].float().mean()
            # Released before the next microbatch is built.
            del out, closs

        metrics: Dict[str, float] = {
            name: float(value) for name, value in totals.items()}
        metrics["imagined_transitions"] = float(count * int(self.config.execute))
        if shaped:
            metrics["shaping_reward"] = float(shaping_total)
            metrics["progress_beta"] = float(self.config.progress_beta)
        if not warming and float(self.config.demo_anchor):
            # Summed into the same .grad as the RL term, clipped with it and
            # consumed by the one actor step below.
            if on_progress is not None:
                on_progress("anchor")
            with phases("anchor"):
                metrics |= self._anchored_backward(trainable, device,
                                                   instruction)
        if on_progress is not None:
            on_progress("optimizer")
        with phases("step"):
            if warming:
                # A random critic gives the actor a gradient toward noise, so
                # the policy is left alone until the value head means
                # something.
                metrics["actor_loss"] = float("nan")
            else:
                # Measured from .grad rather than taken from clip_grad_norm_'s
                # return, beside how many parameters received one: a single
                # number cannot tell "zero" from "nothing was measured".
                with torch.no_grad():
                    populated = [p.grad for p in trainable
                                 if p.grad is not None]
                    raw_norm = (float(torch.sqrt(sum(
                        (g.detach().float() ** 2).sum() for g in populated)))
                        if populated else 0.0)
                torch.nn.utils.clip_grad_norm_(trainable,
                                               self.config.grad_clip)
                self.actor_opt.step()
                self.actor_steps += 1
                metrics |= {"actor_grad_norm": raw_norm,
                            "actor_params_with_grad": float(len(populated)),
                            "actor_params_trainable": float(len(trainable))}
            torch.nn.utils.clip_grad_norm_(self.critic.net.parameters(),
                                           self.config.grad_clip)
            self.critic_opt.step()
            self.critic.update_target()
        self.actor_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        metrics["actor_steps"] = float(self.actor_steps)
        metrics |= phases.metrics()
        return metrics

    @staticmethod
    def _gradient_norm(params) -> float:
        with torch.no_grad():
            populated = [p.grad for p in params if p.grad is not None]
            if not populated:
                return 0.0
            return float(torch.sqrt(
                sum((g.detach().float() ** 2).sum() for g in populated)))

    def _anchored_backward(self, trainable, device, instruction=None
                           ) -> Dict[str, float]:
        """Add the weighted anchor gradient, every so often measuring it alone.

        The anchor has to be *measured* apart from the RL term: subtracting
        the RL gradient from the sum reads an anchor rounded away in float32
        as an anchor of zero. So on a reporting step the RL gradient is set
        aside, the anchor is accumulated into an empty ``.grad`` and measured,
        and the RL gradient is added back -- the same sum, in another order.
        ``retained_anchor_grad_norm`` is how much of the anchor survived being
        added to the RL term.
        """
        every = int(self.config.grad_report_every)
        if not (every > 0 and self.actor_steps % every == 0):
            return self._anchor_backward(device, instruction)

        rl_norm = self._gradient_norm(trainable)
        with torch.no_grad():
            stashed = [(p, None if p.grad is None else p.grad.detach().clone())
                       for p in trainable]
            for parameter, _previous in stashed:
                parameter.grad = None
        metrics = self._anchor_backward(device, instruction)
        with torch.no_grad():
            anchor_norm = self._gradient_norm(trainable)
            retained = 0.0
            for parameter, previous in stashed:
                if previous is None:
                    continue
                if parameter.grad is None:
                    parameter.grad = previous
                    continue
                combined = parameter.grad.detach() + previous
                retained += float(((combined - previous).float() ** 2).sum())
                parameter.grad = combined
            del stashed
        return metrics | {
            "rl_grad_norm": rl_norm,
            "anchor_grad_norm": anchor_norm,
            "retained_anchor_grad_norm": float(retained ** 0.5),
            "anchor_grad_ratio": (anchor_norm / rl_norm if rl_norm > 0
                                  else float("inf"))}

    def _anchor_backward(self, device, instruction=None) -> Dict[str, float]:
        """Stage 1B's flow-matching loss on fresh demonstration rows.

        Windows are drawn with a chunk of action lookahead, as Stage 1B draws
        them, and the sampler's own lookahead is restored afterwards: the
        world model's mixed batches are drawn without one.
        """
        from ..models.flow_sampler import flow_matching_loss
        from .train_imitation import prepare_imitation_rows

        chunk = int(self.actor.chunk_size)
        previous = getattr(self.demo_sampler, "lookahead", None)
        if previous is not None:
            self.demo_sampler.lookahead = chunk
        try:
            windows = self.demo_sampler.batch(int(self.config.anchor_windows))
        finally:
            if previous is not None:
                self.demo_sampler.lookahead = previous
        rows = prepare_imitation_rows(
            self.world_model, self.to_model_batch(windows), chunk,
            max_rows=int(self.config.anchor_rows))
        if rows is None:
            raise RuntimeError(
                f"demo_anchor={self.config.demo_anchor} but none of "
                f"{self.config.anchor_windows} demonstration windows held an "
                "eligible row; dropping the anchor silently would change the "
                "objective")
        feat, targets, mask = rows
        # One denominator and one noise/time draw for the whole selection, made
        # before it is cut into groups, so the loss does not depend on the cut.
        valid = float(mask.sum()) * float(targets.shape[-1])
        noise = torch.randn_like(targets)
        times = torch.rand((targets.shape[0],), device=targets.device,
                           dtype=targets.dtype)
        weight = float(self.config.demo_anchor)
        group = int(self.config.anchor_microbatch) or int(feat.shape[0])
        total = torch.zeros((), device=device, dtype=torch.float32)
        for offset in range(0, int(feat.shape[0]), group):
            stop = offset + group
            with autocast(device, self.config.precision):
                cond = self.actor.condition(feat[offset:stop], instruction)
                loss, _ = flow_matching_loss(
                    self.actor.velocity_fn(), targets[offset:stop], cond,
                    mask=mask[offset:stop], denominator=valid,
                    noise=noise[offset:stop], times=times[offset:stop])
            (weight * loss).backward()
            total += loss.detach().float().to(total.device)
            del loss, cond
        return {"anchor_loss": float(total),
                "anchor_rows": float(feat.shape[0]),
                "demo_anchor": weight}
