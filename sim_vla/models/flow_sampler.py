"""Flow matching: the imitation loss, and a sampler gradients survive.

Both halves take a velocity function rather than a policy, so neither depends
on how the pretrained checkpoint spells its internals, and both can be tested
against a velocity field whose answer is known.

The convention is LeRobot's, not the textbook's, because the pretrained
expert was trained under LeRobot's and the two run in opposite directions::

    x_t    = t * noise + (1 - t) * actions,   noise ~ N(0, I),  t ~ U(0, 1)
    target = noise - actions
    loss   = || v(x_t, t, cond) - target ||^2

So ``t = 0`` is the clean action and ``t = 1`` is noise, and integration to
sample runs from 1 down to 0. Writing it the other way round gives a loss that
descends, a sampler that runs, and a policy that produces noise -- which is why
this is stated here rather than left to the reader of two files.

It is a regression onto a velocity. It is **not** a log-probability, and it
must not be substituted into an objective that expects one: weighting it by an
advantage would be weighting a reconstruction error.

Two objectives are built on that fact rather than around it. The ``pathwise``
actor update differentiates through the sampled action, which needs no density
at all. The ``flow_reinforce`` update gets a real density by sampling a
different, explicitly stochastic policy -- see :func:`sample_flow_path` and
:func:`transition_log_prob` at the end of this module -- whose per-transition
Gaussian log density is exact. The deterministic sampler below still has none,
and scoring it as though it did is refused rather than approximated.

The sampler integrates the same field forward from noise. The ordinary
inference path runs under ``no_grad``; the online actor update cannot, because
the gradient of the return has to reach the adapter through the action that
produced it. :func:`sample_actions` therefore keeps the graph by default and
the caller opts out, rather than the reverse -- a silently detached action
gives an actor loss that runs, produces no learning, and reports nothing wrong.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

import torch

# (x_t, t, cond) -> velocity, shaped like x_t.
VelocityFn = Callable[[torch.Tensor, torch.Tensor, object], torch.Tensor]


def flow_matching_loss(velocity_fn: VelocityFn, actions: torch.Tensor,
                       cond: object, *, mask: Optional[torch.Tensor] = None,
                       dim_mask: Optional[torch.Tensor] = None,
                       denominator: Optional[Any] = None,
                       noise: Optional[torch.Tensor] = None,
                       times: Optional[torch.Tensor] = None,
                       generator: Optional[torch.Generator] = None
                       ) -> Tuple[torch.Tensor, dict]:
    """Conditional flow-matching regression onto the demonstrated chunk.

    ``mask`` is per (batch, step) and excludes padding past the end of an
    episode; ``dim_mask`` is per action dimension, for a robot whose action
    vector has entries this task does not drive. Both are applied before the
    mean so that a chunk half past the end does not count as half a chunk of
    error.

    ``denominator`` replaces this call's own ``weight.sum()``. It exists for
    the Stage 2 anchor, which scores one selected set of rows in several
    microbatches: dividing each group by its *own* valid-target count and
    adding the results is a mean of unequal masked means, so the total would
    depend on how the rows happened to be partitioned. Passing the whole
    selection's count makes every partition give the same sum. Left at None,
    this is exactly the Stage 1B loss.

    ``noise`` and ``times`` supply the path sample instead of drawing it, which
    is what makes the anchor's partition-independence testable at all: this
    loss is a random function of ``(z, tau)``, so two partitions that each draw
    their own differ because of the draw and not because of the normalization.
    Fixing them per row lets full-batch and microbatched *gradients* be
    compared directly.
    """
    if actions.dim() != 3:
        raise ValueError(
            f"expected actions (batch, chunk, dim), got {tuple(actions.shape)}")
    batch, chunk, dim = actions.shape
    device, dtype = actions.device, actions.dtype

    if noise is None:
        noise = torch.randn(actions.shape, device=device, dtype=dtype,
                            generator=generator)
    elif noise.shape != actions.shape:
        raise ValueError(
            f"supplied noise {tuple(noise.shape)} does not match the action "
            f"chunk {tuple(actions.shape)}")
    else:
        noise = noise.to(device=device, dtype=dtype)
    # One time per chunk, not per step: a chunk is one sample of the path.
    if times is None:
        t = torch.rand((batch, 1, 1), device=device, dtype=dtype,
                       generator=generator)
    else:
        t = times.reshape(batch, 1, 1).to(device=device, dtype=dtype)
    x_t = t * noise + (1.0 - t) * actions
    target = noise - actions

    predicted = velocity_fn(x_t, t.reshape(batch), cond)
    if predicted.shape != actions.shape:
        raise ValueError(
            f"velocity {tuple(predicted.shape)} does not match the action "
            f"chunk {tuple(actions.shape)}")

    error = (predicted - target) ** 2
    weight = torch.ones_like(error)
    if dim_mask is not None:
        weight = weight * dim_mask.to(dtype).reshape(1, 1, dim)
    if mask is not None:
        weight = weight * mask.to(dtype).reshape(batch, chunk, 1)
    if denominator is None:
        scale = weight.sum().clamp(min=1.0)
    else:
        scale = (denominator if torch.is_tensor(denominator)
                 else error.new_tensor(float(denominator)))
        scale = scale.to(error.dtype).clamp(min=1.0)
    loss = (error * weight).sum() / scale
    return loss, {"flow_loss": loss.detach(),
                  "flow_scale": (error.detach() * weight).max(),
                  "flow_valid_targets": weight.sum().detach()}


def sample_actions(velocity_fn: VelocityFn, cond: object, *, batch: int,
                   chunk: int, dim: int, steps: int = 10,
                   device=None, dtype=torch.float32,
                   differentiable: bool = True,
                   generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Integrate the velocity field from noise at t=1 to an action at t=0.

    Euler, uniformly spaced, and *backwards*: under LeRobot's convention noise
    lives at ``t = 1`` and the action at ``t = 0``, so the step is ``-dt`` and
    the loop counts down. Integrating forwards instead converges on noise.

    ``differentiable=True`` keeps the graph so an actor update can push the
    return's gradient back through every integration step into the
    conditioning; only the starting noise is detached, being a sample rather
    than a parameter.
    """
    x = torch.randn((batch, chunk, dim), device=device, dtype=dtype,
                    generator=generator)
    steps = max(int(steps), 1)
    dt = 1.0 / float(steps)
    context = torch.enable_grad() if differentiable else torch.no_grad()
    with context:
        for index in range(steps):
            t = torch.full((batch,), 1.0 - index * dt, device=device, dtype=dtype)
            x = x - dt * velocity_fn(x, t, cond)
    return x


def assert_gradient_reaches(action: torch.Tensor, *parameters: torch.Tensor
                            ) -> None:
    """Raise unless the sampled action carries gradient to ``parameters``.

    Used in tests and once at the start of an online run. A flow actor whose
    sampler was built under ``no_grad`` trains silently and forever without
    improving, and this is the cheapest place to find that out.
    """
    if not action.requires_grad:
        raise RuntimeError(
            "the sampled action carries no gradient; the flow sampler was run "
            "under no_grad or its conditioning was detached")
    grads = torch.autograd.grad(action.sum(), parameters, retain_graph=True,
                                allow_unused=True)
    missing = [i for i, g in enumerate(grads) if g is None]
    if missing:
        raise RuntimeError(
            f"no gradient reached parameter(s) {missing} through the sampled "
            "action; the conditioning path is broken")


# --------------------------------------------------------------------------
# Stochastic sampler for the score-function actor objective.
#
# The deterministic sampler above has no tractable density: its transitions are
# Dirac, and the flow-matching regression is a squared error, not a log
# probability. Neither may be substituted into ``advantage * log pi``.
#
# So the ``flow_reinforce`` objective samples from an explicitly different
# policy: the same Euler field, with fixed Gaussian noise added at every
# transition. Each transition is then a Gaussian whose log density is exact and
# cheap, and the *path* -- not the final action -- is what carries the score.
#
# This is a prototype schedule. It is not ReinFlow's learned noise and not
# RL-100's DDIM sigma. It is a valid discrete stochastic policy, but it is a
# different policy from the pretrained deterministic flow: it is not guaranteed
# to preserve that flow's action distribution or its task success. The starting
# policy's quality is a validation gate, not an assumption.
# --------------------------------------------------------------------------

# Versioned, and recorded in the checkpoint: a run scored under one schedule
# cannot be compared with a run scored under another, and the scale alone does
# not say which was used.
FLOW_NOISE_SCHEDULE = "constant_per_step_scaled_by_sqrt_k"

_LOG_2PI = math.log(2.0 * math.pi)


def flow_sigmas(flow_noise_std: float, steps: int, *,
                schedule: str = FLOW_NOISE_SCHEDULE,
                device=None, dtype=torch.float32) -> torch.Tensor:
    """One standard deviation per transition, ``K`` of them.

    ``flow_noise_std`` is a nominal *aggregate* injected-noise scale in
    normalized action coordinates: ``K`` independent transitions each
    contributing ``std/sqrt(K)`` accumulate to roughly ``std`` if the field
    were the identity. It is not a guarantee about the final action's variance,
    which the velocity field also shapes.

    Strictly positive and finite is a requirement, not a preference: a zero
    sigma makes the transition deterministic and its density singular, and
    deterministic sampling belongs to the pathwise path only.
    """
    if schedule != FLOW_NOISE_SCHEDULE:
        raise ValueError(
            f"unknown flow noise schedule {schedule!r}; this implementation "
            f"provides {FLOW_NOISE_SCHEDULE!r} only")
    steps = int(steps)
    if steps < 1:
        raise ValueError(f"flow steps must be at least 1, got {steps}")
    std = float(flow_noise_std)
    if not math.isfinite(std) or std <= 0.0:
        raise ValueError(
            f"flow_noise_std must be finite and strictly positive for the "
            f"flow_reinforce objective, got {flow_noise_std!r}; a zero or "
            "non-finite sigma has no Gaussian density to score")
    return torch.full((steps,), std / math.sqrt(steps),
                      device=device, dtype=dtype)


def flow_times(steps: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """``t_k = 1 - k/K`` for ``k = 0 .. K-1``: noise at 1, action at 0."""
    steps = int(steps)
    if steps < 1:
        raise ValueError(f"flow steps must be at least 1, got {steps}")
    dt = 1.0 / float(steps)
    return torch.tensor([1.0 - index * dt for index in range(steps)],
                        device=device, dtype=dtype)


def transition_mean(velocity_fn: VelocityFn, u: torch.Tensor, t: torch.Tensor,
                    cond: object, dt: float) -> torch.Tensor:
    """One Euler step's mean: the single definition both paths call.

    Sampling and scoring have to agree exactly or the recorded sample is not a
    draw from the distribution being differentiated, so they share this rather
    than each spelling the update out. The velocity is cast to ``u``'s dtype
    because the transformer may have produced it under autocast while the
    density arithmetic downstream is float32.

    The sign matches :func:`sample_actions`: under LeRobot's convention noise
    is at ``t = 1`` and the action at ``t = 0``, so integration steps *down*.
    """
    velocity = velocity_fn(u, t, cond)
    if velocity.shape != u.shape:
        raise ValueError(
            f"velocity {tuple(velocity.shape)} does not match the flow state "
            f"{tuple(u.shape)}")
    return u - float(dt) * velocity.to(u.dtype)


def check_sigmas(sigma) -> None:
    """Refuse a sigma with no Gaussian density. Synchronizes; call it once.

    ``isfinite(...).all()`` and ``(sigma <= 0).any()`` are reductions over a
    device tensor, and reading them as Python bools blocks until the queue
    drains. That is the right price at a boundary -- once per update, where
    the schedule is fixed for the whole update anyway -- and the wrong price
    inside a loop that runs thousands of times per update for the same
    unchanged values.
    """
    if not torch.is_tensor(sigma):
        sigma = torch.as_tensor(float(sigma))
    if not bool(torch.isfinite(sigma).all()) or bool((sigma <= 0).any()):
        raise ValueError(
            "a finite, strictly positive sigma is required; a deterministic "
            "transition has no Gaussian density and must not be scored as one "
            f"(got {sigma})")


def transition_log_prob(u_next: torch.Tensor, mean: torch.Tensor,
                        sigma, *, validate: bool = True) -> torch.Tensor:
    """``log N(u_next; mean, sigma^2 I)`` summed over the sampled coordinates.

    Summed, never averaged. All ``C * D`` coordinates of the chunk are one
    sample of one transition, so their densities multiply; averaging would
    divide the log density by ``C * D`` and silently rescale the entire actor
    loss against the imitation anchor.

    The whole chunk is scored even though only its first action is executed.
    The later rows are sampled random variables the expert attends over, so
    they are part of what this transition chose; masking them would score a
    distribution the policy never sampled from.

    ``validate=False`` skips the sigma check, which is a device-tensor
    reduction read as a Python bool and therefore a host synchronization. The
    scoring loop passes False *after* calling :func:`check_sigmas` once on the
    schedule for the whole update -- the values do not change between
    microbatches, so checking them thousands of times only adds stalls. The
    shape check stays: it is pure Python and costs nothing.
    """
    mean = mean.float()
    u_next = u_next.float()
    if u_next.shape != mean.shape:
        raise ValueError(
            f"sampled state {tuple(u_next.shape)} does not match its mean "
            f"{tuple(mean.shape)}")
    if not torch.is_tensor(sigma):
        sigma = mean.new_tensor(float(sigma))
    sigma = sigma.float()
    if validate:
        check_sigmas(sigma)
    residual = (u_next - mean) / sigma
    per_coordinate = -0.5 * residual ** 2 - torch.log(sigma) - 0.5 * _LOG_2PI
    # (..., C, D) -> (...): one scalar per sampled transition.
    return per_coordinate.flatten(start_dim=-2).sum(-1)


def sample_flow_path(velocity_fn: VelocityFn, cond: object, *, batch: int,
                     chunk: int, dim: int, steps: int, sigmas: torch.Tensor,
                     device=None, dtype=torch.float32,
                     generator: Optional[torch.Generator] = None
                     ) -> Dict[str, torch.Tensor]:
    """Integrate the noisy Euler chain and record every state it passed through.

    Deliberately establishes no gradient context of its own, unlike
    :func:`sample_actions`. The score-function objective collects this under
    ``torch.no_grad()`` and recomputes the means later with gradients, so a
    helper that re-enabled grad here would rebuild exactly the sequential
    expert graph this objective exists to avoid.

    Returns the ``K+1`` states and the ``K`` transitions' means, times and
    sigmas -- everything needed to score the same path again later.
    """
    steps = int(steps)
    if steps < 1:
        raise ValueError(f"flow steps must be at least 1, got {steps}")
    if tuple(sigmas.shape) != (steps,):
        raise ValueError(
            f"expected one sigma per transition ({steps},), got "
            f"{tuple(sigmas.shape)}")
    dt = 1.0 / float(steps)
    times = flow_times(steps, device=device, dtype=dtype)
    u = torch.randn((batch, chunk, dim), device=device, dtype=dtype,
                    generator=generator)
    states = [u]
    means = []
    for index in range(steps):
        t = times[index].expand(batch)
        mean = transition_mean(velocity_fn, u, t, cond, dt)
        noise = torch.randn(u.shape, device=device, dtype=dtype,
                            generator=generator)
        u = mean + sigmas[index].to(mean.dtype) * noise
        states.append(u)
        means.append(mean)
    return {
        "chunk": u,                                  # (B, C, D), = u_K
        "states": torch.stack(states, 1),            # (B, K+1, C, D)
        "means": torch.stack(means, 1),              # (B, K,   C, D)
        "times": times,                              # (K,)
        "sigmas": sigmas,                            # (K,)
    }
