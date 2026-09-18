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
must not be substituted into an objective that expects one: a Dreamer actor
loss weighting ``log pi(a) * advantage`` is meaningless here, which is why the
online actor update differentiates through the sampled action instead.

The sampler integrates the same field forward from noise. The ordinary
inference path runs under ``no_grad``; the online actor update cannot, because
the gradient of the return has to reach the adapter through the action that
produced it. :func:`sample_actions` therefore keeps the graph by default and
the caller opts out, rather than the reverse -- a silently detached action
gives an actor loss that runs, produces no learning, and reports nothing wrong.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

# (x_t, t, cond) -> velocity, shaped like x_t.
VelocityFn = Callable[[torch.Tensor, torch.Tensor, object], torch.Tensor]


def flow_matching_loss(velocity_fn: VelocityFn, actions: torch.Tensor,
                       cond: object, *, mask: Optional[torch.Tensor] = None,
                       dim_mask: Optional[torch.Tensor] = None,
                       generator: Optional[torch.Generator] = None
                       ) -> Tuple[torch.Tensor, dict]:
    """Conditional flow-matching regression onto the demonstrated chunk.

    ``mask`` is per (batch, step) and excludes padding past the end of an
    episode; ``dim_mask`` is per action dimension, for a robot whose action
    vector has entries this task does not drive. Both are applied before the
    mean so that a chunk half past the end does not count as half a chunk of
    error.
    """
    if actions.dim() != 3:
        raise ValueError(
            f"expected actions (batch, chunk, dim), got {tuple(actions.shape)}")
    batch, chunk, dim = actions.shape
    device, dtype = actions.device, actions.dtype

    noise = torch.randn(actions.shape, device=device, dtype=dtype,
                        generator=generator)
    # One time per chunk, not per step: a chunk is one sample of the path.
    t = torch.rand((batch, 1, 1), device=device, dtype=dtype, generator=generator)
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
    loss = (error * weight).sum() / weight.sum().clamp(min=1.0)
    return loss, {"flow_loss": loss.detach(),
                  "flow_scale": (error.detach() * weight).max()}


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
