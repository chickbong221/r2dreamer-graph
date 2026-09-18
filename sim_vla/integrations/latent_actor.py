"""SmolVLA behind explicit sampling hooks, and nothing it cannot honestly do.

Both backends' native policies are Gaussian and both native objectives read
more than an action off them. TD-MPC2's ``update_pi`` weights ``entropy_coef *
log_pi - Q``; SOLD's actor loss adds ``-entropy`` and its ``reinforce`` branch
multiplies ``log_prob`` by a detached advantage. A conditional flow model
offers none of that:

* there is no tractable density. The sampler integrates a velocity field; the
  change-of-variables term would need the divergence of that field at every
  Euler step, which is neither cheap nor what either objective assumes.
* the flow-matching loss is a **regression onto a velocity**. Substituting it
  where a log-probability is expected produces a number with the wrong sign
  convention, the wrong units and no relationship to the density. It runs, and
  it optimises nothing.

So this wrapper deliberately does **not** present the Gaussian interface. It
exposes sampling hooks, and ``log_prob``/``entropy`` raise rather than return
a plausible-looking tensor. The backends' actor objectives are then adjusted at
exactly one place each, in SmolVLA mode only, and those adjustments are listed
in the integration's deviation report.

What is preserved on both sides is the *return* term: TD-MPC2 keeps the scaled
Q objective with its ``rho`` weighting, SOLD keeps the discounted lambda-return
advantage under dynamics gradients. Only the entropy/log-probability terms are
switched off, and only when SmolVLA is the policy.

Gradients
---------

``differentiable=True`` keeps the integration graph so a return gradient
reaches the adapter and the action expert through the sampled action. The
frozen transformer is frozen with ``requires_grad_(False)``, never
``detach()``: gradients flow *through* it into the adapter, which is the only
reason conditioning a frozen model works at all.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn


class ActorCapabilityError(NotImplementedError):
    """Asked for something a flow policy does not have."""


class LatentActor(nn.Module):
    """A latent-conditioned action-chunk policy, with sampling hooks only.

    Wraps anything that provides ``condition(features, instruction)`` and
    ``velocity_fn()`` -- in practice
    :class:`sim_vla.models.smolvla_actor.SmolVLAActor` -- so the backends never
    touch the pretrained stack directly and so a test double is a drop-in.
    """

    def __init__(self, actor, *, instruction: Optional[str] = None,
                 flow_steps: Optional[int] = None):
        super().__init__()
        for required in ("condition", "velocity_fn"):
            if not callable(getattr(actor, required, None)):
                raise TypeError(
                    f"a latent actor needs {required}(); got "
                    f"{type(actor).__name__}")
        self.actor = actor
        self.instruction = instruction
        self.chunk_size = int(getattr(actor, "chunk_size", 1))
        self.action_dim = int(getattr(actor, "action_dim", 0))
        self.flow_steps = int(flow_steps or getattr(actor, "flow_steps", 10))
        if self.chunk_size < 1 or self.action_dim < 1:
            raise ValueError(
                f"chunk_size={self.chunk_size} and action_dim={self.action_dim} "
                "must both be at least 1")

    # ----------------------------------------------------------------- basics
    @property
    def device(self) -> torch.device:
        actor_device = getattr(self.actor, "device", None)
        if actor_device is not None:
            return torch.device(actor_device)
        for parameter in self.parameters():
            return parameter.device
        return torch.device("cpu")

    @property
    def adapter(self):
        return getattr(self.actor, "adapter", None)

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def condition(self, features: torch.Tensor,
                  instruction: Optional[str] = None) -> Dict[str, Any]:
        return self.actor.condition(
            features, self.instruction if instruction is None else instruction)

    # ---------------------------------------------------------------- sampling
    def sample_chunk(self, features: torch.Tensor, *,
                     differentiable: bool = False,
                     steps: Optional[int] = None,
                     generator: Optional[torch.Generator] = None
                     ) -> torch.Tensor:
        """``(B, feature) -> (B, chunk, action_dim)`` in actor coordinates.

        ``differentiable=False`` is the rollout and proposal path, run under
        ``no_grad``; ``True`` is the policy-optimization path, where the
        backend's own return objective differentiates through every Euler step
        into the conditioning token.
        """
        from ..models.flow_sampler import sample_actions

        cond = self.condition(features)
        return sample_actions(
            self.actor.velocity_fn(), cond, batch=int(features.shape[0]),
            chunk=self.chunk_size, dim=self.action_dim,
            steps=int(steps or self.flow_steps),
            device=self.device, dtype=torch.float32,
            differentiable=bool(differentiable), generator=generator)

    def sample_action(self, features: torch.Tensor, *, index: int = 0,
                      differentiable: bool = False,
                      steps: Optional[int] = None) -> torch.Tensor:
        """One action out of a freshly sampled chunk.

        ``index`` is which action of the chunk to take, not a planning
        horizon: a proposal at a latent is the chunk's *first* action, and the
        next proposal comes from the latent that action leads to. The two are
        separate quantities and conflating them is how a chunk length gets
        mistaken for a planning horizon.
        """
        chunk = self.sample_chunk(features, differentiable=differentiable,
                                  steps=steps)
        if not 0 <= int(index) < self.chunk_size:
            raise IndexError(
                f"chunk index {index} is outside the policy's chunk of "
                f"{self.chunk_size}")
        return chunk[:, int(index)]

    def flow_loss(self, features: torch.Tensor, targets: torch.Tensor,
                  mask: Optional[torch.Tensor] = None,
                  generator: Optional[torch.Generator] = None):
        """Conditional flow-matching regression onto a demonstrated chunk."""
        from ..models.flow_sampler import flow_matching_loss

        cond = self.condition(features)
        return flow_matching_loss(self.actor.velocity_fn(), targets, cond,
                                  mask=mask, generator=generator)

    # --------------------------------------------------------- what it is not
    def log_prob(self, *args, **kwargs):
        raise ActorCapabilityError(
            "a flow-matching policy has no tractable log-probability. The "
            "flow-matching loss is a regression onto a velocity, not a "
            "density, and substituting it here would produce a number with no "
            "relationship to log pi(a). Use the return term of the native "
            "objective and disable the log-probability term in SmolVLA mode.")

    def entropy(self, *args, **kwargs):
        raise ActorCapabilityError(
            "a flow-matching policy has no analytic entropy. The native "
            "entropy bonus is disabled in SmolVLA mode rather than replaced by "
            "a stand-in regularizer.")

    def rsample(self, *args, **kwargs):
        raise ActorCapabilityError(
            "there is no reparameterised Gaussian here. Call sample_chunk or "
            "sample_action, which keep the gradient through the sampler when "
            "asked to.")

    # ------------------------------------------------------------------ report
    def report(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "chunk_size": self.chunk_size,
            "action_dim": self.action_dim,
            "flow_steps": self.flow_steps,
            "instruction": self.instruction,
            "log_prob": "unavailable (flow policy)",
            "entropy": "unavailable (flow policy)",
        }
        inner = getattr(self.actor, "trainable_report", None)
        if callable(inner):
            out["actor"] = inner()
        adapter = self.adapter
        if adapter is not None and hasattr(adapter, "parameter_report"):
            out["adapter"] = adapter.parameter_report()
        return out


def assert_gradient_reaches(action: torch.Tensor, parameters: Sequence[nn.Parameter],
                            *, what: str = "the sampled action") -> None:
    """Raise unless a sampled action carries gradient into ``parameters``.

    Run once at the start of an online run as well as in tests. A sampler built
    under ``no_grad``, or a conditioning path that was detached, gives an actor
    update that runs forever, costs the same, and learns nothing -- and reports
    no error anywhere.
    """
    from ..models.flow_sampler import assert_gradient_reaches as _check

    wanted = [p for p in parameters if p.requires_grad]
    if not wanted:
        raise RuntimeError(
            f"nothing trainable was passed to the gradient check for {what}")
    _check(action, *wanted)
