"""SmolVLA as SOLD's actor, at the two places SOLD samples one.

SOLD asks its actor for an action in exactly two places:

``imagine_ahead``    once per imagined step, on the slot context built so far,
                     with the context detached. The return of the resulting
                     rollout is then differentiated *through* that action --
                     ``actor_gradients: dynamics``.
``select_action``    once per environment step, on the episode's slot history.

Both go through this class. The native Gaussian actor is left in place and
untouched; with no policy attached it is what runs.

What is different, and why
--------------------------

**No entropy bonus.** ``compute_actor_loss`` adds
``-actor_entropy_loss_weight * mean(discounts * entropy)``. A flow policy has
no analytic entropy, so in SmolVLA mode that term is dropped and nothing is put
in its place. The lambda-return term -- the advantage under dynamics gradients,
with the same discounting and the same return normalisation -- is unchanged.

**REINFORCE is refused.** ``actor_gradients: reinforce`` is
``log pi(a) * advantage``. There is no tractable ``log pi(a)`` here and the
flow-matching loss is a regression onto a velocity, not a density.
``SOLDModule.attach_policy`` raises rather than substituting one.

**The conditioning context is bounded.** See ``adapter.py``: SOLD's ALiBi mask
makes the attention profile depend on the window length, and upstream trains
the actor on windows of 3..18 frames while running it on a history that grows
to the whole episode. A pretrained expert conditioned through one token cannot
absorb that, so the adapter reads a fixed number of trailing frames in
imitation, imagination and inference alike.

**One action per call.** SmolVLA predicts a chunk; the first action of it is
what is used, and the next call is made from the next slot state. Executing
more of the chunk would make the imagined policy and the executed policy
different policies, which is the thing the imagined return is supposed to be
about.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from ..latent_actor import LatentActor


class LatentSlotPolicy(nn.Module):
    """A slot-history-conditioned action-chunk policy behind SOLD's hooks."""

    def __init__(self, actor: LatentActor, converter, *, lr: float = 1e-5,
                 execute: int = 1, max_batch: int = 256,
                 min_num_context: int = 1,
                 descriptor: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.actor = actor
        self.converter = converter
        self.max_batch = int(max_batch)
        self._meta = dict(descriptor or {})

        if int(execute) != 1:
            raise ValueError(
                f"execute={execute} is not supported for SOLD: imagine_ahead "
                "re-plans at every imagined step, so executing more than one "
                "action per chunk online would optimise a policy that is not "
                "the one being run. Use execute=1, or give imagination "
                "matching execution semantics first.")
        self.execute = 1

        adapter = getattr(actor, "adapter", None)
        context = int(getattr(adapter, "context", 1))
        if context > int(min_num_context):
            raise ValueError(
                f"the adapter reads {context} frames of slot history but "
                f"imagination starts from {min_num_context}. The first "
                "imagined step would then see a shorter window than every "
                "later one, which is exactly the inconsistency the bounded "
                "context exists to remove. Set the adapter's context to at "
                f"most {min_num_context}, or raise num_context.")
        self.context = context

        trainable = self.actor.trainable_parameters()
        if not trainable:
            raise RuntimeError(
                "nothing in the actor is trainable; the adapter and the action "
                "expert are supposed to be")
        self.optimizer = torch.optim.AdamW(trainable, lr=float(lr))
        self.lr = float(lr)
        self.calls = {"imagine": 0, "act": 0}
        self.rows = {"imagine": 0, "act": 0}

    # --------------------------------------------------------------- sampling
    def chunk(self, slots: torch.Tensor, *, grad: bool,
              generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """One chunk per row of ``slots``'s batch, in the actor's coordinates.

        The slot window *is* the conditioning feature: the adapter is the
        actor's adapter, so ``condition(features)`` runs it and the prefix is
        built exactly once, by the same code, here and in imitation.
        """
        pieces = []
        size = max(int(self.max_batch), 1)
        for start in range(0, int(slots.shape[0]), size):
            pieces.append(self.actor.sample_chunk(
                slots[start:start + size], differentiable=bool(grad),
                generator=generator))
        return torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]

    def sample(self, slots: torch.Tensor, *, start: Optional[int] = None,
               grad: bool = False, deterministic: bool = False,
               site: str = "imagine") -> torch.Tensor:
        """``(B, S, slots, dim) -> (B, action_dim)`` in SOLD's own units.

        ``start`` is accepted because the native actor takes it, and only the
        last row is supported: a conditioning row is a moment in time and the
        adapter always reads the history ending at it.
        """
        if slots.dim() != 4:
            raise ValueError(
                f"expected slots (batch, steps, num_slots, slot_dim), got "
                f"{tuple(slots.shape)}")
        last = int(slots.shape[1]) - 1
        if start is not None and int(start) != last:
            raise ValueError(
                f"start={start} but this policy conditions on the history "
                f"ending at row {last}; reading an earlier row out of a longer "
                "window would give it a different amount of history than it "
                "gets online")
        generator = None
        if deterministic:
            # Not the distribution's mode -- a flow policy has none -- but a
            # fixed noise draw, which makes the action a deterministic,
            # reproducible function of the slot history. That is what an
            # evaluation rollout needs, and it is not the mean.
            generator = torch.Generator(device=self.actor.device)
            generator.manual_seed(0)
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            sampled = self.chunk(slots.to(self.actor.device), grad=grad,
                                 generator=generator)
            native = self.converter.to_env(sampled[:, 0])
        self.calls[site] += 1
        self.rows[site] += int(slots.shape[0])
        return native

    def act(self, slot_history: torch.Tensor, *, deterministic: bool = False
            ) -> torch.Tensor:
        """One action for the current episode state, for ``select_action``.

        The history may be shorter than ``context`` in an episode's first few
        steps; the adapter takes what there is, which is what a training window
        at an episode start would also give it.
        """
        return self.sample(slot_history, grad=False,
                           deterministic=deterministic, site="act")

    # ------------------------------------------------------------- optimizing
    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def step(self, grad_clip: float) -> float:
        clipped = torch.nn.utils.clip_grad_norm_(
            self.actor.trainable_parameters(), float(grad_clip))
        self.optimizer.step()
        return float(clipped)

    # ------------------------------------------------------------ persistence
    def descriptor(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "context": self.context,
            "execute": self.execute,
            "chunk_size": int(self.actor.chunk_size),
            "action_dim": int(self.actor.action_dim),
            "flow_steps": int(self.actor.flow_steps),
            "online_lr": self.lr,
            "action_normalization": self.converter.mode,
            "log_prob": "unavailable (flow policy); entropy term disabled",
            "actor_gradients": "dynamics (reinforce is refused)",
        }
        out.update(self._meta)
        return out

    def check_descriptor(self, stored: Mapping[str, Any]) -> None:
        current = self.descriptor()
        for key in ("context", "chunk_size", "action_dim",
                    "action_normalization", "revision"):
            left, right = dict(stored).get(key), current.get(key)
            if left in (None, "", 0) or right in (None, "", 0):
                continue
            if left != right:
                raise SystemExit(
                    f"refusing to load this policy: {key} is {left!r} in the "
                    f"checkpoint and {right!r} in this run.")

    def usage(self) -> Dict[str, Any]:
        return {"calls": dict(self.calls), "rows": dict(self.rows)}
