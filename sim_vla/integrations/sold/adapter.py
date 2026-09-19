"""SOLD's causal slot history to one SmolVLA conditioning token.

The conditioning feature for a slot world model is not a vector. It is a set of
slots per frame over a history of frames, and turning that into one token is
exactly the problem SOLD's own actor already solves: a
``SlotAggregationTransformer`` with an ALiBi causal mask over
``(time x slots + cls)`` tokens, reading the cls token of the requested row.

So the adapter **is** that -- ``modeling.sold.prediction.Predictor``, upstream's
class, with the output width set to SmolVLA's conditioning width. No new
dynamics module, no second encoder, no extra modality: the same architecture
family reading the same slots the native actor reads.

The bounded context, and why
----------------------------

The one thing that is not upstream's is that the adapter reads a **fixed
number of trailing frames**, and it is worth being precise about why.

``AlibiMask`` builds ``mask[h, i, j] = slope_h * j`` under a causal triangle,
and ``AttentionMask.forward`` slices the bottom-right ``L x L`` corner of it.
Slicing from the corner subtracts a per-row constant, which softmax ignores, so
the bias a query at within-window index ``i`` puts on a key at within-window
index ``j`` is ``slope * j`` -- **absolute within the window**, not relative to
the query. The consequence is that the attention profile depends on how long
the window is.

Upstream lives with that: ``imagine_ahead`` calls the actor on a context that
grows from ``num_context`` to ``num_context + imagination_horizon``, while
``select_action`` calls it on a history that grows to the whole episode. The
actor is therefore trained at one set of window lengths and run at another.

A policy that is *pretrained elsewhere* and conditioned through one token
cannot absorb that. So the adapter fixes the window: it reads the last
``context`` frames wherever it is called -- imitation, imagination, and
inference -- and the three see the same attention profile over the same slots.
This is an intentional actor-side difference from the native actor and is
reported as one.

It does not truncate slot *identity*. The slots themselves are produced by
SAVi's recurrence over the whole episode, so a slot at time ``t`` still carries
everything before it; what is bounded is how many frames the conditioning head
attends over.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..vendor import SOLD as VENDOR


def _predictor_class():
    return VENDOR.get("modeling.sold.prediction", "Predictor")


class SlotHistoryAdapter(nn.Module):
    """``(B, S, slots, dim) -> (B, 1, token_dim)``: one conditioning token."""

    def __init__(self, *, num_slots: int, slot_dim: int, token_dim: int,
                 context: int, max_episode_steps: int,
                 hidden_dim: int = 512, num_heads: int = 8,
                 num_layers: int = 3, num_mlp_layers: int = 1,
                 head_token_dim: int = 256, num_register_tokens: int = 0):
        super().__init__()
        self.num_slots = int(num_slots)
        self.slot_dim = int(slot_dim)
        self.token_dim = int(token_dim)
        self.context = int(context)
        if self.context < 1:
            raise ValueError("the adapter needs at least one frame of context")
        if self.context > int(max_episode_steps):
            raise ValueError(
                f"context={self.context} exceeds max_episode_steps="
                f"{max_episode_steps}; the attention mask is built for that "
                "many frames")

        predictor = _predictor_class()
        self.predictor = predictor(
            max_episode_steps=int(max_episode_steps), num_slots=self.num_slots,
            slot_dim=self.slot_dim, token_dim=int(head_token_dim),
            num_heads=int(num_heads), num_layers=int(num_layers),
            hidden_dim=int(hidden_dim), output_dim=self.token_dim,
            num_register_tokens=int(num_register_tokens),
            num_mlp_layers=int(num_mlp_layers))
        # Small, so a pretrained expert starts near the behaviour its own state
        # embedding produced rather than being kicked by a random projection.
        last = self.predictor.mlp[-1]
        nn.init.normal_(last.weight, std=0.02)
        nn.init.zeros_(last.bias)

    # ------------------------------------------------------------------ input
    def window(self, slots: torch.Tensor) -> torch.Tensor:
        """The trailing ``context`` frames. Fewer only at an episode's start."""
        if slots.dim() != 4:
            raise ValueError(
                f"expected slots (batch, steps, num_slots, slot_dim), got "
                f"{tuple(slots.shape)}")
        if slots.shape[-2] != self.num_slots or slots.shape[-1] != self.slot_dim:
            raise ValueError(
                f"the adapter was built for {self.num_slots} slots of "
                f"{self.slot_dim} and got {slots.shape[-2]}x{slots.shape[-1]}")
        return slots[:, -self.context:]

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """One token for the **last** frame of the given history.

        Always the last: a conditioning row is a moment in time, and reading an
        earlier row out of a longer window would give it a different amount of
        history than the same row gets at inference.
        """
        window = self.window(slots)
        tokens = self.predictor(window, start=window.shape[1] - 1)
        return tokens                                      # (B, 1, token_dim)

    def select_windows(self, slots: torch.Tensor,
                       index: torch.Tensor) -> torch.Tensor:
        """The trailing history of each selected conditioning row.

        ``slots`` is ``(B, T, N, D)`` and ``index`` selects flat ``b * T + t``
        rows. The result is ``(rows, context, N, D)``: each selected row gets
        *its own* trailing ``context`` frames, so a row in the middle of a
        training window sees exactly what the same row would see online.

        Returned as slot windows rather than as tokens, so the ordinary
        ``condition(features)`` path runs them -- one code path builds the
        conditioning in imitation, in imagination and at inference.

        Rows with fewer than ``context`` frames behind them are refused: the
        caller excludes them with burn-in, because a row conditioned on a
        shorter history is a different question from the one the policy will be
        asked at run time.
        """
        batch, steps, num_slots, slot_dim = slots.shape
        if self.context > steps:
            raise ValueError(
                f"the window has {steps} frames and the adapter reads "
                f"{self.context}; lengthen the imitation window")
        # (B, T - context + 1, context, N, D)
        unfolded = slots.unfold(1, self.context, 1).permute(0, 1, 4, 2, 3)
        offset = self.context - 1
        starts = torch.div(index, steps, rounding_mode="floor")
        within = index - starts * steps
        if bool((within < offset).any()):
            raise ValueError(
                f"{int((within < offset).sum())} row(s) have fewer than "
                f"{self.context} frames of history. Set burn_in to at least "
                f"{offset} so every eligible row has a full context.")
        return unfolded[starts, within - offset]

    # ----------------------------------------------------------------- report
    def parameter_report(self) -> Dict[str, Any]:
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "num_slots": self.num_slots,
            "slot_dim": self.slot_dim,
            "token_dim": self.token_dim,
            "context": self.context,
        }
