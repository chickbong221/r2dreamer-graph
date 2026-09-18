"""The shared imitation contract, for backends that are not Dreamer.

The rules are the same three the Dreamer integration settled on, and they are
restated here because they are properties of *chunked imitation*, not of any
one world model:

**Conditioning is causal.** At row ``t`` the policy may see observations
through ``o_t`` and no further. Nothing in this module reads a future
observation; the lookahead it uses carries *actions* only.

**The target is the whole chunk.** Row ``t`` is supervised on
``[a_t, ..., a_(t+H-1)]``, taken from ``action_target``. It is not supervised on
``action``, which is ``a_(t-1)`` -- the action the latent at ``t`` already
consumed. Supervising on that puts the target inside its own input.

**Availability is not eligibility.** Two masks answer two questions.
``loss_mask`` says this row is scored (burn-in excluded); ``action_valid`` says
a real action was loaded at this row. A row may be conditioned on only if both
hold, while the chunk *offsets* past it are masked by availability alone -- and
those live on the longer target axis, which is what lets the last scored row of
a full interior window still be supervised on a complete chunk.

**Eligible rows are selected before the actor runs.** The flow forward is the
expensive part of a step, and a ``B x T`` grid that is then mostly masked away
pays for rows that contribute nothing.

:func:`chunk_targets` is imported from the Dreamer integration rather than
reimplemented. It is the piece that clamps the gather, cuts the chunk at the
first unavailable offset with a cumulative product, and indexes availability on
the target axis; a second copy of that is a second place for it to go wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from ..training.train_imitation import chunk_targets

TARGET_KEY = "action_target"
AVAILABLE_KEY = "action_valid"
SCORED_KEY = "loss_mask"


@dataclass
class ChunkSelection:
    """Flattened, eligible conditioning rows and the chunk they are scored on."""

    rows: torch.Tensor          # (N,) flat index into batch * steps
    targets: torch.Tensor       # (N, chunk, action_dim), actor coordinates
    mask: torch.Tensor          # (N, chunk) bool: a real action at this offset
    eligible: torch.Tensor      # (B, T) bool, before flattening
    batch: int
    steps: int

    @property
    def empty(self) -> bool:
        return int(self.rows.numel()) == 0

    def features(self, feature: torch.Tensor) -> torch.Tensor:
        """Pick the eligible rows out of a ``(B, T, D)`` feature grid."""
        if feature.shape[:2] != (self.batch, self.steps):
            raise ValueError(
                f"feature grid is {tuple(feature.shape[:2])} but the masks are "
                f"({self.batch}, {self.steps}); the conditioning states and "
                "the targets must be indexed by the same rows")
        return feature.reshape(self.batch * self.steps, -1)[self.rows]

    def stats(self) -> Dict[str, float]:
        if self.empty:
            return {"eligible_rows": 0.0, "target_fraction": 0.0}
        return {"eligible_rows": float(self.rows.numel()),
                "target_fraction": float(self.mask.float().mean())}


def eligibility(scored: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    """Rows the actor may be trained at, on the observation axis.

    ``available`` runs on the target axis, which is ``lookahead`` rows longer,
    so it is truncated here rather than at the call site -- the one place that
    truncation is easy to forget.
    """
    steps = int(scored.shape[1])
    return scored.bool() & available.bool()[:, :steps]


def select(batch: Mapping[str, Any], chunk: int, *,
           converter=None) -> ChunkSelection:
    """Gather the eligible conditioning rows and their supervised chunks.

    ``converter`` maps the stored (native) actions into the actor's
    coordinates. It is the same object the policy's output path uses, so
    imitation targets and policy samples cannot end up in different units.
    """
    for key in (TARGET_KEY, AVAILABLE_KEY, SCORED_KEY):
        if key not in batch:
            raise KeyError(
                f"the window has no {key!r}; every source must build windows "
                "through sim_vla.data.layout.assemble, which is what splits "
                "a_(t-1) from a_t and availability from eligibility. Present: "
                f"{sorted(batch)[:12]}")
    targets = batch[TARGET_KEY]
    if targets.dim() != 3:
        raise ValueError(
            f"{TARGET_KEY} should be (batch, target_rows, action_dim), got "
            f"{tuple(targets.shape)}")
    targets = targets.float()
    if converter is not None:
        targets = converter.to_actor(targets)

    scored = batch[SCORED_KEY].bool()
    available = batch[AVAILABLE_KEY].bool()
    gathered, mask, eligible = chunk_targets(
        targets, available, eligibility(scored, available), int(chunk))

    batch_size, steps = eligible.shape
    flat = eligible.reshape(batch_size * steps)
    rows = torch.nonzero(flat, as_tuple=False).squeeze(-1)
    return ChunkSelection(
        rows=rows,
        targets=gathered.reshape(batch_size * steps, int(chunk),
                                 gathered.shape[-1])[rows],
        mask=mask.reshape(batch_size * steps, int(chunk))[rows],
        eligible=eligible, batch=batch_size, steps=steps)
