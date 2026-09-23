"""Linear warmup, then cosine decay, for the two offline stages.

Stage 1A and Stage 1B used to hold their learning rate constant to the last
step, so the weights they saved were wherever the final full-size updates left
them. A schedule ramps the rate up linearly over ``warmup`` steps and then
decays it by cosine to ``final`` at the stage's last step -- the shape of
LeRobot's SmolVLA fine-tuning schedule.

``warmup=0`` with ``final=None`` is the constant rate, so a run that sets
neither trains exactly as before.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LRSchedule:
    peak: float
    total: int
    warmup: int = 0
    final: Optional[float] = None

    def __post_init__(self):
        if not (math.isfinite(float(self.peak)) and float(self.peak) > 0):
            raise ValueError(f"peak learning rate {self.peak} must be positive")
        if int(self.warmup) < 0:
            raise ValueError(f"warmup={self.warmup} must be nonnegative")
        if self.final is not None and not (
                math.isfinite(float(self.final))
                and 0 < float(self.final) <= float(self.peak)):
            raise ValueError(
                f"final learning rate {self.final} must be positive and at "
                f"most the peak {self.peak}: it is where the decay ends")

    def at(self, step: int) -> float:
        """The rate for optimizer step ``step``, counted from 0."""
        step, warmup = int(step), int(self.warmup)
        peak = float(self.peak)
        if step < warmup:
            return peak * (step + 1) / warmup
        if self.final is None:
            return peak
        final = float(self.final)
        # The last step, total - 1, lands exactly on final.
        span = max(int(self.total) - warmup - 1, 1)
        progress = min((step - warmup) / span, 1.0)
        return final + 0.5 * (peak - final) * (1.0 + math.cos(math.pi * progress))

    def apply(self, optimizer, step: int) -> float:
        """Set every parameter group to step ``step``'s rate, and return it."""
        lr = self.at(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr

    def describe(self) -> str:
        text = f"{float(self.peak):g}"
        if int(self.warmup):
            text += f", linear warmup over {int(self.warmup)} steps"
        if self.final is None:
            return text + (", then constant" if int(self.warmup)
                           else " constant")
        return text + (f", cosine decay to {float(self.final):g} at step "
                       f"{int(self.total)}")
