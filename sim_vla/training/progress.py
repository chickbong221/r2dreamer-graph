"""Graph-derived progress shaping, as a separate and optional reward stream.

Three things keep this from contaminating the primary comparison.

It requires the graph. Progress targets come from the task schedule the graph
is mined against, so ``progress.enabled`` without ``graph.enabled`` is refused
at config load (``sim_vla/config.py``) rather than silently producing zeros.

Its reward is never added to the environment's. The two are carried and logged
apart, and evaluation reports environment success and environment return; a
shaped run that scored better only on its own shaping would be visible as
exactly that.

The shaping is potential-based: ``beta * (gamma * phi(s') - phi(s))``. That form
leaves the optimal policy unchanged, which is what makes the arm a comparison
of learning speed rather than of a different objective.

The baseline arm receives none of this -- no targets, no head, no reward term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

import networks


@dataclass
class ProgressConfig:
    enabled: bool = False
    beta: float = 0.1
    warmup_start: int = 400_000
    warmup_end: int = 900_000


def beta_at(config: ProgressConfig, step: int) -> float:
    """Linear warm-up, so shaping does not dominate an untrained value head."""
    if not config.enabled:
        return 0.0
    if step <= config.warmup_start:
        return 0.0
    if step >= config.warmup_end:
        return float(config.beta)
    span = max(config.warmup_end - config.warmup_start, 1)
    return float(config.beta) * (step - config.warmup_start) / span


class ProgressHead(nn.Module):
    """Predicts scalar task progress from a latent feature.

    Only ever constructed for the graph arm: its target is the graph schedule's
    phase, and there is no schedule without a graph.
    """

    def __init__(self, config, feature_dim: int):
        super().__init__()
        self.net = networks.MLPHead(config.critic, int(feature_dim))

    def potential(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).mode().squeeze(-1)

    def loss(self, feat: torch.Tensor, target: torch.Tensor,
             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        error = (self.potential(feat) - target.detach()) ** 2
        if mask is None:
            return error.mean()
        weight = mask.to(error.dtype)
        return (error * weight).sum() / weight.sum().clamp(min=1.0)


def shaping_reward(head: ProgressHead, feat: torch.Tensor, discount: float
                   ) -> torch.Tensor:
    """``gamma * phi(s') - phi(s)`` over an imagined rollout.

    Potential-based, so it cannot change which policy is optimal -- only how
    quickly one is found.
    """
    phi = head.potential(feat)
    return discount * phi[1:] - phi[:-1]


def build_progress(config, feature_dim: int, *, graph_enabled: bool,
                   progress_enabled: bool) -> Optional[ProgressHead]:
    """A head for the graph arm with shaping on, and None otherwise."""
    if not progress_enabled:
        return None
    if not graph_enabled:
        raise SystemExit(
            "progress shaping requires the graph arm: its targets come from "
            "the graph schedule, and a baseline has no schedule to read.")
    return ProgressHead(config, feature_dim)
