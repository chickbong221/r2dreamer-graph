"""Value critic with a slow target, for both arms.

The critic regresses the lambda-return on detached targets; the slow copy that
produces the bootstrap is updated by an exponential move toward the live one,
so the target a value is trained against does not move at the same rate as the
value.

The actor objective this serves is not the one ``dreamer.py`` uses. That actor
is a distribution and its update weights ``log pi(a)`` by an advantage; a flow
policy has no tractable log-probability, and substituting the flow-matching
regression loss for one would be weighting a reconstruction error by an
advantage, which means nothing. Instead the return is differentiated with
respect to the sampled action -- see
:func:`~sim_vla.training.actor_critic.actor_loss`. Both arms use this same
objective, so the comparison between them is not also a comparison between two
update rules.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn

import networks


class ValueCritic(nn.Module):
    """A value head over world-model features, plus its slow target."""

    def __init__(self, config, feature_dim: int, *, slow_fraction: float = 0.02):
        super().__init__()
        self.net = networks.MLPHead(config.critic, int(feature_dim))
        self.target = copy.deepcopy(self.net)
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.slow_fraction = float(slow_fraction)
        self.feature_dim = int(feature_dim)

    def value(self, feat: torch.Tensor) -> torch.Tensor:
        # symexp_twohot, whose mode() is a method -- unlike the binary
        # continuation head, whose mode is a property. dreamer.py:1282 reads
        # the value the same way.
        return self.net(feat).mode().squeeze(-1)

    def target_value(self, feat: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.target(feat).mode().squeeze(-1)

    def loss(self, feat: torch.Tensor, returns: torch.Tensor,
             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Regress detached returns. The actor's gradient does not come here."""
        dist = self.net(feat)
        error = -dist.log_prob(returns.detach().unsqueeze(-1))
        if mask is None:
            return error.mean()
        weight = mask.to(error.dtype)
        return (error * weight).sum() / weight.sum().clamp(min=1.0)

    @torch.no_grad()
    def update_target(self) -> None:
        for slow, live in zip(self.target.parameters(), self.net.parameters()):
            slow.mul_(1.0 - self.slow_fraction).add_(
                live.detach() * self.slow_fraction)


class ProgressCritic(ValueCritic):
    """A second value head for the progress reward stream.

    Separate rather than a second output of the value critic, so the two
    streams keep separate targets and separate logs: the evaluation reports
    environment return, and a progress term folded into the same head would be
    invisible in it.
    """
