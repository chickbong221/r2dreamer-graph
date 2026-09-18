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


# What a trained progress head needs, and where each piece would come from.
# Checked before any expensive work rather than discovered after Stage 1A.
REQUIRED_PROGRESS_CONTRACT = (
    ("model.progress.schedule", "the task schedule naming the stages whose "
     "satisfaction defines progress; the repository's progress.load_stages "
     "reads one, and no sim_vla config supplies a path to it"),
    ("dataset.metadata['progress']", "per-step progress targets recorded "
     "alongside the demonstrations; sim_vla/data/collect.py records graphs "
     "and rewards but no schedule phase, so there is nothing to regress the "
     "head onto"),
    ("graph decoder relation probabilities", "TaskScheduleReplayPotential "
     "consumes per-relation probabilities; SimpleGraphDecoder produces "
     "reconstruction losses and does not expose them"),
)


def preflight(cfg, metadata=None) -> None:
    """Refuse ``graph_progress`` before anything expensive runs.

    The arm is declared in the configs and the head can be constructed, but
    nothing in this package trains it and nothing supplies its targets. Running
    anyway would produce a third arm that is bit-for-bit the second one while
    being reported as a different method -- which is worse than not having it,
    because the comparison would look like a null result rather than a missing
    feature.

    Inventing labels would be worse still. So this says exactly what is absent
    and stops.
    """
    progress = dict((cfg.get("model") or {}).get("progress") or {})
    if not bool(progress.get("enabled")):
        return

    recorded = dict((metadata or {}).get("progress") or {})
    missing = []
    if not progress.get("schedule"):
        missing.append(REQUIRED_PROGRESS_CONTRACT[0])
    if not recorded:
        missing.append(REQUIRED_PROGRESS_CONTRACT[1])
    missing.append(REQUIRED_PROGRESS_CONTRACT[2])

    detail = "\n".join(f"  - {name}: {why}" for name, why in missing)
    raise SystemExit(
        "refusing to run the graph_progress arm: the progress supervision "
        "contract is not available.\n" + detail + "\n"
        "Run --experiment graph instead, or supply the contract above. This "
        "stops here rather than training a head on invented targets or "
        "silently running plain graph training under the graph_progress name.")
