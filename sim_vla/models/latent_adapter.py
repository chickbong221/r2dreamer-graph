"""World-model features to one SmolVLA conditioning token.

A single state token, by decision. An earlier draft also projected a block of
learned context tokens; that is not built here, and the adapter's whole output
is one token that sits beside the fixed task-instruction embeddings.

The width of the input is the only thing that differs between arms::

    baseline  : feature_dim(h, z)
    graph arm : feature_dim(h, z, g)

Everything after the first linear layer -- trunk width, depth, output token
width -- is identical, so the comparison is between what the feature contains
and not between two differently shaped adapters. The extra input width is
extra capacity, and :meth:`parameter_report` exists so that it is stated rather
than discovered.

The feature is normalized before the trunk. ``h``, ``z`` and ``g`` are produced
by different mechanisms and do not share a scale -- ``z`` is a flattened
one-hot, ``h`` is a GRU state, ``g`` is a projected graph embedding -- and
concatenating them raw lets whichever happens to be largest dominate the first
layer.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class LatentAdapter(nn.Module):
    """``(h, z[, g]) -> one conditioning token`` of the actor's width."""

    def __init__(self, feature_dim: int, token_dim: int, *, hidden: int = 1024,
                 layers: int = 2, act: str = "SiLU"):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.token_dim = int(token_dim)
        self.hidden = int(hidden)

        activation = getattr(nn, act)
        self.input_norm = nn.LayerNorm(self.feature_dim)
        trunk: list[nn.Module] = []
        width = self.feature_dim
        for _ in range(max(int(layers), 1)):
            trunk += [nn.Linear(width, self.hidden), nn.LayerNorm(self.hidden),
                      activation()]
            width = self.hidden
        self.trunk = nn.Sequential(*trunk)
        self.state_token = nn.Linear(self.hidden, self.token_dim)
        # Small, so the pretrained actor starts near the behaviour its own
        # state embedding produced rather than being kicked by a random
        # projection on the first step.
        nn.init.normal_(self.state_token.weight, std=0.02)
        nn.init.zeros_(self.state_token.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``(..., feature_dim) -> (..., 1, token_dim)``: one token."""
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"adapter built for feature_dim={self.feature_dim} got "
                f"{features.shape[-1]}; a baseline feature is (h,z) and a graph "
                f"feature is (h,z,g), and they are different widths")
        hidden = self.trunk(self.input_norm(features))
        return self.state_token(hidden).unsqueeze(-2)

    def parameter_report(self) -> Dict[str, int]:
        """Counted, because the arms do not have the same number."""
        total = sum(p.numel() for p in self.parameters())
        first = self.feature_dim * self.hidden + self.hidden
        return {"total": total, "input_layer": first,
                "feature_dim": self.feature_dim, "token_dim": self.token_dim}
