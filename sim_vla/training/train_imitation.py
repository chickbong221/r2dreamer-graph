"""Stage 1B: adapter and action expert, on a frozen world model.

The world model is loaded from its arm's own Stage 1A checkpoint and frozen.
Episodes are encoded causally -- posterior states from the reset observation
forward, never from a future one -- and the adapter and action expert are
trained to reproduce the demonstrated action chunk with a flow-matching loss.

Chunking happens here rather than in the dataset. A stored dataset of
overlapping chunks is the same data written ``chunk_size`` times, and the
alignment that matters (a chunk starts at ``t``, is masked past the end of its
episode, and never crosses into the next one) is a property of how it is cut,
not of how it is stored.

Caches are per arm and per checkpoint. A latent cache is a function of the
world model that produced it, so it carries that checkpoint's identity and is
refused against any other -- sharing one between arms would mean the graph
arm's policy trained on the baseline's states.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ..models.flow_sampler import flow_matching_loss


def chunk_targets(actions: torch.Tensor, valid: torch.Tensor, chunk: int
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(B, T, A) -> (B, T, chunk, A)`` with a mask for the overrun.

    The chunk starting at the last step of an episode is one real action and
    ``chunk - 1`` steps of nothing. Padding it with the final action and
    masking the pad keeps the tensor rectangular without inventing a target
    the policy is scored against.
    """
    batch, steps, dim = actions.shape
    index = torch.arange(chunk, device=actions.device).reshape(1, 1, chunk)
    base = torch.arange(steps, device=actions.device).reshape(1, steps, 1)
    offset = (base + index).clamp(max=steps - 1)
    gather = offset.expand(batch, steps, chunk)
    targets = torch.gather(
        actions.unsqueeze(2).expand(batch, steps, chunk, dim), 1,
        gather.unsqueeze(-1).expand(batch, steps, chunk, dim))
    within = (base + index) < steps
    mask = within & torch.gather(valid.unsqueeze(2).expand(batch, steps, chunk),
                                 1, gather)
    return targets, mask


@dataclass
class ImitationConfig:
    chunk_size: int = 8
    flow_steps: int = 10
    lr: float = 1e-4
    steps: int = 20_000
    batch_size: int = 16
    log_every: int = 100
    grad_clip: float = 1.0


class ImitationTrainer:
    """Trains the adapter and the action expert; never the world model."""

    def __init__(self, world_model, actor, sampler, config: ImitationConfig,
                 *, device="cuda", normalizer=None):
        self.world_model = world_model.eval()
        for parameter in self.world_model.parameters():
            parameter.requires_grad_(False)
        self.actor = actor
        self.sampler = sampler
        self.config = config
        self.device = torch.device(device)
        self.normalizer = normalizer
        trainable = [p for p in self.actor.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                "nothing in the actor is trainable; the adapter and the action "
                "expert are supposed to be")
        self.optimizer = torch.optim.AdamW(trainable, lr=config.lr)
        self.step = 0

    def to_torch(self, batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        out = {}
        for key, value in batch.items():
            tensor = torch.as_tensor(np.asarray(value))
            if tensor.dtype == torch.float64:
                tensor = tensor.float()
            out[key] = tensor.to(self.device)
        return out

    def features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Causal posterior features. No gradient reaches the world model."""
        with torch.no_grad():
            out = self.world_model.observe(batch)
            return self.world_model.features(out["post"])

    def loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        feat = self.features(batch)
        targets, mask = chunk_targets(
            batch["action"], batch["loss_mask"].bool(), self.config.chunk_size)
        batch_size, steps = targets.shape[:2]
        flat_feat = feat.reshape(batch_size * steps, -1)
        cond = self.actor.condition(flat_feat, batch.get("instruction"))
        loss, metrics = flow_matching_loss(
            self.actor.velocity_fn(),
            targets.reshape(batch_size * steps, self.config.chunk_size, -1),
            cond,
            mask=mask.reshape(batch_size * steps, self.config.chunk_size),
        )
        return loss, metrics

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        tensors = self.to_torch(batch)
        loss, metrics = self.loss(tensors)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clipped = torch.nn.utils.clip_grad_norm_(
            [p for p in self.actor.parameters() if p.requires_grad],
            self.config.grad_clip)
        self.optimizer.step()
        self.step += 1
        return {"loss": float(loss.detach()), "grad_norm": float(clipped),
                **{k: float(v) for k, v in metrics.items()}}

    def fit(self, sampler, steps: Optional[int] = None) -> Dict[str, float]:
        total = int(steps or self.config.steps)
        last: Dict[str, float] = {}
        for _ in range(total):
            last = self.update(sampler.batch(self.config.batch_size))
            if self.step % self.config.log_every == 0:
                print(f"[imitation] step {self.step} loss {last['loss']:.4f}",
                      flush=True)
        return last
