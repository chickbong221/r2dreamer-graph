"""Stage 2: collect, update the world model, imagine, train critic and actor.

The loop is identical for both arms. The only primary difference is whether the
world model's inference and dynamics carry the graph branch, which is decided
when the arm's world model is constructed and not here.

Two things this is careful about.

Posterior states are recomputed every update from the current world model. A
Stage 1 latent cache is a function of the checkpoint that produced it, and the
world model is being trained here, so reusing that cache would condition the
policy on states the model no longer produces.

Demonstrations and online experience are mixed by sequence count, and the
mixture waits for the replay to hold enough episodes to be worth sampling --
four online rollouts treated as half the distribution is worse than training on
demonstrations alone for another few minutes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..data.replay import OnlineEpisode, OnlineReplay, mixed_batch
from ..runtime.checkpoint import CheckpointMeta, save
from .actor_critic import ActorCriticConfig, ActorCriticTrainer
from .imagination import flatten_start


@dataclass
class OnlineConfig:
    total_steps: int = 1_000_000
    episodes_per_collect: int = 2
    updates_per_collect: int = 8
    demo_fraction: float = 0.5
    batch_size: int = 16
    sequence_length: int = 64
    burn_in: int = 8
    imagination_batch: int = 256
    checkpoint_every: int = 10_000
    actor_every: int = 1


def collect_episode(env, policy, *, max_steps: int = 150,
                    seed: Optional[int] = None) -> OnlineEpisode:
    """One rollout, keeping the final observation before the reset.

    ``policy`` takes an observation dict and returns an action. The final
    observation is appended after the loop: a buffer whose observation count
    equals its action count has lost the state the last action led to, and the
    replay refuses such an episode rather than storing it.
    """
    episode = OnlineEpisode()
    obs = env.reset(seed)
    episode.add_observation(obs)
    for _ in range(int(max_steps)):
        action = policy(obs)
        out = env.step(action)
        episode.add_transition(action, out["reward"], out["is_terminal"],
                               out["is_last"], out["success"])
        episode.add_observation(out["obs"])
        obs = out["obs"]
        if out["is_last"]:
            break
    return episode


class OnlineTrainer:
    """World-model updates, imagined actor-critic updates, and checkpoints."""

    def __init__(self, world_model, actor, critic, demo_sampler, *,
                 config: OnlineConfig, ac_config: ActorCriticConfig,
                 world_lr: float = 1e-4, device="cuda",
                 progress_head=None, checkpoint_dir: Optional[Path] = None,
                 meta: Optional[CheckpointMeta] = None):
        self.world_model = world_model
        self.actor = actor
        self.critic = critic
        self.demo_sampler = demo_sampler
        self.config = config
        self.device = torch.device(device)
        self.progress_head = progress_head
        self.replay = OnlineReplay()
        self.world_opt = torch.optim.AdamW(world_model.parameters(), lr=world_lr)
        self.ac = ActorCriticTrainer(world_model, actor, critic, ac_config)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.meta = meta
        self.env_steps = 0
        self.updates = 0

    def to_torch(self, batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        out = {}
        for key, value in batch.items():
            tensor = torch.as_tensor(np.asarray(value))
            if tensor.dtype == torch.float64:
                tensor = tensor.float()
            out[key] = tensor.to(self.device)
        if "actions" in out:
            out["action"] = out.pop("actions").float()
        if "rewards" in out:
            out["reward"] = out.pop("rewards").float().unsqueeze(-1)
        return out

    def update(self) -> Dict[str, float]:
        """One world-model step, then one actor-critic step on fresh states."""
        batch = self.to_torch(mixed_batch(
            self.demo_sampler, self.replay, self.config.batch_size,
            self.config.sequence_length, self.config.burn_in,
            self.config.demo_fraction))

        total, _losses, aux = self.world_model.loss(batch)
        self.world_opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
        self.world_opt.step()
        metrics = {"world_loss": float(total.detach())}

        # Recomputed, never cached: the model that produced these states was
        # updated one line ago.
        with torch.no_grad():
            start = flatten_start(aux["post"], self.world_model.graph_enabled)
        self.updates += 1
        if self.updates % max(int(self.config.actor_every), 1) == 0:
            metrics |= self.ac.update(start)
        return metrics

    def checkpoint(self, tag: str = "latest") -> Optional[Path]:
        if self.checkpoint_dir is None or self.meta is None:
            return None
        meta = CheckpointMeta(**{**self.meta.__dict__, "stage": "online",
                                 "step": self.env_steps})
        return save(self.checkpoint_dir / f"online_{tag}.pt", meta,
                    {"world_model": self.world_model, "actor": self.actor,
                     "critic": self.critic, "progress": self.progress_head},
                    {"world": self.world_opt, "actor": self.ac.actor_opt,
                     "critic": self.ac.critic_opt})
