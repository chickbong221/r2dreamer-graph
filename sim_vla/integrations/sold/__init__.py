"""SOLD with a latent-conditioned SmolVLA actor.

SAVi, the slot dynamics, the reward and value heads, the imagined rollout, the
lambda-return objective, the critic regularization and the target updates are
upstream's and are unchanged. What is added is an actor-side adapter over the
existing causal slot history, a chunk-imitation stage, and a hook in
``sim_vla/sold/sold/train_sold.py`` where the Gaussian actor is sampled.
"""

from __future__ import annotations

__all__ = ["config", "data", "adapter", "policy", "stages", "run"]
