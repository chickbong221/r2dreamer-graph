"""Evaluate a policy in the simulator, on seeds no demonstration used.

Fresh seeds rather than held-out episodes: the demonstrations are all used for
training, as asked, and generalisation is measured against initial states the
collection never visited. The collection blocks start at 0 and run to a few
thousand, so evaluation starts far above them and the number is recorded.

What is reported is the environment's own verdict -- success and return -- and
never a shaped reward. A progress-shaped arm that scored better only on its own
shaping would be visible here as a run whose shaped return rose while its
environment return did not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class EpisodeResult:
    seed: int
    success: bool
    steps: int
    steps_to_success: Optional[int]
    env_return: float
    clipped_actions: int


def evaluate_policy(env, policy: Callable, *, episodes: int = 20,
                    seed_start: int = 900_000, max_steps: int = 150,
                    action_low: float = -1.0, action_high: float = 1.0
                    ) -> Dict[str, Any]:
    """Roll the policy out and report what the task says about it."""
    results: List[EpisodeResult] = []
    reset_policy = getattr(policy, "reset", None)
    # A policy that applies the environment's bounds itself returns the command
    # it already fed back to its own recurrent state; re-clipping it here would
    # desynchronise the two.
    bounded = bool(getattr(policy, "bounded", False))
    for index in range(int(episodes)):
        seed = int(seed_start) + index
        # A recurrent policy carries a state the env reset does not clear.
        # Without this the second episode starts from the first one's final
        # latent, which still produces a rollout and still reports a number.
        if callable(reset_policy):
            reset_policy()
        obs = env.reset(seed)
        total, clipped, first_success = 0.0, 0, None
        step = 0
        for step in range(1, int(max_steps) + 1):
            action = np.asarray(policy(obs), dtype=np.float32)
            # Counted, not silently saturated: a policy living on the action
            # bounds is a policy whose normalization is wrong.
            clipped += int(np.any((action < action_low) | (action > action_high)))
            if not bounded:
                # Only clip for a policy that does not bound itself. Clipping
                # a LatentPolicy's output here would send the environment a
                # different command than the one the policy fed back to its own
                # posterior as a_(t-1), and the recurrence would drift.
                action = np.clip(action, action_low, action_high)
            out = env.step(action)
            total += float(out["reward"])
            obs = out["obs"]
            if out["success"] and first_success is None:
                first_success = step
            if out["is_last"]:
                break
        results.append(EpisodeResult(
            seed=seed, success=first_success is not None, steps=step,
            steps_to_success=first_success, env_return=total,
            clipped_actions=clipped))

    successes = [r for r in results if r.success]
    return {
        "episodes": len(results),
        "success_rate": len(successes) / max(len(results), 1),
        "env_return_mean": float(np.mean([r.env_return for r in results])),
        "steps_to_success_median": (
            float(np.median([r.steps_to_success for r in successes]))
            if successes else None),
        "clipped_fraction": float(np.mean(
            [r.clipped_actions / max(r.steps, 1) for r in results])),
        "seed_start": int(seed_start),
        "per_episode": [vars(r) for r in results],
    }
