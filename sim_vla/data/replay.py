"""Online replay, holding whole episodes so windows never cross a boundary.

Sequences are drawn the same way the demonstration sampler draws them --
inside one episode, with burn-in and a mask -- because the world model does not
care which buffer a sequence came from and must not be able to tell by its
alignment.

The mixture with demonstrations is a ratio of *sequences*, not of episodes or
of steps. Half a batch from each is what "50/50" means for an update, and the
buffer reports its own occupancy so a run can wait for it to fill rather than
training on four online episodes as if they were half the world.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional

import numpy as np


@dataclass
class OnlineEpisode:
    """One rollout, in the dataset's own field layout."""

    obs: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    action: List[np.ndarray] = field(default_factory=list)
    reward: List[float] = field(default_factory=list)
    is_terminal: List[bool] = field(default_factory=list)
    is_last: List[bool] = field(default_factory=list)
    success: List[bool] = field(default_factory=list)

    @property
    def steps(self) -> int:
        return len(self.action)

    def add_observation(self, obs: Dict[str, np.ndarray]) -> None:
        for key, value in obs.items():
            if key == "is_first":
                continue
            self.obs.setdefault(key, []).append(np.asarray(value))

    def add_transition(self, action, reward, is_terminal, is_last, success
                       ) -> None:
        self.action.append(np.asarray(action, dtype=np.float32))
        self.reward.append(float(reward))
        self.is_terminal.append(bool(is_terminal))
        self.is_last.append(bool(is_last))
        self.success.append(bool(success))

    def arrays(self) -> Dict[str, np.ndarray]:
        out = {key: np.stack(values) for key, values in self.obs.items()}
        out |= {
            "action": np.stack(self.action),
            "reward": np.asarray(self.reward, dtype=np.float32),
            "is_terminal": np.asarray(self.is_terminal, dtype=bool),
            "is_last": np.asarray(self.is_last, dtype=bool),
            "success": np.asarray(self.success, dtype=bool),
        }
        return out


class OnlineReplay:
    """Fixed-capacity episode buffer with window sampling."""

    def __init__(self, capacity: int = 600, seed: int = 0):
        self.capacity = int(capacity)
        self.episodes: Deque[Dict[str, np.ndarray]] = deque(maxlen=self.capacity)
        self._rng = np.random.default_rng(int(seed))

    def add(self, episode: OnlineEpisode) -> None:
        if episode.steps <= 0:
            return
        arrays = episode.arrays()
        for key, value in arrays.items():
            if key in ("action", "reward", "is_terminal", "is_last", "success"):
                continue
            # The observation count has to be one more than the action count,
            # or the final observation was dropped by the reset that followed.
            if value.shape[0] != episode.steps + 1:
                raise ValueError(
                    f"{key} has {value.shape[0]} rows for {episode.steps} "
                    "actions; the final observation was not captured before "
                    "the reset")
        self.episodes.append(arrays)

    def __len__(self) -> int:
        return len(self.episodes)

    @property
    def steps(self) -> int:
        return sum(int(ep["action"].shape[0]) for ep in self.episodes)

    def sample(self, batch: int, length: int, burn_in: int = 0
               ) -> Dict[str, np.ndarray]:
        """``batch`` windows of ``length`` transitions, inside one episode."""
        if not self.episodes:
            raise RuntimeError("online replay is empty")
        picks = []
        for _ in range(int(batch)):
            episode = self.episodes[int(self._rng.integers(len(self.episodes)))]
            steps = int(episode["action"].shape[0])
            span = min(int(length), steps)
            start = int(self._rng.integers(0, max(steps - span, 0) + 1))
            picks.append(self._window(episode, start, start + span, int(length),
                                      int(burn_in)))
        return {key: np.stack([p[key] for p in picks]) for key in picks[0]}

    @staticmethod
    def _window(episode, start: int, stop: int, length: int, burn_in: int
                ) -> Dict[str, np.ndarray]:
        steps = stop - start
        pad = max(0, length - steps)
        out: Dict[str, np.ndarray] = {}
        for key, value in episode.items():
            is_obs = key not in ("action", "reward", "is_terminal", "is_last",
                                 "success")
            block = value[start:stop + 1] if is_obs else value[start:stop]
            if pad:
                block = np.concatenate(
                    [block, np.repeat(block[-1:], pad, axis=0)], axis=0)
            out[key] = block
        valid = np.zeros(length, dtype=bool)
        valid[:steps] = True
        scored = valid.copy()
        scored[:min(burn_in, steps)] = False
        is_first = np.zeros(length, dtype=bool)
        is_first[0] = start == 0
        out |= {"valid": valid, "loss_mask": scored, "is_first": is_first}
        return out


def mixed_batch(demo_sampler, replay: OnlineReplay, batch: int, length: int,
                burn_in: int, demo_fraction: float = 0.5
                ) -> Dict[str, np.ndarray]:
    """A batch drawn from both sources, by sequence count.

    Falls back to demonstrations alone while the replay is too small to sample
    a meaningful mixture from -- training on four online episodes as if they
    were half the distribution is worse than waiting.
    """
    demo_n = int(round(batch * float(demo_fraction)))
    online_n = batch - demo_n
    if len(replay) < 4 or online_n <= 0:
        return demo_sampler.batch(batch)
    demo = demo_sampler.batch(demo_n) if demo_n else None
    online = replay.sample(online_n, length, burn_in)
    if demo is None:
        return online
    shared = [k for k in demo if k in online]
    return {k: np.concatenate([demo[k], online[k]], axis=0) for k in shared}
