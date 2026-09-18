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

from . import layout
from .batch import STEP_KEYS, WINDOW_REQUIRED


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
        # Storage naming, matching the demonstration dataset. A mixed batch
        # concatenates the keys both sources share, so a replay that emitted
        # "action" while the sampler emitted "actions" would drop both.
        out |= {
            "actions": np.stack(self.action),
            "rewards": np.asarray(self.reward, dtype=np.float32),
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
            if key in STEP_KEYS:
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
        return sum(int(ep["actions"].shape[0]) for ep in self.episodes)

    def sample(self, batch: int, length: int, burn_in: int = 0,
               lookahead: int = 0, *, ignore_terminations: bool = True
               ) -> Dict[str, np.ndarray]:
        """``batch`` windows of ``length`` transitions, inside one episode.

        The span is chosen the way ``sequences.plan_windows`` chooses it: a
        scored start anywhere in the episode, up to ``length`` scored
        transitions after it, and up to ``burn_in`` real transitions before it.
        Sampling ``length + burn_in`` transitions and then calling
        ``min(burn_in, start)`` of them burn-in scored ``length + burn_in``
        transitions on every window that began at step 0 -- a longer scored
        span than any demonstration window has, mixed into the same batch.
        """
        if not self.episodes:
            raise RuntimeError("online replay is empty")
        picks = []
        for _ in range(int(batch)):
            episode = self.episodes[int(self._rng.integers(len(self.episodes)))]
            steps = int(episode["actions"].shape[0])
            scored_start = self._scored_start(steps, int(burn_in))
            scored_stop = min(scored_start + int(length), steps)
            available = min(int(burn_in), scored_start)
            picks.append(self._window(
                episode, scored_start - available, scored_stop, int(length),
                int(burn_in), burn=available, lookahead=int(lookahead),
                ignore_terminations=bool(ignore_terminations)))
        return {key: np.stack([p[key] for p in picks]) for key in picks[0]}

    def _scored_start(self, steps: int, burn_in: int) -> int:
        """Where the scored span begins: at the reset, or clear of the burn-in.

        ``plan_windows`` strides by ``length``, which is larger than
        ``burn_in``, so a demonstration window either starts at the reset with
        nothing to burn in, or starts far enough in to have the full burn-in
        behind it. Drawing a scored start uniformly would also produce the
        third case -- a window whose first row *is* the reset and which then
        burns that row in -- and a mixed batch would contain two kinds of
        window that the model cannot tell apart but that score different spans.
        """
        if steps <= 0:
            return 0
        if burn_in <= 0:
            return int(self._rng.integers(0, steps))
        # {0} union [burn_in + 1, steps), so a nonzero start always leaves at
        # least one real transition ahead of the scored span.
        span = max(steps - burn_in - 1, 0)
        pick = int(self._rng.integers(0, span + 1))
        return 0 if pick == 0 else burn_in + pick

    @staticmethod
    def _window(episode, start: int, stop: int, length: int, burn_in: int,
                *, burn: int, lookahead: int = 0,
                ignore_terminations: bool = True) -> Dict[str, np.ndarray]:
        """The same layout the demonstration sampler builds.

        Assembled through ``layout.assemble`` rather than by a second
        hand-written slicing: the two used to disagree about both the row count
        and which action the posterior consumes, and a mixed batch is exactly
        where that shows up.
        """
        real = stop - start
        steps = int(episode["actions"].shape[0])
        observations = {key: value[start:stop + 1]
                        for key, value in episode.items()
                        if key not in STEP_KEYS}
        prev_action = prev_reward = None
        if start > 0:
            prev_action = episode["actions"][start - 1]
            prev_reward = float(episode["rewards"][start - 1])
        lookahead = max(int(lookahead), 0)
        ahead = None
        ahead_stop = min(stop + lookahead, steps)
        if ahead_stop > stop:
            ahead = episode["actions"][stop:ahead_stop]
        piece = layout.Slice(start=start, real=real, burn=int(burn),
                             episode_end=stop >= steps)
        out = layout.assemble(
            observations=observations,
            actions=episode["actions"][start:stop],
            rewards=episode["rewards"][start:stop],
            prev_action=prev_action, prev_reward=prev_reward,
            piece=piece, length=length, burn_in=burn_in, lookahead=lookahead,
            lookahead_actions=ahead)
        # Terminations follow the same switch the loader and the env use. A
        # replay that honoured them while the dataset ignored them would train
        # one continuation head on two different conventions.
        out["is_terminal"] = np.zeros_like(out["valid"])
        if not ignore_terminations:
            terminated = np.asarray(episode["is_terminal"], dtype=bool)[start:stop]
            incoming = np.concatenate([[False], terminated])[: real + 1]
            out["is_terminal"] = layout._pad_to(
                incoming, out["valid"].shape[0]) & out["valid"]
        layout.check(out, length, burn_in, lookahead)
        return out


def mixed_batch(demo_sampler, replay: OnlineReplay, batch: int, length: int,
                burn_in: int, demo_fraction: float = 0.5, *,
                lookahead: int = 0, ignore_terminations: bool = True,
                min_replay: int = 4) -> Dict[str, np.ndarray]:
    """A batch drawn from both sources, by sequence count.

    Falls back to demonstrations alone while the replay is too small to sample
    a meaningful mixture from -- training on four online episodes as if they
    were half the distribution is worse than waiting.

    The two sources are required to agree on *every* key, not merely to
    overlap. Intersecting them silently is how a mixed batch lost its actions
    and its rewards and went on training on observations alone: the run keeps
    going, the loss keeps descending, and the model learns no dynamics. A key
    either source has and the other lacks is a bug in whichever source drifted,
    and it is reported as one.
    """
    demo_n = int(round(batch * float(demo_fraction)))
    online_n = batch - demo_n
    if len(replay) < int(min_replay) or online_n <= 0:
        return demo_sampler.batch(batch)
    demo = demo_sampler.batch(demo_n) if demo_n else None
    online = replay.sample(online_n, length, burn_in, lookahead,
                           ignore_terminations=ignore_terminations)
    if demo is None:
        return online

    missing_online = sorted(set(demo) - set(online))
    missing_demo = sorted(set(online) - set(demo))
    if missing_online or missing_demo:
        raise KeyError(
            "the demonstration sampler and the online replay disagree about "
            f"the batch contract: online is missing {missing_online}, "
            f"demonstrations are missing {missing_demo}. Both assemble "
            "through sim_vla.data.layout.assemble; a difference here means "
            "one of them stopped.")
    # Checked anyway, so that a contract both sources broke the same way is
    # still caught rather than agreed upon.
    absent = [name for name in WINDOW_REQUIRED if name not in demo]
    if absent:
        raise KeyError(
            f"{absent} are missing from both sources; a batch without them "
            "trains on observations alone")
    # Same key set is not the same shape. Stage 1B sets the demonstration
    # sampler's lookahead from the actor's chunk, so a caller that does not
    # pass the same lookahead here gets target axes of two different lengths.
    misaligned = {
        key: (demo[key].shape[1:], online[key].shape[1:])
        for key in demo
        if np.asarray(demo[key]).shape[1:] != np.asarray(online[key]).shape[1:]}
    if misaligned:
        raise ValueError(
            f"the two sources disagree about shape: {misaligned} "
            f"(demonstrations, online). The demonstration sampler's lookahead "
            f"is {getattr(demo_sampler, 'lookahead', None)} and this call "
            f"passed {lookahead}; they have to be the same number.")
    return {k: np.concatenate([demo[k], online[k]], axis=0) for k in demo}
