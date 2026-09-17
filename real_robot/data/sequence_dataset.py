"""Burn-in plus learning windows over packed episodes, in the RSSM's convention.

The online buffer stores ``(o_t, r_t, a_t)`` per step and shifts actions back
by one at sampling time, so that step ``t`` carries the action that *led to*
``o_t``. That shift is correct only because the simulator auto-resets and
``is_first`` separates episodes inside one stream. Recorded episodes have no
stream, so the convention is rebuilt here explicitly, per field:

=================  =========================================================
``prev_action``    ``a_{t-1}``; zero on the first frame of an episode
``reward_in``      reward for the transition ``t-1 -> t`` (the reward head at
                   step ``t`` predicts it); zero and masked where no such
                   transition exists
``is_first``       the episode's first recorded frame
``is_terminal``    arriving at ``o_t`` completed the task
``obs_valid``      a recorded observation exists at this position
``graph_valid``    its graph is complete
``learn``          position is in the learning segment (after burn-in)
=================  =========================================================

A final recorded frame is an ordinary valid observation; the online path's
``is_last`` graph masking has no counterpart here. Positions past the end of
an episode, and past its terminal frame, are padding: not observations, never
supervised, and never carried into another episode's state.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from scenegraph.adapters.graph_pack import GRAPH_KEYS

from .episode_dataset import BuiltEpisodeStore

CONTROL_KEYS = ("prev_action", "reward_in", "reward_in_valid", "is_first", "is_terminal",
                "obs_valid", "graph_valid", "cont_valid", "learn")


def last_observation(arrays: Dict[str, np.ndarray]) -> int:
    """Index of the last frame that belongs to the episode: terminal, or final."""
    terminal = np.flatnonzero(arrays["task_terminal"])
    valid = np.flatnonzero(arrays["obs_valid"])
    if valid.size == 0:
        return -1
    last = int(valid[-1])
    return min(last, int(terminal[0])) if terminal.size else last


def episode_window(arrays: Dict[str, np.ndarray], image_keys: Sequence[str], start: int,
                   burn_in: int, length: int) -> Dict[str, np.ndarray]:
    """One window whose learning segment starts at recorded frame ``start``."""
    total = int(burn_in) + int(length)
    last = last_observation(arrays)
    positions = np.arange(int(start) - int(burn_in), int(start) + int(length))
    real = (positions >= 0) & (positions <= last)
    index = np.clip(positions, 0, max(last, 0))

    out: Dict[str, np.ndarray] = {}
    for key in list(image_keys) + ["state"] + list(GRAPH_KEYS):
        source = arrays[key]
        gathered = np.asarray(source[index])
        mask = real.reshape((total,) + (1,) * (gathered.ndim - 1))
        out[key] = np.where(mask, gathered, np.zeros_like(gathered))

    action = arrays["action"]
    prev_index = np.clip(positions - 1, 0, max(last, 0))
    has_prev = real & (positions >= 1)
    transition = np.asarray(arrays["transition_valid"], dtype=bool)
    has_prev &= transition[prev_index]
    out["prev_action"] = np.where(has_prev[:, None], action[prev_index], 0.0).astype(np.float32)
    reward = np.nan_to_num(np.asarray(arrays["reward"], dtype=np.float32), nan=0.0)
    out["reward_in"] = np.where(has_prev, reward[prev_index], 0.0).astype(np.float32)
    out["reward_in_valid"] = has_prev
    out["is_first"] = real & (positions == 0)
    out["is_terminal"] = real & np.asarray(arrays["task_terminal"], dtype=bool)[index]
    out["obs_valid"] = real & np.asarray(arrays["obs_valid"], dtype=bool)[index]
    out["graph_valid"] = out["obs_valid"] & np.asarray(arrays["graph_valid"], dtype=bool)[index]
    # Continuation is supervised wherever the episode's termination is known.
    # Every packed episode carries a validated outcome, so that is every real
    # observation; the flag stays separate so a future unannotated source can
    # clear it without touching the rest.
    out["cont_valid"] = out["obs_valid"].copy()
    out["learn"] = np.arange(total) >= int(burn_in)
    out["frame"] = np.where(real, positions, -1).astype(np.int32)
    return out


class SequenceSampler:
    """Windows uniform over frames, each inside one episode, deterministic given the seed.

    ``episodes`` is a selection name (``training``, ``diagnostic``) or a list of
    episode ids. A window never crosses an episode boundary: its positions are
    consecutive frames of one episode, padded before the first frame and after
    the last, and burn-in rebuilds the recurrent state from that same episode.
    """

    def __init__(self, store: BuiltEpisodeStore, episodes: Union[str, Sequence[int]], burn_in: int, length: int,
                 batch_size: int, seed: int = 0, begin_fraction: float = 0.0):
        self.store = store
        self.image_keys = store.manifest.image_keys
        self.burn_in, self.length, self.batch_size = int(burn_in), int(length), int(batch_size)
        self.begin_fraction = float(begin_fraction)
        self.rng = np.random.default_rng(int(seed))
        name = episodes if isinstance(episodes, str) else "given"
        chosen = store.episodes(episodes) if isinstance(episodes, str) else [int(e) for e in episodes]
        self.episodes: List[int] = []
        self.last: List[int] = []
        for episode in chosen:
            last = last_observation(store.load(episode, images=False))
            if last >= 0:
                self.episodes.append(int(episode))
                self.last.append(last)
        if not self.episodes:
            raise ValueError(f"episode selection {name!r} has no packed episodes with observations")
        lengths = np.asarray(self.last, dtype=np.float64) + 1.0
        self.weights = lengths / lengths.sum()

    def draw(self) -> Tuple[int, int]:
        """One ``(episode, start)``: where a window's learning segment begins."""
        slot = int(self.rng.choice(len(self.episodes), p=self.weights))
        episode, last = self.episodes[slot], self.last[slot]
        if self.rng.random() < self.begin_fraction:
            start = 0
        else:
            start = int(self.rng.integers(0, max(last - self.length + 1, 0) + 1))
        return episode, start

    def window(self, episode: int, start: int) -> Dict[str, np.ndarray]:
        window = episode_window(self.store.load(episode), self.image_keys, start, self.burn_in, self.length)
        window["episode"] = np.full(self.burn_in + self.length, int(episode), dtype=np.int32)
        return window

    def batch(self, draws: Sequence[Tuple[int, int]]) -> Dict[str, np.ndarray]:
        windows = [self.window(episode, start) for episode, start in draws]
        return {key: np.stack([w[key] for w in windows]) for key in windows[0]}

    def sample(self) -> Dict[str, np.ndarray]:
        return self.batch([self.draw() for _ in range(self.batch_size)])


def full_episode(store: BuiltEpisodeStore, episode: int) -> Dict[str, np.ndarray]:
    """The whole episode as one ``(1, T, ...)`` batch with no burn-in."""
    arrays = store.load(episode)
    last = last_observation(arrays)
    window = episode_window(arrays, store.manifest.image_keys, 0, 0, last + 1)
    window["episode"] = np.full(last + 1, int(episode), dtype=np.int32)
    return {key: value[None] for key, value in window.items()}
