"""Fixed-length training windows that never cross an episode boundary.

A recurrent world model is trained on sequences, and every sequence carries a
claim about what the recurrent state was when it started. Three things have to
be right or the claim is false, and none of them fails loudly:

**Boundaries.** A window drawn across the join between two episodes describes a
transition that never happened. Windows are sampled inside one episode, always.

**Burn-in.** A window starting mid-episode inherits a recurrent state the batch
does not contain. It is given ``burn_in`` extra leading steps which rebuild that
state and are then excluded from the loss -- present in the arrays, absent from
``loss_mask``. A window at the episode start needs none, and gets ``is_first``
instead.

**Padding.** The last window of a short episode is padded to the fixed length.
The padding repeats the final row rather than inserting zeros, because a zero
observation is a state the model will try to explain; what keeps it out of the
loss is the mask either way.

The alignment this preserves is::

    o_t -> posterior s_t -> a_t -> r_t, o_{t+1}

so for every action in the window the observation before it and the observation
after it are both present, and the reward is the one that action earned. Future
actions may be imitation targets; no future observation enters the state at
``t``, which is a property of how the model consumes this, not of the window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np

from .dataset import DemoDataset, EpisodeRef


@dataclass(frozen=True)
class Window:
    """One training sequence's place in one episode."""

    episode_id: int
    start: int              # first transition, burn-in included
    stop: int               # one past the last real transition
    burn_in: int            # leading transitions excluded from the loss
    pad: int                # trailing rows repeated to reach the fixed length

    @property
    def length(self) -> int:
        return self.stop - self.start + self.pad

    @property
    def first_scored(self) -> int:
        return self.start + self.burn_in


def plan_windows(ref: EpisodeRef, length: int, burn_in: int,
                 stride: Optional[int] = None) -> List[Window]:
    """Cover one episode with windows of ``length`` scored transitions.

    ``burn_in`` is requested, not guaranteed: near the start of an episode
    there is less history than asked for, and the honest response is to use
    what exists and mark the window as starting at the episode's own beginning.
    """
    length, burn_in = int(length), max(int(burn_in), 0)
    stride = int(stride or length)
    if length <= 0 or stride <= 0:
        raise ValueError("window length and stride must be positive")

    windows: List[Window] = []
    scored_start = 0
    while scored_start < ref.steps:
        scored_stop = min(scored_start + length, ref.steps)
        available = min(burn_in, scored_start)
        start = scored_start - available
        real = scored_stop - start
        windows.append(Window(
            episode_id=ref.episode_id, start=start, stop=scored_stop,
            burn_in=available, pad=max(0, (length + available) - real),
        ))
        scored_start += stride
    return windows


def _pad(array: np.ndarray, rows: int) -> np.ndarray:
    """Extend by repeating the final row; the mask is what excludes it."""
    if rows <= 0:
        return array
    tail = np.repeat(array[-1:], rows, axis=0)
    return np.concatenate([array, tail], axis=0)


def load_window(data: DemoDataset, ref: EpisodeRef, window: Window,
                ) -> Dict[str, np.ndarray]:
    """One window's arrays, with its masks and episode flags.

    Transition-indexed arrays have ``length`` rows and observation-indexed ones
    have ``length + 1``: the observation each action led to is the next action's
    input, and the final one has no action of its own.
    """
    raw = data.read(ref, window.start, window.stop)
    steps = window.stop - window.start
    total = steps + window.pad

    out: Dict[str, np.ndarray] = {}
    for key, array in raw.items():
        kind_is_obs = key not in data.fields.supervision
        out[key] = _pad(array, window.pad)
        if kind_is_obs and out[key].shape[0] != total + 1:
            raise ValueError(
                f"{key} has {out[key].shape[0]} rows, expected {total + 1}")

    valid = np.zeros(total, dtype=bool)
    valid[: steps] = True
    scored = valid.copy()
    scored[: window.burn_in] = False

    is_first = np.zeros(total, dtype=bool)
    is_first[0] = window.start == 0

    terminated = np.asarray(out["terminated"], dtype=bool)
    truncated = np.asarray(out["truncated"], dtype=bool)
    # The episode's own end, not the window's: a window that stops early
    # because it ran out of length has not reached a last step.
    reached_end = window.stop >= ref.steps
    is_last = np.zeros(total, dtype=bool)
    if reached_end and steps > 0:
        is_last[steps - 1] = True
    # Under the online policy nothing terminates: the env is built with
    # ignore_terminations=True, so the recorded terminal flags describe a
    # signal that never reaches the trainer, and every episode end -- horizon
    # or collector cut -- is a bootstrap. Honouring the recording instead would
    # make the demonstrations the only place a terminal state exists.
    if data.ignore_terminations:
        is_terminal = np.zeros(total, dtype=bool)
    else:
        is_terminal = terminated & valid
        # A cut that the collector made is still not a termination.
        if reached_end and not ref.terminal and steps > 0:
            is_terminal[steps - 1] = False

    out |= {
        "valid": valid,
        "loss_mask": scored,
        "is_first": is_first,
        "is_last": is_last,
        "is_terminal": is_terminal,
        "is_truncated": truncated & valid,
    }
    return out


class SequenceSampler:
    """Windows over a whole dataset, in order or at random.

    Deterministic given a seed, because two arms trained on different windows
    of the same demonstrations are not a controlled comparison.
    """

    def __init__(self, data: DemoDataset, *, length: int = 64, burn_in: int = 8,
                 stride: Optional[int] = None, seed: int = 0):
        self.data = data
        self.length = int(length)
        self.burn_in = int(burn_in)
        self.refs = {ref.episode_id: ref for ref in data.episodes}
        self.windows: List[Window] = [
            window for ref in data.episodes
            for window in plan_windows(ref, self.length, self.burn_in, stride)
        ]
        self._rng = np.random.default_rng(int(seed))

    def __len__(self) -> int:
        return len(self.windows)

    def load(self, window: Window) -> Dict[str, np.ndarray]:
        return load_window(self.data, self.refs[window.episode_id], window)

    def iter_epoch(self, shuffle: bool = True) -> Iterator[Dict[str, np.ndarray]]:
        order = np.arange(len(self.windows))
        if shuffle:
            self._rng.shuffle(order)
        for index in order:
            yield self.load(self.windows[int(index)])

    def batch(self, size: int) -> Dict[str, np.ndarray]:
        """``size`` windows stacked on a leading batch axis."""
        picks = self._rng.integers(0, len(self.windows), size=int(size))
        loaded = [self.load(self.windows[int(i)]) for i in picks]
        return {key: np.stack([item[key] for item in loaded])
                for key in loaded[0]}
