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

from . import layout
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
    """Cover one episode with windows of a single fixed shape.

    ``burn_in`` is requested, not guaranteed: near the start of an episode
    there is less history than asked for. What used to vary with that was the
    window's *length* -- 64 transitions at the start and 72 afterwards, which
    do not stack. Now only the mask varies; every window has
    ``layout.rows(length, burn_in)`` rows.
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
        windows.append(Window(
            episode_id=ref.episode_id, start=start, stop=scored_stop,
            burn_in=available,
            pad=max(0, layout.rows(length, burn_in) - (scored_stop - start) - 1),
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
                length: Optional[int] = None, burn_in: Optional[int] = None,
                lookahead: int = 0) -> Dict[str, np.ndarray]:
    """One window's arrays, with its masks and episode flags.

    Transition-indexed arrays have ``length`` rows and observation-indexed ones
    have ``length + 1``: the observation each action led to is the next action's
    input, and the final one has no action of its own.

    ``lookahead`` reads up to that many *actions* past the window, still inside
    the same episode, so a chunked policy can be supervised on a full chunk at
    the window's last eligible rows. Only actions are read; no observation is
    extended, so nothing can condition on a future observation.
    """
    length = int(length if length is not None
                 else window.stop - window.start - window.burn_in)
    burn_in = int(burn_in if burn_in is not None else window.burn_in)

    raw = data.read(ref, window.start, window.stop)
    real = window.stop - window.start

    # What preceded the window. A window starting mid-episode has a real
    # previous action and a real incoming reward; one at a reset has neither,
    # and the masks say so rather than a zero being learned as an action.
    prev_action = prev_reward = None
    if window.start > 0:
        before = data.read(ref, window.start - 1, window.start)
        prev_action = np.asarray(before["actions"])[-1]
        prev_reward = float(np.asarray(before["rewards"])[-1])

    # Action-only lookahead, stopping at the episode boundary. A window that
    # already reaches the end of its episode has nothing to look ahead to, and
    # its target axis is padded and masked instead.
    lookahead = max(int(lookahead), 0)
    ahead = None
    ahead_stop = min(window.stop + lookahead, ref.steps)
    if ahead_stop > window.stop:
        ahead = np.asarray(
            data.read(ref, window.stop, ahead_stop)["actions"])

    observations = {key: value for key, value in raw.items()
                    if key not in data.fields.supervision}
    piece = layout.Slice(start=window.start, real=real, burn=window.burn_in,
                         episode_end=window.stop >= ref.steps)
    out = layout.assemble(
        observations=observations,
        actions=np.asarray(raw["actions"]),
        rewards=np.asarray(raw["rewards"]),
        prev_action=prev_action, prev_reward=prev_reward,
        piece=piece, length=length, burn_in=burn_in, lookahead=lookahead,
        lookahead_actions=ahead)

    # Terminations follow the online env; the recorded flags stay diagnostics.
    out["is_terminal"] = np.zeros_like(out["valid"])
    if not data.ignore_terminations:
        terminated = np.asarray(raw["terminated"], dtype=bool)
        incoming = np.concatenate([[False], terminated])[: real + 1]
        out["is_terminal"] = layout._pad_to(
            incoming, out["valid"].shape[0]) & out["valid"]
    layout.check(out, length, burn_in, lookahead)
    return out


class SequenceSampler:
    """Windows over a whole dataset, in order or at random.

    Deterministic given a seed, because two arms trained on different windows
    of the same demonstrations are not a controlled comparison.
    """

    def __init__(self, data: DemoDataset, *, length: int = 64, burn_in: int = 8,
                 stride: Optional[int] = None, seed: int = 0,
                 lookahead: int = 0):
        self.data = data
        self.length = int(length)
        self.burn_in = int(burn_in)
        # Set by Stage 1B to ``chunk_size - 1`` once the actor's chunk is
        # known. Zero means "supervise only what the window loaded", which is
        # still correctly masked -- lookahead is better supervision, not a
        # precondition for valid supervision.
        self.lookahead = int(lookahead)
        self.refs = {ref.episode_id: ref for ref in data.episodes}
        self.windows: List[Window] = [
            window for ref in data.episodes
            for window in plan_windows(ref, self.length, self.burn_in, stride)
        ]
        self.rows = layout.rows(self.length, self.burn_in)
        self._rng = np.random.default_rng(int(seed))

    def __len__(self) -> int:
        return len(self.windows)

    def load(self, window: Window) -> Dict[str, np.ndarray]:
        return load_window(self.data, self.refs[window.episode_id], window,
                           self.length, self.burn_in, self.lookahead)

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
