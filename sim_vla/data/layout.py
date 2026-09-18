"""One window layout, and the causal contract it encodes.

Both sources of training sequences -- recorded demonstrations and the online
replay -- assemble windows here, so a mixed batch stacks. Three things were
wrong before and all of them were silent in isolation.

**Shape.** Windows were ``length`` transitions at an episode's start and
``length + burn_in`` afterwards, because the burn-in was prepended only when it
existed. 64 and 72 do not stack, and a demonstration/online mixture failed on
observation lengths 73 against 65. Every window is now exactly
``ROWS = burn_in + length + 1`` rows; what varies is the mask, not the shape.

**Alignment.** ``RSSM.obs_step(stoch, deter, prev_action, embed, ...)`` takes
the *previous* action. Feeding it the action taken *at* the current
observation puts ``a_t`` into the posterior that the actor is then trained to
predict ``a_t`` from -- the target in its own input. So the arrays are named
for what they are:

=================  ===========================================================
``action``         ``a_(t-1)``: executed before arriving at ``o_t``. What the
                   posterior consumes.
``action_target``  ``a_t``: executed *at* ``o_t``. What the actor predicts.
``reward``         ``r_(t-1)``: earned by the transition that arrived at
                   ``o_t``. Undefined at a reset, and masked there.
=================  ===========================================================

**Availability is not eligibility.** The final observation of a window has no
action loaded for it, whether or not the episode ended there. Marking that row
valid meant the actor was scored against an unloaded zero on every interior
window. The two questions are now asked by two masks:

``valid``          this row is a real observation, not padding
``loss_mask``      ...and it is scored: a conditioning row the actor is
                   trained *at* (burn-in excluded)
``action_valid``   a real ``action_target`` was loaded at this row. This is
                   *availability* -- it is true on lookahead rows, which are
                   not conditioning rows and are excluded by ``loss_mask``
``reward_valid``   an incoming reward exists here (false at a reset)

**Lookahead.** A chunked policy at row ``t`` is supervised on
``[a_t .. a_(t+H-1)]``. For a window that stops in the middle of an episode
those later actions exist but were not loaded, so the suffix used to be masked
away and the last rows of every window trained on one action instead of a
chunk. ``lookahead`` carries up to ``H - 1`` extra *actions* past the window,
filling rows that are otherwise padding. No observation is extended: nothing
downstream can condition on a future observation, because there is not one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .batch import STEP_KEYS

# Arrays this module produces that are not observations.
MASK_KEYS = ("valid", "loss_mask", "action_valid", "reward_valid", "is_first",
             "is_last", "is_terminal")


def rows(length: int, burn_in: int) -> int:
    """Observation rows in every window, whatever the episode looked like."""
    return int(burn_in) + int(length) + 1


# The two arrays indexed by the *target* axis rather than the observation axis.
TARGET_KEYS = ("action_target", "action_valid")


def target_rows(length: int, burn_in: int, lookahead: int = 0) -> int:
    """Rows on the target axis: the observation rows plus the lookahead.

    A full interior window has no padding at all -- every observation row is
    real -- so squeezing the lookahead into the observation array left exactly
    one free slot no matter how long the chunk was. With ``H = 4`` the last
    rows of such a window were supervised on 4, 4, 3, 2 and 1 actions: masked
    correctly, but not the whole-chunk supervision the lookahead was added to
    provide. The target axis is therefore its own axis, ``lookahead`` rows
    longer, and eligibility is still indexed by the observation axis.
    """
    return rows(length, burn_in) + max(int(lookahead), 0)


@dataclass(frozen=True)
class Slice:
    """Where a window sits in an episode, before any padding."""

    start: int          # first transition index covered
    real: int           # transitions actually present
    burn: int           # leading real transitions excluded from the loss
    episode_end: bool   # the window reaches the end of the episode

    @property
    def obs_rows(self) -> int:
        return self.real + 1


def _pad_to(array: np.ndarray, target: int) -> np.ndarray:
    """Repeat the final row up to ``target``. The mask is what excludes it."""
    missing = target - array.shape[0]
    if missing <= 0:
        return array[:target]
    return np.concatenate(
        [array, np.repeat(array[-1:], missing, axis=0)], axis=0)


def assemble(*, observations: Mapping[str, np.ndarray],
             actions: np.ndarray, rewards: np.ndarray,
             prev_action: Optional[np.ndarray],
             prev_reward: Optional[float],
             piece: Slice, length: int, burn_in: int,
             lookahead: int = 0,
             lookahead_actions: Optional[np.ndarray] = None,
             extras: Optional[Mapping[str, np.ndarray]] = None,
             ) -> Dict[str, np.ndarray]:
    """Build one window in the canonical layout.

    ``observations`` hold ``piece.real + 1`` rows each; ``actions`` and
    ``rewards`` hold ``piece.real``, indexed so that ``actions[i]`` was taken
    at ``observations[i]`` and ``rewards[i]`` was earned arriving at
    ``observations[i + 1]``.

    ``prev_action`` / ``prev_reward`` are what preceded the window. They exist
    for a window starting mid-episode and are None at a reset, where the
    posterior has no previous action to consume and no incoming reward.

    ``lookahead`` is the target axis's extra capacity -- a fixed number, so
    that every window in a batch has the same shape -- and
    ``lookahead_actions`` is what was actually available there: actions taken
    *after* the window's last transition and still inside the same episode.
    They extend ``action_target`` only; no observation is extended, so nothing
    downstream can condition on a future observation.
    """
    total = rows(length, burn_in)
    capacity = target_rows(length, burn_in, lookahead)
    real_obs = piece.obs_rows
    action_dim = int(actions.shape[-1])

    out: Dict[str, np.ndarray] = {}
    for key, value in observations.items():
        out[key] = _pad_to(np.asarray(value), total)

    # a_(t-1) per observation: what preceded the window, then the window's own
    # actions shifted by one. The last recorded action is not a *previous*
    # action for any observation inside the window.
    lead = (np.zeros((1, action_dim), dtype=np.float32) if prev_action is None
            else np.asarray(prev_action, dtype=np.float32).reshape(1, action_dim))
    previous = np.concatenate([lead, np.asarray(actions, dtype=np.float32)],
                              axis=0)[:real_obs]
    out["action"] = _pad_to(previous, total)

    # a_t per target row. Rows 0..real-1 come from the window's own actions.
    # Row ``real`` onward is the lookahead, if any: real actions from the same
    # episode that no observation in this window was loaded for.
    ahead = (np.zeros((0, action_dim), dtype=np.float32)
             if lookahead_actions is None
             else np.asarray(lookahead_actions,
                             dtype=np.float32).reshape(-1, action_dim))
    # Capped by the target axis, not by the observation axis: that cap was the
    # bug, and on a full interior window it left exactly one slot.
    ahead = ahead[: max(capacity - piece.real, 0)]
    targets = np.concatenate(
        [np.asarray(actions, dtype=np.float32), ahead], axis=0)
    loaded_targets = int(targets.shape[0])
    if loaded_targets == 0:
        targets = np.zeros((1, action_dim), dtype=np.float32)
    out["action_target"] = _pad_to(targets, capacity)

    # r_(t-1) per observation: undefined at a reset.
    lead_reward = np.asarray(
        [0.0 if prev_reward is None else float(prev_reward)], dtype=np.float32)
    incoming = np.concatenate(
        [lead_reward, np.asarray(rewards, dtype=np.float32)], axis=0)[:real_obs]
    out["reward"] = _pad_to(incoming, total)

    valid = np.zeros(total, dtype=bool)
    valid[:real_obs] = True
    scored = valid.copy()
    scored[:piece.burn] = False

    # Availability, not eligibility: true exactly where a real action was
    # loaded for this row. Row ``real`` is the window's final observation and
    # has no action of its own unless the lookahead supplied one. Indexed by
    # the target axis, so it is ``lookahead`` rows longer than the masks.
    action_valid = np.zeros(capacity, dtype=bool)
    action_valid[:min(loaded_targets, capacity)] = True

    reward_valid = scored.copy()
    if prev_reward is None:
        reward_valid[0] = False

    is_first = np.zeros(total, dtype=bool)
    is_first[0] = piece.start == 0
    is_last = np.zeros(total, dtype=bool)
    if piece.episode_end and real_obs - 1 < total:
        is_last[real_obs - 1] = True

    out |= {
        "valid": valid,
        "loss_mask": scored,
        "action_valid": action_valid,
        "reward_valid": reward_valid,
        "is_first": is_first,
        "is_last": is_last,
    }
    for key, value in (extras or {}).items():
        out[key] = _pad_to(np.asarray(value), total)
    return out


def conditioning_rows(window: Mapping[str, np.ndarray]) -> np.ndarray:
    """Rows the actor may be trained at: scored, and with a target loaded.

    Indexed by the observation axis. ``action_valid`` runs on the longer target
    axis, so it is truncated to the masks' length here.
    """
    scored = np.asarray(window["loss_mask"], dtype=bool)
    available = np.asarray(window["action_valid"], dtype=bool)
    return scored & available[: scored.shape[0]]


def check(window: Mapping[str, np.ndarray], length: int, burn_in: int,
          lookahead: int = 0) -> None:
    """Two row counts, and every array is on one axis or the other.

    Cheap, and it is the property that lets two sources stack: observation-axis
    arrays all have ``rows(...)`` rows and the target-axis arrays all have
    ``target_rows(...)``.
    """
    total = rows(length, burn_in)
    capacity = target_rows(length, burn_in, lookahead)
    wrong = {}
    for key, value in window.items():
        expected = capacity if key in TARGET_KEYS else total
        observed = int(np.asarray(value).shape[0])
        if observed != expected:
            wrong[key] = (observed, expected)
    if wrong:
        raise ValueError(
            f"window rows are wrong: {wrong} (observed, expected) with "
            f"{total} observation rows and {capacity} target rows. Every "
            "source must use sim_vla.data.layout.assemble.")
