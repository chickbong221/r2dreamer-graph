"""One window layout, and the causal contract it encodes.

Both sources of training sequences -- recorded demonstrations and the online
replay -- assemble windows here, so a mixed batch stacks. Two things were wrong
before and both were silent in isolation.

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

Every array has one row per observation, including the final one, so indices
line up without a caller ever slicing. The masks say which rows mean anything:

``valid``          this row is a real observation, not padding
``loss_mask``      ...and it is scored (not burn-in)
``action_valid``   an ``action_target`` exists here (false at the final
                   observation, which no action was taken at)
``reward_valid``   an incoming reward exists here (false at a reset)
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
    """
    total = rows(length, burn_in)
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

    # a_t per observation: the action taken here. The final observation has
    # none, so its row is padding and action_valid says so.
    targets = np.concatenate(
        [np.asarray(actions, dtype=np.float32),
         np.zeros((1, action_dim), dtype=np.float32)], axis=0)[:real_obs]
    out["action_target"] = _pad_to(targets, total)

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

    action_valid = valid.copy()
    # No action was taken at the final observation of an episode.
    if piece.episode_end and real_obs - 1 < total:
        action_valid[real_obs - 1] = False
    action_valid &= scored

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


def check(window: Mapping[str, np.ndarray], length: int, burn_in: int) -> None:
    """Every array in a window has the same number of rows. Cheap, and it is
    the property that lets two sources stack."""
    total = rows(length, burn_in)
    wrong = {key: int(np.asarray(value).shape[0])
             for key, value in window.items()
             if int(np.asarray(value).shape[0]) != total}
    if wrong:
        raise ValueError(
            f"window rows differ from {total}: {wrong}. Every source must use "
            "sim_vla.data.layout.assemble.")
