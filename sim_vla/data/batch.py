"""One naming convention for a learning batch, and one place that applies it.

The dataset stores ``actions`` and ``rewards``; the RSSM and the heads read
``action`` and ``reward``. That rename has to happen exactly once, in one
place, or two things go wrong and neither of them fails loudly.

Both did. ``ImitationTrainer`` converted a batch to tensors without renaming,
so ``observe`` would not find ``action``. And the online replay emitted the
singular names while the demonstration sampler emitted the plural ones, so
``mixed_batch`` -- which concatenates only the keys the two sources share --
dropped the actions and the rewards from every mixed batch, leaving a batch
that trains on observations alone.

So the names live here, the replay writes the storage names, and every trainer
converts through :func:`to_model_batch`.

Units
-----

This is also the one place that moves an action between coordinate systems.
See :mod:`sim_vla.models.action_space` for what the three systems are. On the
way out of here:

===================  =======================================================
``action``           **dynamics** coordinates. It is consumed by
                     ``RSSM.obs_step`` as ``a_(t-1)`` and by nothing else.
``action_target``    **normalized** coordinates. It is the actor's supervision
                     and must be in the space the actor predicts.
``proprio``          standardised by the same normalizer.
===================  =======================================================

Idempotence
-----------

``_preprocessed`` marks a batch that has already been through here. It is
*kept* on the way out, not consumed: popping it made the second call a no-op
and the third call a second normalization, which is the opposite of what a
guard is for.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

# Stored name -> the name the model reads.
RENAMES = {"actions": "action", "rewards": "reward"}
# Arrays indexed by transition, in storage naming. Everything else in a stored
# episode is indexed by observation and carries one extra row. Windows do not
# use this split -- layout.assemble gives every array one row per observation.
STEP_KEYS = ("actions", "rewards", "terminated", "truncated", "success",
             "is_terminal", "is_last")
# What a window must carry for a batch to be trainable. mixed_batch checks it.
WINDOW_REQUIRED = ("action", "action_target", "reward", "loss_mask", "valid",
                   "action_valid", "reward_valid", "is_first", "is_last",
                   "is_terminal")

MARKER = "_preprocessed"


def storage_keys(batch: Mapping[str, Any]) -> Dict[str, Any]:
    """Assert a batch uses the storage names, and return it unchanged.

    Cheap, and it catches a source that has drifted to the model names before
    that source is mixed with one that has not.
    """
    wrong = [name for name in RENAMES.values() if name in batch]
    if wrong:
        raise KeyError(
            f"{wrong} are model-side names; a batch at this point should use "
            f"{sorted(RENAMES)}. Rename once, in to_model_batch.")
    return dict(batch)


IMAGE_PREFIX = "image_"


def _as_tensor(value, device=None):
    """A tensor on ``device``, without a CPU round trip for a live tensor."""
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(np.asarray(value))
    if tensor.dtype == torch.float64:
        tensor = tensor.float()
    if device is not None and tensor.device != torch.device(device):
        tensor = tensor.to(device)
    return tensor


def is_preprocessed(batch: Mapping[str, Any]) -> bool:
    """Whether this batch has already been converted."""
    marker = batch.get(MARKER)
    if marker is None:
        return False
    try:
        return bool(marker)
    except RuntimeError:                                   # multi-element
        return True


def to_model_batch(batch: Mapping[str, Any], device=None,
                   *, reward_dim: bool = True, normalizer=None,
                   coords=None, already_preprocessed: bool = False
                   ) -> Dict[str, Any]:
    """Tensors the world model can consume, preprocessed exactly once.

    Four things happen here and nowhere else, so they cannot happen twice.

    **Images.** Recorded RGB is ``uint8`` in [0, 255]; the convolutional
    encoder expects floats around [0, 1]. Left as bytes, the encoder sees
    inputs two orders of magnitude outside the range its initialisation
    assumes, which does not fail -- it just trains badly.

    **Proprioception.** Standardised by the shared normalizer when one is
    given, so that both arms use statistics fitted once from the
    demonstrations they have in common.

    **Actions.** Standardised, and then ``action`` alone is mapped into
    dynamics coordinates because that is the only thing that reads it. See the
    module docstring.

    **The trailing reward axis** the reward head's distribution expects.

    ``already_preprocessed``, and the ``_preprocessed`` key a previous call
    left behind, both skip the numeric work. The marker survives the call, so
    a third pass is a no-op too.
    """
    import torch

    from ..models.action_space import ActionCoordinates, scaler_for

    seen = is_preprocessed(batch) or already_preprocessed

    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if key == MARKER:
            continue
        out[RENAMES.get(key, key)] = _as_tensor(value, device)

    if seen:
        out[MARKER] = torch.ones((), dtype=torch.bool, device=device)
        return out

    for key, tensor in list(out.items()):
        if key.startswith(IMAGE_PREFIX):
            out[key] = (tensor.float() / 255.0 if tensor.dtype == torch.uint8
                        else tensor.float())

    if coords is None and normalizer is not None:
        coords = ActionCoordinates(normalizer, device=device)

    if normalizer is not None and "proprio" in out:
        # Through the same scaler the actions use: it clamps the divisor and
        # honours the mode. Dividing by a raw statistic here produced inf and
        # nan on every dimension the robot never moves.
        scaler = scaler_for(normalizer, "proprio", device=device)
        if scaler is not None:
            out["proprio"] = scaler.normalize(out["proprio"].float())

    if "action" in out:
        raw = out["action"].float()
        # a_(t-1): standardise, then into dynamics coordinates, because the
        # RSSM projects anything outside the unit ball onto it.
        out["action"] = (coords.raw_to_dynamics(raw) if coords is not None
                         else raw)
    if "action_target" in out:
        # a_t: the actor's supervision stays in the actor's coordinates.
        target = out["action_target"].float()
        out["action_target"] = (coords.normalize(target) if coords is not None
                                else target)
    if "reward" in out:
        out["reward"] = out["reward"].float()
        if reward_dim and out["reward"].dim() == 2:
            out["reward"] = out["reward"].unsqueeze(-1)

    anchor = out.get("action")
    out[MARKER] = torch.ones(
        (), dtype=torch.bool,
        device=anchor.device if anchor is not None else device)
    return out
