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
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

# Stored name -> the name the model reads.
RENAMES = {"actions": "action", "rewards": "reward"}
# Arrays indexed by transition, in storage naming. Everything else in a stored
# episode is indexed by observation and carries one extra row. Windows do not
# use this split -- layout.assemble gives every array one row per observation.
STEP_KEYS = ("actions", "rewards", "terminated", "truncated", "success",
             "is_terminal", "is_last")
# What a window must carry for a batch to be trainable. mixed_batch checks it.
WINDOW_REQUIRED = ("action", "action_target", "reward", "loss_mask", "valid")


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


def to_model_batch(batch: Mapping[str, np.ndarray], device=None,
                   *, reward_dim: bool = True, normalizer=None,
                   already_preprocessed: bool = False) -> Dict[str, Any]:
    """Tensors the world model can consume, preprocessed exactly once.

    Three things happen here and nowhere else, so they cannot happen twice.

    **Images.** Recorded RGB is ``uint8`` in [0, 255]; the convolutional
    encoder expects floats around [0, 1]. Left as bytes, the encoder sees
    inputs two orders of magnitude outside the range its initialisation
    assumes, which does not fail -- it just trains badly.

    **Proprioception and actions.** Scaled by the shared normalizer when one is
    given, so that both arms use statistics fitted once from the demonstrations
    they have in common.

    **The trailing reward axis** the reward head's distribution expects.

    ``already_preprocessed`` marks a batch that has been through here before,
    which is how double normalization is prevented rather than hoped against.
    """
    import torch

    out: Dict[str, Any] = {}
    for key, value in batch.items():
        tensor = torch.as_tensor(np.asarray(value))
        if tensor.dtype == torch.float64:
            tensor = tensor.float()
        out[RENAMES.get(key, key)] = tensor if device is None else tensor.to(device)

    if out.pop("_preprocessed", None) is not None or already_preprocessed:
        return out

    for key, tensor in list(out.items()):
        if key.startswith(IMAGE_PREFIX):
            out[key] = (tensor.float() / 255.0 if tensor.dtype == torch.uint8
                        else tensor.float())

    if normalizer is not None:
        for key, field in (("proprio", "proprio"), ("action", "actions"),
                           ("action_target", "actions")):
            if key in out and field in normalizer.fields:
                shape = out[key].shape
                flat = out[key].reshape(-1, shape[-1]).cpu().numpy()
                scaled = normalizer.normalize(field, flat)
                out[key] = torch.as_tensor(
                    scaled, device=out[key].device).reshape(shape).float()

    if "action" in out:
        out["action"] = out["action"].float()
    if "reward" in out:
        out["reward"] = out["reward"].float()
        if reward_dim and out["reward"].dim() == 2:
            out["reward"] = out["reward"].unsqueeze(-1)
    # Marks this batch as having been through preprocessing, so a second pass
    # is a no-op rather than a second normalization.
    out["_preprocessed"] = torch.ones((), dtype=torch.bool,
                                      device=out["action"].device
                                      if "action" in out else None)
    return out
