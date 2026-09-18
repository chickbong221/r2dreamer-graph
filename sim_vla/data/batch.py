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
# Arrays indexed by transition, in storage naming. Everything else in a window
# is indexed by observation and carries one extra row.
STEP_KEYS = ("actions", "rewards", "terminated", "truncated", "success",
             "is_terminal", "is_last")


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


def to_model_batch(batch: Mapping[str, np.ndarray], device=None,
                   *, reward_dim: bool = True) -> Dict[str, Any]:
    """Tensors under the names the world model reads.

    ``reward_dim`` adds the trailing axis the reward head's distribution
    expects; the stored array is one value per step.
    """
    import torch

    out: Dict[str, Any] = {}
    for key, value in batch.items():
        tensor = torch.as_tensor(np.asarray(value))
        if tensor.dtype == torch.float64:
            tensor = tensor.float()
        out[RENAMES.get(key, key)] = tensor if device is None else tensor.to(device)

    if "action" in out:
        out["action"] = out["action"].float()
    if "reward" in out:
        out["reward"] = out["reward"].float()
        if reward_dim and out["reward"].dim() == 2:
            out["reward"] = out["reward"].unsqueeze(-1)
    return out
