"""One boundary between the actor's action units and a backend's own.

The Dreamer integration has three coordinate systems because the RSSM projects
any action outside the unit ball onto it, so a standardised action had to be
squashed with ``tanh(x / 3)`` before the dynamics could tell two of them apart.
**Neither backend here has that problem and neither gets that squashing.**

* TD-MPC2 feeds the action straight into ``_dynamics``, ``_reward`` and
  ``_Qs`` as a plain concatenation, and its own planner clamps proposals to
  ``[-1, 1]``. Its native action units are the controller's.
* SOLD feeds the action into the OCVP dynamics through a linear
  ``action_encoder`` and its ``GaussianPredictor`` emits ``tanh(mean)``
  clamped to the action space's bounds. Its native action units are the
  controller's too.

So there are two systems, not three:

============  ==============================================================
**native**    what ``env.step`` receives, what the demonstrations store, what
              the backend's dynamics, reward and value heads consume. For the
              ManiSkill controllers used here that is ``[-1, 1]``.
**actor**     what the flow model regresses onto and samples in.
============  ==============================================================

``ActionConverter`` is the only place that crosses between them, so that
imitation targets and policy outputs cannot end up in different units --
supervising on standardised actions and then feeding raw samples to the
planner descends a loss and learns nothing.

Modes
-----

``identity``   the two systems coincide. The default, and the honest one for
               these backends: the native space is already the controller's
               normalised ``[-1, 1]``, which is also roughly the range
               SmolVLA's released checkpoint was trained on, and an extra
               affine layer buys nothing while adding a place for training and
               inference to disagree.
``mean_std``   standardise with the demonstrations' per-dimension statistics.
``range``      map the fitted percentile band to ``[-1, 1]``.

The last two are built on :class:`sim_vla.models.action_space.FieldScaler`,
which floors the divisor at ``EPS``. That flooring is not cosmetic: a joint the
task never moves has a fitted ``std`` of exactly zero, and dividing by it gives
``inf`` on that dimension and ``nan`` on the whole loss one step later. Any new
scaler added here must go through ``FieldScaler`` for the same reason.

Clipping
--------

The command sent to the environment is clipped to the controller's bounds, so
the action the next latent is conditioned on has to be the clipped one.
:meth:`ActionConverter.executed` does that in actor coordinates with a
straight-through estimator: clipped forward, identity backward, because a
saturated dimension is the one the actor most needs to be pushed back from. A
hard ``clamp`` would zero exactly that gradient.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

MODES = ("identity", "mean_std", "range")


class ActionConverter:
    """actor <-> native, for one task, with the controller's bounds attached."""

    def __init__(self, *, action_dim: int, mode: str = "identity",
                 normalizer=None, low=None, high=None, device=None):
        import torch

        from ..models.action_space import scaler_for

        self.mode = str(mode)
        if self.mode not in MODES:
            raise ValueError(
                f"action normalization {self.mode!r} is not implemented; "
                f"choose one of {list(MODES)}")
        self.action_dim = int(action_dim)
        self.device = torch.device(device) if device is not None else None
        self.normalizer = normalizer

        self.scaler = None
        if self.mode != "identity":
            if normalizer is None:
                raise ValueError(
                    f"action normalization {self.mode!r} needs statistics; fit "
                    "a Normalizer from the demonstrations or use 'identity'")
            fitted = str(getattr(normalizer, "mode", "mean_std"))
            if fitted != self.mode:
                raise ValueError(
                    f"the normalizer was fitted in {fitted!r} mode and "
                    f"{self.mode!r} was requested; one object decides this, or "
                    "training and inference use different arithmetic")
            self.scaler = scaler_for(normalizer, "actions", device=self.device)
            if self.scaler is None:
                raise ValueError(
                    "the normalizer has no 'actions' field to scale with")

        low = (np.full(self.action_dim, -1.0, np.float32) if low is None
               else np.broadcast_to(np.asarray(low, np.float32),
                                    (self.action_dim,)).copy())
        high = (np.full(self.action_dim, 1.0, np.float32) if high is None
                else np.broadcast_to(np.asarray(high, np.float32),
                                     (self.action_dim,)).copy())
        self.low_native = low
        self.high_native = high
        self._low = torch.as_tensor(low, device=self.device)
        self._high = torch.as_tensor(high, device=self.device)

    # --------------------------------------------------------------- scaling
    @property
    def normalizes(self) -> bool:
        return self.scaler is not None

    def to_actor(self, native):
        """native -> actor. Used on imitation targets."""
        if self.scaler is None:
            return native
        return self.scaler.normalize(native)

    def to_native(self, actor):
        """actor -> native. The one conversion on the policy's output path."""
        if self.scaler is None:
            return actor
        return self.scaler.denormalize(actor)

    # ---------------------------------------------------------------- bounds
    def _aligned(self, reference, tensor):
        return tensor.to(device=reference.device,
                         dtype=reference.dtype)[..., :reference.shape[-1]]

    def actor_bounds(self, reference):
        """The controller's bounds expressed in actor coordinates.

        Through the same scaler the actions go through. Both supported scalers
        are strictly increasing, so low stays below high.
        """
        low = self._aligned(reference, self._low)
        high = self._aligned(reference, self._high)
        if self.scaler is None:
            return low, high
        return self.scaler.normalize(low), self.scaler.normalize(high)

    def executed(self, actor_action):
        """What the environment will run, still in actor coordinates."""
        from ..models.action_space import straight_through_clamp

        low, high = self.actor_bounds(actor_action)
        return straight_through_clamp(actor_action, low, high)

    def to_env(self, actor_action):
        """The single explicit boundary: actor sample -> backend action.

        Clip first, in the actor's own units, then convert. Clipping after the
        conversion would hand the dynamics a command the environment would not
        have run.
        """
        return self.to_native(self.executed(actor_action))

    def clip_native(self, native):
        import torch

        if isinstance(native, torch.Tensor):
            low = self._aligned(native, self._low)
            high = self._aligned(native, self._high)
            return torch.maximum(torch.minimum(native, high), low)
        width = np.asarray(native).shape[-1]
        return np.clip(np.asarray(native, np.float32),
                       self.low_native[:width], self.high_native[:width])

    # ---------------------------------------------------------------- record
    def descriptor(self) -> Dict[str, Any]:
        """What a checkpoint records so its weights stay interpretable."""
        out: Dict[str, Any] = {
            "mode": self.mode,
            "action_dim": self.action_dim,
            "bounds_low": self.low_native.tolist(),
            "bounds_high": self.high_native.tolist(),
        }
        if self.normalizer is not None and self.mode != "identity":
            out["statistics"] = self.normalizer.statistics_fingerprint()
            out["dataset"] = dict(getattr(self.normalizer, "identity", {}))
        return out


def bounds_from_metadata(metadata: Optional[Any], action_dim: int):
    """The recorded controller bounds, or the normalised default, stated."""
    from ..models.action_space import ActionBounds

    recorded = ActionBounds.from_metadata(metadata, int(action_dim))
    return recorded.low, recorded.high


def converter_for(metadata: Optional[Any], *, action_dim: int,
                  mode: str = "identity", normalizer=None,
                  device=None) -> ActionConverter:
    low, high = bounds_from_metadata(metadata, action_dim)
    return ActionConverter(action_dim=action_dim, mode=mode,
                           normalizer=normalizer, low=low, high=high,
                           device=device)
