"""Three action coordinate systems, and the conversions between them.

An action in this pipeline is written in one of three coordinate systems, and
every previous bug in this area was a value crossing a boundary without being
converted. They are named here once so that every call site can say which one
it means.

=============  =============================================================
**raw**        Environment command units. What ``env.step`` receives, what the
               dataset stores in ``actions``, and what the replay keeps. Bounded
               by the controller's own limits (``[-1, 1]`` for the ManiSkill
               normalised controllers used here).
**normalized** Standardised by the shared :class:`~sim_vla.data.normalization.
               Normalizer` fitted once on the demonstrations. This is the
               actor's coordinate system: flow matching regresses onto it and
               the sampler emits it. Mean 0, unit scale, unbounded.
**dynamics**   What the RSSM consumes as ``prev_action``. Bounded to
               ``(-1, 1)``.
=============  =============================================================

Why ``dynamics`` is a separate system at all
--------------------------------------------

``rssm.py:65`` normalises its action input::

    action = action / torch.clip(torch.abs(action), min=1.0).detach()

For ``|a| <= 1`` this is the identity, which is the regime the original Dreamer
pipeline runs in: its actions are environment commands already inside the
controller bounds. For ``|a| > 1`` it projects onto the unit ball, so
standardised values of 2 and 4 both arrive as 1 -- two distinct, valid physical
actions that the dynamics can no longer tell apart.

``rssm.py`` is the existing simulator's code and is left exactly as it is. The
incompatibility is resolved on this side of the boundary instead: normalized
actions are mapped into ``(-1, 1)`` by :func:`to_dynamics` before the RSSM sees
them, so the clip is always the identity and no two distinct actions collapse.
The map is ``tanh(x / squash)``, which is smooth, strictly increasing and
therefore injective, and differentiable everywhere -- the online actor update
pushes a gradient back through it.

``squash`` sets how much of the standardised range is spent before ``tanh``
flattens. At the default 3.0, +-1 sigma lands at +-0.32 and +-3 sigma at
+-0.76, so the ordinary range of the data is in the near-linear part.

Clipping to the environment's bounds
------------------------------------

The command actually sent to the environment is clipped to the controller's
bounds, so the action the next posterior is conditioned on must be that clipped
command and not what the policy asked for. :func:`clip_normalized` does this in
normalized coordinates with a straight-through estimator: the forward value is
clipped, the backward pass is the identity. A hard ``clamp`` would be forward-
correct and would zero the gradient of every saturated dimension, which is
exactly the dimension the actor most needs to be pushed back from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch

# How much of the standardised range is spent before tanh flattens. Not a
# tuning knob so much as a statement about the data: at 3.0 the bulk of a
# standardised distribution stays in the near-linear part of the map.
DEFAULT_SQUASH = 3.0

# What the field is called in the shared normalizer.
ACTION_FIELD = "actions"


def to_dynamics(normalized: torch.Tensor, squash: float = DEFAULT_SQUASH
                ) -> torch.Tensor:
    """normalized -> dynamics. Injective, differentiable, inside (-1, 1)."""
    return torch.tanh(normalized / float(squash))


def from_dynamics(dynamics: torch.Tensor, squash: float = DEFAULT_SQUASH,
                  *, eps: float = 1e-6) -> torch.Tensor:
    """dynamics -> normalized. The exact inverse of :func:`to_dynamics`.

    Used by tests and diagnostics rather than by the training path, which only
    ever converts in the forward direction.
    """
    clamped = dynamics.clamp(-1.0 + eps, 1.0 - eps)
    return torch.atanh(clamped) * float(squash)


def straight_through_clamp(value: torch.Tensor, low: torch.Tensor,
                           high: torch.Tensor) -> torch.Tensor:
    """Clamped forward, identity backward.

    The environment really does clip, so the forward value has to be the
    clipped one or imagination models a transition that cannot happen. The
    gradient is passed through because a saturated dimension is precisely the
    one the actor has to be pushed back from, and a zero there says nothing.
    """
    clamped = torch.maximum(torch.minimum(value, high), low)
    return value + (clamped - value).detach()


class FieldScaler:
    """Tensor mirror of :meth:`Normalizer.normalize` for one field.

    It exists so there is exactly one set of formulas. The numpy
    :class:`~sim_vla.data.normalization.Normalizer` clamps its divisor --
    ``np.maximum(std, EPS)`` -- and the tensor path used to divide by the raw
    statistic. A dimension the robot never moves has ``std == 0``, and the
    fitter returns that honestly, so the tensor path produced ``inf`` and
    ``nan`` for every such dimension while the numpy path beside it returned
    finite values. Constant dimensions are common: an unused gripper axis, a
    locked joint.

    The mode switch is mirrored for the same reason. A ``Normalizer`` in
    ``range`` mode maps the fitted percentile band to ``[-1, 1]``; a tensor
    path that standardised regardless would disagree with the very object it
    was built from, and inference would denormalize with different arithmetic
    than training normalized with.
    """

    MODES = ("mean_std", "range")

    def __init__(self, stats, mode: str = "mean_std", device=None,
                 dtype=torch.float32):
        from ..data.normalization import EPS

        self.mode = str(mode)
        if self.mode not in self.MODES:
            raise ValueError(
                f"normalization mode {self.mode!r} is not implemented here; "
                f"this mirrors Normalizer and supports {list(self.MODES)}. A "
                "mode that is read and then ignored puts training and "
                "inference in different units.")
        mean, std, low, high = stats.as_arrays()
        make = lambda a: torch.as_tensor(                  # noqa: E731
            np.asarray(a, dtype=np.float32), device=device, dtype=dtype)
        self.mean = make(mean)
        # Clamped once, here, and never used unclamped again.
        self.std = make(np.maximum(np.asarray(std, np.float32), EPS))
        self.low = make(low)
        self.span = make(np.maximum(
            np.asarray(high, np.float32) - np.asarray(low, np.float32), EPS))
        self.width = int(self.mean.shape[-1])

    def _align(self, reference: torch.Tensor, tensor: torch.Tensor
               ) -> torch.Tensor:
        return tensor.to(device=reference.device,
                         dtype=reference.dtype)[..., :reference.shape[-1]]

    def normalize(self, raw: torch.Tensor) -> torch.Tensor:
        if self.mode == "mean_std":
            return (raw - self._align(raw, self.mean)) / self._align(raw,
                                                                     self.std)
        return 2.0 * (raw - self._align(raw, self.low)) / self._align(
            raw, self.span) - 1.0

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        if self.mode == "mean_std":
            return value * self._align(value, self.std) + self._align(
                value, self.mean)
        return (value + 1.0) * 0.5 * self._align(
            value, self.span) + self._align(value, self.low)


def scaler_for(normalizer, field: str, device=None) -> Optional[FieldScaler]:
    """A scaler for one field of a normalizer, or None if it has no such field."""
    fields = getattr(normalizer, "fields", None) or {}
    if field not in fields:
        return None
    return FieldScaler(fields[field], getattr(normalizer, "mode", "mean_std"),
                       device=device)


@dataclass(frozen=True)
class ActionBounds:
    """Per-dimension environment command bounds, in raw units."""

    low: np.ndarray
    high: np.ndarray

    @classmethod
    def from_metadata(cls, metadata: Any, action_dim: int) -> "ActionBounds":
        """The recorded controller bounds, or the normalised default.

        ManiSkill's normalised controllers act on ``[-1, 1]``; a recording that
        says otherwise is believed over that default, and a recording that says
        nothing falls back to it explicitly rather than by omission.
        """
        controller = dict((metadata or {}).get("controller") or {})
        low = controller.get("action_low")
        high = controller.get("action_high")
        if low is None or high is None:
            return cls(low=np.full(int(action_dim), -1.0, dtype=np.float32),
                       high=np.full(int(action_dim), 1.0, dtype=np.float32))
        return cls(low=np.broadcast_to(
                       np.asarray(low, dtype=np.float32),
                       (int(action_dim),)).copy(),
                   high=np.broadcast_to(
                       np.asarray(high, dtype=np.float32),
                       (int(action_dim),)).copy())


class ActionCoordinates:
    """Conversions between the three systems, for one task and one normalizer.

    Holds the normalizer's statistics as tensors so that the conversions are
    differentiable and stay on the device. A ``None`` normalizer means the
    identity between raw and normalized -- stated explicitly, because a silent
    identity is how an unnormalised run gets mistaken for a normalised one.
    """

    def __init__(self, normalizer=None, bounds: Optional[ActionBounds] = None,
                 *, squash: float = DEFAULT_SQUASH, device=None,
                 action_dim: Optional[int] = None):
        self.squash = float(squash)
        self.normalizer = normalizer
        self.device = torch.device(device) if device is not None else None
        self.action_dim = int(action_dim) if action_dim else None

        self.scaler = scaler_for(normalizer, ACTION_FIELD, device=self.device)
        if self.scaler is not None and self.action_dim is None:
            self.action_dim = self.scaler.width

        if bounds is None and self.action_dim:
            bounds = ActionBounds.from_metadata(None, self.action_dim)
        self.bounds = bounds
        self._low: Optional[torch.Tensor] = None
        self._high: Optional[torch.Tensor] = None
        if bounds is not None:
            self._low = torch.as_tensor(bounds.low, device=self.device)
            self._high = torch.as_tensor(bounds.high, device=self.device)

    # ------------------------------------------------------------ raw <-> norm
    @property
    def normalizes(self) -> bool:
        """Whether this actually standardises, rather than passing through."""
        return self.scaler is not None

    def _aligned(self, reference: torch.Tensor, tensor: torch.Tensor
                 ) -> torch.Tensor:
        return tensor.to(device=reference.device,
                         dtype=reference.dtype)[..., :reference.shape[-1]]

    def normalize(self, raw: torch.Tensor) -> torch.Tensor:
        if not self.normalizes:
            return raw
        return self.scaler.normalize(raw)

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        if not self.normalizes:
            return normalized
        return self.scaler.denormalize(normalized)

    # -------------------------------------------------------------- bounds
    def normalized_bounds(self, reference: torch.Tensor):
        """The environment's raw bounds expressed in normalized coordinates.

        Through the same scaler the actions go through, so the bounds land in
        the coordinates the policy actually emits. Both supported modes are
        strictly increasing, so the order survives and low stays below high.
        """
        if self._low is None or self._high is None:
            return None, None
        low = self._aligned(reference, self._low)
        high = self._aligned(reference, self._high)
        if not self.normalizes:
            return low, high
        return self.scaler.normalize(low), self.scaler.normalize(high)

    def clip_normalized(self, normalized: torch.Tensor) -> torch.Tensor:
        """Apply the environment's clipping, in the actor's coordinates."""
        low, high = self.normalized_bounds(normalized)
        if low is None:
            return normalized
        return straight_through_clamp(normalized, low, high)

    def clip_raw(self, raw: np.ndarray) -> np.ndarray:
        """Clip a raw command to the environment's bounds."""
        if self.bounds is None:
            return raw
        width = raw.shape[-1]
        return np.clip(raw, self.bounds.low[:width], self.bounds.high[:width])

    # ------------------------------------------------------- norm <-> dynamics
    def to_dynamics(self, normalized: torch.Tensor) -> torch.Tensor:
        return to_dynamics(normalized, self.squash)

    def from_dynamics(self, dynamics: torch.Tensor) -> torch.Tensor:
        return from_dynamics(dynamics, self.squash)

    def raw_to_dynamics(self, raw: torch.Tensor) -> torch.Tensor:
        """The full path a stored action takes to reach the RSSM."""
        return self.to_dynamics(self.normalize(raw))

    def executed(self, normalized: torch.Tensor) -> torch.Tensor:
        """What the environment will actually run, in normalized coordinates.

        Clipping happens here and only here, so the action fed back to the
        posterior is the command that was sent -- not the one requested.
        """
        return self.clip_normalized(normalized)

    def describe(self) -> dict:
        return {
            "normalizes": self.normalizes,
            "mode": None if self.scaler is None else self.scaler.mode,
            "squash": self.squash,
            "action_dim": self.action_dim,
            "bounds_low": None if self.bounds is None
            else self.bounds.low.tolist(),
            "bounds_high": None if self.bounds is None
            else self.bounds.high.tolist(),
        }
