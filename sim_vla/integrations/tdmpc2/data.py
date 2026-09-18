"""The shared demonstrations, in TD-MPC2's own batch shapes.

One dataset feeds all three backends, so the windowing code is the tested one
from ``sim_vla.data`` rather than a third copy. What differs is the *reward
convention*, and that difference is applied here, once, in one function.

The two conventions
-------------------

``sim_vla.data.layout`` is written for a Dreamer-style model, where row ``t``
holds the observation ``o_t`` and::

    action[t]         = a_(t-1)   the action that led here
    action_target[t]  = a_t       the action taken here
    reward[t]         = r_(t-1)   the reward earned arriving here

TD-MPC2 reads a batch where index ``t`` pairs ``o_t`` with the action taken
there and the reward that action earned -- ``_estimate_value`` computes
``reward(z_t, a_t)`` and only then advances ``z`` -- so::

    obs[t]     = o_t                          = observation row t
    action[t]  = a_t                          = action_target[t]
    reward[t]  = r_t  (earned by a_t)         = layout reward[t + 1]

Both are correct for their own model and neither is a fix for the other. The
shift is written out in :func:`native_batch` so there is exactly one place that
has to be right, and ``tests/test_tdmpc2_timeline.py`` asserts it against a
hand-built episode.

Windows are drawn **unpadded**. ``TDMPC2.update`` has no mask: it averages its
consistency, reward and value losses over the whole batch, and upstream's own
``SliceSampler`` is built with ``strict_length=True`` for the same reason. A
padded window would train the dynamics on a repeated frame and a zero action
and report nothing.

Images cross into the model as bytes. ``layers.conv`` begins with ``ShiftAug``
then ``PixelPreprocess``, which is where ``/255 - 0.5`` happens; anything
scaled here would be scaled twice. The random shift is applied on every
encode, including at action-selection time -- that is upstream's behaviour and
it is preserved, so a TD-MPC2 latent is a stochastic function of its
observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from ...data.dataset import DemoDataset
from ...data.sequences import SequenceSampler, plan_windows
from ..observations import ImageContract, contract_for

RGB_KEY = "rgb"
STATE_KEY = "rgb-state"


@dataclass
class DemoSource:
    """The demonstrations, plus everything derived from their metadata."""

    dataset: DemoDataset
    images: ImageContract
    include_state: bool
    action_dim: int
    proprio_dim: int
    episode_length: int

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self.dataset.metadata)

    def obs_shape(self) -> Dict[str, Tuple[int, ...]]:
        """What ``cfg.obs_shape`` has to say, from the data rather than a guess."""
        shape: Dict[str, Tuple[int, ...]] = {RGB_KEY: self.images.shape}
        if self.include_state:
            shape[STATE_KEY] = (int(self.proprio_dim),)
        return shape

    def identity(self) -> Dict[str, Any]:
        from ...data.normalization import dataset_identity

        return dataset_identity(self.metadata, len(self.dataset.episodes))

    def contract(self) -> Dict[str, Any]:
        """What an online environment has to reproduce to be the same task."""
        controller = dict(self.metadata.get("controller") or {})
        return {
            "env_id": str(self.metadata.get("env_id") or ""),
            "control_mode": controller.get("control_mode"),
            "action_dim": int(self.action_dim),
            "reward_mode": self.metadata.get("reward_mode"),
            "cameras": list(self.images.cameras),
            "render_size": list(self.images.size),
            "rgb_channels": self.images.channels,
            "include_state": bool(self.include_state),
            "proprio_dim": int(self.proprio_dim) if self.include_state else 0,
            "episode_length": int(self.episode_length),
        }


def open_demos(path: str | Path, *, render_size: int, include_state: bool,
               cameras: Optional[Sequence[str]] = None,
               ignore_terminations: bool = True) -> DemoSource:
    """Open the demonstrations and read the contract off them."""
    data = DemoDataset(path, graph_enabled=False,
                       ignore_terminations=ignore_terminations)
    if not data.episodes:
        raise SystemExit(f"{path} holds no episodes")
    images = contract_for(data.metadata, backend="tdmpc2",
                          size=(int(render_size), int(render_size)),
                          cameras=cameras)
    ref = data.episodes[0]
    probe = data.read(ref, 0, 1)
    action_dim = int(np.asarray(probe["actions"]).shape[-1])
    proprio_dim = int(np.asarray(probe["proprio"]).shape[-1])
    length = max(int(r.steps) for r in data.episodes)
    return DemoSource(dataset=data, images=images,
                      include_state=bool(include_state),
                      action_dim=action_dim, proprio_dim=proprio_dim,
                      episode_length=length)


def _full_windows(sampler: SequenceSampler) -> List[Any]:
    """Only the windows that are entirely real.

    ``plan_windows`` pads the last window of an episode to the fixed shape;
    ``TDMPC2.update`` has no mask and would train on the padding.
    """
    return [w for w in sampler.windows if w.pad == 0]


class DemoWindows:
    """Unpadded windows of a fixed transition count, drawn inside episodes."""

    def __init__(self, source: DemoSource, *, horizon: int, seed: int = 0,
                 stride: int = 1, lookahead: int = 0, burn_in: int = 0):
        self.source = source
        self.horizon = int(horizon)
        self.sampler = SequenceSampler(
            source.dataset, length=int(horizon), burn_in=int(burn_in),
            stride=int(stride), seed=int(seed), lookahead=int(lookahead))
        self.windows = _full_windows(self.sampler)
        if not self.windows:
            raise SystemExit(
                f"no episode is long enough for a window of {horizon} "
                f"transitions (+ {burn_in} burn-in); the longest is "
                f"{max(r.steps for r in source.dataset.episodes)} steps")
        self._rng = np.random.default_rng(int(seed))

    @property
    def lookahead(self) -> int:
        return int(self.sampler.lookahead)

    @lookahead.setter
    def lookahead(self, value: int) -> None:
        self.sampler.lookahead = int(value)

    def __len__(self) -> int:
        return len(self.windows)

    def raw_batch(self, size: int) -> Dict[str, np.ndarray]:
        picks = self._rng.integers(0, len(self.windows), size=int(size))
        loaded = [self.sampler.load(self.windows[int(i)]) for i in picks]
        return {key: np.stack([item[key] for item in loaded])
                for key in loaded[0]}


def observations(batch: Mapping[str, Any], source: DemoSource, *,
                 device=None) -> Any:
    """The observation half of a batch, in TD-MPC2's layout.

    ``(B, T, ...)`` in, ``(T, B, ...)`` out, because upstream indexes time
    first. Images stay ``uint8``; the encoder scales them.
    """
    rgb = source.images.apply(batch).movedim(0, 1)          # (T, B, C, H, W)
    if device is not None:
        rgb = rgb.to(device, non_blocking=True)
    if not source.include_state:
        return rgb
    state = batch["proprio"]
    state = (state if isinstance(state, torch.Tensor)
             else torch.as_tensor(np.asarray(state))).float().movedim(0, 1)
    if device is not None:
        state = state.to(device, non_blocking=True)
    from tensordict.tensordict import TensorDict

    return TensorDict({RGB_KEY: rgb, STATE_KEY: state},
                      batch_size=tuple(rgb.shape[:2]),
                      device=rgb.device)


def native_batch(batch: Mapping[str, Any], source: DemoSource, *,
                 horizon: int, device=None, converter=None):
    """``(obs, action, reward, task)`` exactly as ``Buffer.sample`` returns it.

    The reward shift is here. See the module docstring: TD-MPC2's ``reward[t]``
    is what ``a_t`` earned, which in the shared window layout is the reward
    recorded one row later.
    """
    obs = observations(batch, source, device=device)
    targets = batch["action_target"]
    targets = (targets if isinstance(targets, torch.Tensor)
               else torch.as_tensor(np.asarray(targets))).float()
    action = targets[:, :horizon].movedim(0, 1)             # (T, B, A)

    rewards = batch["reward"]
    rewards = (rewards if isinstance(rewards, torch.Tensor)
               else torch.as_tensor(np.asarray(rewards))).float()
    if rewards.dim() == 3 and rewards.shape[-1] == 1:
        rewards = rewards.squeeze(-1)
    # layout reward[t] is r_(t-1); TD-MPC2 wants r_t beside a_t.
    reward = rewards[:, 1:horizon + 1].movedim(0, 1).unsqueeze(-1)

    if converter is not None:
        # Stored actions are native units already; this is the identity unless
        # a run has deliberately fitted statistics, and going through the same
        # object as the policy is what keeps the two in one coordinate system.
        action = converter.clip_native(action)
    if device is not None:
        action = action.to(device, non_blocking=True)
        reward = reward.to(device, non_blocking=True)
    return obs, action, reward, None


class DemoBuffer:
    """A replay buffer's interface over a fixed set of demonstrations.

    ``TDMPC2.update(buffer)`` asks for one thing, ``buffer.sample()``, and gets
    back what ``common.buffer.Buffer`` returns. Supplying that here means
    Stage 1 is upstream's own update -- encoder, dynamics, reward, Q ensemble
    and the Gaussian prior, coupled exactly as they are online -- run against
    demonstrations instead of a live rollout.
    """

    def __init__(self, source: DemoSource, *, horizon: int, batch_size: int,
                 device="cuda", seed: int = 0, converter=None,
                 stride: int = 1):
        self.source = source
        self.horizon = int(horizon)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.converter = converter
        self.windows = DemoWindows(source, horizon=int(horizon), seed=int(seed),
                                   stride=int(stride))

    def __len__(self) -> int:
        return len(self.windows)

    def sample(self):
        raw = self.windows.raw_batch(self.batch_size)
        return native_batch(raw, self.source, horizon=self.horizon,
                            device=self.device, converter=self.converter)
