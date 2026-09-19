"""The shared demonstrations, in SOLD's own batch shapes.

SOLD's replay writes one row per *observation*::

    obs[i]      o_i
    action[i]   a_(i-1)   NaN at i = 0
    reward[i]   r_(i-1)   NaN at i = 0

which is the same convention ``sim_vla.data.layout`` uses, so no shift is
needed here -- unlike TD-MPC2, whose ``reward[t]`` is what ``a_t`` earned. The
one thing that has to be reproduced is the **NaN**: ``compute_reward_loss``
builds its mask with ``is_firsts = torch.isnan(rewards)``, so a window whose
first row has no incoming reward has to say so with a NaN and not with a zero,
or the reward head is trained to predict 0 at every episode start.

The actions are read the same way SOLD reads them: ``actions[:, 1:]`` is the
action taken *at* each frame, which is what ``SAVi.encode`` advances its slot
predictor with and what the dynamics model is conditioned on.

Images arrive as ``uint8`` at the SAVi encoder's own resolution, single camera.
``train_sold`` divides by 255 itself, so nothing is scaled here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from ...data.dataset import DemoDataset
from ...data.sequences import SequenceSampler
from ..observations import ImageContract, contract_for


@dataclass
class SoldDemoSource:
    """The demonstrations, plus what SOLD needs to know about them."""

    dataset: Any
    images: ImageContract
    action_dim: int
    episode_length: int

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self.dataset.metadata)

    def identity(self) -> Dict[str, Any]:
        from ...data.normalization import dataset_identity

        return dataset_identity(self.metadata, len(self.dataset.episodes))

    def contract(self) -> Dict[str, Any]:
        controller = dict(self.metadata.get("controller") or {})
        return {
            "env_id": str(self.metadata.get("env_id") or ""),
            "control_mode": controller.get("control_mode"),
            "action_dim": int(self.action_dim),
            "reward_mode": self.metadata.get("reward_mode"),
            "cameras": list(self.images.cameras),
            "image_size": list(self.images.size),
            "episode_length": int(self.episode_length),
        }


def open_demos(path: str | Path, *, image_size: Sequence[int],
               camera: Optional[str] = None,
               ignore_terminations: bool = True) -> SoldDemoSource:
    """Open the target demonstrations for SOLD.

    One camera. SAVi's encoder is a three-channel single-view autoencoder and
    its positional-embedding grid is built for one image size; feeding it a
    channel-stacked pair of views would be a different modality, which is not
    what this integration adds. Which camera is a decision, so it is named
    rather than being whichever key sorts first by accident.
    """
    data = DemoDataset(path, graph_enabled=False,
                       ignore_terminations=ignore_terminations)
    if not data.episodes:
        raise SystemExit(f"{path} holds no episodes")
    images = contract_for(data.metadata, backend="sold",
                          size=(int(image_size[0]), int(image_size[1])),
                          cameras=[camera] if camera else None, max_cameras=1)
    probe = data.read(data.episodes[0], 0, 1)
    return SoldDemoSource(
        dataset=data, images=images,
        action_dim=int(np.asarray(probe["actions"]).shape[-1]),
        episode_length=max(int(r.steps) for r in data.episodes))


class SoldWindows:
    """Fixed-length windows in SOLD's batch layout."""

    def __init__(self, source: SoldDemoSource, *, length: int, burn_in: int = 0,
                 seed: int = 0, stride: int = 1, lookahead: int = 0,
                 full_only: bool = True):
        self.source = source
        self.sampler = SequenceSampler(
            source.dataset, length=int(length), burn_in=int(burn_in),
            stride=int(stride), seed=int(seed), lookahead=int(lookahead))
        self.windows = ([w for w in self.sampler.windows if w.pad == 0]
                        if full_only else list(self.sampler.windows))
        if not self.windows:
            raise SystemExit(
                f"no episode is long enough for a window of {length} "
                f"transitions (+ {burn_in} burn-in)")
        self._rng = np.random.default_rng(int(seed))

    def __len__(self) -> int:
        return len(self.windows)

    def raw_batch(self, size: int) -> Dict[str, np.ndarray]:
        picks = self._rng.integers(0, len(self.windows), size=int(size))
        loaded = [self.sampler.load(self.windows[int(i)]) for i in picks]
        return {key: np.stack([item[key] for item in loaded])
                for key in loaded[0]}


def _tensor(value, device=None, dtype=None):
    out = (value if isinstance(value, torch.Tensor)
           else torch.as_tensor(np.asarray(value)))
    if dtype is not None:
        out = out.to(dtype)
    if device is not None:
        out = out.to(device, non_blocking=True)
    return out


def sold_batch(batch: Mapping[str, Any], source: SoldDemoSource, *,
               device=None, converter=None) -> Dict[str, torch.Tensor]:
    """``{obs, action, reward}`` in SOLD's shapes and its NaN convention.

    ``obs`` is ``uint8 (B, T, 3, H, W)``; ``train_sold`` divides by 255.
    ``action[t]`` is ``a_(t-1)`` and ``reward[t]`` is ``r_(t-1)``, both NaN
    where the window has no such thing -- which is what
    ``compute_reward_loss`` tests for.
    """
    obs = source.images.apply(batch)
    if device is not None:
        obs = obs.to(device, non_blocking=True)

    action = _tensor(batch["action"], device, torch.float32)
    if converter is not None:
        action = converter.clip_native(action)
    reward = _tensor(batch["reward"], device, torch.float32)
    if reward.dim() == 3 and reward.shape[-1] == 1:
        reward = reward.squeeze(-1)

    valid = _tensor(batch["reward_valid"], device).bool()
    nan = torch.full_like(reward, float("nan"))
    reward = torch.where(valid, reward, nan)

    # The first row's previous action is undefined in the same way. SOLD drops
    # it with `actions[:, 1:]` everywhere it is read, so this only matters if
    # something starts reading index 0 -- and then it should see a NaN rather
    # than a zero that looks like a real command.
    first = _tensor(batch["is_first"], device).bool()
    action = torch.where(first.unsqueeze(-1),
                         torch.full_like(action, float("nan")), action)
    return {"obs": obs, "action": action, "reward": reward}


class DemoLoader:
    """A replay buffer's interface over the demonstrations.

    ``RingBufferDataset.sample()`` returns a dict of stacked sequences; this
    returns the same three keys in the same shapes, so upstream's
    ``training_step`` can be run against demonstrations without changing it.
    """

    def __init__(self, source: SoldDemoSource, *, sequence_length: int,
                 batch_size: int, device="cuda", seed: int = 0,
                 converter=None):
        self.source = source
        self.sequence_length = int(sequence_length)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.converter = converter
        # A SOLD sequence of length L covers L observation rows, which is
        # L - 1 transitions in the window planner's terms.
        self.windows = SoldWindows(source, length=int(sequence_length) - 1,
                                   seed=int(seed), stride=1)

    def __len__(self) -> int:
        return len(self.windows)

    def sample(self) -> Dict[str, torch.Tensor]:
        raw = self.windows.raw_batch(self.batch_size)
        return sold_batch(raw, self.source, device=self.device,
                          converter=self.converter)
