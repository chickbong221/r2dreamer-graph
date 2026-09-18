"""Demonstration frames into each backend's own image contract, once.

The collected demonstrations store one ``uint8`` array per camera, shaped
``(T, H, W, 3)`` at whatever resolution the collector recorded -- 112x112 by
default. Neither backend reads that:

============  ==============================================================
TD-MPC2       one ``uint8`` tensor ``(T, 3 * cameras, H, W)``. ``layers.conv``
              asserts ``H == W`` and ``H in {64, 128}``, and its first two
              modules are ``ShiftAug`` then ``PixelPreprocess``, which is where
              ``/255 - 0.5`` happens. Anything divided here would be divided
              twice.
SOLD          one ``uint8`` tensor ``(T, 3, H, W)`` from a single camera, at
              the resolution the SAVi encoder's positional-embedding grid was
              built for. ``train_sold`` divides by 255 itself.
============  ==============================================================

So this module changes layout, camera count and resolution and **nothing
else**. Both backends receive bytes and do their own scaling; that is the
single rule that keeps a frame from being normalized twice.

Resizing
--------

The recorded resolution and the backend's resolution rarely match, and the
same conversion has to apply to a stored demonstration and to a live
observation or the policy sees two different pictures of the same scene. There
is therefore one function, :func:`resize_uint8`, used by the replay path and
the rollout path alike, and the target size is recorded in the checkpoint.

It never writes into its input. ``PixelPreprocess.forward`` is ``x.div_(255.)``
-- in place -- and it is only safe because ``ShiftAug`` copies first with
``x.float()``. An array handed out of here that aliased replay storage would
be a corruption waiting for the first path that skips the copy.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

IMAGE_PREFIX = "image_"


def image_keys(batch: Mapping[str, Any]) -> Tuple[str, ...]:
    """Camera keys in a stable order. Sorted, so two runs agree."""
    return tuple(sorted(k for k in batch if k.startswith(IMAGE_PREFIX)))


def resize_uint8(images, size: Tuple[int, int]):
    """``(..., C, H, W)`` bytes to ``(..., C, h, w)`` bytes, out of place.

    Bilinear with antialiasing, which is what makes a downscale from 112 to 64
    keep the thin geometry these tasks turn on. Torch rather than PIL so the
    same call works on a batch that is already on the device, and so the stored
    and live paths cannot drift into two different resamplers.
    """
    import torch
    import torch.nn.functional as F

    tensor = images if isinstance(images, torch.Tensor) else torch.as_tensor(
        np.asarray(images))
    height, width = int(size[0]), int(size[1])
    if tensor.shape[-2] == height and tensor.shape[-1] == width:
        # Still a copy: the caller is entitled to mutate what it gets back.
        return tensor.clone()
    lead = tensor.shape[:-3]
    flat = tensor.reshape(-1, *tensor.shape[-3:]).float()
    resized = F.interpolate(flat, size=(height, width), mode="bilinear",
                            align_corners=False, antialias=True)
    return resized.round().clamp_(0, 255).to(torch.uint8).reshape(
        *lead, tensor.shape[-3], height, width)


def to_chw(images):
    """``(..., H, W, C)`` to ``(..., C, H, W)`` without copying the data."""
    import torch

    tensor = images if isinstance(images, torch.Tensor) else torch.as_tensor(
        np.asarray(images))
    if tensor.dim() < 3:
        raise ValueError(
            f"expected an image with at least (H, W, C), got {tuple(tensor.shape)}")
    return tensor.movedim(-1, -3)


class ImageContract:
    """How one backend wants the recorded cameras presented.

    ``cameras`` is the ordered list of dataset keys to use. TD-MPC2 stacks all
    of them on the channel axis, which is what ``FlattenRGBDObservationWrapper``
    does upstream. SOLD's SAVi is a single-view autoencoder with a
    three-channel encoder, so it takes one; which one is a decision and is
    named rather than defaulted to "whichever sorts first" silently.
    """

    def __init__(self, cameras: Sequence[str], size: Tuple[int, int],
                 *, max_cameras: Optional[int] = None, backend: str = ""):
        self.cameras = tuple(str(c) for c in cameras)
        self.size = (int(size[0]), int(size[1]))
        self.backend = str(backend)
        if not self.cameras:
            raise ValueError(
                f"{self.backend or 'this backend'} needs at least one camera "
                "key from the dataset")
        if max_cameras is not None and len(self.cameras) > int(max_cameras):
            raise ValueError(
                f"{self.backend} takes at most {max_cameras} camera(s) and was "
                f"given {list(self.cameras)}. Pick one explicitly rather than "
                "letting the extra views be dropped silently.")

    @property
    def channels(self) -> int:
        return 3 * len(self.cameras)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.channels, self.size[0], self.size[1])

    def apply(self, batch: Mapping[str, Any]):
        """Stack, transpose and resize the cameras this contract names.

        Accepts a window ``(..., H, W, 3)`` per key and returns one ``uint8``
        tensor ``(..., 3 * cameras, h, w)``.
        """
        import torch

        missing = [key for key in self.cameras if key not in batch]
        if missing:
            raise KeyError(
                f"{self.backend or 'this backend'} was configured for cameras "
                f"{list(self.cameras)} but the batch has "
                f"{sorted(k for k in batch if k.startswith(IMAGE_PREFIX))}; "
                f"{missing} are absent")
        frames = []
        for key in self.cameras:
            value = batch[key]
            tensor = value if isinstance(value, torch.Tensor) else \
                torch.as_tensor(np.asarray(value))
            if tensor.dtype != torch.uint8:
                raise TypeError(
                    f"{key} arrived as {tensor.dtype}; the backends do their "
                    "own /255, so images cross this boundary as bytes. A "
                    "float here is a frame that has already been scaled once.")
            frames.append(resize_uint8(to_chw(tensor), self.size))
        return torch.cat(frames, dim=-3)

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.backend, "cameras": list(self.cameras),
                "size": list(self.size), "channels": self.channels}


def contract_for(metadata: Mapping[str, Any], *, backend: str,
                 size: Tuple[int, int], cameras: Optional[Sequence[str]] = None,
                 max_cameras: Optional[int] = None) -> ImageContract:
    """Build a contract from the dataset's own recorded camera set."""
    recorded = [str(v) for v in
                (dict(metadata.get("camera_keys") or {})).values()]
    recorded = sorted(recorded)
    if not recorded:
        raise SystemExit(
            "the dataset records no camera_keys; there is nothing for "
            f"{backend} to encode")
    chosen = list(cameras) if cameras else recorded
    unknown = [c for c in chosen if c not in recorded]
    if unknown:
        raise SystemExit(
            f"{backend}: cameras {unknown} are not in this dataset, which "
            f"recorded {recorded}")
    if max_cameras is not None and len(chosen) > int(max_cameras):
        chosen = chosen[: int(max_cameras)]
    return ImageContract(chosen, size, max_cameras=max_cameras, backend=backend)
