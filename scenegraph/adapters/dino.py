"""Frozen DINOv2 patch features, the source of every node's appearance.

One batched forward per environment step covering every camera, under
``inference_mode`` with every parameter frozen. Appearance is therefore a
property of the pixels alone: nothing the agent trains can move it, so the
vector stored in replay stays valid for the whole run and the appearance
reconstruction loss cannot chase a target that drifts underneath it.

The register variant is the default: registers absorb the high-norm artifact
tokens that otherwise contaminate patch-level features, and patches are the
only thing pooled here. ``x_norm_patchtokens`` already excludes both the class
token and the registers.

The segmentation that selects a node's patches exists only inside this step and
is never packed into an observation.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional

import torch
import torch.nn.functional as F

# Feature width per official checkpoint. Patch size is 14 for all of them.
_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
    "dinov2_vits14_reg": 384,
    "dinov2_vitb14_reg": 768,
    "dinov2_vitl14_reg": 1024,
    "dinov2_vitg14_reg": 1536,
}
_PATCH = 14
_HUB = "facebookresearch/dinov2"

# DINOv2 was trained on ImageNet-normalised inputs.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _checksum(model) -> str:
    """SHA-256 over the frozen weights.

    Hashing the loaded tensors rather than a file pins the exact checkpoint a
    run trained against whether it came from the hub cache or a local path.
    """
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().to(torch.float32).numpy().tobytes())
    return h.hexdigest()


class DinoFeatures:
    """Frozen DINOv2 encoder plus masked average pooling over its patch grid."""

    def __init__(
        self,
        name: str = "dinov2_vits14_reg",
        *,
        res: int = 112,
        device: Optional[str] = None,
        weights_path: Optional[str] = None,
    ):
        if name not in _DIMS:
            raise ValueError(
                f"unknown dino model {name!r}; expected one of {sorted(_DIMS)}"
            )
        if res % _PATCH:
            raise ValueError(
                f"dino_res={res} is not a multiple of the patch size {_PATCH}"
            )
        self.name = name
        self.res = int(res)
        self.dim = _DIMS[name]
        self.grid = self.res // _PATCH
        self.patches = self.grid * self.grid
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))

        model = torch.hub.load(
            _HUB, name, pretrained=weights_path is None, trust_repo=True)
        if weights_path:
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.checksum = _checksum(model)
        self._model = model.to(self.device)

        shape = lambda v: torch.tensor(v, device=self.device).view(1, 3, 1, 1)
        self._mean, self._std = shape(_MEAN), shape(_STD)
        print(f"[graph] frozen appearance encoder: {self.metadata}", flush=True)

    @property
    def metadata(self) -> Dict[str, object]:
        """Run/replay pin: features are only comparable across runs that share
        every one of these."""
        return dict(
            model=self.name, res=self.res, patch=_PATCH,
            grid=self.grid, dim=self.dim, checksum=self.checksum)

    @torch.inference_mode()
    def patch_tokens(self, rgb: torch.Tensor) -> torch.Tensor:
        """uint8 ``[B, C, H, W, 3]`` -> patch tokens ``[B, C, P, dim]``.

        Cameras ride along in the batch dimension so the whole step costs one
        forward.
        """
        B, C = rgb.shape[:2]
        x = rgb.reshape((B * C, *rgb.shape[2:]))
        x = x.permute(0, 3, 1, 2).to(self.device, torch.float32) / 255.0
        if x.shape[-2:] != (self.res, self.res):
            x = F.interpolate(
                x, (self.res, self.res), mode="bilinear",
                align_corners=False, antialias=True)
        x = (x - self._mean) / self._std
        out = self._model.forward_features(x)["x_norm_patchtokens"].float()
        return out.reshape(B, C, self.patches, self.dim)

    @torch.inference_mode()
    def pool(self, tokens: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Masked average pool of ``[B, C, P, dim]`` under ``[B, C, N, P]``.

        A node with no patches in a camera pools to exactly zero, which is what
        the model reads back as "appearance unknown" for that camera; L2
        normalisation leaves that row at zero rather than blowing it up.
        """
        B, C, N = weights.shape[:3]
        t = tokens.reshape(B * C, self.patches, self.dim)
        w = weights.reshape(B * C, N, self.patches).to(t.device, t.dtype)
        feat = torch.bmm(w, t) / w.sum(-1, keepdim=True).clamp(min=1e-6)
        return F.normalize(feat, dim=-1).reshape(B, C, N, self.dim)
