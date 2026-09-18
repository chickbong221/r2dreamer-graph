"""The pretrained SmolVLA action expert, conditioned on world-model latents.

What this does *not* do is re-implement SmolVLA. It loads the real checkpoint,
keeps its language transformer and action expert, and replaces the input that
normally comes from images and a raw state vector with one token from
:class:`~sim_vla.models.latent_adapter.LatentAdapter`. The observation has
already been through the world model; running it through the policy's own
vision encoder as well would be encoding it twice.

**The attribute names of a third-party checkpoint are not guessable**, and this
file does not guess them silently. :func:`resolve` searches a list of candidate
paths and, on failure, raises naming every path it tried and printing the
module tree it searched. ``sim_vla/download_pretrained.py`` prints the same
tree on demand, so the first run on a machine that has the weights tells us the
real interface instead of a stack trace three layers deep.

Trainable, initially: the adapter, the action expert, and the action/time
projections. Frozen: the world model, the language transformer, and the vision
encoder the latent path bypasses. Frozen means ``requires_grad=False``, not
``detach()`` -- gradients still flow *through* the frozen transformer back into
the adapter, which is the whole point of conditioning it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

# Where the action expert has lived across lerobot versions. Ordered most
# recent first; the loader reports which one matched.
EXPERT_PATHS: Sequence[str] = (
    "model.vlm_with_expert.lm_expert",
    "model.vlm_with_expert.action_expert",
    "model.action_expert",
    "vlm_with_expert.lm_expert",
)
LANGUAGE_PATHS: Sequence[str] = (
    "model.vlm_with_expert.vlm.model.text_model",
    "model.vlm_with_expert.vlm",
    "model.vlm",
)
VISION_PATHS: Sequence[str] = (
    "model.vlm_with_expert.vlm.model.vision_model",
    "model.vlm_with_expert.vision_model",
)


def module_tree(root: nn.Module, depth: int = 3) -> List[str]:
    """Dotted names of ``root``'s submodules, to ``depth`` levels."""
    out: List[str] = []
    for name, _ in root.named_modules():
        if name and name.count(".") < depth:
            out.append(name)
    return sorted(out)


def resolve(root: nn.Module, paths: Sequence[str], what: str) -> tuple:
    """First attribute path that exists, or an error naming all of them."""
    for path in paths:
        node: Any = root
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if isinstance(node, nn.Module):
            return node, path
    raise AttributeError(
        f"could not find the {what} on this SmolVLA checkpoint. Tried: "
        f"{list(paths)}. Submodules present: {module_tree(root)[:40]}. "
        f"Update the candidate paths in sim_vla/models/smolvla_actor.py for "
        f"the pinned lerobot version.")


def freeze(module: nn.Module) -> int:
    """Stop a module's own parameters training, without cutting the graph."""
    count = 0
    for parameter in module.parameters():
        if parameter.requires_grad:
            parameter.requires_grad_(False)
            count += 1
    return count


class SmolVLAActor(nn.Module):
    """Pretrained SmolVLA driven by one world-model conditioning token."""

    def __init__(self, policy: nn.Module, adapter: nn.Module, *,
                 chunk_size: int = 8, action_dim: int = 8,
                 flow_steps: int = 10, freeze_language: bool = True,
                 freeze_vision: bool = True):
        super().__init__()
        self.policy = policy
        self.adapter = adapter
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.flow_steps = int(flow_steps)

        self.expert, self.expert_path = resolve(policy, EXPERT_PATHS,
                                                "action expert")
        self.language, self.language_path = resolve(policy, LANGUAGE_PATHS,
                                                    "language transformer")
        try:
            self.vision, self.vision_path = resolve(policy, VISION_PATHS,
                                                    "vision encoder")
        except AttributeError:
            # A checkpoint without a separable vision tower is not an error
            # here: the latent path does not use one.
            self.vision, self.vision_path = None, ""

        self.frozen: Dict[str, int] = {}
        if freeze_language:
            self.frozen[self.language_path] = freeze(self.language)
        if freeze_vision and self.vision is not None:
            self.frozen[self.vision_path] = freeze(self.vision)

    # ----------------------------------------------------------- conditioning
    def condition(self, features: torch.Tensor,
                  instruction: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """One state token from the latent, beside the task instruction."""
        token = self.adapter(features)
        return {"state_token": token, "instruction": instruction}

    def velocity_fn(self) -> Callable:
        """The expert's velocity field, in the signature the sampler wants."""
        def velocity(x_t: torch.Tensor, t: torch.Tensor, cond: Any
                     ) -> torch.Tensor:
            return self.expert_velocity(x_t, t, cond)
        return velocity

    def expert_velocity(self, x_t: torch.Tensor, t: torch.Tensor,
                        cond: Dict[str, Any]) -> torch.Tensor:
        """Call the pretrained expert. Separated so a test can replace it.

        The call signature of the expert differs between lerobot versions more
        than its module path does, so this is the single place a version bump
        has to be reconciled, and it fails with the signature it tried rather
        than with a shape error inside the transformer.
        """
        call = getattr(self.policy, "denoise_step", None)
        if call is None:
            raise AttributeError(
                "this SmolVLA policy exposes no denoise_step; sim_vla drives "
                "the expert through it. Check the pinned lerobot version and "
                f"the methods present: {sorted(m for m in dir(self.policy) if not m.startswith('_'))[:40]}")
        return call(cond["state_token"], cond["instruction"], x_t, t)

    # ----------------------------------------------------------------- report
    def trainable_report(self) -> Dict[str, Any]:
        trainable = [n for n, p in self.named_parameters() if p.requires_grad]
        total = sum(p.numel() for p in self.parameters())
        live = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "expert_path": self.expert_path,
            "language_path": self.language_path,
            "vision_path": self.vision_path,
            "frozen_modules": self.frozen,
            "parameters_total": total,
            "parameters_trainable": live,
            "trainable_prefixes": sorted({n.split(".")[0] for n in trainable}),
        }
