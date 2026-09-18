"""Shared fixtures for the sim_vla server tests.

Every module here needs torch, and several need a GPU, the pretrained
checkpoint or ManiSkill. A missing dependency is reported as a skip with the
reason named -- never as a pass -- and ``run_tests.sh`` treats a stage that
skipped everything as incomplete rather than successful.

The world models built here are deliberately tiny. These are interface and
gradient tests: what they check is that the arms construct, that gradients
reach what they should, that a frozen module stays frozen, and that a
checkpoint round-trips. Whether the model learns anything is a training
question and training is not run from the test suite.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def require(module: str) -> Any:
    """Import or skip, naming the dependency that was missing."""
    try:
        return __import__(module)
    except Exception as exc:                               # noqa: BLE001
        raise unittest.SkipTest(f"{module} unavailable: {exc}")


def require_torch():
    return require("torch")


def require_cuda():
    torch = require_torch()
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no CUDA device")
    return torch


def small_model_config(graph_enabled: bool, *, n_max: int = 4, e_max: int = 16):
    """A model config small enough to build in a second on CPU."""
    from sim_vla.config import load_config
    from sim_vla.models.model_config import load_model_config

    cfg = load_config("pickcube", "graph" if graph_enabled else "dreamer")
    model = load_model_config(cfg)
    model.rssm.deter = 64
    model.rssm.hidden = 64
    model.rssm.stoch = 4
    model.rssm.discrete = 4
    model.rssm.blocks = 1
    model.graph.n_max = n_max
    model.graph.e_max = e_max
    model.graph.semantic_dim = 16
    model.graph.simple_units = 32
    model.graph.decoder_units = 32
    model.device = "cpu"
    return cfg, model


def fake_batch(*, graph_enabled: bool, batch: int = 2, steps: int = 6,
               image: int = 16, proprio: int = 9, action: int = 8,
               n_max: int = 4, e_max: int = 16) -> Dict[str, Any]:
    """A window in the loader's layout, as tensors."""
    torch = require_torch()
    rng = np.random.default_rng(0)
    out: Dict[str, Any] = {
        "image_base": torch.as_tensor(
            rng.random((batch, steps, image, image, 3), dtype=np.float32)),
        "proprio": torch.as_tensor(
            rng.standard_normal((batch, steps, proprio)).astype(np.float32)),
        "action": torch.as_tensor(
            rng.standard_normal((batch, steps, action)).astype(np.float32)),
        "reward": torch.as_tensor(
            rng.standard_normal((batch, steps, 1)).astype(np.float32)),
        "is_first": torch.zeros((batch, steps), dtype=torch.bool),
        "is_terminal": torch.zeros((batch, steps), dtype=torch.bool),
        "loss_mask": torch.ones((batch, steps), dtype=torch.bool),
    }
    out["is_first"][:, 0] = True
    if graph_enabled:
        for key, high, shape in (
            ("graph_node_ent", 4, (batch, steps, n_max)),
            ("graph_node_target", 2, (batch, steps, n_max)),
            ("graph_edge_src", n_max, (batch, steps, e_max)),
            ("graph_edge_dst", n_max, (batch, steps, e_max)),
            ("graph_edge_rel", 3, (batch, steps, e_max)),
            ("graph_edge_abs", 4, (batch, steps, e_max)),
            ("graph_edge_temp", 3, (batch, steps, e_max)),
        ):
            out[key] = torch.as_tensor(
                rng.integers(0, high, shape).astype(np.int64))
        out["graph_node_bbox"] = torch.as_tensor(
            rng.random((batch, steps, n_max, 4), dtype=np.float32))
        out["graph_node_centroid"] = torch.as_tensor(
            rng.random((batch, steps, n_max, 3), dtype=np.float32))
    return out


def obs_shapes(batch: Dict[str, Any]) -> Dict[str, tuple]:
    return {k: tuple(v.shape[2:]) for k, v in batch.items() if v.dim() >= 2}


class DummyExpert:
    """A velocity field with a known answer, standing in for SmolVLA.

    Lets the flow loss, the sampler and the imagined actor update be tested for
    gradient flow and shape without the pretrained checkpoint. It does not
    stand in for the checkpoint in the test that loads it.
    """

    def __init__(self, token_dim: int, action_dim: int):
        torch = require_torch()
        self.linear = torch.nn.Linear(token_dim + action_dim + 1, action_dim)

    def __call__(self, x_t, t, cond):
        torch = require_torch()
        token = cond["state_token"].squeeze(-2)
        batch, chunk, dim = x_t.shape
        token = token.unsqueeze(1).expand(batch, chunk, token.shape[-1])
        time = t.reshape(batch, 1, 1).expand(batch, chunk, 1)
        return self.linear(torch.cat([x_t, token, time], dim=-1))

    def parameters(self):
        return self.linear.parameters()
