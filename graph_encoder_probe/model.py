"""The encoder wired straight into the decoder, and nothing else.

``GraphEncoder`` normally hands its pooled token to the RSSM, which is where the
token acquires a dynamics loss, a prior, a KL and a policy. None of that is in
this file. The token goes to ``SimpleGraphDecoder`` as its semantic input, the
decoder asks for the graph back, and the only gradient either module sees is
reconstruction.

Both classes are imported unchanged from ``graph.py``. What this module owns is
the config they read -- the widths the experiment fixes, and the vocabulary
sizes and camera count, which come from the collected data rather than from a
number written down twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Mapping, Optional

import torch
from torch import nn

from graph import GraphEncoder, GraphEncoding, SimpleGraphDecoder
from scenegraph.adapters.graph_vocab import (
    build_absolute_vocab,
    build_relation_vocab,
    build_temporal_vocab,
)

# The four reconstruction terms. Order fixed so the logged columns are stable.
LOSS_KEYS = ("node", "nodetgt", "relabs", "reltemp")


def graph_config(model_cfg: Mapping, dataset_meta: Mapping) -> SimpleNamespace:
    """The config object ``GraphEncoder`` and ``SimpleGraphDecoder`` read.

    Widths come from the experiment's config; ``n_cams`` and ``entity_vocab``
    come from the cache, because both are properties of the task that was
    collected. The three label vocabularies are derived from the shared tables
    rather than copied -- the decoder asserts its own sizes against those tables
    at construction, so a number written here could only ever disagree.
    """
    sizes = dict(dataset_meta.get("vocab_sizes") or {})
    for key in ("entity", "relation", "absolute", "temporal"):
        if key not in sizes:
            raise KeyError(f"dataset meta is missing vocab_sizes.{key}")
    n_cams = int(dataset_meta.get("n_cams", 0))
    if n_cams <= 0:
        raise ValueError(f"dataset meta reports n_cams={n_cams}")

    derived = {
        "relation": len(build_relation_vocab()),
        "absolute": len(build_absolute_vocab()),
        "temporal": len(build_temporal_vocab()),
    }
    disagreed = {k: (sizes[k], v) for k, v in derived.items() if int(sizes[k]) != v}
    if disagreed:
        raise ValueError(
            "the cache was collected against different label vocabularies than "
            f"this checkout defines: {disagreed} (cache, checkout). Re-collect, "
            "or check out the revision the cache records."
        )

    units = int(model_cfg["simple_units"])
    return SimpleNamespace(
        simple_units=units,
        semantic_dim=units,
        decoder_units=int(model_cfg["decoder_units"]),
        layers=int(model_cfg["layers"]),
        embed=int(model_cfg["embed"]),
        bbox_query_dim=int(model_cfg["bbox_query_dim"]),
        bbox_beta=float(model_cfg["bbox_beta"]),
        reverse_edges=bool(model_cfg["reverse_edges"]),
        act=str(model_cfg["act"]),
        centroid_origin=list(model_cfg["centroid_origin"]),
        centroid_scale=model_cfg["centroid_scale"],
        n_cams=n_cams,
        entity_vocab=int(sizes["entity"]),
        n_rel=derived["relation"],
        n_abs=derived["absolute"],
        n_temp=derived["temporal"],
    )


@dataclass
class ProbeOutput:
    """One forward pass: the measured token, the four terms, and their sum."""

    token: torch.Tensor
    total: torch.Tensor
    losses: dict[str, torch.Tensor]
    metrics: dict[str, torch.Tensor]


class GraphProbe(nn.Module):
    """``G -> GraphEncoder -> z -> SimpleGraphDecoder -> G``.

    ``z`` is the pooled token exactly as the RSSM would receive it. It is handed
    to the decoder unprojected and unnormalised: any adapter in between would be
    a third module whose gradients also shape the thing being measured.
    """

    def __init__(self, config, loss_scales: Optional[Mapping[str, float]] = None):
        super().__init__()
        self.config = config
        self.encoder = GraphEncoder(config)
        self.decoder = SimpleGraphDecoder(config, semantic_dim=int(config.simple_units))
        scales = dict(loss_scales or {})
        self.loss_scales = {key: float(scales.get(key, 1.0)) for key in LOSS_KEYS}

    @property
    def token_dim(self) -> int:
        return int(self.config.simple_units)

    def encode(self, batch: Mapping[str, torch.Tensor]) -> GraphEncoding:
        return self.encoder(batch)

    def token(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The pooled readout alone -- what every distance in this experiment is
        measured on."""
        return self.encoder(batch).token

    def forward(self, batch: Mapping[str, torch.Tensor]) -> ProbeOutput:
        encoding = self.encoder(batch)
        graphs = int(encoding.token.shape[0])
        # Every frame in this pool is a real observation, so nothing is masked
        # out; the argument exists because replay hands the decoder padded time
        # steps, which this experiment does not have.
        step_valid = torch.ones(graphs, dtype=torch.bool, device=encoding.token.device)
        losses, metrics = self.decoder(encoding.token, encoding.compact, step_valid)
        missing = [key for key in LOSS_KEYS if key not in losses]
        if missing:
            raise KeyError(f"the decoder did not emit {missing}; it returned {sorted(losses)}")
        total = sum(losses[key] * self.loss_scales[key] for key in LOSS_KEYS)
        return ProbeOutput(encoding.token, total, dict(losses), dict(metrics))


def build_model(model_cfg: Mapping, dataset_meta: Mapping, *, loss_scales=None, device=None):
    config = graph_config(model_cfg, dataset_meta)
    model = GraphProbe(config, loss_scales)
    if device is not None:
        model = model.to(device)
    return model


def save_checkpoint(path: str, model: GraphProbe, extra: Optional[Mapping] = None) -> str:
    """State dict plus the config it was built from.

    The config travels with the weights because rebuilding the model is what a
    reload does, and a checkpoint that needs a yaml file to be interpretable is
    one the yaml file can drift away from.
    """
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": vars(model.config),
            "loss_scales": dict(model.loss_scales),
            "extra": dict(extra or {}),
        },
        path,
    )
    return path


def load_checkpoint(path: str, *, device=None) -> tuple[GraphProbe, dict]:
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    config = SimpleNamespace(**payload["config"])
    model = GraphProbe(config, payload.get("loss_scales"))
    model.load_state_dict(payload["state_dict"])
    if device is not None:
        model = model.to(device)
    return model, dict(payload.get("extra") or {})
