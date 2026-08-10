"""Compact real-edge scene graph encoder and auxiliary decoder.

Replay keeps fixed-width arrays because they are simple to store and sample.
This module removes padding immediately after sampling: node slots stay dense
(there are only ten), while every edge MLP and aggregation runs only on rows
whose relation id is non-zero. The graph path is intentionally plain PyTorch
so this repository has no runtime dependency on ReLDreamer or a GNN package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from tools import weight_init_


GRAPH_KEYS = (
    "graph_node_ent",
    "graph_node_app",
    "graph_node_bbox",
    "graph_node_target",
    "graph_edge_src",
    "graph_edge_dst",
    "graph_edge_rel",
    "graph_edge_abs",
    "graph_edge_temp",
)


def graph_from(data: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return and validate the graph observation subset."""
    missing = [key for key in GRAPH_KEYS if key not in data]
    if missing:
        raise KeyError(f"graph.enabled requires observation keys: {missing}")
    return {key: data[key] for key in GRAPH_KEYS}


@dataclass
class CompactGraph:
    """Dense small node table plus a compact list of real directed facts."""

    batch_shape: tuple[int, ...]
    num_nodes: int
    node_ent: torch.Tensor
    node_app: torch.Tensor
    node_bbox: torch.Tensor
    node_target: torch.Tensor
    node_valid: torch.Tensor
    appearance_known: torch.Tensor
    camera_visible: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_rel: torch.Tensor
    edge_abs: torch.Tensor
    edge_temp: torch.Tensor
    edge_graph: torch.Tensor

    @property
    def graph_count(self) -> int:
        return reduce(mul, self.batch_shape, 1)

    @property
    def edge_count(self) -> int:
        return int(self.edge_rel.numel())


@dataclass
class GraphEncoding:
    nodes: torch.Tensor
    token: torch.Tensor
    compact: CompactGraph


def compact_graph(graph: Mapping[str, torch.Tensor]) -> CompactGraph:
    """Strip padded edge rows on device without creating Python graph objects."""
    graph = graph_from(graph)
    ent = graph["graph_node_ent"]
    if ent.ndim < 2:
        raise ValueError(f"graph_node_ent must have batch and node axes, got {ent.shape}")
    batch_shape = tuple(ent.shape[:-1])
    graph_count = reduce(mul, batch_shape, 1)
    num_nodes = int(ent.shape[-1])

    node_ent = ent.reshape(graph_count, num_nodes).long()
    app_tail = graph["graph_node_app"].shape[-2:]
    bbox_tail = graph["graph_node_bbox"].shape[-2:]
    node_app = graph["graph_node_app"].reshape(graph_count, num_nodes, *app_tail)
    node_bbox = graph["graph_node_bbox"].reshape(graph_count, num_nodes, *bbox_tail)
    node_target = graph["graph_node_target"].reshape(graph_count, num_nodes).long()
    node_valid = node_ent.ne(0)
    appearance_known = node_app.float().abs().sum(-1).ne(0)
    camera_visible = (
        (node_bbox[..., 1] > node_bbox[..., 0])
        & (node_bbox[..., 3] > node_bbox[..., 2])
    )

    rel = graph["graph_edge_rel"]
    edge_width = int(rel.shape[-1])
    rel = rel.reshape(graph_count, edge_width).long()
    edge_valid = rel.ne(0)
    edge_graph = torch.arange(graph_count, device=rel.device)[:, None].expand(-1, edge_width)[edge_valid]

    def select(key: str) -> torch.Tensor:
        return graph[key].reshape(graph_count, edge_width)[edge_valid].long()

    local_src = select("graph_edge_src")
    local_dst = select("graph_edge_dst")
    offset = edge_graph * num_nodes
    return CompactGraph(
        batch_shape=batch_shape,
        num_nodes=num_nodes,
        node_ent=node_ent,
        node_app=node_app,
        node_bbox=node_bbox,
        node_target=node_target,
        node_valid=node_valid,
        appearance_known=appearance_known,
        camera_visible=camera_visible,
        edge_src=local_src + offset,
        edge_dst=local_dst + offset,
        edge_rel=rel[edge_valid],
        edge_abs=select("graph_edge_abs"),
        edge_temp=select("graph_edge_temp"),
        edge_graph=edge_graph,
    )


class GraphMLP(nn.Module):
    def __init__(self, inp: int, out: int, act: str):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, out),
            nn.RMSNorm(out, eps=1e-4, dtype=torch.float32),
            getattr(nn, act)(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphEncoder(nn.Module):
    """Edge-feature GNN with real-edge mean aggregation and attention pooling."""

    def __init__(self, config):
        super().__init__()
        self.units = int(config.units)
        self.layers = int(config.layers)
        self.n_cams = int(config.n_cams)
        self.app_dim = int(config.app_dim)
        self.embed_dim = int(config.embed)
        self.reverse_edges = bool(config.reverse_edges)

        self.app_proj = nn.ModuleList(
            nn.Linear(self.app_dim, int(config.app)) for _ in range(self.n_cams)
        )
        self.bbox_proj = nn.ModuleList(
            nn.Linear(4, int(config.bbox)) for _ in range(self.n_cams)
        )
        self.entity = nn.Embedding(int(config.entity_vocab), self.embed_dim)
        self.target = nn.Embedding(2, self.embed_dim)
        node_in = self.n_cams * (int(config.app) + int(config.bbox)) + 2 * self.embed_dim
        self.node = GraphMLP(node_in, self.units, str(config.act))

        self.relation = nn.Embedding(int(config.n_rel), self.embed_dim)
        self.absolute = nn.Embedding(int(config.n_abs), self.embed_dim)
        self.temporal = nn.Embedding(int(config.n_temp), self.embed_dim)
        self.fact = GraphMLP(3 * self.embed_dim, self.units, str(config.act))
        self.messages = nn.ModuleList(
            GraphMLP(2 * self.units + 1, self.units, str(config.act))
            for _ in range(self.layers)
        )
        self.updates = nn.ModuleList(
            GraphMLP(2 * self.units, self.units, str(config.act))
            for _ in range(self.layers)
        )
        self.query = nn.Parameter(torch.empty(self.units))
        self.key = nn.Linear(self.units, self.units)
        self.value = nn.Linear(self.units, self.units)
        self.out = nn.Linear(self.units, self.units)
        self.apply(weight_init_)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, graph: Mapping[str, torch.Tensor]) -> GraphEncoding:
        compact = compact_graph(graph)
        valid = compact.node_valid
        # Replay stores these arrays as float16. Normalize the storage boundary
        # to float32; the shared acting/training autocast policy selects compute
        # dtype for the learned projections.
        node_app = compact.node_app.to(self.app_proj[0].weight.dtype)
        node_bbox = compact.node_bbox.to(self.bbox_proj[0].weight.dtype)
        parts = []
        for camera in range(self.n_cams):
            app = self.app_proj[camera](node_app[..., camera, :])
            parts.append(app * compact.appearance_known[..., camera, None])
            box = self.bbox_proj[camera](node_bbox[..., camera, :])
            parts.append(box * compact.camera_visible[..., camera, None])
        parts.extend((self.entity(compact.node_ent), self.target(compact.node_target)))
        nodes = self.node(torch.cat(parts, -1)) * valid[..., None]

        temporal = self.temporal(compact.edge_temp)
        temporal = temporal * compact.edge_temp.ne(0)[..., None]
        facts = self.fact(
            torch.cat(
                [self.relation(compact.edge_rel), self.absolute(compact.edge_abs), temporal],
                -1,
            )
        )

        src, dst = compact.edge_src, compact.edge_dst
        direction = torch.zeros((compact.edge_count, 1), device=nodes.device, dtype=nodes.dtype)
        if self.reverse_edges:
            src, dst = torch.cat([src, dst]), torch.cat([dst, src])
            facts = torch.cat([facts, facts], 0)
            direction = torch.cat([direction, torch.ones_like(direction)], 0)

        flat = nodes.reshape(-1, self.units)
        counts = torch.zeros(flat.shape[0], device=flat.device, dtype=torch.float32)
        counts = counts.index_add(
            0, dst, torch.ones(dst.shape[0], device=flat.device, dtype=torch.float32)
        )
        for message, update in zip(self.messages, self.updates):
            msg = message(torch.cat([flat.index_select(0, src), facts, direction], -1))
            total = torch.zeros_like(flat).index_add(0, dst, msg)
            agg = total / counts.clamp_min(1).to(total.dtype)[:, None]
            flat = update(torch.cat([flat, agg], -1))
            flat = flat * valid.reshape(-1, 1)
        nodes = flat.reshape(compact.graph_count, compact.num_nodes, self.units)

        score = (self.key(nodes) * self.query).sum(-1) / math.sqrt(self.units)
        # Apply masks in float32: -1e9 is outside float16's finite range and
        # mixed-precision autocast can otherwise fail before softmax.
        score = score.float().masked_fill(~valid, -1e9)
        attention = torch.softmax(score, -1).to(nodes.dtype) * valid
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-6)
        token = self.out((attention[..., None] * self.value(nodes)).sum(1))
        token = token * valid.any(-1, keepdim=True)
        nodes = nodes.reshape(*compact.batch_shape, compact.num_nodes, self.units)
        token = token.reshape(*compact.batch_shape, self.units)
        return GraphEncoding(nodes, token, compact)


def _relation_masks(n_rel: int, n_abs: int) -> torch.Tensor:
    """Legal absolute labels for the fixed ten-relation vocabulary."""
    if n_rel != 11 or n_abs != 17:
        raise ValueError("the graph decoder expects relation vocab 11 and absolute vocab 17")
    mask = torch.zeros(n_rel, n_abs, dtype=torch.bool)
    mask[1:5, 1:3] = True
    mask[5, 3:8] = True
    mask[6, 8:13] = True
    mask[7:11, 13:17] = True
    return mask


class GraphDecoder(nn.Module):
    """Auxiliary graph heads with the same per-frame weighting as the JAX model."""

    def __init__(self, config, sem_dim: int):
        super().__init__()
        self.units = int(config.units)
        self.app_dim = int(config.app_dim)
        self.n_cams = int(config.n_cams)
        self.entity_vocab = int(config.entity_vocab)
        self.n_abs = int(config.n_abs)
        self.n_temp = int(config.n_temp)
        self.bbox_beta = float(config.bbox_beta)

        self.app = nn.Linear(self.units, self.n_cams * self.app_dim)
        self.bbox = nn.Linear(self.units, self.n_cams * 4)
        self.visibility = nn.Linear(self.units, self.n_cams)
        self.reltype = nn.Embedding(int(config.n_rel), int(config.embed))
        self.pair = GraphMLP(
            2 * self.units + int(config.embed), self.units, str(config.act)
        )
        self.abs_head = nn.Linear(self.units, self.n_abs)
        self.temp_head = nn.Linear(self.units, self.n_temp)
        self.target_in = GraphMLP(sem_dim, self.units, str(config.act))
        self.target_head = nn.Linear(self.units, self.entity_vocab)
        self.register_buffer(
            "abs_valid", _relation_masks(int(config.n_rel), self.n_abs), persistent=False
        )
        self.apply(weight_init_)

    def forward(
        self,
        encoded: GraphEncoding,
        sem: torch.Tensor,
        step_valid: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        compact = encoded.compact
        graph_count = compact.graph_count
        nodes = encoded.nodes.reshape(graph_count, compact.num_nodes, self.units)
        step = step_valid.reshape(graph_count).bool()
        valid = compact.node_valid & step[:, None]
        known = compact.appearance_known
        visible = compact.camera_visible
        losses: dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor] = {}

        pred_app = self.app(nodes).reshape(
            graph_count, compact.num_nodes, self.n_cams, self.app_dim
        )
        target_app = compact.node_app.detach()
        cosine = self._cosine(pred_app, target_app)
        app_mask = valid[..., None] & known
        node_app = self._frame_mean(1 - cosine, app_mask).mean()
        metrics["node_app_cos"] = self._masked(cosine, app_mask)
        metrics["node_app_var"] = self._spread(target_app, app_mask)

        pred_bbox = self.bbox(nodes).reshape(
            graph_count, compact.num_nodes, self.n_cams, 4
        )
        bbox_mask = valid[..., None] & visible
        bbox_error = F.smooth_l1_loss(
            pred_bbox.float(), compact.node_bbox.float(), reduction="none", beta=self.bbox_beta
        ).sum(-1)
        node_bbox = self._frame_mean(bbox_error, bbox_mask).mean()
        metrics["node_bbox_iou"] = self._masked(
            self._iou(pred_bbox, compact.node_bbox), bbox_mask
        )

        visibility_logit = self.visibility(nodes)
        visibility_mask = valid[..., None].expand_as(visibility_logit)
        visibility_loss = F.binary_cross_entropy_with_logits(
            visibility_logit.float(), visible.float(), reduction="none"
        )
        node_visibility = self._frame_mean(visibility_loss, visibility_mask).mean()
        metrics["node_vis_acc"] = self._masked(
            visibility_logit.gt(0).eq(visible), visibility_mask
        )
        losses["node"] = (node_app + node_bbox + node_visibility) / 3

        edge_step = step.index_select(0, compact.edge_graph)
        flat_nodes = nodes.reshape(-1, self.units)
        pair = self.pair(
            torch.cat(
                [
                    flat_nodes.index_select(0, compact.edge_src),
                    flat_nodes.index_select(0, compact.edge_dst),
                    self.reltype(compact.edge_rel),
                ],
                -1,
            )
        )
        losses["relabs"], metrics["relabs_acc"] = self._edge_categorical(
            self.abs_head(pair),
            compact.edge_abs,
            compact.edge_graph,
            edge_step,
            self.abs_valid.index_select(0, compact.edge_rel),
            graph_count,
        )
        temp_mask = edge_step & compact.edge_temp.ne(0)
        temp_logits = self.temp_head(pair)
        temp_classes = torch.ones_like(temp_logits, dtype=torch.bool)
        temp_classes[:, 0] = False
        losses["reltemp"], metrics["reltemp_acc"] = self._edge_categorical(
            temp_logits,
            compact.edge_temp,
            compact.edge_graph,
            temp_mask,
            temp_classes,
            graph_count,
        )

        flag = compact.node_target.bool() & valid
        present = flag.any(-1)
        label = (compact.node_target * compact.node_ent).sum(-1)
        safe_label = torch.where(present, label, torch.ones_like(label))
        target_logits = self.target_head(self.target_in(sem.reshape(graph_count, -1)))
        target_classes = torch.arange(self.entity_vocab, device=target_logits.device).ne(0)
        # Cross entropy is intentionally evaluated in float32. Mask there too,
        # because -1e9 cannot be converted to float16 under autocast.
        target_logits = target_logits.float().masked_fill(~target_classes, -1e9)
        target_loss = F.cross_entropy(target_logits, safe_label, reduction="none")
        losses["semtgt"] = (target_loss * present).mean()
        metrics["semtgt_acc"] = self._masked(target_logits.argmax(-1).eq(label), present)
        metrics["semtgt_frac"] = present.float().mean()
        metrics["graph_real_edges"] = torch.as_tensor(
            compact.edge_count / max(graph_count, 1),
            device=nodes.device,
            dtype=torch.float32,
        )
        return losses, metrics

    def _edge_categorical(self, logits, target, frame, mask, classes, graph_count):
        if logits.shape[0] == 0:
            zero = logits.sum() * 0
            return zero, zero.detach()
        logits = logits.float().masked_fill(~classes, -1e9)
        safe_target = torch.where(mask, target, torch.ones_like(target))
        values = F.cross_entropy(logits, safe_target, reduction="none")
        numerator = torch.zeros(graph_count, device=logits.device).index_add(
            0, frame, values * mask
        )
        denominator = torch.zeros(graph_count, device=logits.device).index_add(
            0, frame, mask.float()
        )
        loss = (numerator / denominator.clamp_min(1)).mean()
        accuracy = self._masked(logits.argmax(-1).eq(target), mask)
        return loss, accuracy

    @staticmethod
    def _frame_mean(values, mask):
        axes = tuple(range(1, values.ndim))
        numerator = (values.float() * mask).sum(axes)
        denominator = mask.float().sum(axes).clamp_min(1)
        return numerator / denominator

    @staticmethod
    def _masked(values, mask):
        mask = mask.float()
        return (values.float() * mask).sum() / mask.sum().clamp_min(1)

    @staticmethod
    def _cosine(pred, target):
        pred, target = pred.float(), target.float()
        numerator = (pred * target).sum(-1)
        pred_norm = pred.square().sum(-1).clamp_min(1e-12).sqrt()
        target_norm = target.square().sum(-1).clamp_min(1e-12).sqrt()
        return numerator / (pred_norm * target_norm)

    @staticmethod
    def _iou(pred, target):
        pred, target = pred.float(), target.float()
        px0, px1, py0, py1 = pred.unbind(-1)
        tx0, tx1, ty0, ty1 = target.unbind(-1)
        width = (torch.minimum(px1, tx1) - torch.maximum(px0, tx0)).clamp_min(0)
        height = (torch.minimum(py1, ty1) - torch.maximum(py0, ty0)).clamp_min(0)
        intersection = width * height
        pred_area = (px1 - px0).clamp_min(0) * (py1 - py0).clamp_min(0)
        target_area = (tx1 - tx0).clamp_min(0) * (ty1 - ty0).clamp_min(0)
        return intersection / (pred_area + target_area - intersection).clamp_min(1e-6)

    @staticmethod
    def _spread(target, mask):
        target = target.float()
        weight = mask[..., None].float()
        axes = tuple(range(target.ndim - 1))
        mean = (target * weight).sum(axes, keepdim=True) / weight.sum(
            axes, keepdim=True
        ).clamp_min(1)
        return ((target - mean).square() * weight).sum() / (
            weight.sum() * target.shape[-1]
        ).clamp_min(1)
