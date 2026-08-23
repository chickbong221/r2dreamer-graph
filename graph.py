"""Compact real-edge scene graph encoder and auxiliary decoder.

Replay keeps fixed-width arrays because they are simple to store and sample.
This module removes padding immediately after sampling: node slots stay dense
(there are only ten), while every edge MLP and aggregation runs only on rows
whose relation id is non-zero. The graph path is intentionally plain PyTorch
so this repository has no runtime dependency on ReLDreamer or a GNN package.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from tools import weight_init_
from scenegraph.adapters.graph_vocab import build_absolute_vocab
from scenegraph.core.relation_rules import (
    RELATION_TYPES,
    abs_labels_for,
)

# Relation ids match build_relation_vocab: index 0 is pad.
_RELATION_IDS = {name: i + 1 for i, name in enumerate(RELATION_TYPES)}


GRAPH_KEYS = (
    "graph_node_ent",
    "graph_node_bbox",
    "graph_node_centroid",
    "graph_node_target",
    "graph_edge_src",
    "graph_edge_dst",
    "graph_edge_rel",
    "graph_edge_abs",
    "graph_edge_temp",
)

# Keys retired schemas emitted. The pixel/state encoder excludes the whole set,
# so a stale wrapper cannot feed one to the MLP encoder.
RESERVED_GRAPH_KEYS = frozenset(GRAPH_KEYS) | {"graph_node_uid", "graph_node_app"}


def graph_keys() -> tuple:
    return GRAPH_KEYS


def graph_from(data: Mapping[str, torch.Tensor]) -> dict:
    """Return and validate the graph observation subset."""
    keys = GRAPH_KEYS
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"graph.enabled requires observation keys: {missing}")
    return {key: data[key] for key in keys}


@dataclass
class CompactGraph:
    """Dense small node table plus a compact list of real directed facts."""

    batch_shape: tuple[int, ...]
    num_nodes: int
    node_ent: torch.Tensor
    node_target: torch.Tensor
    node_valid: torch.Tensor
    node_bbox: torch.Tensor
    # Boxes vanish when a node goes invisible; this does not, which is what
    # lets a retained target keep a position through occlusion.
    node_centroid: torch.Tensor
    camera_visible: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    # Endpoints before the per-graph offset is added. The progress teacher has
    # to ask "is this edge's source row zero", which the offset destroys.
    edge_src_local: torch.Tensor
    edge_dst_local: torch.Tensor
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

    def bbox_feature(self, dtype: torch.dtype) -> torch.Tensor:
        """(G, N, 5C): every camera's box, zeroed where invalid, then its bit.

        One flat block so the encoder input and the decoder query are each a
        single GEMM. Boxes arrive already normalised to [0, 1] by the node
        builder. Validity is derived, never stored.
        """
        seen = self.camera_visible.to(dtype)
        box = self.node_bbox.to(dtype) * seen[..., None]
        return torch.cat([box.flatten(-2), seen], -1)

    def centroid_feature(
        self, dtype: torch.dtype, origin: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """(G, N, 3) world centroid on fixed bounds, zero on padded rows.

        Fixed bounds, not batch statistics: the same object in the same place
        has to encode identically in every episode, and a running normaliser
        would make a stationary target's position drift as the batch changes.
        Padded rows stay exactly zero so an unfilled row cannot be read as a
        position at the origin.
        """
        normed = (self.node_centroid.float() - origin) / scale
        return (normed * self.node_valid[..., None]).to(dtype)


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
    bbox_tail = graph["graph_node_bbox"].shape[-2:]
    node_bbox = graph["graph_node_bbox"].reshape(graph_count, num_nodes, *bbox_tail)
    # A camera that never saw the node leaves its row zero, so an empty box is
    # exactly "not visible here". Derived rather than stored.
    camera_visible = (
        (node_bbox[..., 1] > node_bbox[..., 0])
        & (node_bbox[..., 3] > node_bbox[..., 2])
    )
    node_centroid = graph["graph_node_centroid"].reshape(graph_count, num_nodes, 3)
    node_target = graph["graph_node_target"].reshape(graph_count, num_nodes).long()
    node_valid = node_ent.ne(0)

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
        node_target=node_target,
        node_valid=node_valid,
        node_bbox=node_bbox,
        node_centroid=node_centroid,
        camera_visible=camera_visible,
        edge_src=local_src + offset,
        edge_dst=local_dst + offset,
        edge_src_local=local_src,
        edge_dst_local=local_dst,
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
    """Edge-feature GNN with real-edge mean aggregation and a pooled readout.

    The readout is a masked mean plus the normalised node count: uniform
    coefficients, one reduction and one projection over the flattened batch.
    The count is what keeps two sets with the same mean and different
    cardinality apart.
    """

    def __init__(self, config):
        super().__init__()
        self.units = int(config.simple_units)
        self.layers = int(config.layers)
        self.n_cams = int(config.n_cams)
        self.embed_dim = int(config.embed)
        self.reverse_edges = bool(config.reverse_edges)

        self.entity = nn.Embedding(int(config.entity_vocab), self.embed_dim)
        self.target = nn.Embedding(2, self.embed_dim)
        # Boxes plus their per-camera visible bits, plus the world-frame
        # centroid. The boxes say where a node is on each screen and go dark
        # when it leaves; the centroid says where it is in the world and does
        # not. Both are fed raw into the same projection.
        node_in = 2 * self.embed_dim + 5 * self.n_cams + 3
        self.node = GraphMLP(node_in, self.units, str(config.act))
        origin, scale = _centroid_bounds(config)
        self.register_buffer("centroid_origin", origin, persistent=False)
        self.register_buffer("centroid_scale", scale, persistent=False)

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
        # [masked mean, count / n_max] -> one token of the same width, so the
        # RSSM's graph-token input is unchanged.
        self.pool = nn.Linear(self.units + 1, self.units)
        self.apply(weight_init_)

    def forward(self, graph: Mapping[str, torch.Tensor]) -> GraphEncoding:
        compact = compact_graph(graph)
        valid = compact.node_valid
        parts = []
        parts.extend((self.entity(compact.node_ent), self.target(compact.node_target)))
        parts.append(compact.bbox_feature(parts[0].dtype))
        parts.append(
            compact.centroid_feature(
                parts[0].dtype, self.centroid_origin, self.centroid_scale
            )
        )
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

        # Nodes are already zeroed where invalid, so the sum needs no second
        # mask. Every admitted node gets the same 1/n coefficient.
        count = valid.sum(-1, keepdim=True).to(nodes.dtype)
        mean = nodes.sum(1) / count.clamp_min(1)
        ratio = count / float(compact.num_nodes)
        token = self.pool(torch.cat([mean, ratio], -1)) * count.gt(0)
        nodes = nodes.reshape(*compact.batch_shape, compact.num_nodes, self.units)
        token = token.reshape(*compact.batch_shape, self.units)
        return GraphEncoding(nodes, token, compact)


def _centroid_bounds(config) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed world-frame normalisation, shared by the encoder and the decoder
    query. They address the same node, so they must agree exactly."""
    origin = torch.as_tensor(
        getattr(config, "centroid_origin", (0.0, 0.0, 0.0)), dtype=torch.float32
    ).reshape(3)
    scale = torch.as_tensor(
        getattr(config, "centroid_scale", 5.0), dtype=torch.float32
    ).reshape(-1)
    if scale.numel() == 1:
        scale = scale.expand(3).clone()
    if not bool((scale > 0).all()):
        raise ValueError(f"graph.centroid_scale must be positive, got {scale}")
    return origin, scale


def _relation_masks(
    n_rel: int, n_abs: int,
) -> torch.Tensor:
    """Legal absolute labels per relation, built from the shared tables.

    Derived rather than hardcoded: the packer's ``abs_valid`` already comes
    from ``ABS_LABELS``, so a hand-written mask here is a second source of
    truth. Directional support/contain labels are also non-contiguous in sigma,
    which no slice expresses.
    """
    absolute = build_absolute_vocab()
    labels = abs_labels_for()
    want_rel, want_abs = len(RELATION_TYPES) + 1, len(absolute)
    if n_rel != want_rel or n_abs != want_abs:
        raise ValueError(
            f"relation vocab is {want_rel} and "
            f"absolute vocab {want_abs}; config says n_rel={n_rel}, "
            f"n_abs={n_abs}"
        )
    mask = torch.zeros(n_rel, n_abs, dtype=torch.bool)
    for name in RELATION_TYPES:
        rid = _RELATION_IDS[name]
        for label in labels[name]:
            mask[rid, absolute.encode(label)] = True
    return mask


def _frame_mean(values, mask):
    """Mean over one frame's valid entries, so a frame with seven nodes does
    not outweigh a frame with one."""
    axes = tuple(range(1, values.ndim))
    numerator = (values.float() * mask).sum(axes)
    denominator = mask.float().sum(axes).clamp_min(1)
    return numerator / denominator


def _masked(values, mask):
    mask = mask.float()
    return (values.float() * mask).sum() / mask.sum().clamp_min(1)


def _edge_categorical(
    logits, target, frame, mask, classes, graph_count, only_present=False
):
    """Per-frame-averaged cross entropy over the legal label set.

    ``only_present`` averages over frames that actually carry a supervised fact
    instead of over every frame. Use it where a missing fact means "no label",
    not "no progress" -- a frame with nothing to supervise must not be counted
    as a frame with zero loss.
    """
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
    per_frame = numerator / denominator.clamp_min(1)
    if only_present:
        present = denominator.gt(0).float()
        loss = (per_frame * present).sum() / present.sum().clamp_min(1)
    else:
        loss = per_frame.mean()
    accuracy = _masked(logits.argmax(-1).eq(target), mask)
    return loss, accuracy


class SimpleGraphDecoder(nn.Module):
    """Conditional graph reconstruction from the pooled semantic state.

    Every node is read back out of the single posterior vector ``g`` by querying
    it with that node's current box, so the losses measure what ``g`` retained
    rather than what the encoder's own node vectors still hold. A box is a cheap
    current-frame content address: it works the frame a node first appears and
    it keeps episode-random identity codes out of the global dynamics.

    The fused EE->target progress head lives here too. Keeping it inside the
    decoder is what makes the optimizer, the checkpoint and ``clone_and_freeze``
    pick it up automatically, so imagination reads exactly the parameters
    training wrote instead of a second copy that has to be kept in step.

    Everything is one flattened call over ``B * T`` frames, ``B * T * N`` nodes
    or every real edge in the batch. No loop over nodes, edges or relations.
    """

    def __init__(self, config, semantic_dim: int):
        super().__init__()
        self.units = int(getattr(config, "decoder_units", config.simple_units))
        self.n_cams = int(config.n_cams)
        self.n_abs = int(config.n_abs)
        self.n_temp = int(config.n_temp)
        self.entity_vocab = int(config.entity_vocab)
        self.bbox_beta = float(config.bbox_beta)
        act = str(config.act)

        # Node signature: every camera's box plus the world centroid. One
        # Linear, no norm and no activation -- this addresses a node, it does
        # not represent one, and the narrow output is a deliberate bottleneck on
        # how much geometry reaches the box head directly rather than through g.
        #
        # The centroid is what separates nodes the boxes cannot. Under
        # unconditional retention every node without pixels has an all-zero box
        # and all-zero visibility bits, so a box-only query is identical for all
        # of them and the decoder would be asked for several different entities
        # from one input. Bounds match the encoder's exactly.
        self.query = nn.Linear(5 * self.n_cams + 3, int(config.bbox_query_dim))
        origin, scale = _centroid_bounds(config)
        self.register_buffer("centroid_origin", origin, persistent=False)
        self.register_buffer("centroid_scale", scale, persistent=False)
        # W_g runs once per frame and broadcasts over the node axis. Projecting
        # the full semantic vector separately for all eight nodes would be the
        # same arithmetic done N times.
        self.global_proj = nn.Linear(int(semantic_dim), self.units)
        self.query_proj = nn.Linear(int(config.bbox_query_dim), self.units)
        self.node_proj = nn.Linear(self.units, self.units)
        self.act = getattr(nn, act)()
        # Entity logits, the target logit and every camera's box come out of one
        # projection and are split afterwards: one GEMM over the node rows
        # instead of three.
        self.node_head = nn.Linear(
            self.units, self.entity_vocab + 1 + 4 * self.n_cams
        )

        self.reltype = nn.Embedding(int(config.n_rel), int(config.embed))
        self.pair = GraphMLP(2 * self.units + int(config.embed), self.units, act)
        # Absolute and temporal labels likewise share one projection over the
        # edge rows; their legal masks and losses stay separate.
        self.edge_head = nn.Linear(self.units, self.n_abs + self.n_temp)

        self.register_buffer(
            "abs_valid", _relation_masks(int(config.n_rel), self.n_abs), persistent=False
        )
        self.apply(weight_init_)

    def node_features(self, sem: torch.Tensor, compact: CompactGraph) -> torch.Tensor:
        """(G, N, U) per-node representation conditioned on ``sem`` and the
        node's box-and-centroid signature."""
        dtype = self.query.weight.dtype
        signature = torch.cat(
            [
                compact.bbox_feature(dtype),
                compact.centroid_feature(
                    dtype, self.centroid_origin, self.centroid_scale
                ),
            ],
            -1,
        )
        query = self.query_proj(self.query(signature))
        frame = self.global_proj(sem.reshape(compact.graph_count, -1).to(query.dtype))
        return self.node_proj(self.act(frame[:, None, :] + query))

    def forward(
        self,
        sem: torch.Tensor,
        compact: CompactGraph,
        step_valid: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        graph_count = compact.graph_count
        num_nodes = compact.num_nodes
        step = step_valid.reshape(graph_count).bool()
        valid = compact.node_valid & step[:, None]
        losses: dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor] = {}

        nodes = self.node_features(sem, compact) * valid[..., None]
        entity_logit, target_logit, bbox_pred = self.node_head(nodes).split(
            [self.entity_vocab, 1, 4 * self.n_cams], dim=-1
        )

        # --- node attributes -------------------------------------------------
        entity_logit = entity_logit.float()
        entity_error = F.cross_entropy(
            entity_logit.reshape(-1, self.entity_vocab),
            compact.node_ent.reshape(-1),
            reduction="none",
        ).reshape(graph_count, num_nodes)
        node_ent = _frame_mean(entity_error, valid).mean()
        metrics["node_ent_acc"] = _masked(
            entity_logit.argmax(-1).eq(compact.node_ent), valid
        )

        # Averaged over the four coordinates, not summed, so the term does not
        # inherit a factor of four from the box layout. Masked per camera: a
        # camera that never saw the node has nothing to regress to.
        bbox_pred = bbox_pred.reshape(graph_count, num_nodes, self.n_cams, 4).float()
        bbox_mask = valid[..., None] & compact.camera_visible
        bbox_error = F.smooth_l1_loss(
            bbox_pred,
            compact.node_bbox.float(),
            reduction="none",
            beta=self.bbox_beta,
        ).mean(-1)
        node_bbox = _frame_mean(bbox_error, bbox_mask).mean()

        # One configured scale covers both halves; the components are logged as
        # metrics rather than emitted as losses, which would each need their own
        # scale key. Note this is not comparable with full mode's ``node``,
        # which averages appearance, boxes and visibility instead.
        losses["node"] = 0.5 * (node_ent + node_bbox)
        metrics["node_ent_loss"] = node_ent.detach()
        metrics["node_bbox_loss"] = node_bbox.detach()

        # Slot zero is the end effector and never carries the target flag.
        target_logit = target_logit.squeeze(-1).float()
        target_mask = valid.clone()
        target_mask[:, 0] = False
        target_flag = compact.node_target.bool()
        target_error = F.binary_cross_entropy_with_logits(
            target_logit, target_flag.float(), reduction="none"
        )
        positive = target_mask & target_flag
        negative = target_mask & ~target_flag
        has_positive = positive.any(-1)
        has_negative = negative.any(-1)
        class_count = has_positive.float() + has_negative.float()
        losses["nodetgt"] = (
            (
                _frame_mean(target_error, positive) * has_positive
                + _frame_mean(target_error, negative) * has_negative
            )
            / class_count.clamp_min(1)
        ).mean()

        target_index = target_flag.long().argmax(-1)
        selected = target_logit.masked_fill(~target_mask, -1e9).argmax(-1)
        metrics["node_target_acc"] = _masked(selected.eq(target_index), has_positive)
        metrics["node_target_frac"] = has_positive.float().mean()

        # --- posterior relations ---------------------------------------------
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
        abs_logits, temp_logits = self.edge_head(pair).split(
            [self.n_abs, self.n_temp], dim=-1
        )
        losses["relabs"], metrics["relabs_acc"] = _edge_categorical(
            abs_logits,
            compact.edge_abs,
            compact.edge_graph,
            edge_step,
            self.abs_valid.index_select(0, compact.edge_rel),
            graph_count,
        )
        temp_mask = edge_step & compact.edge_temp.ne(0)
        temp_classes = torch.ones_like(temp_logits, dtype=torch.bool)
        temp_classes[:, 0] = False
        losses["reltemp"], metrics["reltemp_acc"] = _edge_categorical(
            temp_logits,
            compact.edge_temp,
            compact.edge_graph,
            temp_mask,
            temp_classes,
            graph_count,
        )

        metrics["graph_real_edges"] = torch.as_tensor(
            compact.edge_count / max(graph_count, 1),
            device=nodes.device,
            dtype=torch.float32,
        )
        metrics["graph_nodes_per_frame"] = valid.float().sum(-1).mean()
        return losses, metrics
