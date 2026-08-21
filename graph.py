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
from scenegraph.adapters.graph_vocab import build_absolute_vocab
from scenegraph.core.relation_rules import (
    EDGE_CONTRACT_CANONICAL,
    EDGE_CONTRACT_LEGACY,
    RELATION_TYPES,
    abs_labels_for,
)

# Relation ids match build_relation_vocab: index 0 is pad.
_RELATION_IDS = {name: i + 1 for i, name in enumerate(RELATION_TYPES)}


def _edge_contract(config) -> str:
    return str(getattr(config, "edge_contract", None) or EDGE_CONTRACT_LEGACY)


# Mirrors ``scenegraph.core.graph_builder``. Duplicated rather than imported so
# the model package never reaches into the simulator-side package; the tests
# assert the two constants agree.
UID_PAD = 0
UID_EE = 1

# ``pooled``: the shipped model. One attention-pooled token becomes the single
# semantic vector g. ``slots``: no pooling anywhere in the dynamics path -- the
# GNN's six node vectors are the semantic state.
STATE_MODES = ("pooled", "slots")


def graph_state_mode(config) -> str:
    mode = str(getattr(config, "state_mode", "pooled"))
    if mode not in STATE_MODES:
        raise ValueError(
            f"graph.state_mode={mode!r} is not one of {STATE_MODES}"
        )
    return mode


# Three observation schemas. ``simple`` alone no longer selects one: pooled
# graph-simple addresses a node by the box it currently occupies and never sees
# an identity code, while slot graph-simple aligns by UID across frames and
# carries no boxes at all. Emitting both fields to both modes would put a key in
# replay that one of them must never read.
SCHEMA_FULL = "full"
SCHEMA_SIMPLE_POOLED = "simple_pooled_bbox"
SCHEMA_SIMPLE_SLOT = "simple_slot_uid"
GRAPH_SCHEMAS = (SCHEMA_FULL, SCHEMA_SIMPLE_POOLED, SCHEMA_SIMPLE_SLOT)

_EDGE_KEYS = (
    "graph_edge_src",
    "graph_edge_dst",
    "graph_edge_rel",
    "graph_edge_abs",
    "graph_edge_temp",
)

FULL_GRAPH_KEYS = (
    "graph_node_ent",
    "graph_node_app",
    "graph_node_bbox",
    "graph_node_target",
    *_EDGE_KEYS,
)

# Pooled graph-simple. No appearance and no UID: the per-camera box is both the
# node's distinguishing content and the decoder's address. The world-frame
# centroid rides alongside because the box cannot survive occlusion and the
# retained target needs a position that does.
SIMPLE_POOLED_GRAPH_KEYS = (
    "graph_node_ent",
    "graph_node_bbox",
    "graph_node_centroid",
    "graph_node_target",
    *_EDGE_KEYS,
)

# Slot graph-simple. Relation-only; ``graph_node_uid`` names a node across
# frames, which is the only thing :class:`SlotAligner` may match on.
SIMPLE_SLOT_GRAPH_KEYS = (
    "graph_node_ent",
    "graph_node_uid",
    "graph_node_target",
    *_EDGE_KEYS,
)

SCHEMA_KEYS = {
    SCHEMA_FULL: FULL_GRAPH_KEYS,
    SCHEMA_SIMPLE_POOLED: SIMPLE_POOLED_GRAPH_KEYS,
    SCHEMA_SIMPLE_SLOT: SIMPLE_SLOT_GRAPH_KEYS,
}

# Every key any schema has ever emitted, whether or not the active one reads it.
# The pixel/state encoder excludes this whole set, so a key the active schema
# dropped -- ``graph_node_uid`` under the pooled schema -- cannot reach the MLP
# encoder from a stale wrapper and quietly train an identity-conditioned model.
RESERVED_GRAPH_KEYS = frozenset(
    key for keys in SCHEMA_KEYS.values() for key in keys
) | {
    "graph_node_uid",
    "graph_node_app",
    "graph_node_bbox",
    "graph_node_centroid",
}

GRAPH_KEYS = FULL_GRAPH_KEYS
# Retained for callers that still speak of "the simple contract"; slot mode is
# the one that kept the relation-only key set unchanged.
SIMPLE_GRAPH_KEYS = SIMPLE_SLOT_GRAPH_KEYS


def graph_schema(simple: bool, state_mode: str = "pooled") -> str:
    """Pick the observation schema from the two existing config switches."""
    if not simple:
        return SCHEMA_FULL
    if str(state_mode) == "slots":
        return SCHEMA_SIMPLE_SLOT
    return SCHEMA_SIMPLE_POOLED


def graph_keys(schema: str) -> tuple[str, ...]:
    try:
        return SCHEMA_KEYS[schema]
    except KeyError:
        raise ValueError(
            f"unknown graph schema {schema!r}; expected one of {GRAPH_SCHEMAS}"
        ) from None


def graph_from(
    data: Mapping[str, torch.Tensor], schema: str = SCHEMA_FULL
) -> dict[str, torch.Tensor]:
    """Return and validate the graph observation subset for the active schema."""
    keys = graph_keys(schema)
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
    # Only the fields the active schema emits are populated. The rest stay None
    # so a schema mix-up raises rather than silently reading zeros: the slot
    # schema has a UID and no boxes, the pooled schema the reverse, and only the
    # full schema has appearance.
    node_uid: torch.Tensor | None
    node_app: torch.Tensor | None
    node_bbox: torch.Tensor | None
    # World-frame object position, pooled schema only. Boxes vanish when a node
    # goes invisible; this does not, which is what lets the retained target
    # keep a position through occlusion.
    node_centroid: torch.Tensor | None
    appearance_known: torch.Tensor | None
    camera_visible: torch.Tensor | None
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
        single GEMM. The full schema's per-camera ``ModuleList`` is deliberately
        not reused here -- it pays one small matmul per camera for nothing.

        Boxes arrive already normalised to [0, 1] by the node builder, so there
        is no second normalisation. Validity is derived, never stored.
        """
        if self.node_bbox is None:
            raise RuntimeError("this graph schema carries no boxes")
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
        if self.node_centroid is None:
            raise RuntimeError("this graph schema carries no centroids")
        normed = (self.node_centroid.float() - origin) / scale
        return (normed * self.node_valid[..., None]).to(dtype)


@dataclass
class SlotObservation:
    """One frame's graph as six observed slots plus their structural labels.

    Every field is indexed by the *observation* node position, which the vertex
    registry reuses; ``uid`` is what names an entity across frames and the only
    thing :class:`SlotAligner` is allowed to match on.
    """

    slots: torch.Tensor  # (..., n, D) contextual node embeddings
    uid: torch.Tensor  # (..., n) long
    ent: torch.Tensor  # (..., n) long
    target: torch.Tensor  # (..., n) long
    mask: torch.Tensor  # (..., n) bool

    def __getitem__(self, key) -> "SlotObservation":
        """Index every field the same way."""
        return SlotObservation(
            self.slots[key],
            self.uid[key],
            self.ent[key],
            self.target[key],
            self.mask[key],
        )

    def step(self, index: int) -> "SlotObservation":
        """Select one time index from a (B, T, ...) observation."""
        return self[:, index]

    def keep(self, valid: torch.Tensor) -> "SlotObservation":
        """Blank whole frames, e.g. auto-reset terminal observations.

        A blanked frame is not "an empty scene": it is *no observation*, so the
        aligner matches nothing, every carried slot keeps its identity, and the
        posterior falls back to the prior for all of them.
        """
        while valid.ndim < self.uid.ndim:
            valid = valid.unsqueeze(-1)
        return SlotObservation(
            self.slots * valid[..., None].to(self.slots.dtype),
            self.uid * valid.long(),
            self.ent * valid.long(),
            self.target * valid.long(),
            self.mask & valid.bool(),
        )


@dataclass
class GraphEncoding:
    nodes: torch.Tensor
    # ``None`` in slot mode: no pooled token is computed at all.
    token: torch.Tensor | None
    compact: CompactGraph
    slots: SlotObservation | None = None


def compact_graph(
    graph: Mapping[str, torch.Tensor], schema: str = SCHEMA_FULL
) -> CompactGraph:
    """Strip padded edge rows on device without creating Python graph objects."""
    graph = graph_from(graph, schema)
    ent = graph["graph_node_ent"]
    if ent.ndim < 2:
        raise ValueError(f"graph_node_ent must have batch and node axes, got {ent.shape}")
    batch_shape = tuple(ent.shape[:-1])
    graph_count = reduce(mul, batch_shape, 1)
    num_nodes = int(ent.shape[-1])

    node_ent = ent.reshape(graph_count, num_nodes).long()
    node_uid = node_app = node_bbox = node_centroid = None
    appearance_known = camera_visible = None
    if schema == SCHEMA_SIMPLE_SLOT:
        node_uid = graph["graph_node_uid"].reshape(graph_count, num_nodes).long()
    else:
        bbox_tail = graph["graph_node_bbox"].shape[-2:]
        node_bbox = graph["graph_node_bbox"].reshape(graph_count, num_nodes, *bbox_tail)
        # A camera that never saw the node leaves its row zero, so an empty box
        # is exactly "not visible here". Derived rather than stored.
        camera_visible = (
            (node_bbox[..., 1] > node_bbox[..., 0])
            & (node_bbox[..., 3] > node_bbox[..., 2])
        )
        if schema == SCHEMA_FULL:
            app_tail = graph["graph_node_app"].shape[-2:]
            node_app = graph["graph_node_app"].reshape(
                graph_count, num_nodes, *app_tail
            )
            appearance_known = node_app.float().abs().sum(-1).ne(0)
        else:
            node_centroid = graph["graph_node_centroid"].reshape(
                graph_count, num_nodes, 3
            )
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
        node_uid=node_uid,
        node_app=node_app,
        node_bbox=node_bbox,
        node_centroid=node_centroid,
        appearance_known=appearance_known,
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

    The readout depends on the schema. Full mode keeps the shipped attention
    pooling. Pooled graph-simple replaces it with a masked mean plus the
    normalised node count: uniform coefficients, one reduction and one
    projection over the flattened batch instead of two per-node 512-wide
    projections and a softmax. Slot mode pools nowhere.
    """

    def __init__(self, config):
        super().__init__()
        self.simple = bool(getattr(config, "simple", False))
        self.state_mode = graph_state_mode(config)
        self.slot_mode = self.state_mode == "slots"
        self.schema = graph_schema(self.simple, self.state_mode)
        if self.slot_mode and not self.simple:
            raise ValueError(
                "graph.state_mode=slots is a relation-only mode; set "
                "model.graph_simple=true (graph.simple) as well"
            )
        if self.slot_mode:
            self.units = int(config.slot_dim)
        else:
            self.units = int(config.simple_units if self.simple else config.units)
        self.layers = int(config.layers)
        self.n_cams = int(config.n_cams)
        self.app_dim = int(config.app_dim)
        self.embed_dim = int(config.embed)
        self.reverse_edges = bool(config.reverse_edges)

        self.entity = nn.Embedding(int(config.entity_vocab), self.embed_dim)
        self.target = nn.Embedding(2, self.embed_dim)
        if self.slot_mode:
            # No UID embedding. A UID is an episode-random code; letting it into
            # the node representation would make two identical scenes embed
            # differently. Identity is used for alignment only.
            self.uid = None
            self.app_proj = None
            self.bbox_proj = None
            node_in = 2 * self.embed_dim
        elif self.simple:
            # No UID anywhere in the pooled path. The per-camera box separates
            # same-type siblings and is what the decoder later addresses a node
            # with, so it is fed raw into the node MLP: one flat (4 + 1) * C
            # block folded into the projection that was already there, rather
            # than a per-camera projection per node.
            self.uid = None
            self.app_proj = None
            self.bbox_proj = None
            # Boxes plus their per-camera visible bits, plus the world-frame
            # centroid. The boxes say where a node is on each screen and go
            # dark when it leaves; the centroid says where it is in the world
            # and does not. Both are fed raw into the same projection.
            node_in = 2 * self.embed_dim + 5 * self.n_cams + 3
        else:
            self.uid = None
            self.app_proj = nn.ModuleList(
                nn.Linear(self.app_dim, int(config.app)) for _ in range(self.n_cams)
            )
            self.bbox_proj = nn.ModuleList(
                nn.Linear(4, int(config.bbox)) for _ in range(self.n_cams)
            )
            node_in = (
                self.n_cams * (int(config.app) + int(config.bbox))
                + 2 * self.embed_dim
            )
        self.node = GraphMLP(node_in, self.units, str(config.act))
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
        self.query = self.key = self.value = self.out = None
        self.pool = None
        if self.slot_mode:
            # Attention pooling exists only as a head readout in slot mode, and
            # it lives next to the heads. Building it here would leave unused
            # parameters in the optimizer and invite pooling back into the
            # dynamics path by accident.
            pass
        elif self.simple:
            # [masked mean, count / n_max] -> one token of the same width, so
            # the RSSM's graph-token input is unchanged. The count is what keeps
            # two sets with the same mean and different cardinality apart.
            self.pool = nn.Linear(self.units + 1, self.units)
        else:
            self.query = nn.Parameter(torch.empty(self.units))
            self.key = nn.Linear(self.units, self.units)
            self.value = nn.Linear(self.units, self.units)
            self.out = nn.Linear(self.units, self.units)
        self.apply(weight_init_)
        if self.query is not None:
            nn.init.normal_(self.query, std=0.02)

    def forward(self, graph: Mapping[str, torch.Tensor]) -> GraphEncoding:
        compact = compact_graph(graph, self.schema)
        valid = compact.node_valid
        parts = []
        if self.schema == SCHEMA_FULL:
            # Replay stores these arrays as float16. Normalize the storage
            # boundary to float32; the shared acting/training autocast policy
            # selects compute dtype for the learned projections.
            node_app = compact.node_app.to(self.app_proj[0].weight.dtype)
            node_bbox = compact.node_bbox.to(self.bbox_proj[0].weight.dtype)
            for camera in range(self.n_cams):
                app = self.app_proj[camera](node_app[..., camera, :])
                parts.append(app * compact.appearance_known[..., camera, None])
                box = self.bbox_proj[camera](node_bbox[..., camera, :])
                parts.append(box * compact.camera_visible[..., camera, None])
        parts.extend((self.entity(compact.node_ent), self.target(compact.node_target)))
        if self.schema == SCHEMA_SIMPLE_POOLED:
            parts.append(compact.bbox_feature(parts[0].dtype))
            parts.append(
                compact.centroid_feature(
                    parts[0].dtype, self.centroid_origin, self.centroid_scale
                )
            )
        if self.uid is not None:
            parts.append(self.uid(compact.node_uid))
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

        if self.slot_mode:
            shape = (*compact.batch_shape, compact.num_nodes)
            slots = SlotObservation(
                slots=nodes.reshape(*shape, self.units),
                uid=compact.node_uid.reshape(shape),
                ent=compact.node_ent.reshape(shape),
                target=compact.node_target.reshape(shape),
                mask=valid.reshape(shape),
            )
            return GraphEncoding(slots.slots, None, compact, slots)

        if self.pool is not None:
            # Nodes are already zeroed where invalid, so the sum needs no second
            # mask. Every admitted node gets the same 1/n coefficient.
            count = valid.sum(-1, keepdim=True).to(nodes.dtype)
            mean = nodes.sum(1) / count.clamp_min(1)
            ratio = count / float(compact.num_nodes)
            token = self.pool(torch.cat([mean, ratio], -1)) * count.gt(0)
        else:
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


def _relation_masks(
    n_rel: int, n_abs: int, contract: str = EDGE_CONTRACT_LEGACY,
) -> torch.Tensor:
    """Legal absolute labels per relation, built from the contract's tables.

    Derived rather than hardcoded: the packer's ``abs_valid`` already comes
    from ``ABS_LABELS``, so a hand-written mask here is a second source of
    truth. canonical_v2 also makes support/contain non-contiguous in sigma,
    which no slice expresses.
    """
    absolute = build_absolute_vocab(contract)
    labels = abs_labels_for(contract)
    want_rel, want_abs = len(RELATION_TYPES) + 1, len(absolute)
    if n_rel != want_rel or n_abs != want_abs:
        raise ValueError(
            f"edge_contract {contract!r} has relation vocab {want_rel} and "
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


def slot_target_logits(logits, alive, null_logit: float = 0.0):
    """Categorical logits over object slots plus one null-target class.

    Slot zero is the end effector and is never a target class, so the layout is
    ``[slot 1, ..., slot n-1, null]`` and the null index is ``n - 1``. Inactive
    object slots are masked out; when none is alive the distribution collapses
    onto null, which is the honest answer rather than an arbitrary object.
    """
    objects = logits[..., 1:].float()
    objects = objects.masked_fill(
        ~alive[..., 1:].bool(), torch.finfo(torch.float32).min
    )
    null = objects.new_full((*objects.shape[:-1], 1), float(null_logit))
    return torch.cat([objects, null], -1)


def slot_target_label(target_flag, alive):
    """Teacher label for the slot-or-null objective.

    Latched, not read off the current frame: with history-off graphs the target
    leaves the observation whenever it is occluded, and a label that went null
    there would teach "is the target visible" instead of "which slot is it".
    Requires no additional state -- the flag is already latched in ``slot_meta``.
    """
    live = target_flag.bool() & alive.bool()
    objects = live[..., 1:]
    count = objects.long().sum(-1)
    if bool(count.gt(1).any()):
        raise ValueError(
            "more than one live slot carries the target flag; the subtask "
            "target is unique per episode, so this is an alignment bug rather "
            "than something to resolve with argmax"
        )
    null = objects.shape[-1]
    label = torch.where(
        count.eq(1), objects.long().argmax(-1), torch.full_like(count, null)
    )
    return label, count.eq(1)


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

    def __init__(self, config, semantic_dim: int, progress_relations):
        super().__init__()
        self.units = int(getattr(config, "decoder_units", config.simple_units))
        self.n_cams = int(config.n_cams)
        self.n_abs = int(config.n_abs)
        self.n_temp = int(config.n_temp)
        self.entity_vocab = int(config.entity_vocab)
        self.bbox_beta = float(config.bbox_beta)
        act = str(config.act)

        # Box signature. One Linear, no norm and no activation: this addresses a
        # node, it does not represent one. The narrow output is a deliberate
        # continuous bottleneck on how much geometry reaches the box head
        # directly rather than through g.
        self.query = nn.Linear(5 * self.n_cams, int(config.bbox_query_dim))
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

        relations = torch.as_tensor(progress_relations, dtype=torch.long).reshape(-1)
        self.n_progress = int(relations.numel())
        # One linear from g_hat to every scorer relation at once. The cumulative
        # stages reuse these distributions; they never decode one apiece.
        self.progress_head = nn.Linear(
            int(semantic_dim), self.n_progress * self.n_abs
        )
        self.register_buffer("progress_relations", relations, persistent=False)
        self.register_buffer(
            "abs_valid", _relation_masks(int(config.n_rel), self.n_abs, _edge_contract(config)), persistent=False
        )
        # Relation id -> row of the fused progress output, -1 for relations the
        # scorer does not read. Built from the scorer's own relation order,
        # because the relation id is not the row.
        row = torch.full((int(config.n_rel),), -1, dtype=torch.long)
        row[relations] = torch.arange(self.n_progress, dtype=torch.long)
        self.register_buffer("progress_row", row, persistent=False)
        self.register_buffer(
            "progress_valid", self.abs_valid.index_select(0, relations), persistent=False
        )
        self.apply(weight_init_)

    def node_features(self, sem: torch.Tensor, compact: CompactGraph) -> torch.Tensor:
        """(G, N, U) per-node representation conditioned on ``sem`` and the box."""
        box = compact.bbox_feature(self.query.weight.dtype)
        query = self.query_proj(self.query(box))
        frame = self.global_proj(sem.reshape(compact.graph_count, -1).to(query.dtype))
        return self.node_proj(self.act(frame[:, None, :] + query))

    def progress_logits(self, sem: torch.Tensor) -> torch.Tensor:
        """(..., R, n_abs) EE->target logits for the scorer's relations."""
        return self.progress_head(sem).reshape(
            *sem.shape[:-1], self.n_progress, self.n_abs
        )

    def progress_probs(self, sem: torch.Tensor) -> torch.Tensor:
        """Legal-masked distributions in float32, one call for the whole batch."""
        logits = self.progress_logits(sem).float()
        return torch.softmax(logits.masked_fill(~self.progress_valid, -1e9), -1)

    def forward(
        self,
        sem: torch.Tensor,
        compact: CompactGraph,
        step_valid: torch.Tensor,
        prior_sem: torch.Tensor | None = None,
        prior_valid: torch.Tensor | None = None,
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

        # --- causal EE->target prior -----------------------------------------
        if prior_sem is not None:
            losses["prior_progress_relabs"], progress_metrics = self.prior_progress(
                prior_sem, compact, has_positive, target_index, prior_valid
            )
            metrics.update(progress_metrics)

        metrics["graph_real_edges"] = torch.as_tensor(
            compact.edge_count / max(graph_count, 1),
            device=nodes.device,
            dtype=torch.float32,
        )
        metrics["graph_nodes_per_frame"] = valid.float().sum(-1).mean()
        return losses, metrics

    def prior_progress(
        self,
        prior_sem: torch.Tensor,
        compact: CompactGraph,
        has_target: torch.Tensor,
        target_index: torch.Tensor,
        prior_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Cross entropy on the EE->target facts the progress scorer reads.

        The observed target flag only selects *which* edge supplies the label.
        Neither it, nor the boxes, nor any observed relation reaches the head,
        whose sole input is the predicted ``g_hat``.

        The mask is causal in its own right: ``step_valid`` alone still admits
        the reset frame, which ``g_hat`` has no preceding episode to predict
        from. A target admitted on any later frame is supervised immediately --
        being newly admitted is not a reason to mask it. When the target is
        occluded the flag and its edges vanish together and the frame drops out
        on its own, which is why no ``target_resolved`` key has to reach replay.
        """
        graph_count = compact.graph_count
        frame = compact.edge_graph
        row = self.progress_row.index_select(0, compact.edge_rel)
        teacher = (
            prior_valid.reshape(graph_count).bool().index_select(0, frame)
            & has_target.index_select(0, frame)
            & compact.edge_src_local.eq(0)
            & compact.edge_dst_local.eq(target_index.index_select(0, frame))
            & row.ge(0)
        )
        # Dense over every frame, then gathered: one GEMM plus one gather beats
        # a dynamically shaped select, and B * T * 102 costs nothing.
        logits = self.progress_logits(prior_sem).reshape(
            graph_count, self.n_progress, self.n_abs
        )
        gathered = logits[frame, row.clamp_min(0)]
        loss, accuracy = _edge_categorical(
            gathered,
            compact.edge_abs,
            frame,
            teacher,
            self.abs_valid.index_select(0, compact.edge_rel),
            graph_count,
            only_present=True,
        )
        metrics = {
            "prior_progress_acc": accuracy,
            "prior_progress_facts": teacher.float().sum() / max(graph_count, 1),
        }
        return loss, metrics


@dataclass
class SlotAlignment:
    """Where each observed node belongs in the recurrent slot table."""

    # (B, n) destination slot per *observed* node; ``num_slots`` means dropped.
    dest: torch.Tensor
    present: torch.Tensor  # (B, n) slot received an observation this step
    matched: torch.Tensor  # (B, n) slot kept its identity and was observed
    born: torch.Tensor  # (B, n) inactive slot that received a fresh node
    replaced: torch.Tensor  # (B, n) live slot overwritten by a different entity
    slots: torch.Tensor  # (B, n, D) observed embeddings, slot-ordered
    uid: torch.Tensor  # (B, n) long
    ent: torch.Tensor  # (B, n) long
    target: torch.Tensor  # (B, n) long
    alive: torch.Tensor  # (B, n) float posterior presence
    overflow: torch.Tensor  # (B,) observed nodes with nowhere to go


class SlotAligner(nn.Module):
    """Observation-to-slot correspondence. No parameters, no categories.

    Assignment order, in full:

    1. the end effector takes slot zero;
    2. an observed UID returns to whatever slot it already held;
    3. a newly observed *target* takes a free or replaceable non-target slot;
    4. a newly observed non-target prefers an inactive slot;
    5. otherwise a live non-target slot is replaced;
    6. the retained target slot is never given to a non-target.

    With births enabled, step 4 is decided by matching predicted birth
    candidates against the fresh observations instead of by index order. The
    graph-side registry normally guarantees the observation fits, so step 5 is a
    rare fallback and an unplaceable row is dropped and counted rather than
    displacing the target.

    Entity category is never consulted -- two interchangeable cans would swap
    latents whenever the registry reordered them -- and UID is used here and
    nowhere else in the model.
    """

    def __init__(self, num_slots: int, uid_ee: int = UID_EE):
        super().__init__()
        self.num_slots = int(num_slots)
        self.uid_ee = int(uid_ee)

    @staticmethod
    def _place(values: torch.Tensor, dest: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """Scatter observation-ordered rows into slot order.

        Rows with nowhere to go address one extra scratch row that is sliced
        off, so no padded observation can land on a real slot.
        """
        batch, count = dest.shape
        trailing = values.shape[2:]
        view = (batch, count) + (1,) * len(trailing)
        out = values.new_zeros(batch, count + 1, *trailing)
        index = dest.reshape(view).expand(batch, count, *trailing)
        out.scatter_(1, index, values * keep.reshape(view).to(values.dtype))
        return out[:, :count]

    @staticmethod
    def _flag(reference: torch.Tensor, hit: torch.Tensor, position: torch.Tensor):
        """One-hot ``position`` along the last axis, gated by ``hit``."""
        index = torch.arange(reference.shape[-1], device=reference.device)
        return hit[:, None] & index[None, :].eq(position[:, None])

    def _birth_cost(self, candidates: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        """Cost of explaining observed node j with birth candidate i.

        float32 and detached: the assignment is a discrete decision, and the
        gradient belongs to the losses that use the assignment, not to the
        arg-min that produced it.
        """
        with torch.no_grad():
            left = observed.float().unsqueeze(2)  # (B, obs, 1, D)
            right = candidates.float().unsqueeze(1)  # (B, 1, slot, D)
            cosine = 1.0 - F.cosine_similarity(left, right, dim=-1, eps=1e-6)
            l1 = F.smooth_l1_loss(
                left.expand(-1, -1, right.shape[2], -1),
                right.expand(-1, left.shape[1], -1, -1),
                reduction="none",
                beta=1.0,
            ).mean(-1)
        return cosine + l1

    def _greedy(self, cost: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        """Batched greedy assignment of rows to columns.

        A fixed column-count loop over an ``[B, rows*cols]`` arg-min. No
        ``.item()``, no host synchronisation, and each graph is matched inside
        its own row -- costs never cross frames.
        """
        batch, rows, cols = cost.shape
        big = torch.finfo(torch.float32).max
        cost = cost.float().masked_fill(~allowed, big)
        used_row = torch.zeros(batch, rows, dtype=torch.bool, device=cost.device)
        used_col = torch.zeros(batch, cols, dtype=torch.bool, device=cost.device)
        dest = torch.full((batch, rows), cols, dtype=torch.long, device=cost.device)
        for _ in range(cols):
            blocked = used_row[:, :, None] | used_col[:, None, :]
            flat = cost.masked_fill(blocked, big).reshape(batch, -1)
            best = flat.argmin(-1)
            found = flat.gather(1, best[:, None]).squeeze(1).lt(big * 0.5)
            row, col = best.div(cols, rounding_mode="floor"), best.remainder(cols)
            picked = self._flag(used_row, found, row)
            dest = torch.where(picked, col[:, None].expand_as(dest), dest)
            used_row = used_row | picked
            used_col = used_col | self._flag(used_col, found, col)
        return dest

    def forward(
        self,
        obs: SlotObservation,
        prev_uid: torch.Tensor,
        prev_ent: torch.Tensor,
        prev_target: torch.Tensor,
        prev_alive: torch.Tensor,
        prior_slot: torch.Tensor | None = None,
        births: bool = False,
    ) -> SlotAlignment:
        count = self.num_slots
        if obs.uid.shape[-1] != count:
            raise ValueError(
                f"slot mode needs one latent slot per observation row: got "
                f"{obs.uid.shape[-1]} nodes for {count} slots. Set "
                f"graph.n_max to the slot count."
            )
        device = obs.uid.device
        index = torch.arange(count, device=device)
        valid = obs.mask.bool()
        # A padded row's UID is meaningless; zero it so it cannot match.
        uid = obs.uid.long() * valid
        prev_uid = prev_uid.long()
        # Occupancy is now an explicit state, not ``uid != 0``.
        alive = prev_alive.bool()
        idx = index[None, :].expand_as(alive)

        is_ee = valid & uid.eq(self.uid_ee)
        obj = valid & ~is_ee
        # Slot zero belongs to the end effector, so object matching never
        # considers it even if a stale UID somehow sits there.
        same = (
            uid.unsqueeze(-1).eq(prev_uid.unsqueeze(1))
            & obj.unsqueeze(-1)
            & (alive & index.gt(0)).unsqueeze(1)
        )
        matched_obs = same.any(-1)
        matched_slot = same.long().argmax(-1)
        taken = same.any(1) | index.eq(0)

        fresh = obj & ~matched_obs
        # The retained target keeps its slot even while unobserved, and no
        # non-target may take it. This is the one asymmetry in the table.
        retained_target = alive & prev_target.bool() & ~taken
        inactive = ~alive & ~taken & index.gt(0)
        stale = alive & ~taken & index.gt(0) & ~retained_target

        birth_dest = torch.full_like(matched_slot, count)
        if births and prior_slot is not None:
            cost = self._birth_cost(prior_slot, obs.slots)
            # A fresh target is assigned before any non-target, so a capacity
            # shortfall can never cost us the target.
            cost = cost - (fresh & obs.target.bool()).float()[..., None] * 1e6
            birth_dest = self._greedy(cost, fresh[..., None] & inactive[:, None, :])
        born_here = fresh & birth_dest.lt(count)
        claimed = self._place(
            torch.ones_like(birth_dest, dtype=torch.bool), birth_dest, born_here
        )

        # Whatever birth matching did not place falls through to index order:
        # inactive slots first, then replaceable live non-targets.
        free = inactive & ~claimed
        spare = stale & ~claimed
        usable = free | spare
        order = torch.where(
            free, idx, torch.where(spare, idx + count, idx + 4 * count)
        ).argsort(-1)
        left = fresh & ~born_here
        is_target = left & obs.target.bool()
        rank = torch.where(
            is_target,
            is_target.long().cumsum(-1) - 1,
            is_target.long().sum(-1, keepdim=True)
            + (left & ~is_target).long().cumsum(-1)
            - 1,
        )
        room = usable.long().sum(-1, keepdim=True)
        fits = left & rank.lt(room)
        fallback = order.gather(1, rank.clamp(0, count - 1))

        dropped = torch.full_like(matched_slot, count)
        dest = torch.where(
            is_ee,
            torch.zeros_like(matched_slot),
            torch.where(
                matched_obs,
                matched_slot,
                torch.where(
                    born_here,
                    birth_dest,
                    torch.where(fits, fallback, dropped),
                ),
            ),
        )
        keep = valid & dest.lt(count)

        present = self._place(torch.ones_like(dest, dtype=torch.bool), dest, keep)
        slots = self._place(obs.slots, dest, keep)
        uid_slot = self._place(uid, dest, keep)
        ent_slot = self._place(obs.ent.long(), dest, keep)
        target_slot = self._place(obs.target.long(), dest, keep)

        # An unobserved slot keeps its identity; a replaced one is overwritten
        # wholesale, which is what "clear the old latent" means here.
        new_uid = torch.where(present, uid_slot, prev_uid)
        new_ent = torch.where(present, ent_slot, prev_ent.long())
        new_target = torch.where(present, target_slot, prev_target.long())
        matched = torch.where(
            index.eq(0).unsqueeze(0),
            (is_ee.any(-1) & prev_uid[:, 0].eq(self.uid_ee)).unsqueeze(-1),
            same.any(1),
        )
        return SlotAlignment(
            dest=dest,
            present=present,
            matched=matched & present,
            born=present & ~alive & index.gt(0),
            # A live slot handed to a different entity: the prior predicted the
            # old one, so no dynamics loss has a correspondence this step.
            replaced=present & alive & ~matched & index.gt(0),
            slots=slots,
            uid=new_uid,
            ent=new_ent,
            target=new_target,
            # Presence is monotone inside the bounded memory: absence from the
            # graph is not death, and only a capacity replacement changes who
            # occupies a live slot.
            alive=(alive | present).to(torch.float32),
            overflow=(fresh & ~keep).sum(-1),
        )


class SlotReadout(nn.Module):
    """Learned-query attention pooling, for heads only.

    This is the one place pooling is allowed in slot mode. It never feeds the
    slot transition, is never a prediction target, and never decides identity;
    it exists so the ordinary Dreamer heads keep a fixed-width input.
    """

    def __init__(self, slot_dim: int, out_dim: int):
        super().__init__()
        self.slot_dim = int(slot_dim)
        self.out_dim = int(out_dim)
        self.query = nn.Parameter(torch.empty(self.slot_dim))
        self.key = nn.Linear(self.slot_dim, self.slot_dim)
        self.value = nn.Linear(self.slot_dim, self.slot_dim)
        self.out = nn.Linear(self.slot_dim, self.out_dim)
        self.apply(weight_init_)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, slots: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # (..., n, D), (..., n)
        mask = mask.bool()
        score = (self.key(slots) * self.query).sum(-1) / math.sqrt(self.slot_dim)
        # float32 for the mask: -1e9 is outside float16's finite range.
        score = score.float().masked_fill(~mask, -1e9)
        attention = torch.softmax(score, -1).to(slots.dtype) * mask
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-6)
        token = self.out((attention[..., None] * self.value(slots)).sum(-2))
        return token * mask.any(-1, keepdim=True)


class SlotGraphDecoder(nn.Module):
    """Slot-native reconstruction of the relation-only graph.

    Unlike :class:`SimpleGraphDecoder` there is no global vector to query and no
    UID embedding to query it with: a node's prediction comes from its own slot,
    and a fact's prediction from its two endpoint slots plus the relation type.
    The same heads read posterior slots and predicted prior slots, which is what
    makes relations decodable from an imagined rollout.
    """

    def __init__(self, config):
        super().__init__()
        self.units = int(config.slot_dim)
        self.n_abs = int(config.n_abs)
        self.n_temp = int(config.n_temp)
        act = str(config.act)

        self.node = GraphMLP(self.units, self.units, act)
        self.target = nn.Linear(self.units, 1)
        self.reltype = nn.Embedding(int(config.n_rel), int(config.embed))
        self.pair = GraphMLP(2 * self.units + int(config.embed), self.units, act)
        self.abs_head = nn.Linear(self.units, self.n_abs)
        self.temp_head = nn.Linear(self.units, self.n_temp)
        self.register_buffer(
            "abs_valid", _relation_masks(int(config.n_rel), self.n_abs, _edge_contract(config)), persistent=False
        )
        self.apply(weight_init_)

    def _pairs(self, slots, src, dst, relation):
        """Fact features for flattened endpoint indices."""
        flat = self.node(slots).reshape(-1, self.units)
        return self.pair(
            torch.cat(
                [
                    flat.index_select(0, src),
                    flat.index_select(0, dst),
                    self.reltype(relation),
                ],
                -1,
            )
        )

    def target_logits(self, slots):
        """Per-slot target logit. One head, one interpretation everywhere."""
        return self.target(self.node(slots)).squeeze(-1)

    def target_loss(self, slots, alive, target_flag, step_valid):
        """Categorical slot-or-null cross entropy.

        The same objective supervises the posterior slots and the predicted
        prior slots, so the logits mean ``P(target identity | S)`` in both and
        there is no scale conflict between a per-node BCE and a categorical
        head.
        """
        logits = slot_target_logits(self.target_logits(slots), alive)
        label, has_target = slot_target_label(target_flag, alive)
        keep = step_valid.reshape(-1).bool()
        flat = logits.reshape(-1, logits.shape[-1])
        error = F.cross_entropy(flat, label.reshape(-1), reduction="none")
        weight = keep.float()
        loss = (error * weight).sum() / weight.sum().clamp_min(1)
        with torch.no_grad():
            picked = flat.argmax(-1)
            accuracy = _masked(
                picked.eq(label.reshape(-1)), keep & has_target.reshape(-1)
            )
        return loss, accuracy

    def relation_probs(self, src_slot, dst_slot, relations):
        """Admissible-label distributions for one ordered pair of slots.

        ``src_slot``/``dst_slot`` are (..., D) and ``relations`` is a 1-D list
        of relation ids; the result is (..., R, n_abs). Used by the progress
        scorer, which must read predicted facts and never observed ones.
        """
        relations = relations.long()
        count = int(relations.numel())
        shape = src_slot.shape[:-1]
        src = self.node(src_slot).unsqueeze(-2).expand(*shape, count, self.units)
        dst = self.node(dst_slot).unsqueeze(-2).expand(*shape, count, self.units)
        rel = self.reltype(relations).expand(*shape, count, -1)
        pair = self.pair(torch.cat([src, dst, rel], -1))
        logits = self.abs_head(pair).float()
        legal = self.abs_valid.index_select(0, relations).expand(*shape, count, self.n_abs)
        return torch.softmax(logits.masked_fill(~legal, -1e9), -1)

    def forward(
        self,
        post_slots: torch.Tensor,
        prior_slots: torch.Tensor,
        compact: CompactGraph,
        slot_index: torch.Tensor,
        slot_alive: torch.Tensor,
        slot_target: torch.Tensor,
        step_valid: torch.Tensor,
        relations: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        graph_count = compact.graph_count
        count = post_slots.shape[-2]
        step = step_valid.reshape(graph_count).bool()
        post_slots = post_slots.reshape(graph_count, count, self.units)
        prior_slots = prior_slots.reshape(graph_count, count, self.units)
        alive = slot_alive.reshape(graph_count, count)
        target_flag = slot_target.reshape(graph_count, count)
        valid = alive.bool() & step[:, None]
        losses: dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor] = {}

        losses["nodetgt"], metrics["node_target_acc"] = self.target_loss(
            post_slots, alive, target_flag, step
        )
        # The prior target head is masked by *posterior* presence during
        # training. Masking by its own prediction would let a slot the prior
        # believes dead carry the correct label at an infinite loss.
        losses["prior_nodetgt"], metrics["prior_target_acc"] = self.target_loss(
            prior_slots, alive, target_flag, step
        )
        label, has_target = slot_target_label(target_flag, alive)
        metrics["node_target_frac"] = has_target.float().mean()

        # Edge endpoints address observation rows; route them through the
        # alignment so both branches read the same slots the dynamics carry.
        route = slot_index.reshape(-1)
        src_slot = route.index_select(0, compact.edge_src)
        dst_slot = route.index_select(0, compact.edge_dst)
        routed = src_slot.lt(count) & dst_slot.lt(count)
        offset = compact.edge_graph * count
        src = src_slot.clamp(max=count - 1) + offset
        dst = dst_slot.clamp(max=count - 1) + offset
        edge_step = step.index_select(0, compact.edge_graph) & routed
        legal = self.abs_valid.index_select(0, compact.edge_rel)
        temporal = edge_step & compact.edge_temp.ne(0)

        post_pair = self._pairs(post_slots, src, dst, compact.edge_rel)
        losses["relabs"], metrics["relabs_acc"] = _edge_categorical(
            self.abs_head(post_pair),
            compact.edge_abs,
            compact.edge_graph,
            edge_step,
            legal,
            graph_count,
        )
        temp_logits = self.temp_head(post_pair)
        temp_classes = torch.ones_like(temp_logits, dtype=torch.bool)
        temp_classes[:, 0] = False
        losses["reltemp"], metrics["reltemp_acc"] = _edge_categorical(
            temp_logits,
            compact.edge_temp,
            compact.edge_graph,
            temporal,
            temp_classes,
            graph_count,
        )

        # Prior relations are supervised on the one pair progress is computed
        # from, using the *observed* target identity as the teacher. Letting the
        # predicted target choose which edge supplies the label would make the
        # model gather facts about the wrong object and reinforce its own error.
        teacher = label.clamp(max=count - 1) + 1  # object slot index, 1..n-1
        edge_target = teacher.index_select(0, compact.edge_graph)
        progress_edge = (
            edge_step
            & src_slot.eq(0)
            & dst_slot.eq(edge_target)
            & has_target.index_select(0, compact.edge_graph)
            & (compact.edge_rel.unsqueeze(-1) == relations.reshape(1, -1)).any(-1)
        )
        prior_pair = self._pairs(prior_slots, src, dst, compact.edge_rel)
        (
            losses["prior_progress_relabs"],
            metrics["prior_progress_acc"],
        ) = _edge_categorical(
            self.abs_head(prior_pair),
            compact.edge_abs,
            compact.edge_graph,
            progress_edge,
            legal,
            graph_count,
            only_present=True,
        )

        metrics["graph_real_edges"] = torch.as_tensor(
            compact.edge_count / max(graph_count, 1),
            device=post_slots.device,
            dtype=torch.float32,
        )
        metrics["graph_nodes_per_frame"] = valid.float().sum(-1).mean()
        metrics["slot_progress_facts"] = (
            progress_edge.float().sum() / max(graph_count, 1)
        )
        metrics["slot_unrouted_facts"] = (
            (~routed).float().sum() / max(graph_count, 1)
        )
        return losses, metrics


class GraphDecoder(nn.Module):
    """Auxiliary graph heads with the same per-frame weighting as the JAX model."""

    def __init__(self, config):
        super().__init__()
        self.units = int(config.units)
        self.app_dim = int(config.app_dim)
        self.n_cams = int(config.n_cams)
        self.n_abs = int(config.n_abs)
        self.n_temp = int(config.n_temp)
        self.bbox_beta = float(config.bbox_beta)

        self.app = nn.Linear(self.units, self.n_cams * self.app_dim)
        self.bbox = nn.Linear(self.units, self.n_cams * 4)
        self.visibility = nn.Linear(self.units, self.n_cams)
        self.target = nn.Linear(self.units, 1)
        self.reltype = nn.Embedding(int(config.n_rel), int(config.embed))
        self.pair = GraphMLP(
            2 * self.units + int(config.embed), self.units, str(config.act)
        )
        self.abs_head = nn.Linear(self.units, self.n_abs)
        self.temp_head = nn.Linear(self.units, self.n_temp)
        self.register_buffer(
            "abs_valid", _relation_masks(int(config.n_rel), self.n_abs, _edge_contract(config)), persistent=False
        )
        self.apply(weight_init_)

    def forward(
        self,
        encoded: GraphEncoding,
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

        # Reconstruct the exact per-node target flag. Slot zero is reserved for
        # the end effector, so only valid object rows participate in this loss.
        target_logit = self.target(nodes).squeeze(-1).float()
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
        target_frame_loss = (
            self._frame_mean(target_error, positive) * has_positive
            + self._frame_mean(target_error, negative) * has_negative
        ) / class_count.clamp_min(1)
        losses["nodetgt"] = target_frame_loss.mean()

        # Report exact target-instance selection accuracy on frames that name a
        # target, rather than per-node binary accuracy dominated by negatives.
        target_index = target_flag.long().argmax(-1)
        selected = target_logit.masked_fill(~target_mask, -1e9).argmax(-1)
        metrics["node_target_acc"] = self._masked(selected.eq(target_index), has_positive)
        metrics["node_target_frac"] = has_positive.float().mean()

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

        metrics["graph_real_edges"] = torch.as_tensor(
            compact.edge_count / max(graph_count, 1),
            device=nodes.device,
            dtype=torch.float32,
        )
        return losses, metrics

    def _edge_categorical(self, logits, target, frame, mask, classes, graph_count):
        return _edge_categorical(logits, target, frame, mask, classes, graph_count)

    _frame_mean = staticmethod(_frame_mean)
    _masked = staticmethod(_masked)

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
