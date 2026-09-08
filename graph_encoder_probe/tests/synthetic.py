"""Packed graphs without a simulator.

The real collector needs ManiSkill, a mined whitelist and a GPU-free CPU env;
the properties the probe depends on -- padding is stripped, labels are legal,
an edit lands where the spec says -- are properties of the packed arrays alone.
So the tests build those arrays directly, honouring the same contract
``pack_graph`` writes: index zero is padding everywhere, node rows fill a dense
prefix, edge rows fill a dense prefix, and every absolute label is one its
relation may legally take.
"""

from __future__ import annotations

import numpy as np

from scenegraph.adapters.graph_pack import GRAPH_KEYS
from scenegraph.adapters.graph_vocab import build_relation_vocab, build_temporal_vocab
from scenegraph.core.relation_rules import TEMPORAL_RELATIONS

from ..dataset import FIELD_DTYPES, GraphDataset, GraphFrames
from ..pairs import legal_absolute_mask


def _relation_ids() -> tuple[dict[int, str], set[int]]:
    names = {i: token for token, i in build_relation_vocab().token_to_id.items()}
    temporal = {rid for rid, name in names.items() if name in TEMPORAL_RELATIONS}
    return names, temporal


def make_frames(
    count: int = 32,
    *,
    n_max: int = 8,
    e_max: int = 24,
    n_cams: int = 2,
    entity_vocab: int = 6,
    seed: int = 0,
) -> GraphFrames:
    """``count`` packed graphs with enough variety for all four edits.

    Each frame is guaranteed to carry at least one temporal label and at least
    two edges of one relation that hold different labels over different node
    pairs, so no edit group can fail for want of a candidate.
    """
    rng = np.random.default_rng(int(seed))
    abs_valid = legal_absolute_mask()
    n_rel = int(abs_valid.shape[0])
    n_temp = len(build_temporal_vocab())
    _names, temporal_ids = _relation_ids()
    multi = [r for r in range(1, n_rel) if int(abs_valid[r].sum()) >= 2]
    multi_temporal = [r for r in multi if r in temporal_ids]

    fields = {
        "graph_node_ent": np.zeros((count, n_max), np.uint8),
        "graph_node_bbox": np.zeros((count, n_max, n_cams, 4), np.float16),
        "graph_node_centroid": np.zeros((count, n_max, 3), np.float32),
        "graph_node_target": np.zeros((count, n_max), np.uint8),
        "graph_edge_src": np.zeros((count, e_max), np.uint8),
        "graph_edge_dst": np.zeros((count, e_max), np.uint8),
        "graph_edge_rel": np.zeros((count, e_max), np.uint8),
        "graph_edge_abs": np.zeros((count, e_max), np.uint8),
        "graph_edge_temp": np.zeros((count, e_max), np.uint8),
    }

    for f in range(count):
        n_nodes = int(rng.integers(3, n_max + 1))
        fields["graph_node_ent"][f, :n_nodes] = rng.integers(1, entity_vocab, n_nodes)
        fields["graph_node_centroid"][f, :n_nodes] = rng.uniform(-0.6, 0.6, (n_nodes, 3))
        for node in range(n_nodes):
            for cam in range(n_cams):
                if rng.random() < 0.2:
                    continue                       # this camera does not see it
                x0, y0 = rng.uniform(0.0, 0.7, 2)
                # Layout is [x0, x1, y0, y1] with exclusive maxima; validity is
                # read back as x1 > x0 and y1 > y0.
                fields["graph_node_bbox"][f, node, cam] = [
                    x0, x0 + rng.uniform(0.05, 0.3), y0, y0 + rng.uniform(0.05, 0.3)
                ]

        rows: list[tuple[int, int, int, int, int]] = []

        # Two edges of one relation, different labels over different pairs: the
        # assignment edit has nothing to swap without them.
        relation = int(rng.choice(multi_temporal or multi))
        labels = np.flatnonzero(abs_valid[relation])
        first, second = rng.choice(labels, size=2, replace=False)
        pairs = _distinct_pairs(rng, n_nodes, 2)
        for (src, dst), label in zip(pairs, (first, second)):
            temp = int(rng.integers(1, n_temp)) if relation in temporal_ids else 0
            rows.append((src, dst, relation, int(label), temp))

        # At least one temporal label, wherever the relation above was not one.
        if relation not in temporal_ids:
            spatial = int(rng.choice(sorted(temporal_ids)))
            src, dst = _distinct_pairs(rng, n_nodes, 1)[0]
            rows.append(
                (src, dst, spatial, int(rng.choice(np.flatnonzero(abs_valid[spatial]))),
                 int(rng.integers(1, n_temp)))
            )

        for _ in range(int(rng.integers(2, max(3, e_max // 3)))):
            if len(rows) >= e_max:
                break
            extra = int(rng.integers(1, n_rel))
            src, dst = _distinct_pairs(rng, n_nodes, 1)[0]
            temp = int(rng.integers(1, n_temp)) if extra in temporal_ids else 0
            rows.append(
                (src, dst, extra, int(rng.choice(np.flatnonzero(abs_valid[extra]))), temp)
            )

        for i, (src, dst, rel, label, temp) in enumerate(rows[:e_max]):
            fields["graph_edge_src"][f, i] = src
            fields["graph_edge_dst"][f, i] = dst
            fields["graph_edge_rel"][f, i] = rel
            fields["graph_edge_abs"][f, i] = label
            fields["graph_edge_temp"][f, i] = temp

    return GraphFrames({key: fields[key].astype(FIELD_DTYPES[key], copy=False) for key in GRAPH_KEYS})


def _distinct_pairs(rng, n_nodes: int, count: int) -> list[tuple[int, int]]:
    """``count`` distinct ordered node pairs, source never equal to destination."""
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    while len(out) < count:
        src, dst = (int(v) for v in rng.integers(0, n_nodes, 2))
        if src == dst or (src, dst) in seen:
            continue
        seen.add((src, dst))
        out.append((src, dst))
    return out


def make_dataset(count: int = 32, *, entity_vocab: int = 6, n_cams: int = 2, **kwargs) -> GraphDataset:
    frames = make_frames(count, entity_vocab=entity_vocab, n_cams=n_cams, **kwargs)
    index = {
        "episode": np.zeros(count, np.int32),
        "seed": np.zeros(count, np.int64),
        "frame": np.arange(count, dtype=np.int32),
        "success": np.ones(count, bool),
    }
    meta = {
        "env_id": "synthetic",
        "n_cams": n_cams,
        "cameras": [f"cam{i}" for i in range(n_cams)],
        "vocab_sizes": {
            "entity": entity_vocab,
            "relation": len(build_relation_vocab()),
            "absolute": int(legal_absolute_mask().shape[1]),
            "temporal": len(build_temporal_vocab()),
        },
        "revision": "synthetic",
    }
    return GraphDataset(frames, index, meta)
