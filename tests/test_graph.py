"""Shared graph batch fixture."""

import torch


def graph_batch(edge_width, batch=2, time=3, nodes=8):
    """One padded graph batch under the packed contract."""
    shape = (batch, time)
    out = {
        "graph_node_ent": torch.zeros(*shape, nodes, dtype=torch.uint8),
        "graph_node_bbox": torch.zeros(*shape, nodes, 2, 4, dtype=torch.float16),
        "graph_node_centroid": torch.zeros(*shape, nodes, 3),
        "graph_node_target": torch.zeros(*shape, nodes, dtype=torch.uint8),
        "graph_edge_src": torch.zeros(*shape, edge_width, dtype=torch.uint8),
        "graph_edge_dst": torch.zeros(*shape, edge_width, dtype=torch.uint8),
        "graph_edge_rel": torch.zeros(*shape, edge_width, dtype=torch.uint8),
        "graph_edge_abs": torch.zeros(*shape, edge_width, dtype=torch.uint8),
        "graph_edge_temp": torch.zeros(*shape, edge_width, dtype=torch.uint8),
    }
    out["graph_node_ent"][..., :3] = torch.tensor([1, 2, 3], dtype=torch.uint8)
    out["graph_node_target"][..., 1] = 1
    out["graph_node_bbox"][..., :3, :, :] = torch.tensor(
        [0.1, 0.4, 0.2, 0.6], dtype=torch.float16)
    out["graph_node_centroid"][..., :3, :] = torch.randn(batch, time, 3, 3)
    for key in ("graph_edge_rel", "graph_edge_abs", "graph_edge_temp"):
        out[key][..., :4] = torch.arange(1, 5, dtype=torch.uint8)
    out["graph_edge_dst"][..., :4] = 1
    return out
