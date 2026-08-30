"""Benchmark compact graph forward and backward at different replay widths."""

import argparse
import pathlib
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from graph import GraphEncoder, SimpleGraphDecoder  # noqa: E402


def make_config(args):
    return SimpleNamespace(
        units=args.units,
        layers=args.layers,
        n_cams=2,
        app_dim=384,
        embed=64,
        app=64,
        bbox=8,
        reverse_edges=True,
        entity_vocab=14,
        n_rel=12,
        n_abs=19,
        n_temp=6,
        act="SiLU",
        bbox_beta=0.1,
    )


def make_graph(args, edge_width, device):
    if args.real_edges > edge_width:
        raise ValueError("real_edges cannot exceed edge_width")
    shape = (args.batch, args.length)
    graph = {
        "graph_node_ent": torch.zeros(*shape, args.nodes, dtype=torch.uint16, device=device),
        "graph_node_bbox": torch.zeros(*shape, args.nodes, 2, 4, dtype=torch.float16, device=device),
        "graph_node_target": torch.zeros(*shape, args.nodes, dtype=torch.uint8, device=device),
        "graph_node_centroid": torch.zeros(*shape, args.nodes, 3, dtype=torch.float32, device=device),
        "graph_edge_src": torch.zeros(*shape, edge_width, dtype=torch.uint8, device=device),
        "graph_edge_dst": torch.zeros(*shape, edge_width, dtype=torch.uint8, device=device),
        "graph_edge_rel": torch.zeros(*shape, edge_width, dtype=torch.uint8, device=device),
        "graph_edge_abs": torch.zeros(*shape, edge_width, dtype=torch.uint8, device=device),
        "graph_edge_temp": torch.zeros(*shape, edge_width, dtype=torch.uint8, device=device),
    }
    valid_nodes = min(args.valid_nodes, args.nodes)
    ids = torch.arange(1, valid_nodes + 1, device=device).to(torch.uint16)
    graph["graph_node_ent"][..., :valid_nodes] = ids
    graph["graph_node_bbox"][..., :valid_nodes, :, :] = torch.tensor(
        [0.1, 0.4, 0.2, 0.6], dtype=torch.float16, device=device
    )
    graph["graph_node_target"][..., 1] = 1

    index = torch.arange(args.real_edges, device=device)
    src = (index % valid_nodes).to(torch.uint8)
    dst = ((index + 1) % valid_nodes).to(torch.uint8)
    spatial = index.remainder(2).bool()
    rel = torch.where(spatial, torch.tensor(5, device=device), torch.tensor(1, device=device)).to(torch.uint8)
    absolute = torch.where(spatial, torch.tensor(3, device=device), torch.tensor(2, device=device)).to(torch.uint8)
    temporal = torch.where(spatial, torch.tensor(3, device=device), torch.tensor(0, device=device)).to(torch.uint8)
    graph["graph_edge_src"][..., : args.real_edges] = src
    graph["graph_edge_dst"][..., : args.real_edges] = dst
    graph["graph_edge_rel"][..., : args.real_edges] = rel
    graph["graph_edge_abs"][..., : args.real_edges] = absolute
    graph["graph_edge_temp"][..., : args.real_edges] = temporal
    return graph


def measure(args, encoder, decoder, edge_width, device):
    torch.manual_seed(0)
    graph = make_graph(args, edge_width, device)
    valid = torch.ones(args.batch, args.length, dtype=torch.bool, device=device)

    def step():
        encoder.zero_grad(set_to_none=True)
        decoder.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            encoded = encoder(graph)
            losses, _ = decoder(encoded, valid)
            loss = sum(losses.values()) + encoded.token.square().mean()
        loss.backward()
        return encoded.compact.edge_count

    for _ in range(args.warmup):
        edge_count = step()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.steps):
        edge_count = step()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000 / args.steps
    return elapsed, edge_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-widths", nargs="+", type=int, default=[96, 168])
    parser.add_argument("--real-edges", type=int, default=72)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--valid-nodes", type=int, default=7)
    parser.add_argument("--units", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    device = torch.device("cuda")
    cfg = make_config(args)
    encoder = GraphEncoder(cfg).to(device)
    decoder = SimpleGraphDecoder(cfg, int(cfg.semantic_dim)).to(device)
    results = []
    for width in args.edge_widths:
        milliseconds, count = measure(args, encoder, decoder, width, device)
        results.append(milliseconds)
        print(
            f"e_max={width:4d} real_edges/batch={count:6d} "
            f"forward+backward={milliseconds:8.3f} ms"
        )
    if len(results) == 2:
        print(f"timing ratio={max(results) / min(results):.3f}x (target <= 1.20x)")


if __name__ == "__main__":
    main()
