import unittest
from types import SimpleNamespace

import torch

from graph import GraphDecoder, GraphEncoder, compact_graph


def config(units=16, app_dim=8):
    return SimpleNamespace(
        units=units,
        layers=2,
        n_cams=2,
        app_dim=app_dim,
        embed=4,
        app=4,
        bbox=2,
        reverse_edges=True,
        entity_vocab=14,
        n_rel=11,
        n_abs=17,
        n_temp=6,
        act="SiLU",
        bbox_beta=0.1,
    )


def graph_batch(edge_width, batch=2, time=3, nodes=10, app_dim=8):
    out = {
        "graph_node_ent": torch.zeros(batch, time, nodes, dtype=torch.uint16),
        "graph_node_app": torch.zeros(batch, time, nodes, 2, app_dim),
        "graph_node_bbox": torch.zeros(batch, time, nodes, 2, 4),
        "graph_node_target": torch.zeros(batch, time, nodes, dtype=torch.uint8),
        "graph_edge_src": torch.zeros(batch, time, edge_width, dtype=torch.uint8),
        "graph_edge_dst": torch.zeros(batch, time, edge_width, dtype=torch.uint8),
        "graph_edge_rel": torch.zeros(batch, time, edge_width, dtype=torch.uint8),
        "graph_edge_abs": torch.zeros(batch, time, edge_width, dtype=torch.uint8),
        "graph_edge_temp": torch.zeros(batch, time, edge_width, dtype=torch.uint8),
    }
    out["graph_node_ent"][..., :3] = torch.tensor([1, 2, 3], dtype=torch.uint16)
    out["graph_node_app"][..., :3, :, :] = torch.randn(batch, time, 3, 2, app_dim)
    out["graph_node_bbox"][..., :3, :, :] = torch.tensor([0.1, 0.4, 0.2, 0.6])
    out["graph_node_target"][..., 1] = 1
    out["graph_edge_src"][..., :2] = torch.tensor([0, 1], dtype=torch.uint8)
    out["graph_edge_dst"][..., :2] = torch.tensor([1, 2], dtype=torch.uint8)
    out["graph_edge_rel"][..., :2] = torch.tensor([1, 5], dtype=torch.uint8)
    out["graph_edge_abs"][..., :2] = torch.tensor([2, 3], dtype=torch.uint8)
    out["graph_edge_temp"][..., :2] = torch.tensor([0, 3], dtype=torch.uint8)
    return out


class CompactGraphTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_compaction_keeps_only_real_edges(self):
        compact = compact_graph(graph_batch(270))
        self.assertEqual(compact.edge_count, 2 * 3 * 2)
        self.assertTrue(compact.edge_rel.ne(0).all())
        self.assertEqual(compact.edge_src.max().item(), 5 * 10 + 1)
        self.assertEqual(compact.edge_dst.max().item(), 5 * 10 + 2)

    def test_padding_width_does_not_change_encoder_result(self):
        small = graph_batch(96)
        large = graph_batch(270)
        edge_keys = {"graph_edge_src", "graph_edge_dst", "graph_edge_rel", "graph_edge_abs", "graph_edge_temp"}
        for key in small:
            if key in edge_keys:
                large[key][..., :96].copy_(small[key])
            else:
                large[key].copy_(small[key])
        encoder = GraphEncoder(config()).eval()
        small_out = encoder(small)
        large_out = encoder(large)
        torch.testing.assert_close(small_out.nodes, large_out.nodes, rtol=0, atol=0)
        torch.testing.assert_close(small_out.token, large_out.token, rtol=0, atol=0)
        self.assertEqual(small_out.compact.edge_count, large_out.compact.edge_count)

    def test_decoder_losses_are_finite_and_backpropagate(self):
        encoder = GraphEncoder(config())
        decoder = GraphDecoder(config(), sem_dim=8)
        encoded = encoder(graph_batch(96))
        sem = torch.randn(2, 3, 2, 4, requires_grad=True)
        step_valid = torch.tensor([[True, True, False], [True, True, False]])
        losses, metrics = decoder(encoded, sem, step_valid)
        self.assertEqual(set(losses), {"node", "relabs", "reltemp", "semtgt"})
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))
        total = sum(losses.values()) + encoded.token.square().mean()
        total.backward()
        self.assertIsNotNone(encoder.node.net[0].weight.grad)
        self.assertIsNotNone(sem.grad)


if __name__ == "__main__":
    unittest.main()
