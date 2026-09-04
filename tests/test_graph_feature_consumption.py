"""That entity, box and centroid actually reach the encoder.

The observation carrying a field and the model reading it are different
claims, and only the second one matters. A projection whose input slice is
misaligned, a feature zeroed by a reshape, a buffer that never made it into
``node_in`` -- each leaves the field present in the observation and absent
from every gradient, and nothing downstream reports it. The graph would then
be an expensive way to feed the model a constant.

The test is behavioural rather than structural: perturb exactly one field,
hold every other byte fixed, and require the encoding to move. A field the
encoder does not read cannot change its output.

Needs torch, so it runs on the server with the rest of the graph stack.
"""

import unittest


def _torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("torch is not installed")


class FeatureConsumptionTest(unittest.TestCase):

    N_MAX, E_MAX, N_CAMS = 6, 24, 2

    @classmethod
    def setUpClass(cls):
        _torch()

    def _config(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            simple_units=32, layers=2, n_cams=self.N_CAMS, embed=16,
            reverse_edges=True, entity_vocab=12, n_rel=12, n_abs=19,
            n_temp=6, act="SiLU", centroid_origin=(0.0, 0.0, 0.0),
            centroid_scale=5.0,
        )

    def _encoder(self):
        import torch

        from graph import GraphEncoder

        torch.manual_seed(0)
        return GraphEncoder(self._config()).eval()

    def _graph(self):
        """One batch element: three real nodes and two real facts."""
        import torch

        graph = {
            "graph_node_ent": torch.zeros(1, self.N_MAX, dtype=torch.long),
            "graph_node_target": torch.zeros(1, self.N_MAX, dtype=torch.long),
            "graph_node_bbox": torch.zeros(1, self.N_MAX, self.N_CAMS, 4),
            "graph_node_centroid": torch.zeros(1, self.N_MAX, 3),
            "graph_edge_src": torch.zeros(1, self.E_MAX, dtype=torch.long),
            "graph_edge_dst": torch.zeros(1, self.E_MAX, dtype=torch.long),
            "graph_edge_rel": torch.zeros(1, self.E_MAX, dtype=torch.long),
            "graph_edge_abs": torch.zeros(1, self.E_MAX, dtype=torch.long),
            "graph_edge_temp": torch.zeros(1, self.E_MAX, dtype=torch.long),
        }
        # ee, target, site -- the protected rows.
        graph["graph_node_ent"][0, :3] = torch.tensor([1, 5, 7])
        graph["graph_node_target"][0, 1] = 1
        graph["graph_node_bbox"][0, :3] = 0.25
        graph["graph_node_centroid"][0, :3] = torch.tensor(
            [[0.1, 0.2, 0.9], [0.3, 0.1, 0.5], [0.0, 0.4, 0.7]])
        graph["graph_edge_src"][0, :2] = torch.tensor([0, 0])
        graph["graph_edge_dst"][0, :2] = torch.tensor([1, 2])
        graph["graph_edge_rel"][0, :2] = torch.tensor([3, 4])
        graph["graph_edge_abs"][0, :2] = torch.tensor([2, 6])
        return graph

    def _encode(self, encoder, graph):
        import torch

        with torch.no_grad():
            out = encoder(graph)
        # ``token`` is the pooled readout the RSSM consumes -- the one place
        # every node and every fact has to have arrived to matter.
        return out.token

    def _moves(self, mutate):
        """Whether changing one field changes the encoding."""
        import torch

        encoder = self._encoder()
        base = self._graph()
        changed = {k: v.clone() for k, v in base.items()}
        mutate(changed)
        self.assertFalse(
            all(torch.equal(base[k], changed[k]) for k in base),
            "the mutation changed no input, so the test would pass vacuously")
        return not torch.allclose(
            self._encode(encoder, base), self._encode(encoder, changed),
            atol=1e-7)

    # ---- the three node features ---------------------------------------- #
    def test_entity_identity_reaches_the_encoder(self):
        """Which object a row is, not merely that a row is occupied."""
        self.assertTrue(self._moves(
            lambda g: g["graph_node_ent"].__setitem__((0, 1), 9)))

    def test_the_bounding_box_reaches_the_encoder(self):
        """Where a node is on each screen, and where it goes when it leaves."""
        self.assertTrue(self._moves(
            lambda g: g["graph_node_bbox"].__setitem__((0, 1), 0.75)))

    def test_the_centroid_reaches_the_encoder(self):
        """Where a node is in the world, which the box cannot say once the
        object is out of frame."""
        self.assertTrue(self._moves(
            lambda g: g["graph_node_centroid"].__setitem__(
                (0, 1), __import__("torch").tensor([2.0, -1.0, 0.4]))))

    def test_the_target_flag_reaches_the_encoder(self):
        """Which of several same-category instances the subtask acts on."""
        self.assertTrue(self._moves(
            lambda g: (g["graph_node_target"].__setitem__((0, 1), 0),
                       g["graph_node_target"].__setitem__((0, 2), 1))))

    # ---- per-camera boxes are not collapsed ------------------------------ #
    def test_each_camera_box_is_read_separately(self):
        """A reshape that folded the camera axis would leave one camera's box
        unreadable while both still arrived."""
        for cam in range(self.N_CAMS):
            with self.subTest(camera=cam):
                self.assertTrue(self._moves(
                    lambda g, c=cam: g["graph_node_bbox"].__setitem__(
                        (0, 1, c), 0.9)))

    def test_each_centroid_axis_is_read_separately(self):
        for axis in range(3):
            with self.subTest(axis=axis):
                self.assertTrue(self._moves(
                    lambda g, a=axis: g["graph_node_centroid"].__setitem__(
                        (0, 1, a), 3.0)))

    # ---- the edge features ----------------------------------------------- #
    def test_the_relation_reaches_the_encoder(self):
        self.assertTrue(self._moves(
            lambda g: g["graph_edge_rel"].__setitem__((0, 0), 7)))

    def test_the_absolute_label_reaches_the_encoder(self):
        """The whole point of the graph: what the fact *says*, not that a
        fact exists."""
        self.assertTrue(self._moves(
            lambda g: g["graph_edge_abs"].__setitem__((0, 0), 11)))

    def test_the_endpoints_reach_the_encoder(self):
        self.assertTrue(self._moves(
            lambda g: g["graph_edge_dst"].__setitem__((0, 0), 2)))

    # ---- padding must not ------------------------------------------------ #
    def test_padded_node_rows_are_ignored(self):
        """A row with the pad entity id is not a vertex, and writing features
        into one must not move the encoding."""
        self.assertFalse(self._moves(
            lambda g: (g["graph_node_bbox"].__setitem__((0, 5), 0.9),
                       g["graph_node_centroid"].__setitem__((0, 5), 7.0))))

    def test_padded_edge_rows_are_ignored(self):
        self.assertFalse(self._moves(
            lambda g: (g["graph_edge_abs"].__setitem__((0, 10), 5),
                       g["graph_edge_src"].__setitem__((0, 10), 1))))

    # ---- gradients actually flow ----------------------------------------- #
    def test_the_continuous_features_receive_gradient(self):
        """Reading a value and learning from it are different; a detached
        feature would pass every test above."""
        encoder = self._encoder()
        graph = self._graph()
        for key in ("graph_node_bbox", "graph_node_centroid"):
            with self.subTest(field=key):
                tensor = graph[key].clone().requires_grad_(True)
                fed = dict(graph)
                fed[key] = tensor
                encoder(fed).token.sum().backward()
                self.assertIsNotNone(tensor.grad)
                self.assertGreater(float(tensor.grad.abs().sum()), 0.0)

    def test_the_embeddings_receive_gradient(self):
        encoder = self._encoder()
        encoder(self._graph()).token.sum().backward()
        for name in ("entity", "target", "relation", "absolute"):
            with self.subTest(embedding=name):
                weight = getattr(encoder, name).weight
                self.assertIsNotNone(weight.grad)
                self.assertGreater(float(weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
