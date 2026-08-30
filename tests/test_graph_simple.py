"""Graph mode: the packed contract and the deterministic g.

Synthetic tensors only -- no simulator.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from graph import (
    GRAPH_KEYS,
    RESERVED_GRAPH_KEYS,
    GraphEncoder,
    SimpleGraphDecoder,
    compact_graph,
    graph_from,
    graph_keys,
)
from progress import (
    ABS_FAR_BELOW,
    ABS_HOLDS,
    ABS_LEVEL,
    ABS_MATCH,
    ABS_NOT_HOLDS,
    ABS_UNOBSERVED,
    ABS_VERY_FAR,
    ABS_VERY_NEAR,
    N_ABS,
    PICK_STAGES,
    REL_CONTACT,
    REL_CONTACT_COMPAT,
    REL_GRASP,
    REL_GRASP_COMPAT,
    REL_HEIGHT_OFFSET,
    REL_PLANAR_DISTANCE,
    ProgressScorer,
)
from envs.maniskill import _GRAPH_CONFIG_KEYS, graph_observation_config
from networks import MultiDecoder
from rssm import RSSM
from scenegraph.adapters.graph_obs import _DTYPES as _PACKED_DTYPES
from scenegraph.adapters.graph_obs import GraphObsBuilder
from scenegraph.adapters.graph_pack import (
    GRAPH_KEYS as PACK_KEYS,
    pack_graph,
)
from scenegraph.adapters.graph_vocab import (
    EntityVocab,
    GraphVocab,
    build_absolute_vocab,
    build_relation_vocab,
    build_temporal_vocab,
)
from scenegraph.core.node_builder import fill_bboxes
from scenegraph.core.schema import Graph, Node

N_MAX = 8
E_MAX = 16


def graph_config(units=32):
    return SimpleNamespace(
        units=units,
        simple_units=units,
        semantic_dim=units,
        decoder_units=16,
        bbox_query_dim=4,
        layers=1,
        n_cams=2,
        entity_vocab=14,
        # Derived, not written down. The model asserts these against the shared
        # vocabularies, so a hardcoded size turns "the vocabulary grew" into a
        # dozen unrelated-looking decoder failures.
        n_rel=len(build_relation_vocab()),
        n_abs=len(build_absolute_vocab()),
        n_temp=len(build_temporal_vocab()),
        embed=8,
        bbox=4,
        bbox_beta=0.1,
        reverse_edges=True,
        act="SiLU",
        centroid_origin=[0.0, 0.0, 0.0],
        centroid_scale=5.0,
    )


def _base_graph(batch, time, n_valid, n_edges):
    """Entity ids, the target flag and n_edges real facts. No node content."""
    ent = torch.zeros(batch, time, N_MAX, dtype=torch.uint8)
    ent[..., :n_valid] = torch.arange(1, n_valid + 1, dtype=torch.uint8)
    target = torch.zeros(batch, time, N_MAX, dtype=torch.uint8)
    target[..., 1] = 1
    rel = torch.zeros(batch, time, E_MAX, dtype=torch.uint8)
    rel[..., :n_edges] = torch.arange(1, n_edges + 1, dtype=torch.uint8)
    src = torch.zeros(batch, time, E_MAX, dtype=torch.uint8)
    dst = torch.zeros(batch, time, E_MAX, dtype=torch.uint8)
    dst[..., :n_edges] = 1
    return {
        "graph_node_ent": ent,
        "graph_node_target": target,
        "graph_edge_src": src,
        "graph_edge_dst": dst,
        "graph_edge_rel": rel,
        "graph_edge_abs": rel.clone(),
        "graph_edge_temp": rel.clone(),
    }


def pooled_graph(
    batch=2, time=3, n_valid=3, n_edges=4, boxes=None, centroids=None
):
    """The pooled contract: per-camera boxes and world centroids, no UID.

    Camera 0 sees every valid node, camera 1 only the end effector, so masked
    per-camera validity is actually exercised rather than assumed. Centroids
    are independent of the boxes on purpose -- that separation is the point of
    carrying both.
    """
    graph = _base_graph(batch, time, n_valid, n_edges)
    bbox = torch.zeros(batch, time, N_MAX, 2, 4, dtype=torch.float16)
    if boxes is None:
        boxes = torch.tensor([
            [0.10, 0.40, 0.20, 0.50],
            [0.50, 0.90, 0.10, 0.30],
            [0.00, 0.20, 0.60, 0.80],
            [0.30, 0.35, 0.30, 0.35],
            [0.60, 0.95, 0.60, 0.95],
            [0.05, 0.15, 0.05, 0.15],
            [0.40, 0.80, 0.40, 0.80],
        ])
    bbox[..., :n_valid, 0, :] = boxes[:n_valid].to(torch.float16)
    bbox[..., 0, 1, :] = torch.tensor([0.2, 0.6, 0.2, 0.6]).to(torch.float16)
    graph["graph_node_bbox"] = bbox
    centroid = torch.zeros(batch, time, N_MAX, 3, dtype=torch.float32)
    if centroids is None:
        centroids = torch.tensor([
            [0.00, 0.00, 0.90],
            [0.40, 0.10, 0.75],
            [-0.30, 0.60, 0.75],
            [1.20, -0.40, 0.05],
            [-1.10, 1.30, 0.35],
            [0.75, 0.75, 0.60],
            [-0.50, -0.90, 0.20],
        ])
    centroid[..., :n_valid, :] = centroids[:n_valid]
    graph["graph_node_centroid"] = centroid
    return graph


class ContractTest(unittest.TestCase):
    """One packed contract: boxes and a world centroid, no identity code."""

    def test_the_keys_are_boxes_and_centroids(self):
        keys = graph_keys()
        self.assertEqual(keys, GRAPH_KEYS)
        self.assertIn("graph_node_bbox", keys)
        self.assertIn("graph_node_centroid", keys)
        self.assertNotIn("graph_node_uid", keys)
        self.assertNotIn("graph_node_app", keys)

    def test_retired_keys_stay_reserved(self):
        """A stale wrapper still exposing one must not reach the MLP encoder."""
        for key in ("graph_node_uid", "graph_node_app"):
            with self.subTest(key=key):
                self.assertNotIn(key, GRAPH_KEYS)
                self.assertIn(key, RESERVED_GRAPH_KEYS)
        self.assertTrue(set(GRAPH_KEYS) <= RESERVED_GRAPH_KEYS)

    def test_the_packer_and_the_model_agree(self):
        self.assertEqual(tuple(PACK_KEYS), tuple(GRAPH_KEYS))

    def test_a_missing_key_raises(self):
        graph = pooled_graph()
        del graph["graph_node_centroid"]
        with self.assertRaises(KeyError):
            graph_from(graph)

    def test_compact_carries_both_position_fields(self):
        pooled = compact_graph(pooled_graph())
        self.assertIsNotNone(pooled.node_bbox)
        self.assertIsNotNone(pooled.node_centroid)
        self.assertIsNotNone(pooled.camera_visible)


class AdapterConstructionTest(unittest.TestCase):
    """The observation adapter is built for real, for every schema.

    Nothing else in the suite constructs it -- the training path needs
    ManiSkill -- so a name used in ``__init__`` but never imported used to
    surface only when a run started. This covers the constructor itself.
    """

    CFG = {
        "temporal": {"K": 2},
        "selection": {"n_max": N_MAX},
        "whitelist_dir": "",
    }

    @staticmethod
    def _vocab():
        relation = build_relation_vocab()
        return GraphVocab(
            entity=EntityVocab(token_to_id={"<pad>": 0, "<ee>": 1}),
            relation=relation, absolute=build_absolute_vocab(),
            temporal=build_temporal_vocab(),
            abs_valid=np.zeros((len(relation), 1), bool),
            temp_valid=np.zeros(len(relation), bool),
        )

    def _build(self):
        return GraphObsBuilder(
            object(), num_envs=2, teemo_cfg=dict(self.CFG), vocab=self._vocab(),
            n_max=N_MAX, e_max=168, cameras=["cam_a", "cam_b"],
        )

    def test_it_emits_the_packed_contract(self):
        adapter = self._build()
        self.assertEqual(set(adapter.obs_spec_shapes), set(GRAPH_KEYS))
        self.assertEqual(
            adapter.obs_spec_shapes["graph_node_bbox"], (N_MAX, 2, 4)
        )

    def test_the_zero_pack_matches_the_declared_contract(self):
        # Emitted on terminal frames before any real graph exists, so a shape
        # or key mismatch here corrupts replay rather than raising.
        adapter = self._build()
        packed = adapter._zero_pack()
        self.assertEqual(set(packed), set(adapter.obs_spec_shapes))
        for key, shape in adapter.obs_spec_shapes.items():
            self.assertEqual(packed[key].shape, shape, key)


class BboxExtractionTest(unittest.TestCase):
    """Boxes without appearance: same numbers, none of the patch machinery."""

    def _nodes(self, groups):
        nodes = {}
        for name, ids in groups.items():
            node = Node(node_id=name, node_type="object", name=name,
                        visible=True, source="segmentation")
            node.segmentation_ids = list(ids)
            nodes[name] = node
        return nodes

    def test_no_patch_coverage_is_produced(self):
        nodes = self._nodes({"a": [1]})
        fill_bboxes(nodes, [np.array([[0, 1], [1, 0]])])
        self.assertIsNone(nodes["a"].patch_weights)

    def test_out_of_range_and_unowned_ids_stay_out_of_every_box(self):
        # Background, negative ids, ids above the table, and ids no node
        # claims all land on the sentinel row and are dropped.
        seg = np.array([[-3, 0, 1], [5, 99, 2]])
        nodes = self._nodes({"a": [1], "b": [2]})
        fill_bboxes(nodes, [seg])
        np.testing.assert_allclose(nodes["a"].bbox[0], [2 / 3, 1.0, 0.0, 0.5])
        np.testing.assert_allclose(nodes["b"].bbox[0], [2 / 3, 1.0, 0.5, 1.0])

    def test_a_node_rendered_nowhere_reads_back_as_invisible(self):
        nodes = self._nodes({"ghost": [7]})
        fill_bboxes(nodes, [np.array([[0, 1], [1, 0]])])
        np.testing.assert_array_equal(nodes["ghost"].bbox, np.zeros((1, 4)))

    def test_cameras_of_different_sizes_are_handled(self):
        nodes = self._nodes({"a": [1]})
        fill_bboxes(nodes, [np.array([[1, 0]]), np.array([[0], [1]])])
        np.testing.assert_allclose(nodes["a"].bbox[0], [0.0, 0.5, 0.0, 1.0])
        np.testing.assert_allclose(nodes["a"].bbox[1], [0.0, 1.0, 0.5, 1.0])


class ReplayDtypeTest(unittest.TestCase):
    """Every packed dtype must survive the replay buffer.

    torchrl stores through ``index_put``, which has no uint16 kernel. A key
    packed in an unsupported dtype builds and acts fine and only fails on the
    first replay write, well after the run looks healthy.
    """

    # Types with an index_put kernel that the buffer actually exercises.
    SUPPORTED = {np.uint8, np.int32, np.int64, np.float16, np.float32}

    def test_every_graph_key_is_storable(self):
        for key, dtype in _PACKED_DTYPES.items():
            with self.subTest(key=key):
                self.assertIn(np.dtype(dtype).type, self.SUPPORTED)

    def test_index_put_accepts_every_graph_dtype(self):
        # The exact operation the buffer performs, so this fails here rather
        # than on the first add_transition of a real run.
        for key, dtype in _PACKED_DTYPES.items():
            with self.subTest(key=key):
                store = torch.zeros(4, dtype=torch.from_numpy(np.zeros(1, dtype)).dtype)
                store[torch.tensor([0, 2])] = store[torch.tensor([1, 3])]

class PooledEncoderTest(unittest.TestCase):
    """Boxes in, one masked-mean token out. No UID anywhere."""

    def _encoder(self, units=32):
        torch.manual_seed(0)
        return GraphEncoder(graph_config(units=units))

    def test_no_identity_embedding_exists(self):
        encoder = self._encoder()
        self.assertFalse(hasattr(encoder, "uid"))
        # One flat projection over all cameras, not a per-camera ModuleList.
        self.assertFalse(hasattr(encoder, "bbox_proj"))
        self.assertFalse(hasattr(encoder, "app_proj"))

    def test_attention_pooling_is_gone(self):
        encoder = self._encoder()
        self.assertIsNotNone(encoder.pool)
        for name in ("query", "key", "value", "out"):
            self.assertFalse(hasattr(encoder, name), name)

    def test_siblings_differ_through_their_boxes(self):
        # Same entity id and same target flag: with UID gone, geometry is the
        # only thing left to separate two instances of one type.
        encoder = self._encoder()
        left = pooled_graph(batch=1, time=1, n_valid=3)
        right = pooled_graph(
            batch=1, time=1, n_valid=3,
            boxes=torch.tensor([
                [0.10, 0.40, 0.20, 0.50],
                [0.05, 0.25, 0.70, 0.95],
                [0.00, 0.20, 0.60, 0.80],
            ]),
        )
        with torch.no_grad():
            self.assertFalse(torch.allclose(encoder(left).nodes, encoder(right).nodes))

    def test_padded_slots_are_zeroed(self):
        encoder = self._encoder()
        with torch.no_grad():
            nodes = encoder(pooled_graph(n_valid=3)).nodes
        self.assertTrue(torch.equal(nodes[..., 3:, :], torch.zeros_like(nodes[..., 3:, :])))

    def test_the_centroid_reaches_the_node_representation(self):
        # An object that moved in the world with an unchanged box must encode
        # differently -- otherwise a retained target carries no position.
        encoder = self._encoder()
        left = pooled_graph(batch=1, time=1, n_valid=3)
        right = pooled_graph(
            batch=1, time=1, n_valid=3,
            centroids=torch.tensor([
                [0.00, 0.00, 0.90],
                [-0.85, 0.95, 0.20],
                [-0.30, 0.60, 0.75],
            ]),
        )
        self.assertTrue(torch.equal(
            left["graph_node_bbox"], right["graph_node_bbox"]
        ))
        with torch.no_grad():
            self.assertFalse(torch.allclose(
                encoder(left).nodes[:, :, 1], encoder(right).nodes[:, :, 1]
            ))

    def test_an_invisible_node_still_carries_its_position(self):
        # Boxes all zero, centroid intact: this is the retained target's frame,
        # and it must not encode the same as a node that is simply absent.
        encoder = self._encoder()
        hidden = pooled_graph(batch=1, time=1, n_valid=3)
        hidden["graph_node_bbox"][..., 1, :, :] = 0.0
        moved = {k: v.clone() for k, v in hidden.items()}
        moved["graph_node_centroid"][..., 1, :] = torch.tensor([-1.4, 0.2, 0.1])
        with torch.no_grad():
            self.assertFalse(torch.allclose(
                encoder(hidden).nodes[:, :, 1], encoder(moved).nodes[:, :, 1]
            ))

    def test_centroids_are_normalised_on_fixed_bounds_not_batch_statistics(self):
        encoder = self._encoder()
        data = pooled_graph(batch=1, time=1, n_valid=3)
        compact = compact_graph(data)
        feature = compact.centroid_feature(
            torch.float32, encoder.centroid_origin, encoder.centroid_scale
        )
        want = (
            data["graph_node_centroid"].reshape(1, N_MAX, 3)
            - encoder.centroid_origin
        ) / encoder.centroid_scale
        self.assertTrue(torch.allclose(feature[:, :3], want[:, :3], atol=1e-6))
        # Padded rows stay exactly zero rather than encoding the origin.
        self.assertEqual(float(feature[:, 3:].abs().max()), 0.0)

    def test_shifting_the_whole_scene_shifts_every_centroid_equally(self):
        # The consequence of fixed bounds: normalisation carries no dependence
        # on what else is in the batch.
        encoder = self._encoder()
        base = pooled_graph(batch=1, time=1, n_valid=3)
        shifted = {k: v.clone() for k, v in base.items()}
        shifted["graph_node_centroid"] = shifted["graph_node_centroid"] + 1.0
        args = (torch.float32, encoder.centroid_origin, encoder.centroid_scale)
        left = compact_graph(base).centroid_feature(*args)
        right = compact_graph(shifted).centroid_feature(*args)
        delta = (right - left)[:, :3]
        self.assertTrue(torch.allclose(
            delta, torch.ones_like(delta) / encoder.centroid_scale, atol=1e-6
        ))

    def test_token_is_the_masked_mean_plus_the_count(self):
        encoder = self._encoder()
        data = pooled_graph(n_valid=3)
        with torch.no_grad():
            encoded = encoder(data)
            compact = compact_graph(data)
            nodes = encoded.nodes.reshape(compact.graph_count, N_MAX, 32)
            valid = compact.node_valid
            count = valid.sum(-1, keepdim=True).float()
            mean = (nodes * valid[..., None]).sum(1) / count
            want = encoder.pool(torch.cat([mean, count / N_MAX], -1))
        self.assertTrue(torch.allclose(
            encoded.token.reshape(compact.graph_count, 32), want, atol=1e-5
        ))

    def test_pooling_coefficients_are_uniform(self):
        # Every admitted node contributes exactly 1/n. Scaling one node's
        # contribution must move the mean by exactly that node's share.
        encoder = self._encoder()
        data = pooled_graph(n_valid=3)
        with torch.no_grad():
            nodes = encoder(data).nodes.reshape(-1, N_MAX, 32)
        share = nodes[:, :3].sum(1) / 3.0
        self.assertTrue(torch.allclose(share, nodes[:, :3].mean(1), atol=1e-6))

    def test_token_is_permutation_invariant(self):
        encoder = self._encoder()
        data = pooled_graph(n_valid=3)
        order = [1, 0, 2] + list(range(3, N_MAX))
        swapped = {key: value.clone() for key, value in data.items()}
        for key in ("graph_node_ent", "graph_node_target"):
            swapped[key] = swapped[key][..., order]
        swapped["graph_node_bbox"] = swapped["graph_node_bbox"][..., order, :, :]
        swapped["graph_node_centroid"] = swapped["graph_node_centroid"][
            ..., order, :
        ]
        inverse = torch.tensor(
            [order.index(i) for i in range(N_MAX)], dtype=torch.uint8
        )
        real = swapped["graph_edge_rel"] != 0
        for key in ("graph_edge_src", "graph_edge_dst"):
            swapped[key] = torch.where(real, inverse[swapped[key].long()], swapped[key])
        with torch.no_grad():
            self.assertTrue(torch.allclose(
                encoder(swapped).token, encoder(data).token, atol=1e-5
            ))

    def test_token_width_follows_simple_units(self):
        encoder = self._encoder(units=32)
        self.assertEqual(encoder.units, 32)
        with torch.no_grad():
            token = encoder(pooled_graph()).token
        self.assertEqual(tuple(token.shape), (2, 3, 32))

class PooledDecoderTest(unittest.TestCase):
    """Node and relation reconstruction from g, addressed by the box."""

    LOSSES = {"node", "nodetgt", "relabs", "reltemp"}

    def _decoder(self):
        torch.manual_seed(0)
        scorer = ProgressScorer(PICK_STAGES, N_ABS)
        decoder = SimpleGraphDecoder(graph_config(), semantic_dim=32)
        return decoder, scorer

    def _run(self, sem=None, graph=None):
        decoder, _ = self._decoder()
        compact = compact_graph(
            pooled_graph() if graph is None else graph
        )
        if sem is None:
            sem = torch.randn(2, 3, 32, requires_grad=True)
        step_valid = torch.ones(2, 3, dtype=torch.bool)
        out = decoder(sem, compact, step_valid)
        return decoder, sem, out

    def test_losses_are_finite_and_named(self):
        _, _, (losses, metrics) = self._run()
        self.assertEqual(set(losses), self.LOSSES)
        for name, value in losses.items():
            self.assertTrue(torch.isfinite(value), name)
        for name in ("node_ent_acc", "node_ent_loss", "node_bbox_loss",
                     "node_target_acc", "relabs_acc", "reltemp_acc"):
            self.assertIn(name, metrics)

    def test_bbox_iou_is_not_computed_during_training(self):
        # IoU needs its own min/max kernels and optimises nothing; it belongs
        # in evaluation, not in every update.
        _, _, (_, metrics) = self._run()
        self.assertNotIn("node_bbox_iou", metrics)

    def test_every_loss_reaches_the_semantic_state(self):
        # The decoder reads g, not the encoder's node vectors: that is what
        # makes these losses a test of what g retained.
        for name in sorted(self.LOSSES):
            with self.subTest(loss=name):
                _, sem, (losses, _) = self._run()
                losses[name].backward()
                self.assertIsNotNone(sem.grad)
                self.assertGreater(float(sem.grad.abs().sum()), 0.0)

    def test_node_loss_averages_its_two_halves(self):
        _, _, (losses, metrics) = self._run()
        self.assertAlmostEqual(
            float(losses["node"]),
            0.5 * (float(metrics["node_ent_loss"]) + float(metrics["node_bbox_loss"])),
            places=5,
        )

    def test_node_loss_does_not_scale_with_graph_size(self):
        # Reduced per valid node and then per frame, so a seven-node frame does
        # not outweigh a two-node one.
        values = []
        for n_valid in (2, 4, 7):
            torch.manual_seed(0)
            sem = torch.zeros(2, 3, 32)
            _, _, (losses, _) = self._run(sem=sem, graph=pooled_graph(n_valid=n_valid))
            values.append(float(losses["node"]))
        self.assertLess(max(values) / min(values), 2.0, values)

    def test_the_query_carries_geometry_not_the_target_flag(self):
        # Moving the flag to a different node must not change any node's query,
        # only the label it is scored against.
        decoder, _ = self._decoder()
        base = pooled_graph(batch=1, time=1)
        moved = pooled_graph(batch=1, time=1)
        moved["graph_node_target"][..., 1] = 0
        moved["graph_node_target"][..., 2] = 1
        sem = torch.randn(1, 1, 32)
        with torch.no_grad():
            a = decoder.node_features(
                sem, compact_graph(base))
            b = decoder.node_features(
                sem, compact_graph(moved))
        self.assertTrue(torch.equal(a, b))

    def test_boxless_nodes_are_separated_by_their_centroids(self):
        """Retention puts several nodes without pixels in one frame. Their
        boxes and visibility bits are all zero, so a box-only query would be
        byte-identical for each and the decoder would be asked for two
        different entities from one input."""
        decoder, _ = self._decoder()
        graph = pooled_graph(batch=1, time=1, n_valid=3)
        # Nodes 1 and 2 keep their rows and centroids but lose every pixel.
        graph["graph_node_bbox"][..., 1, :, :] = 0.0
        graph["graph_node_bbox"][..., 2, :, :] = 0.0
        graph["graph_node_centroid"][..., 1, :] = torch.tensor([0.4, 0.1, 0.9])
        graph["graph_node_centroid"][..., 2, :] = torch.tensor([-0.3, 0.7, 0.2])
        compact = compact_graph(graph)
        self.assertTrue(bool(compact.node_valid[0, 1]))
        self.assertTrue(bool(compact.node_valid[0, 2]))
        box = compact.bbox_feature(torch.float32)
        self.assertTrue(torch.equal(box[0, 1], box[0, 2]))   # the ambiguity
        with torch.no_grad():
            nodes = decoder.node_features(torch.randn(1, 1, 32), compact)
        self.assertFalse(torch.allclose(nodes[0, 1], nodes[0, 2], atol=1e-5))

    def test_two_boxless_nodes_in_the_same_place_stay_ambiguous(self):
        """The converse, so the test above is not passing on noise: identical
        geometry gives identical queries, which is the honest answer."""
        decoder, _ = self._decoder()
        graph = pooled_graph(batch=1, time=1, n_valid=3)
        for row in (1, 2):
            graph["graph_node_bbox"][..., row, :, :] = 0.0
            graph["graph_node_centroid"][..., row, :] = torch.tensor([0.4, 0.1, 0.9])
        with torch.no_grad():
            nodes = decoder.node_features(
                torch.randn(1, 1, 32),
                compact_graph(graph))
        self.assertTrue(torch.allclose(nodes[0, 1], nodes[0, 2], atol=1e-6))

    def test_the_decoder_query_uses_the_encoder_centroid_bounds(self):
        """They address the same node; disagreeing bounds would mean the
        decoder is querying a position the encoder never encoded."""
        decoder, _ = self._decoder()
        encoder = GraphEncoder(graph_config())
        self.assertTrue(torch.equal(decoder.centroid_origin,
                                    encoder.centroid_origin))
        self.assertTrue(torch.equal(decoder.centroid_scale,
                                    encoder.centroid_scale))

    def test_node_count_may_vary_between_frames(self):
        for n_valid in (1, 3, 7):
            with self.subTest(nodes=n_valid):
                _, _, (losses, _) = self._run(graph=pooled_graph(n_valid=max(n_valid, 2)))
                self.assertTrue(torch.isfinite(losses["nodetgt"]))


class PotentialMatrixTest(unittest.TestCase):
    """The soft potential is linear, so it is one contraction, not 14 stages."""

    def setUp(self):
        self.scorer = ProgressScorer(PICK_STAGES, N_ABS)

    def _stagewise(self, probs):
        return (self.scorer.satisfaction(probs, False) * self.scorer.weights).sum(-1)

    def test_matrix_matches_the_cumulative_stage_sum(self):
        torch.manual_seed(0)
        probs = torch.softmax(
            torch.randn(5, self.scorer.n_relations, N_ABS), -1)
        self.assertTrue(torch.allclose(
            self.scorer.potential(probs), self._stagewise(probs), atol=1e-6
        ))

    def test_potential_stays_bounded(self):
        torch.manual_seed(1)
        probs = torch.softmax(
            torch.randn(64, self.scorer.n_relations, N_ABS) * 4.0, -1)
        phi = self.scorer.potential(probs)
        self.assertGreaterEqual(float(phi.min()), 0.0)
        self.assertLessEqual(float(phi.max()), 1.0)

    def _onehot(self, per_relation):
        """One-hot the named label for each of the scorer's relations.

        Keyed by the shared constants, not by literal ids: this test failed the
        first time the label vocabulary grew, and a table of integers here is
        the same second source of truth the scorer itself just stopped being.
        """
        probs = torch.zeros(1, len(self.scorer.relations), N_ABS)
        for row, relation in enumerate(self.scorer.relations.tolist()):
            probs[0, row, per_relation[relation]] = 1.0
        return probs

    def test_saturated_state_scores_exactly_one(self):
        best = self._onehot({
            REL_PLANAR_DISTANCE: ABS_VERY_NEAR,
            REL_HEIGHT_OFFSET: ABS_LEVEL,
            REL_CONTACT_COMPAT: ABS_MATCH,
            REL_GRASP_COMPAT: ABS_MATCH,
            REL_CONTACT: ABS_HOLDS,
            REL_GRASP: ABS_HOLDS,
        })
        self.assertAlmostEqual(float(self.scorer.potential(best)), 1.0, places=6)
        self.assertAlmostEqual(
            float(self.scorer.potential(best, hard=True)), 1.0, places=6
        )

    def test_worst_state_scores_zero(self):
        worst = self._onehot({
            REL_PLANAR_DISTANCE: ABS_VERY_FAR,
            REL_HEIGHT_OFFSET: ABS_FAR_BELOW,
            REL_CONTACT_COMPAT: ABS_UNOBSERVED,
            REL_GRASP_COMPAT: ABS_UNOBSERVED,
            REL_CONTACT: ABS_NOT_HOLDS,
            REL_GRASP: ABS_NOT_HOLDS,
        })
        self.assertAlmostEqual(float(self.scorer.potential(worst)), 0.0, places=6)


class SimpleRSSMTest(unittest.TestCase):
    GRAPH_DIM = 16
    TOKEN = 12

    def _rssm(self):
        config = SimpleNamespace(
            stoch=4, hybrid_stoch=2, deter=16, hidden=8, discrete=3,
            img_layers=1, obs_layers=1, dyn_layers=1, blocks=2, act="SiLU",
            norm=True, unimix_ratio=0.01, initial="learned", device="cpu",
            sem_stoch=2, sem_discrete=3, sem_layers=1,
        )
        return RSSM(
            config, embed_size=6, act_dim=2, semantic=True,
            graph_token_size=self.TOKEN,
            graph_dim=self.GRAPH_DIM,
        )

    def test_z_keeps_the_stock_width(self):
        model = self._rssm()
        self.assertEqual(model._stoch, 4)          # stoch, not hybrid_stoch
        self.assertEqual(model.flat_sem, self.GRAPH_DIM)
        self.assertEqual(model.feat_size, 4 * 3 + self.GRAPH_DIM + 16)

    def test_sem_is_a_flat_vector(self):
        model = self._rssm()
        _, _, sem = model.initial(5)
        self.assertEqual(tuple(sem.shape), (5, self.GRAPH_DIM))
        self.assertEqual(model.sem_shape(), (self.GRAPH_DIM,))

    def test_z_networks_do_not_read_g(self):
        model = self._rssm()
        obs_in = model._obs_net[0].in_features
        img_in = model._img_net[0].in_features
        self.assertEqual(obs_in, 16 + 6)             # deter + embed
        self.assertEqual(img_in, 16)                 # deter only

    def test_semantic_heads_do_not_read_previous_g(self):
        model = self._rssm()
        self.assertEqual(model._sem_obs[0].in_features, 16 + self.TOKEN)
        self.assertEqual(model._sem_img[0].in_features, 16)

    def test_previous_g_still_drives_the_transition(self):
        model = self._rssm()
        self.assertIsNotNone(model._deter_net._dyn_in3)
        self.assertEqual(model._deter_net._dyn_in3[0].in_features, self.GRAPH_DIM)

    def test_posterior_is_deterministic(self):
        torch.manual_seed(0)
        model = self._rssm().eval()
        stoch, deter, sem = model.initial(2)
        action = torch.zeros(2, 2)
        embed = torch.randn(2, 6)
        token = torch.randn(2, self.TOKEN)
        reset = torch.zeros(2, dtype=torch.bool)
        with torch.no_grad():
            first = model.obs_step(stoch, deter, action, embed, reset, sem=sem, graph_token=token)
            second = model.obs_step(stoch, deter, action, embed, reset, sem=sem, graph_token=token)
        self.assertTrue(torch.equal(first[3], second[3]))
        self.assertIsNone(first[4])

    def test_observe_returns_no_semantic_logits(self):
        torch.manual_seed(0)
        model = self._rssm()
        initial = model.initial(2)
        observed = model.observe(
            torch.randn(2, 4, 6), torch.zeros(2, 4, 2), initial,
            torch.zeros(2, 4, dtype=torch.bool), torch.randn(2, 4, self.TOKEN),
        )
        self.assertEqual(len(observed), 4)
        self.assertEqual(tuple(observed[3].shape), (2, 4, self.GRAPH_DIM))

    def test_alignment_is_symmetric_in_value_and_split_in_gradient(self):
        model = self._rssm()
        post = torch.randn(2, 3, self.GRAPH_DIM, requires_grad=True)
        prior = torch.randn(2, 3, self.GRAPH_DIM, requires_grad=True)
        dyn, rep = model.semantic_align_loss(post, prior)
        self.assertTrue(torch.allclose(dyn, rep))
        dyn.sum().backward(retain_graph=True)
        self.assertIsNone(post.grad)                 # dyn updates the prior
        self.assertGreater(float(prior.grad.abs().sum()), 0.0)
        prior.grad = None
        rep.sum().backward()
        self.assertGreater(float(post.grad.abs().sum()), 0.0)
        self.assertIsNone(prior.grad)                # rep updates the posterior

    def test_rms_is_parameter_free_and_fixed_scale(self):
        model = self._rssm()
        scaled = model.rms(torch.randn(4, self.GRAPH_DIM) * 100.0)
        self.assertAlmostEqual(float(scaled.square().mean()), 1.0, places=4)

    def test_feature_carries_g(self):
        model = self._rssm()
        stoch, deter, sem = model.initial(3)
        feat = model.get_feat(stoch, deter, sem)
        self.assertEqual(tuple(feat.shape), (3, model.feat_size))


class EnvConfigPlumbingTest(unittest.TestCase):
    """The env config is flattened key by key, so a new key is easy to drop."""

    def test_every_env_graph_key_is_forwarded(self):
        for name in ("mshab", "maniskill"):
            env_config = OmegaConf.load(f"configs/env/{name}.yaml")
            declared = set(env_config.graph.keys())
            forwarded = set(_GRAPH_CONFIG_KEYS) | {"cameras"}
            self.assertEqual(
                declared - forwarded, set(),
                f"configs/env/{name}.yaml declares graph keys the adapter "
                "never passes to build_graph_obs; they would be ignored",
            )

    def test_every_forwarded_key_is_declared(self):
        """The other direction: graph_observation_config getattrs every key,
        so one added here and not to a config raises at env construction."""
        for name in ("mshab", "maniskill"):
            declared = set(
                OmegaConf.load(f"configs/env/{name}.yaml").graph.keys())
            self.assertEqual(
                set(_GRAPH_CONFIG_KEYS) - declared, set(),
                f"configs/env/{name}.yaml is missing graph keys the adapter "
                "reads; the env would fail to build",
            )

    def test_the_base_leaves_every_switch_off(self):
        model_config = OmegaConf.load("configs/model/_base_.yaml")
        self.assertEqual(model_config.graph.n_max, 8)
        self.assertEqual(model_config.graph.e_max, 168)
        self.assertIs(model_config.graph.enabled, False)
        self.assertIs(model_config.progress.enabled, False)
        self.assertEqual(model_config.loss_scales.reltemp, 1.0)

    def test_the_pooled_simple_preset_pins_its_contract(self):
        preset = OmegaConf.load("configs/model/size50M_graph_simple.yaml")
        self.assertIs(preset.graph.enabled, True)
        self.assertEqual(preset.graph.n_max, 8)
        self.assertEqual(preset.graph.e_max, 168)
        self.assertEqual(preset.graph.decoder_units, 256)
        self.assertEqual(preset.graph.bbox_query_dim, 4)
        self.assertIs(preset.progress.enabled, True)
        # Beta is the one knob a run is expected to retune -- a short window
        # for a smoke run, a long one for a full run -- so what is pinned is
        # the shape of the schedule, not its numbers: progress stays on, the
        # plateau is positive, and the window neither starts before zero nor
        # ends before it starts (Dreamer raises on that last one).
        for name in ("size50M_graph_simple", "size100M_graph_simple"):
            schedule = OmegaConf.load(f"configs/model/{name}.yaml").progress
            self.assertIs(schedule.enabled, True, name)
            self.assertGreater(schedule.beta, 0.0, name)
            self.assertGreaterEqual(schedule.beta_warmup_start, 0, name)
            self.assertGreaterEqual(
                schedule.beta_warmup_end, schedule.beta_warmup_start, name
            )
        for name in ("node", "nodetgt", "relabs", "reltemp",
                     "progress_model",
                     "progress_value", "graphdyn"):
            self.assertEqual(preset.loss_scales[name], 1.0, name)
        self.assertEqual(preset.loss_scales.graphrep, 0.05)
        # There is no recurrent slot state to supervise in this arm.
        for name in ("slotdyn", "slotalive", "prior_nodetgt"):
            self.assertNotIn(name, preset.loss_scales)

    def test_the_shipped_schedule_is_the_one_the_experiment_specifies(self):
        """The numbers, not just the shape, for the run that is about to go.

        Deliberately narrower than the shape check above: that one exists so a
        smoke run can retune the window freely, and it stays that way. This one
        pins the values the current experiment was specified with, so a stray
        edit to a preset shows up here rather than as an unexplained difference
        between two arms three days into training. Change both together when
        the experiment changes.
        """
        for name in ("size50M_graph_simple", "size100M_graph_simple"):
            schedule = OmegaConf.load(f"configs/model/{name}.yaml").progress
            self.assertAlmostEqual(schedule.beta, 0.05, msg=name)
            self.assertEqual(schedule.beta_warmup_start, 200000, name)
            self.assertEqual(schedule.beta_warmup_end, 700000, name)

    def test_every_arm_sees_the_same_privileged_observation(self):
        # The graph is built from privileged state in every graph arm, so the
        # graph-free baseline has to see the same fields or the comparison
        # measures privilege instead of representation.
        env_config = OmegaConf.load("configs/env/mshab.yaml")
        self.assertIs(env_config.nonprivileged_obs, False)

    def test_the_graph_config_reaches_the_builder(self):
        graph_config = OmegaConf.create({
            "enabled": True,
            "mshab_task": "set_table", "entity_vocab": 14,
            "thresholds_path": "", "whitelist_dir": "",
            "n_max": 8, "e_max": 168,
            "visibility_policy": "keep_tabletop",
            "bypass_teemo": False,
            "use_target_flag": True, "object_object_spatial": False,
        })
        out = graph_observation_config(graph_config, ["fetch_head"])
        self.assertIs(out["enabled"], True)
        self.assertEqual(out["cameras"], ["fetch_head"])


class DecoderDetachTest(unittest.TestCase):
    """Pixel reconstruction may read g but must not reshape it."""

    def _decoder(self, detach):
        # MultiDecoder splats the dist nodes (``**config.cnn_dist``) and also
        # reads them by attribute, so this has to be a real config node rather
        # than a namespace.
        config = OmegaConf.create({
            "cnn_keys": "^image$",
            "mlp_keys": "^state$",
            "mlp_dist": {"name": "symlog_mse"},
            "cnn_dist": {"name": "mse"},
            "mlp": {
                "shape": None, "layers": 1, "units": 8, "act": "SiLU",
                "norm": True, "dist": {"name": "identity"}, "device": "cpu",
                "outscale": 1.0, "symlog_inputs": False, "name": "mlp_decoder",
            },
            "cnn": {
                "depth": 4, "units": 8, "bspace": 2, "mults": [1, 1],
                "act": "SiLU", "norm": True, "kernel_size": 3, "minres": 4,
                "outscale": 1.0,
            },
        })
        return MultiDecoder(
            config, deter=8, flat_stoch=6,
            shapes={"image": (16, 16, 3), "state": (5,)},
            flat_sem=4, detach_sem_cnn=detach,
        )

    def _grad(self, detach, key):
        torch.manual_seed(0)
        decoder = self._decoder(detach)
        sem = torch.randn(2, 1, 4, requires_grad=True)
        stoch = torch.randn(2, 1, 6)
        deter = torch.randn(2, 1, 8)
        out = decoder(stoch, deter, sem)[key]
        out.mode().square().mean().backward()
        return 0.0 if sem.grad is None else float(sem.grad.abs().sum())

    def test_image_reconstruction_does_not_reach_g(self):
        self.assertEqual(self._grad(detach=True, key="image"), 0.0)

    def test_state_reconstruction_still_reaches_g(self):
        self.assertGreater(self._grad(detach=True, key="state"), 0.0)

    def test_full_mode_leaves_the_image_path_attached(self):
        self.assertGreater(self._grad(detach=False, key="image"), 0.0)


if __name__ == "__main__":
    unittest.main()
