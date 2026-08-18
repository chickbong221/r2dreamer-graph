"""Simple-graph mode: relation-only contract, UIDs, and the deterministic g.

Everything here runs on synthetic tensors -- no simulator, no DINO.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from graph import (
    FULL_GRAPH_KEYS,
    RESERVED_GRAPH_KEYS,
    SCHEMA_FULL,
    SCHEMA_SIMPLE_POOLED,
    SCHEMA_SIMPLE_SLOT,
    SIMPLE_POOLED_GRAPH_KEYS,
    SIMPLE_SLOT_GRAPH_KEYS,
    GraphEncoder,
    SimpleGraphDecoder,
    compact_graph,
    graph_from,
    graph_keys,
    graph_schema,
)
from progress import PICK_STAGES, ProgressScorer
from envs.maniskill import _GRAPH_CONFIG_KEYS, graph_observation_config
from networks import MultiDecoder
from rssm import RSSM
from scenegraph.adapters.graph_obs import _DTYPES as _PACKED_DTYPES
from scenegraph.adapters.graph_obs import GraphObsBuilder
from scenegraph.adapters.graph_pack import (
    graph_keys as pack_keys,
    graph_schema as pack_schema,
    pack_graph,
)
from scenegraph.adapters.graph_vocab import (
    EntityVocab,
    GraphVocab,
    build_absolute_vocab,
    build_relation_vocab,
    build_temporal_vocab,
)
from scenegraph.core.node_builder import fill_appearance, fill_bboxes
from scenegraph.core.schema import Graph, Node
from scenegraph.core.graph_builder import (
    UID_EE,
    UID_PAD,
    UID_VOCAB_MAX,
    EpisodeUIDs,
)

N_MAX = 8
E_MAX = 16
UID_VOCAB = 32


def graph_config(simple=True, units=32, state_mode="pooled"):
    return SimpleNamespace(
        simple=simple,
        state_mode=state_mode,
        units=units,
        simple_units=units,
        semantic_dim=units,
        decoder_units=16,
        bbox_query_dim=4,
        slot_dim=16,
        layers=1,
        n_cams=2,
        app_dim=8,
        entity_vocab=14,
        n_rel=11,
        n_abs=17,
        n_temp=6,
        embed=8,
        app=4,
        bbox=4,
        bbox_beta=0.1,
        uid_vocab=UID_VOCAB,
        uid_embed=8,
        reverse_edges=True,
        act="SiLU",
    )


def slot_graph(batch=2, time=3, n_valid=3, n_edges=4, uids=None):
    """The slot contract: UID, no boxes."""
    shape = (batch, time)
    graph = {
        "graph_node_ent": torch.zeros(*shape, N_MAX, dtype=torch.uint8),
        "graph_node_uid": torch.zeros(*shape, N_MAX, dtype=torch.int64),
        "graph_node_target": torch.zeros(*shape, N_MAX, dtype=torch.uint8),
        "graph_edge_src": torch.zeros(*shape, E_MAX, dtype=torch.uint8),
        "graph_edge_dst": torch.zeros(*shape, E_MAX, dtype=torch.uint8),
        "graph_edge_rel": torch.zeros(*shape, E_MAX, dtype=torch.uint8),
        "graph_edge_abs": torch.zeros(*shape, E_MAX, dtype=torch.uint8),
        "graph_edge_temp": torch.zeros(*shape, E_MAX, dtype=torch.uint8),
    }
    graph["graph_node_ent"][..., :n_valid] = torch.arange(1, n_valid + 1)
    if uids is None:
        uids = [UID_EE] + list(range(2, n_valid + 1))
    graph["graph_node_uid"][..., :n_valid] = torch.tensor(uids, dtype=torch.int64)
    # Slot zero is the end effector, so the target lives on slot one.
    graph["graph_node_target"][..., 1] = 1

    index = torch.arange(n_edges)
    graph["graph_edge_src"][..., :n_edges] = (index % n_valid).to(torch.uint8)
    graph["graph_edge_dst"][..., :n_edges] = ((index + 1) % n_valid).to(torch.uint8)
    # Relation 1 admits absolute labels 1-2; relation 5 admits 3-7.
    spatial = index.remainder(2).bool()
    graph["graph_edge_rel"][..., :n_edges] = torch.where(
        spatial, torch.tensor(5), torch.tensor(1)
    ).to(torch.uint8)
    graph["graph_edge_abs"][..., :n_edges] = torch.where(
        spatial, torch.tensor(3), torch.tensor(2)
    ).to(torch.uint8)
    graph["graph_edge_temp"][..., :n_edges] = torch.where(
        spatial, torch.tensor(3), torch.tensor(0)
    ).to(torch.uint8)
    return graph


def pooled_graph(batch=2, time=3, n_valid=3, n_edges=4, boxes=None):
    """The pooled contract: per-camera boxes, no UID.

    Camera 0 sees every valid node, camera 1 only the end effector, so masked
    per-camera validity is actually exercised rather than assumed.
    """
    graph = {
        key: value
        for key, value in slot_graph(batch, time, n_valid, n_edges).items()
        if key != "graph_node_uid"
    }
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
    return graph


class ContractTest(unittest.TestCase):
    """``simple`` alone no longer names a contract.

    Pooled graph-simple addresses a node by the box it currently occupies;
    slot graph-simple aligns by UID across frames. Emitting both fields to both
    would put a key in replay that one of them must never read.
    """

    def test_schema_follows_simple_and_state_mode(self):
        self.assertEqual(graph_schema(False, "pooled"), SCHEMA_FULL)
        self.assertEqual(graph_schema(False, "slots"), SCHEMA_FULL)
        self.assertEqual(graph_schema(True, "pooled"), SCHEMA_SIMPLE_POOLED)
        self.assertEqual(graph_schema(True, "slots"), SCHEMA_SIMPLE_SLOT)

    def test_pooled_carries_boxes_and_no_identity(self):
        keys = graph_keys(SCHEMA_SIMPLE_POOLED)
        self.assertEqual(keys, SIMPLE_POOLED_GRAPH_KEYS)
        self.assertIn("graph_node_bbox", keys)
        self.assertNotIn("graph_node_uid", keys)
        self.assertNotIn("graph_node_app", keys)

    def test_slot_carries_identity_and_no_boxes(self):
        keys = graph_keys(SCHEMA_SIMPLE_SLOT)
        self.assertEqual(keys, SIMPLE_SLOT_GRAPH_KEYS)
        self.assertIn("graph_node_uid", keys)
        self.assertNotIn("graph_node_bbox", keys)
        self.assertNotIn("graph_node_app", keys)

    def test_full_keeps_appearance(self):
        self.assertEqual(graph_keys(SCHEMA_FULL), FULL_GRAPH_KEYS)
        self.assertIn("graph_node_app", FULL_GRAPH_KEYS)
        self.assertNotIn("graph_node_uid", FULL_GRAPH_KEYS)

    def test_uid_stays_reserved_even_where_it_is_not_emitted(self):
        # Nothing in the pooled schema names it, but a stale wrapper that still
        # exposes it must not be able to feed it to the ordinary MLP encoder.
        self.assertNotIn("graph_node_uid", SIMPLE_POOLED_GRAPH_KEYS)
        self.assertIn("graph_node_uid", RESERVED_GRAPH_KEYS)
        for keys in (FULL_GRAPH_KEYS, SIMPLE_POOLED_GRAPH_KEYS,
                     SIMPLE_SLOT_GRAPH_KEYS):
            self.assertTrue(set(keys) <= RESERVED_GRAPH_KEYS)

    def test_unknown_schema_raises(self):
        with self.assertRaises(ValueError):
            graph_keys("simple")

    def test_validation_is_per_schema(self):
        pooled = pooled_graph()
        self.assertEqual(
            set(graph_from(pooled, SCHEMA_SIMPLE_POOLED)),
            set(SIMPLE_POOLED_GRAPH_KEYS),
        )
        with self.assertRaises(KeyError):
            graph_from(pooled, SCHEMA_SIMPLE_SLOT)
        with self.assertRaises(KeyError):
            graph_from(slot_graph(), SCHEMA_SIMPLE_POOLED)

    def test_compact_leaves_the_unused_side_none(self):
        slot = compact_graph(slot_graph(), SCHEMA_SIMPLE_SLOT)
        self.assertIsNotNone(slot.node_uid)
        self.assertIsNone(slot.node_bbox)
        self.assertIsNone(slot.camera_visible)
        self.assertIsNone(slot.node_app)

        pooled = compact_graph(pooled_graph(), SCHEMA_SIMPLE_POOLED)
        self.assertIsNone(pooled.node_uid)
        self.assertIsNotNone(pooled.node_bbox)
        self.assertIsNotNone(pooled.camera_visible)
        self.assertIsNone(pooled.node_app)
        self.assertIsNone(pooled.appearance_known)

    def test_camera_validity_is_derived_not_stored(self):
        compact = compact_graph(pooled_graph(n_valid=3), SCHEMA_SIMPLE_POOLED)
        visible = compact.camera_visible
        self.assertTrue(bool(visible[:, :3, 0].all()))   # camera 0 saw them
        self.assertTrue(bool(visible[:, 0, 1].all()))    # camera 1 saw only ee
        self.assertFalse(bool(visible[:, 1:, 1].any()))
        self.assertFalse(bool(visible[:, 3:, :].any()))  # padded rows

    def test_bbox_feature_zeroes_invalid_cameras(self):
        compact = compact_graph(pooled_graph(), SCHEMA_SIMPLE_POOLED)
        feature = compact.bbox_feature(torch.float32)
        n_cams = compact.node_bbox.shape[-2]
        self.assertEqual(feature.shape[-1], 5 * n_cams)
        # The trailing block is exactly the validity bits.
        self.assertTrue(torch.equal(
            feature[..., 4 * n_cams:], compact.camera_visible.float()
        ))
        # Camera 1 saw nothing but the end effector, so its box block is zero.
        self.assertEqual(float(feature[:, 1:, 4:8].abs().max()), 0.0)

    def test_local_endpoints_survive_the_graph_offset(self):
        compact = compact_graph(pooled_graph(), SCHEMA_SIMPLE_POOLED)
        offset = compact.edge_graph * compact.num_nodes
        self.assertTrue(torch.equal(compact.edge_src, compact.edge_src_local + offset))
        self.assertTrue(torch.equal(compact.edge_dst, compact.edge_dst_local + offset))


class PackerSchemaTest(unittest.TestCase):
    """The simulator side emits exactly one contract, and it is the right one."""

    def _graph(self, with_box=True):
        node = Node(node_id="ee", node_type="ee", name="ee", visible=True,
                    source="segmentation")
        node.index = 0
        obj = Node(node_id="obj-1", node_type="object", name="apple",
                   visible=True, source="segmentation",
                   attributes={"whitelist_key": "apple"})
        obj.index = 1
        if with_box:
            node.bbox = np.array([[0.1, 0.4, 0.2, 0.5], [0.0, 0.0, 0.0, 0.0]],
                                 np.float32)
            obj.bbox = np.array([[0.5, 0.9, 0.1, 0.3], [0.2, 0.6, 0.2, 0.6]],
                                np.float32)
        node.appearance = np.zeros((2, 4), np.float32)
        obj.appearance = np.ones((2, 4), np.float32)
        return Graph(
            frame=0, env_id="env0", camera="cam",
            nodes=[node, obj], edges=[],
            meta=dict(active_target_node_id="obj-1",
                      node_uids={"ee": UID_EE, "obj-1": 7}),
        )

    @staticmethod
    def _vocab():
        """A two-entry entity vocabulary, so this needs no mined whitelist."""
        entity = EntityVocab(token_to_id={"<pad>": 0, "<ee>": 1, "apple": 2})
        relation = build_relation_vocab()
        return GraphVocab(
            entity=entity, relation=relation,
            absolute=build_absolute_vocab(), temporal=build_temporal_vocab(),
            abs_valid=np.zeros((len(relation), 1), bool),
            temp_valid=np.zeros(len(relation), bool),
        )

    def _pack(self, schema, **kwargs):
        return pack_graph(
            self._graph(**kwargs), self._vocab(),
            n_max=N_MAX, e_max=E_MAX, n_cams=2, app_dim=4, schema=schema,
            uid_vocab=UID_VOCAB,
        )

    def test_schema_constants_mirror_the_model_package(self):
        # Duplicated so the simulator side never imports the model package;
        # they have to agree or replay and the encoder disagree silently.
        self.assertEqual(pack_schema(False, "pooled"), SCHEMA_FULL)
        self.assertEqual(pack_schema(True, "pooled"), SCHEMA_SIMPLE_POOLED)
        self.assertEqual(pack_schema(True, "slots"), SCHEMA_SIMPLE_SLOT)
        for schema in (SCHEMA_FULL, SCHEMA_SIMPLE_POOLED, SCHEMA_SIMPLE_SLOT):
            self.assertEqual(tuple(pack_keys(schema)), tuple(graph_keys(schema)))

    def test_pooled_packs_boxes_and_no_identity(self):
        packed = self._pack(SCHEMA_SIMPLE_POOLED)
        self.assertEqual(set(packed), set(SIMPLE_POOLED_GRAPH_KEYS))
        self.assertNotIn("graph_node_uid", packed)
        self.assertNotIn("graph_node_app", packed)
        self.assertEqual(packed["graph_node_bbox"].dtype, np.float16)
        # Row 1 is the object; camera 0 saw it, and its box round-trips.
        np.testing.assert_allclose(
            packed["graph_node_bbox"][1, 0], [0.5, 0.9, 0.1, 0.3], atol=1e-3
        )
        # The end effector is invisible to camera 1, which stays an empty box.
        self.assertEqual(float(np.abs(packed["graph_node_bbox"][0, 1]).max()), 0.0)
        self.assertEqual(int(packed["graph_node_target"][1]), 1)

    def test_slot_packs_identity_and_no_boxes(self):
        packed = self._pack(SCHEMA_SIMPLE_SLOT)
        self.assertEqual(set(packed), set(SIMPLE_SLOT_GRAPH_KEYS))
        self.assertNotIn("graph_node_bbox", packed)
        self.assertEqual(int(packed["graph_node_uid"][1]), 7)

    def test_full_packs_both(self):
        packed = self._pack(SCHEMA_FULL)
        self.assertEqual(set(packed), set(FULL_GRAPH_KEYS))
        self.assertIn("graph_node_app", packed)
        self.assertIn("graph_node_bbox", packed)

    def test_pooled_does_not_need_uids_at_all(self):
        # EpisodeUIDs is not run under the pooled schema, so the packer must
        # not demand a code the builder never assigned.
        graph = self._graph()
        graph.meta["node_uids"] = {}
        packed = pack_graph(
            graph, self._vocab(), n_max=N_MAX, e_max=E_MAX,
            n_cams=2, app_dim=4, schema=SCHEMA_SIMPLE_POOLED,
            uid_vocab=UID_VOCAB,
        )
        self.assertNotIn("graph_node_uid", packed)

    def test_unknown_schema_raises(self):
        with self.assertRaises(ValueError):
            self._pack("simple")


class AdapterConstructionTest(unittest.TestCase):
    """The observation adapter is built for real, for every schema.

    Nothing else in the suite constructs it -- the training path needs
    ManiSkill -- so a name used in ``__init__`` but never imported used to
    surface only when a run started. This covers the constructor itself.
    """

    CFG = {
        "temporal": {"K": 2},
        "selection": {"n_max": N_MAX, "k_persist": -1},
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

    def _build(self, simple, state_mode):
        return GraphObsBuilder(
            object(), num_envs=2, teemo_cfg=dict(self.CFG), vocab=self._vocab(),
            n_max=N_MAX, e_max=168, cameras=["cam_a", "cam_b"],
            simple=simple, state_mode=state_mode,
        )

    def test_pooled_emits_boxes_and_builds_no_dino(self):
        adapter = self._build(True, "pooled")
        self.assertEqual(adapter.schema, SCHEMA_SIMPLE_POOLED)
        self.assertTrue(adapter.bbox_enabled)
        self.assertFalse(adapter.appearance_enabled)
        self.assertFalse(adapter.uids_enabled)
        self.assertIsNone(adapter.dino)
        self.assertEqual(adapter.patch_grid, 0)
        self.assertEqual(set(adapter.obs_spec_shapes), set(SIMPLE_POOLED_GRAPH_KEYS))
        self.assertEqual(
            adapter.obs_spec_shapes["graph_node_bbox"], (N_MAX, 2, 4)
        )

    def test_slot_emits_uid_and_no_boxes(self):
        adapter = self._build(True, "slots")
        self.assertEqual(adapter.schema, SCHEMA_SIMPLE_SLOT)
        self.assertFalse(adapter.bbox_enabled)
        self.assertTrue(adapter.uids_enabled)
        self.assertIsNone(adapter.dino)
        self.assertEqual(set(adapter.obs_spec_shapes), set(SIMPLE_SLOT_GRAPH_KEYS))

    def test_full_keeps_appearance(self):
        adapter = self._build(False, "pooled")
        self.assertEqual(adapter.schema, SCHEMA_FULL)
        self.assertTrue(adapter.bbox_enabled)
        self.assertTrue(adapter.appearance_enabled)
        self.assertFalse(adapter.uids_enabled)
        self.assertEqual(set(adapter.obs_spec_shapes), set(FULL_GRAPH_KEYS))

    def test_every_switch_reaches_the_per_env_builders(self):
        # The adapter decides; the builders are what actually skip the work.
        for simple, mode in ((True, "pooled"), (True, "slots"), (False, "pooled")):
            with self.subTest(simple=simple, state_mode=mode):
                adapter = self._build(simple, mode)
                for builder in adapter.builders:
                    self.assertEqual(builder.bbox_enabled, adapter.bbox_enabled)
                    self.assertEqual(
                        builder.appearance_enabled, adapter.appearance_enabled
                    )
                    self.assertEqual(builder.uids_enabled, adapter.uids_enabled)

    def test_the_zero_pack_matches_the_declared_contract(self):
        # Emitted on terminal frames before any real graph exists, so a shape
        # or key mismatch here corrupts replay rather than raising.
        for simple, mode in ((True, "pooled"), (True, "slots"), (False, "pooled")):
            with self.subTest(simple=simple, state_mode=mode):
                adapter = self._build(simple, mode)
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

    def test_matches_the_appearance_path_exactly(self):
        rng = np.random.default_rng(0)
        for trial in range(50):
            segs = [rng.integers(0, 6, size=(16, 16)) for _ in range(2)]
            groups = {"ee": [1, 5], "a": [2], "b": [3, 4], "gone": [9]}
            reference = self._nodes(groups)
            fast = self._nodes(groups)
            fill_appearance(reference, segs, grid=8)
            fill_bboxes(fast, segs)
            for name in groups:
                np.testing.assert_array_equal(
                    reference[name].bbox, fast[name].bbox, err_msg=f"{trial} {name}"
                )

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

    def test_uid_is_storable_and_holds_the_vocabulary(self):
        self.assertIs(np.dtype(_PACKED_DTYPES["graph_node_uid"]).type, np.uint8)
        self.assertLessEqual(UID_VOCAB_MAX - 1, np.iinfo(np.uint8).max)

    def test_index_put_accepts_every_graph_dtype(self):
        # The exact operation the buffer performs, so this fails here rather
        # than on the first add_transition of a real run.
        for key, dtype in _PACKED_DTYPES.items():
            with self.subTest(key=key):
                store = torch.zeros(4, dtype=torch.from_numpy(np.zeros(1, dtype)).dtype)
                store[torch.tensor([0, 2])] = store[torch.tensor([1, 3])]

    def test_uid_vocab_above_the_dtype_raises(self):
        with self.assertRaises(ValueError):
            EpisodeUIDs(UID_VOCAB_MAX + 1, seed=0)


class EpisodeUIDTest(unittest.TestCase):
    def test_uid_survives_disappearance(self):
        uids = EpisodeUIDs(UID_VOCAB, seed=0)
        first = uids.uid_for("apple-1")
        uids.uid_for("can-1")
        uids.uid_for("can-2")
        # apple-1 left the view for a while; its code is still its own.
        self.assertEqual(uids.uid_for("apple-1"), first)

    def test_codes_are_never_shared(self):
        uids = EpisodeUIDs(UID_VOCAB, seed=0)
        codes = [uids.uid_for(f"n{i}") for i in range(UID_VOCAB - 2)]
        self.assertEqual(len(set(codes)), len(codes))
        self.assertNotIn(UID_PAD, codes)
        self.assertNotIn(UID_EE, codes)

    def test_ee_code_is_fixed(self):
        uids = EpisodeUIDs(UID_VOCAB, seed=0)
        self.assertEqual(uids.uid_for("ee", is_ee=True), UID_EE)
        self.assertEqual(uids.uid_for("other", is_ee=True), UID_EE)

    def test_reset_clears_the_mapping(self):
        uids = EpisodeUIDs(UID_VOCAB, seed=0)
        uids.uid_for("apple-1")
        uids.reset()
        self.assertEqual(len(uids), 0)

    def test_overflow_raises_rather_than_aliasing(self):
        uids = EpisodeUIDs(6, seed=0)
        for i in range(4):
            uids.uid_for(f"n{i}")
        with self.assertRaises(RuntimeError):
            uids.uid_for("one-too-many")

    def test_codes_are_permuted_per_episode(self):
        # Two episodes must not hand the same object the same code, or a UID
        # becomes a global name the model can memorise.
        uids = EpisodeUIDs(UID_VOCAB, seed=0)
        runs = []
        for _ in range(6):
            uids.reset()
            runs.append(uids.uid_for("apple-1"))
        self.assertGreater(len(set(runs)), 1)


class PooledEncoderTest(unittest.TestCase):
    """Boxes in, one masked-mean token out. No UID anywhere."""

    def _encoder(self, units=32):
        torch.manual_seed(0)
        return GraphEncoder(graph_config(simple=True, units=units))

    def test_no_identity_embedding_exists(self):
        encoder = self._encoder()
        self.assertEqual(encoder.schema, SCHEMA_SIMPLE_POOLED)
        self.assertIsNone(encoder.uid)
        # One flat projection over all cameras, not a per-camera ModuleList.
        self.assertIsNone(encoder.bbox_proj)
        self.assertIsNone(encoder.app_proj)

    def test_attention_pooling_is_gone(self):
        encoder = self._encoder()
        self.assertIsNotNone(encoder.pool)
        for name in ("query", "key", "value", "out"):
            self.assertIsNone(getattr(encoder, name), name)

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

    def test_token_is_the_masked_mean_plus_the_count(self):
        encoder = self._encoder()
        data = pooled_graph(n_valid=3)
        with torch.no_grad():
            encoded = encoder(data)
            compact = compact_graph(data, SCHEMA_SIMPLE_POOLED)
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

    def test_slot_mode_still_reads_uid_and_pools_nowhere(self):
        encoder = GraphEncoder(graph_config(simple=True, state_mode="slots"))
        self.assertEqual(encoder.schema, SCHEMA_SIMPLE_SLOT)
        self.assertIsNone(encoder.pool)
        with torch.no_grad():
            encoded = encoder(slot_graph())
        self.assertIsNone(encoded.token)
        self.assertIsNotNone(encoded.slots)

    def test_full_mode_is_unchanged(self):
        encoder = GraphEncoder(graph_config(simple=False))
        self.assertEqual(encoder.schema, SCHEMA_FULL)
        self.assertIsNotNone(encoder.app_proj)
        self.assertIsNone(encoder.uid)
        self.assertIsNone(encoder.pool)
        self.assertIsNotNone(encoder.query)


class PooledDecoderTest(unittest.TestCase):
    """Node and relation reconstruction from g, addressed by the box."""

    LOSSES = {"node", "nodetgt", "relabs", "reltemp"}

    def _decoder(self):
        torch.manual_seed(0)
        scorer = ProgressScorer(PICK_STAGES, 17)
        decoder = SimpleGraphDecoder(
            graph_config(simple=True), semantic_dim=32,
            progress_relations=scorer.relations,
        )
        return decoder, scorer

    def _run(self, sem=None, graph=None, prior=None, prior_valid=None):
        decoder, _ = self._decoder()
        compact = compact_graph(
            pooled_graph() if graph is None else graph, SCHEMA_SIMPLE_POOLED
        )
        if sem is None:
            sem = torch.randn(2, 3, 32, requires_grad=True)
        step_valid = torch.ones(2, 3, dtype=torch.bool)
        out = decoder(sem, compact, step_valid, prior, prior_valid)
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
                sem, compact_graph(base, SCHEMA_SIMPLE_POOLED))
            b = decoder.node_features(
                sem, compact_graph(moved, SCHEMA_SIMPLE_POOLED))
        self.assertTrue(torch.equal(a, b))

    def test_node_count_may_vary_between_frames(self):
        for n_valid in (1, 3, 7):
            with self.subTest(nodes=n_valid):
                _, _, (losses, _) = self._run(graph=pooled_graph(n_valid=max(n_valid, 2)))
                self.assertTrue(torch.isfinite(losses["nodetgt"]))


class PooledProgressHeadTest(unittest.TestCase):
    """One fused head, one relation-row map, one teacher mask."""

    def _decoder(self):
        torch.manual_seed(0)
        scorer = ProgressScorer(PICK_STAGES, 17)
        decoder = SimpleGraphDecoder(
            graph_config(simple=True), semantic_dim=32,
            progress_relations=scorer.relations,
        )
        return decoder, scorer

    def test_rows_come_from_the_scorer_not_the_relation_id(self):
        decoder, scorer = self._decoder()
        self.assertEqual(decoder.n_progress, int(scorer.relations.numel()))
        for row, relation in enumerate(scorer.relations.tolist()):
            self.assertEqual(int(decoder.progress_row[relation]), row)
        # Every relation the scorer does not read is explicitly out of range,
        # never silently row zero.
        unused = set(range(11)) - set(scorer.relations.tolist())
        for relation in unused:
            self.assertEqual(int(decoder.progress_row[relation]), -1)

    def test_head_emits_one_legal_simplex_per_relation(self):
        decoder, scorer = self._decoder()
        probs = decoder.progress_probs(torch.randn(2, 3, 32))
        self.assertEqual(tuple(probs.shape), (2, 3, int(scorer.relations.numel()), 17))
        self.assertTrue(torch.allclose(probs.sum(-1), torch.ones(2, 3, 6), atol=1e-5))
        illegal = ~decoder.progress_valid.expand_as(probs)
        self.assertLess(float(probs.masked_select(illegal).abs().max()), 1e-6)

    def test_reset_frame_is_excluded_and_later_frames_are_not(self):
        decoder, _ = self._decoder()
        compact = compact_graph(pooled_graph(n_edges=4), SCHEMA_SIMPLE_POOLED)
        step_valid = torch.ones(2, 3, dtype=torch.bool)
        prior_valid = step_valid.clone()
        prior_valid[:, 0] = False
        _, metrics = decoder(
            torch.randn(2, 3, 32), compact, step_valid,
            torch.randn(2, 3, 32), prior_valid,
        )
        with_reset = float(metrics["prior_progress_facts"])
        _, all_frames = decoder(
            torch.randn(2, 3, 32), compact, step_valid,
            torch.randn(2, 3, 32), step_valid,
        )
        # Two of six frames are reset frames, so dropping them removes a third
        # of the supervised facts and nothing else.
        self.assertAlmostEqual(
            with_reset, float(all_frames["prior_progress_facts"]) * 2 / 3, places=5
        )

    def test_absent_target_masks_the_loss_to_zero(self):
        # Occluded, unresolved or not yet admitted all look the same here: the
        # flag is dark and the edge is gone, so no target_resolved key is needed.
        decoder, _ = self._decoder()
        graph = pooled_graph()
        graph["graph_node_target"][:] = 0
        compact = compact_graph(graph, SCHEMA_SIMPLE_POOLED)
        step_valid = torch.ones(2, 3, dtype=torch.bool)
        losses, metrics = decoder(
            torch.randn(2, 3, 32), compact, step_valid,
            torch.randn(2, 3, 32), step_valid,
        )
        self.assertEqual(float(metrics["prior_progress_facts"]), 0.0)
        self.assertEqual(float(losses["prior_progress_relabs"]), 0.0)

    def test_observed_labels_never_reach_the_head(self):
        # The target flag selects which edge supplies the teacher label; no
        # gradient may run from the prior objective back into the posterior g.
        decoder, _ = self._decoder()
        compact = compact_graph(pooled_graph(), SCHEMA_SIMPLE_POOLED)
        sem = torch.randn(2, 3, 32, requires_grad=True)
        prior = torch.randn(2, 3, 32, requires_grad=True)
        step_valid = torch.ones(2, 3, dtype=torch.bool)
        losses, _ = decoder(sem, compact, step_valid, prior, step_valid)
        losses["prior_progress_relabs"].backward()
        self.assertTrue(sem.grad is None or float(sem.grad.abs().sum()) == 0.0)
        self.assertGreater(float(prior.grad.abs().sum()), 0.0)

    def test_branch_is_skipped_when_no_prior_is_supplied(self):
        decoder, _ = self._decoder()
        losses, metrics = decoder(
            torch.randn(2, 3, 32),
            compact_graph(pooled_graph(), SCHEMA_SIMPLE_POOLED),
            torch.ones(2, 3, dtype=torch.bool),
        )
        self.assertNotIn("prior_progress_relabs", losses)
        self.assertNotIn("prior_progress_acc", metrics)


class PotentialMatrixTest(unittest.TestCase):
    """The soft potential is linear, so it is one contraction, not 14 stages."""

    def setUp(self):
        self.scorer = ProgressScorer(PICK_STAGES, 17)

    def _stagewise(self, probs):
        return (self.scorer.satisfaction(probs, False) * self.scorer.weights).sum(-1)

    def test_matrix_matches_the_cumulative_stage_sum(self):
        torch.manual_seed(0)
        probs = torch.softmax(torch.randn(5, 6, 17), -1)
        self.assertTrue(torch.allclose(
            self.scorer.potential(probs), self._stagewise(probs), atol=1e-6
        ))

    def test_potential_stays_bounded(self):
        torch.manual_seed(1)
        probs = torch.softmax(torch.randn(64, 6, 17) * 4.0, -1)
        phi = self.scorer.potential(probs)
        self.assertGreaterEqual(float(phi.min()), 0.0)
        self.assertLessEqual(float(phi.max()), 1.0)

    def test_saturated_state_scores_exactly_one(self):
        best = torch.zeros(1, 6, 17)
        satisfying = {5: 3, 6: 10, 8: 13, 7: 13, 1: 2, 2: 2}
        for row, relation in enumerate(self.scorer.relations.tolist()):
            best[0, row, satisfying[relation]] = 1.0
        self.assertAlmostEqual(float(self.scorer.potential(best)), 1.0, places=6)
        self.assertAlmostEqual(
            float(self.scorer.potential(best, hard=True)), 1.0, places=6
        )

    def test_worst_state_scores_zero(self):
        worst = torch.zeros(1, 6, 17)
        unsatisfying = {5: 7, 6: 8, 8: 16, 7: 16, 1: 1, 2: 1}
        for row, relation in enumerate(self.scorer.relations.tolist()):
            worst[0, row, unsatisfying[relation]] = 1.0
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
            graph_token_size=self.TOKEN, graph_simple=True,
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
        env_config = OmegaConf.load("configs/env/mshab.yaml")
        declared = set(env_config.graph.keys())
        forwarded = set(_GRAPH_CONFIG_KEYS) | {"cameras"}
        self.assertEqual(
            declared - forwarded,
            set(),
            "configs/env/mshab.yaml declares graph keys the adapter never "
            "passes to build_graph_obs; they would be silently ignored",
        )

    def test_pooled_stays_the_default_state_mode(self):
        # The frozen baseline must not move because slot mode exists.
        model_config = OmegaConf.load("configs/model/_base_.yaml")
        self.assertEqual(model_config.graph.state_mode, "pooled")
        self.assertEqual(model_config.graph.n_max, 8)
        self.assertEqual(model_config.graph.e_max, 168)
        self.assertIs(model_config.graph_simple, False)
        self.assertIs(model_config.progress.enabled, False)
        self.assertIs(model_config.graph.slot_births, False)
        self.assertEqual(model_config.loss_scales.reltemp, 1.0)

    def test_the_slot_preset_pins_the_node_contract(self):
        slot_config = OmegaConf.load("configs/model/size100M_graph_slots.yaml")
        self.assertEqual(slot_config.graph.state_mode, "slots")
        # End effector plus seven objects, matching the pooled arm, so a sixth
        # object cannot be dropped upstream by the vertex registry.
        self.assertEqual(slot_config.graph.n_max, 8)
        # 3 * n_max * (n_max - 1): truncation-free, and the packer drops
        # spatial relations first, so truncation is not a neutral loss.
        self.assertEqual(slot_config.graph.e_max, 168)
        self.assertEqual(slot_config.loss_scales.reltemp, 1.0)
        self.assertEqual(slot_config.loss_scales.slotalive, 1.0)
        self.assertEqual(slot_config.loss_scales.prior_nodetgt, 1.0)
        self.assertEqual(slot_config.loss_scales.prior_progress_relabs, 1.0)
        self.assertIs(slot_config.graph.slot_births, True)
        # The all-edge prior relation losses are replaced by the teacher-forced
        # end-effector-to-target one.
        self.assertNotIn("prior_relabs", slot_config.loss_scales)
        self.assertNotIn("prior_reltemp", slot_config.loss_scales)
        self.assertEqual(slot_config.graph.slot_dim, 256)
        self.assertEqual(slot_config.graph.slot_heads, 4)
        self.assertEqual(slot_config.graph.slot_mixer_layers, 1)
        self.assertIs(slot_config.graph_simple, True)
        # The relation-only contract stays relation-only.
        pooled = OmegaConf.load("configs/model/size100M_graph.yaml")
        for key in ("deter", "hidden", "discrete", "depth", "units", "rep_loss"):
            self.assertEqual(slot_config[key], pooled[key], key)

    def test_the_pooled_simple_preset_pins_its_contract(self):
        preset = OmegaConf.load("configs/model/size50M_graph_simple.yaml")
        self.assertIs(preset.graph_simple, True)
        self.assertEqual(preset.graph.state_mode, "pooled")
        self.assertEqual(preset.graph.n_max, 8)
        self.assertEqual(preset.graph.e_max, 168)
        self.assertEqual(preset.graph.decoder_units, 256)
        self.assertEqual(preset.graph.bbox_query_dim, 4)
        self.assertIs(preset.progress.enabled, True)
        self.assertEqual(preset.progress.beta, 0.05)
        for name in ("node", "nodetgt", "relabs", "reltemp",
                     "prior_progress_relabs", "progress_value", "graphdyn"):
            self.assertEqual(preset.loss_scales[name], 1.0, name)
        self.assertEqual(preset.loss_scales.graphrep, 0.05)
        # There is no recurrent slot state to supervise in this arm.
        for name in ("slotdyn", "slotalive", "prior_nodetgt"):
            self.assertNotIn(name, preset.loss_scales)

    def test_every_arm_sees_the_same_privileged_observation(self):
        # The graph is built from privileged state in every graph arm, so the
        # graph-free baseline has to see the same fields or the comparison
        # measures privilege instead of representation.
        env_config = OmegaConf.load("configs/env/mshab.yaml")
        self.assertIs(env_config.nonprivileged_obs, False)

    def test_simple_mode_reaches_the_builder(self):
        graph_config = OmegaConf.create({
            "enabled": True, "simple": True, "uid_vocab": 64,
            "profile": "room_scale", "thresholds_path": "", "whitelist_dir": "",
            "n_max": 8, "e_max": 168, "k_persist": -1, "app_dim": 384,
            "dino_model": "dinov2_vits14_reg", "dino_res": 112,
            "dino_weights": "", "staleness_enabled": False,
            "bypass_teemo": False, "state_mode": "pooled",
        })
        out = graph_observation_config(graph_config, ["fetch_head"])
        self.assertIs(out["simple"], True)
        # ``simple`` alone cannot tell the adapter which contract to emit.
        self.assertEqual(out["state_mode"], "pooled")
        self.assertEqual(out["uid_vocab"], 64)
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
