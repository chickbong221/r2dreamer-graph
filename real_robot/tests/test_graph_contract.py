"""Orientation, label mirroring, vocabulary and packing against the repository's contract."""

from __future__ import annotations

import copy
import unittest

import numpy as np

from scenegraph.adapters.graph_pack import GRAPH_KEYS
from scenegraph.adapters.graph_vocab import build_absolute_vocab, build_relation_vocab, build_temporal_vocab
from scenegraph.core import schedule as repo_schedule
from scenegraph.core.relation_rules import TEMPORAL_RELATIONS

from ..common import load_config
from ..graphs.pack import build_frame_graph, node_rows, pack_episode, pack_frame
from ..graphs.schema import CHANGE_MIRROR, HEIGHT_MIRROR, GraphSpec, SpecError
from ..graphs.vocabulary import build_vocab, vocab_sizes
from . import synthetic as syn


class Orientation(unittest.TestCase):
    def setUp(self):
        self.spec = syn.graph_spec()

    def test_end_effector_facts_start_at_the_end_effector(self):
        for fact in self.spec.facts:
            if "ee" in (fact.src, fact.dst):
                self.assertEqual(fact.src, "ee", fact)

    def test_object_pairs_follow_key_order(self):
        for fact in self.spec.facts:
            if fact.src != "ee":
                self.assertLess(self.spec.entity(fact.src).key, self.spec.entity(fact.dst).key, fact)
        self.assertIsNotNone(self.spec.fact_index("banana", "pot", "contain"))
        self.assertIsNone(self.spec.fact_index("pot", "banana", "contain"))

    def test_height_offset_mirrors_with_the_swap(self):
        src, dst, label, change = self.spec.canonicalize("height-offset", "pot", "banana", "above", "increase-fast")
        self.assertEqual((src, dst, label, change), ("banana", "pot", "below", "decrease-fast"))

    def test_directional_labels_mirror_with_the_swap(self):
        _, _, label, _ = self.spec.canonicalize("contain", "pot", "banana", "src-holds")
        self.assertEqual(label, "dst-holds")
        _, _, label, _ = self.spec.canonicalize("support", "pot", "lid", "not-holds")
        self.assertEqual(label, "not-holds")

    def test_symmetric_labels_do_not_change(self):
        _, _, label, change = self.spec.canonicalize("planar-distance", "pot", "banana", "near", "decrease-slow")
        self.assertEqual((label, change), ("near", "decrease-slow"))

    def test_mirror_tables_agree_with_the_schedule_compiler(self):
        self.assertEqual(HEIGHT_MIRROR, repo_schedule._MIRROR)
        self.assertEqual({k: CHANGE_MIRROR[CHANGE_MIRROR[k]] for k in CHANGE_MIRROR}, {k: k for k in CHANGE_MIRROR})


class Configuration(unittest.TestCase):
    def config(self):
        return copy.deepcopy(load_config("graph"))

    def test_grasp_between_objects_is_refused(self):
        cfg = self.config()
        cfg["facts"].append({"pair": ["banana", "pot"], "relations": ["grasp"]})
        with self.assertRaises(SpecError):
            GraphSpec.from_config(cfg)

    def test_duplicate_facts_are_refused_in_either_orientation(self):
        cfg = self.config()
        cfg["facts"].append({"pair": ["pot", "banana"], "relations": ["contain"]})
        with self.assertRaises(SpecError):
            GraphSpec.from_config(cfg)

    def test_too_many_facts_for_the_edge_budget(self):
        cfg = self.config()
        cfg["e_max"] = 5
        with self.assertRaises(SpecError):
            GraphSpec.from_config(cfg)


class Vocabulary(unittest.TestCase):
    def test_label_vocabularies_are_the_repository_tables(self):
        vocab = build_vocab(syn.graph_spec())
        self.assertEqual(vocab.relation.token_to_id, build_relation_vocab().token_to_id)
        self.assertEqual(vocab.absolute.token_to_id, build_absolute_vocab().token_to_id)
        self.assertEqual(vocab.temporal.token_to_id, build_temporal_vocab().token_to_id)

    def test_sizes_match_what_the_decoder_masks_expect(self):
        sizes = vocab_sizes(build_vocab(syn.graph_spec()))
        base = load_config("configs/model/_base_.yaml")["graph"]
        self.assertEqual((sizes["n_rel"], sizes["n_abs"], sizes["n_temp"]), (base["n_rel"], base["n_abs"], base["n_temp"]))
        self.assertEqual(sizes["entity_vocab"], 2 + len(syn.graph_spec().object_ids))

    def test_end_effector_has_the_reserved_id(self):
        vocab = build_vocab(syn.graph_spec())
        self.assertEqual(vocab.entity.pad_id, 0)
        self.assertEqual(vocab.entity.ee_id, 1)


class Packing(unittest.TestCase):
    def setUp(self):
        self.spec = syn.graph_spec()
        self.vocab = build_vocab(self.spec)
        self.annotation = syn.annotation(self.spec)
        self.assertTrue(self.annotation.valid, [i.message for i in self.annotation.issues])
        n, e, c = self.annotation.n_frames, len(self.spec.entities), len(self.spec.cameras)
        self.boxes = np.tile(np.array([0.1, 0.4, 0.2, 0.5], dtype=np.float32), (n, e, c, 1))
        self.visible = np.ones((n, e, c), dtype=bool)
        self.centroids = np.random.default_rng(0).normal(size=(n, e, 3)).astype(np.float32)
        self.known = np.ones((n, e), dtype=bool)

    def test_rows_and_dtypes_follow_the_packer(self):
        arrays, valid = pack_episode(self.spec, self.vocab, self.annotation, self.boxes, self.visible,
                                     self.centroids, self.known)
        self.assertTrue(valid.all())
        self.assertEqual(set(arrays), set(GRAPH_KEYS))
        self.assertEqual(arrays["graph_node_ent"].dtype, np.uint8)
        self.assertEqual(arrays["graph_node_bbox"].shape[1:], (self.spec.n_max, len(self.spec.cameras), 4))
        self.assertTrue((arrays["graph_node_ent"][:, 0] == self.vocab.entity.ee_id).all())
        self.assertTrue((arrays["graph_node_target"][:, 1] == 1).all())
        self.assertEqual(int(arrays["graph_node_target"].sum()), self.annotation.n_frames)

    def test_switching_target_keeps_every_identity(self):
        arrays, _ = pack_episode(self.spec, self.vocab, self.annotation, self.boxes, self.visible,
                                 self.centroids, self.known)
        before = node_rows(self.spec, {k: v[0] for k, v in arrays.items()}, self.vocab)
        after = node_rows(self.spec, {k: v[-1] for k, v in arrays.items()}, self.vocab)
        self.assertEqual(before["banana"], 1)
        self.assertEqual(after["lid"], 1)
        self.assertEqual(set(before), set(after))
        # The banana's facts survive the switch with its entity id, wherever its row is.
        index = self.spec.facts.index(next(f for f in self.spec.facts if f.relation == "grasp" and f.dst == "banana"))
        self.assertIsNotNone(self.annotation.absolute[index][-1])

    def test_no_temporal_label_before_the_window(self):
        arrays, _ = pack_episode(self.spec, self.vocab, self.annotation, self.boxes, self.visible,
                                 self.centroids, self.known)
        K = self.spec.temporal_window
        self.assertTrue((arrays["graph_edge_temp"][:K] == 0).all())
        temporal_facts = sum(f.relation in TEMPORAL_RELATIONS for f in self.spec.facts)
        self.assertEqual(int((arrays["graph_edge_temp"][K] > 0).sum()), temporal_facts)

    def test_invisible_camera_leaves_an_empty_box(self):
        visible = self.visible.copy()
        visible[:, :, 1] = False
        arrays, _ = pack_episode(self.spec, self.vocab, self.annotation, self.boxes, visible,
                                 self.centroids, self.known)
        self.assertTrue((arrays["graph_node_bbox"][:, :, 1] == 0).all())
        self.assertTrue((arrays["graph_node_bbox"][:, :5, 0, 1] > 0).all())

    def test_overflowing_edges_raise_rather_than_truncate(self):
        cfg = copy.deepcopy(load_config("graph"))
        spec = GraphSpec.from_config(cfg)
        spec.e_max = 10
        graph = build_frame_graph(spec, self.annotation, 0, self.boxes[0], self.visible[0],
                                  self.centroids[0], self.known[0])
        with self.assertRaises(RuntimeError):
            pack_frame(spec, self.vocab, graph)


if __name__ == "__main__":
    unittest.main()
