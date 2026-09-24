"""The shipped graph configuration, the vocabulary and the packed rows, against the repository's packer."""

from __future__ import annotations

import unittest

import numpy as np

from scenegraph.adapters.graph_vocab import EE_TOKEN, PAD_TOKEN

from ..graphs.pack import pack_episode
from ..graphs.schema import GraphConfig, SpecError
from ..graphs.validate import build_annotation
from ..graphs.vocabulary import build_vocab, vocab_sizes
from . import synthetic


class Configuration(unittest.TestCase):
    def test_every_task_loads_with_the_end_effector_first(self):
        config = synthetic.graph_config()
        self.assertEqual(set(config.tasks), set(synthetic.TASKS))
        for spec in config.tasks.values():
            self.assertEqual(spec.entities[0].id, "ee")
            self.assertLessEqual(len(spec.entities), config.n_max)
            self.assertLessEqual(len(spec.facts), config.e_max)

    def test_dataset_task_strings_map_to_tasks(self):
        config = synthetic.graph_config()
        self.assertEqual(config.task_for("Pick blue cube and place on red cube"), "blue_on_red")
        self.assertEqual(config.task_for("Pick all cubes and place into cup "), "cubes_in_cup")
        with self.assertRaises(KeyError):
            config.task_for("stack the plates")

    def test_a_task_naming_an_unknown_entity_is_refused(self):
        cfg = synthetic.configs()["graph"]
        cfg["tasks"]["blue_on_red"]["entities"].append("plate")
        with self.assertRaises(SpecError):
            GraphConfig.from_config(cfg)

    def test_object_pairs_are_stored_in_key_order_with_mirrored_labels(self):
        spec = synthetic.spec("cubes_in_cup")
        self.assertIsNotNone(spec.fact_index("cup", "red_cube", "contain"))
        self.assertIsNone(spec.fact_index("red_cube", "cup", "contain"))
        self.assertEqual(spec.canonicalize("contain", "red_cube", "cup", "dst-holds"),
                         ("cup", "red_cube", "src-holds", None))
        self.assertEqual(spec.canonicalize("height-offset", "red_cube", "cup", "above", "increase-fast"),
                         ("cup", "red_cube", "below", "decrease-fast"))


class Vocabulary(unittest.TestCase):
    def test_entity_ids_are_shared_by_every_task(self):
        config = synthetic.graph_config()
        vocab = build_vocab(config)
        table = vocab.entity.token_to_id
        self.assertEqual(table[PAD_TOKEN], 0)
        self.assertEqual(table[EE_TOKEN], 1)
        objects = [e.key for e in config.entities if e.type == "object"]
        self.assertEqual([table[key] for key in objects], list(range(2, 2 + len(objects))))
        self.assertEqual(vocab_sizes(vocab)["entity_vocab"], 2 + len(objects))


class Packing(unittest.TestCase):
    def pack(self, task: str, n: int = 90):
        spec = synthetic.spec(task)
        annotation = build_annotation(spec, episode_index=0, n_frames=n, fps=30.0,
                                      answer=synthetic.complete_answer(spec, n), settings=synthetic.settings())
        self.assertTrue(annotation.valid, [i.message for i in annotation.issues])
        vocab = build_vocab(synthetic.graph_config())
        arrays, valid = pack_episode(spec, vocab, annotation)
        return spec, vocab, annotation, arrays, valid

    def test_rows_are_the_repository_contract(self):
        for task in synthetic.TASKS:
            spec, vocab, annotation, arrays, valid = self.pack(task)
            n = annotation.n_frames
            self.assertEqual(arrays["graph_node_ent"].shape, (n, spec.n_max))
            self.assertEqual(arrays["graph_node_bbox"].shape, (n, spec.n_max, len(spec.cameras), 4))
            self.assertEqual(arrays["graph_edge_rel"].shape, (n, spec.e_max))
            self.assertTrue(valid.all())
            self.assertTrue((arrays["graph_node_ent"][:, 0] == vocab.entity.ee_id).all())
            for t in (0, n - 1):
                target = spec.entity(annotation.active_target[t])
                self.assertEqual(arrays["graph_node_ent"][t, 1], vocab.entity.encode(target.key))
                self.assertEqual(arrays["graph_node_target"][t].tolist().index(1), 1)
            self.assertEqual(int((arrays["graph_node_ent"][0] > 0).sum()), len(spec.entities))
            self.assertEqual(int((arrays["graph_edge_rel"][0] > 0).sum()), len(spec.facts))
            self.assertFalse(arrays["graph_node_centroid"].any())

    def test_a_target_switch_moves_the_new_target_to_row_one(self):
        spec, vocab, annotation, arrays, _ = self.pack("banana_pot_lid")
        lid, banana = vocab.entity.encode("actor:lid"), vocab.entity.encode("actor:banana")
        self.assertEqual(arrays["graph_node_ent"][0, 1], banana)
        self.assertEqual(arrays["graph_node_ent"][-1, 1], lid)
        self.assertIn(banana, arrays["graph_node_ent"][-1, 2:].tolist())

    def test_edges_decode_to_the_annotated_labels(self):
        spec, vocab, annotation, arrays, _ = self.pack("blue_on_red")
        t = 50
        rows = {int(ent): row for row, ent in enumerate(arrays["graph_node_ent"][t]) if ent}
        relation = {i: name for name, i in vocab.relation.token_to_id.items()}
        absolute = {i: name for name, i in vocab.absolute.token_to_id.items()}
        temporal = {i: name for name, i in vocab.temporal.token_to_id.items()}
        decoded = {}
        for e in range(spec.e_max):
            if not arrays["graph_edge_rel"][t, e]:
                continue
            decoded[(int(arrays["graph_edge_src"][t, e]), int(arrays["graph_edge_dst"][t, e]),
                     relation[int(arrays["graph_edge_rel"][t, e])])] = (
                absolute[int(arrays["graph_edge_abs"][t, e])], temporal.get(int(arrays["graph_edge_temp"][t, e])))
        for index, fact in enumerate(spec.facts):
            src = rows[vocab.entity.encode(spec.entity(fact.src).key)]
            dst = rows[vocab.entity.encode(spec.entity(fact.dst).key)]
            label, change = decoded[(src, dst, fact.relation)]
            self.assertEqual(label, annotation.absolute[index][t])
            self.assertEqual(change, annotation.temporal[index][t] if fact.temporal else None)

    def test_boxes_are_normalised_x0_x1_y0_y1(self):
        spec, vocab, annotation, arrays, _ = self.pack("blue_on_red")
        box = arrays["graph_node_bbox"][0, 0, 0].astype(np.float32)
        expected = synthetic.gemini_box(0, 0)
        np.testing.assert_allclose(box, [expected[1] / 1000, expected[3] / 1000, expected[0] / 1000,
                                         expected[2] / 1000], atol=1e-3)


if __name__ == "__main__":
    unittest.main()
