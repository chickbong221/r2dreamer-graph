"""The discrete readout has to agree with the loss that produced it.

The strongest available check is that accuracies computed here from argmax match
the decoder's own metrics exactly. If ``SimpleGraphDecoder`` ever changes which
classes it masks, this fails rather than quietly reporting a number the loss
never optimised.
"""

from __future__ import annotations

import csv
import os
import tempfile
import unittest

import numpy as np

from .synthetic import make_dataset
from .test_model import MODEL_CFG

try:
    import torch
except ImportError:                                        # pragma: no cover
    torch = None

DECODER_CFG = {
    "enabled": True, "batch_size": 8, "max_frames": 0, "dump_frames": 4, "examples": 2,
}


@unittest.skipIf(torch is None, "torch is not installed")
class Readout(unittest.TestCase):
    def setUp(self):
        from ..model import build_model

        torch.manual_seed(0)
        self.dataset = make_dataset(40, seed=17)
        self.model = build_model(MODEL_CFG, self.dataset.meta)
        self.frames = self.dataset.frames

    def test_predictions_reproduce_the_decoders_own_metrics(self):
        """Same masks, same numbers -- or the readout measures something else."""
        from ..model import LOSS_KEYS

        out = self.model.predict(self.frames.torch_batch(np.arange(16)))
        valid = out.node_valid

        entity_acc = float(
            (out.node_ent_pred.eq(out.node_ent_true) & valid).sum() / valid.sum().clamp_min(1)
        )
        self.assertAlmostEqual(entity_acc, float(out.metrics["node_ent_acc"]), places=5)

        edges = out.abs_true.numel()
        abs_acc = float(out.abs_pred.eq(out.abs_true).sum() / max(edges, 1))
        self.assertAlmostEqual(abs_acc, float(out.metrics["relabs_acc"]), places=5)

        mask = out.temp_mask
        temp_acc = float(
            (out.temp_pred.eq(out.temp_true) & mask).sum() / mask.sum().clamp_min(1)
        )
        self.assertAlmostEqual(temp_acc, float(out.metrics["reltemp_acc"]), places=5)
        for key in LOSS_KEYS:
            self.assertIn(key, out.losses)

    def test_predicted_labels_are_always_legal_for_their_relation(self):
        """The decoder masks illegal sigma to -1e9; the argmax must inherit that."""
        from ..pairs import legal_absolute_mask

        legal = torch.as_tensor(legal_absolute_mask())
        out = self.model.predict(self.frames.torch_batch(np.arange(20)))
        chosen = legal[out.edge_rel.cpu(), out.abs_pred.cpu()]
        self.assertTrue(bool(chosen.all()), "an illegal absolute label was predicted")
        self.assertTrue(bool(out.abs_pred.gt(0).all()), "padding was predicted")
        self.assertTrue(bool(out.temp_pred.gt(0).all()), "padding was predicted for a delta")

    def test_the_target_row_is_never_the_end_effector(self):
        out = self.model.predict(self.frames.torch_batch(np.arange(8)))
        self.assertFalse(bool(out.target_mask[:, 0].any()))

    def test_predicting_leaves_the_model_in_training_mode(self):
        self.model.train()
        self.model.predict(self.frames.torch_batch(np.arange(4)))
        self.assertTrue(self.model.training)

    def test_confusions_count_every_scored_item(self):
        from ..decoder_eval import evaluate_decoder

        report = evaluate_decoder(
            self.model, self.frames, np.arange(len(self.frames)),
            split="train", update=0, batch_size=8,
        )
        self.assertEqual(int(report.entity_confusion.sum()), report.heads["node_ent"].total)
        self.assertEqual(int(report.absolute_confusion.sum()), report.heads["relabs"].total)
        self.assertEqual(int(report.temporal_confusion.sum()), report.heads["reltemp"].total)
        self.assertEqual(int(np.trace(report.absolute_confusion)), report.heads["relabs"].correct)
        # Padding is never a true or a predicted label.
        self.assertEqual(int(report.absolute_confusion[0].sum()), 0)
        self.assertEqual(int(report.absolute_confusion[:, 0].sum()), 0)

    def test_batching_does_not_change_the_aggregate(self):
        from ..decoder_eval import evaluate_decoder

        whole = evaluate_decoder(
            self.model, self.frames, np.arange(40), split="train", update=0, batch_size=40)
        split = evaluate_decoder(
            self.model, self.frames, np.arange(40), split="train", update=0, batch_size=7)
        for name in ("node_ent", "relabs", "reltemp"):
            self.assertEqual(whole.heads[name].correct, split.heads[name].correct, name)
            self.assertEqual(whole.heads[name].total, split.heads[name].total, name)
        self.assertAlmostEqual(whole.bbox_mae, split.bbox_mae, places=5)

    def test_an_unscorable_head_reports_no_items_not_zero_accuracy(self):
        """PlaceSphere names no target. "Nothing to score" and "never right" are
        different claims and must not print the same."""
        from ..decoder_eval import evaluate_decoder, report_table

        report = evaluate_decoder(
            self.model, self.frames, np.arange(16), split="train", update=0, batch_size=16)
        self.assertEqual(report.heads["node_target"].total, 0)
        self.assertIn("not scored", report_table(report))
        self.assertIn("no node_target", report.summary_line())

    def test_per_relation_scores_add_up_to_the_total(self):
        from ..decoder_eval import evaluate_decoder, label_maps, name_relations

        report = name_relations(
            evaluate_decoder(self.model, self.frames, np.arange(40), split="train",
                             update=0, batch_size=16),
            label_maps(self.dataset.meta),
        )
        self.assertEqual(
            sum(score.total for score in report.per_relation.values()),
            report.heads["relabs"].total,
        )
        self.assertEqual(
            sum(score.correct for score in report.per_relation.values()),
            report.heads["relabs"].correct,
        )
        for name in report.per_relation:
            self.assertFalse(name.isdigit(), "relation ids should be named by then")

    def test_the_item_dump_pairs_every_label_with_its_prediction(self):
        from ..decoder_eval import label_maps, sample_predictions, write_item_rows

        names = label_maps(self.dataset.meta)
        rows = sample_predictions(self.model, self.frames, np.arange(40), names, limit=3)
        self.assertTrue(rows)
        self.assertEqual({row["pool_index"] for row in rows}, {0, 1, 2})
        for row in rows:
            self.assertEqual(int(row["match"]), int(row["true"] == row["pred"]))
            self.assertIsInstance(row["true"], str)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_item_rows(os.path.join(tmp, "items.csv"), rows)
            with open(path) as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), len(rows))


@unittest.skipIf(torch is None, "torch is not installed")
class EvaluationSet(unittest.TestCase):
    def setUp(self):
        from ..pairs import build_pairs
        from ..train import build_pool

        self.dataset = make_dataset(60, seed=23)
        self.pairs = build_pairs(self.dataset.frames, per_group=2, controls=2, seed=23)
        self.pool, self.index_a, self.index_b = build_pool(self.dataset, self.pairs)
        self.pinned = np.concatenate([self.index_a, self.index_b])

    def test_no_holdout_means_the_readout_says_train(self):
        """The experiment has no test set; the label must not pretend otherwise."""
        from ..train import evaluation_indices

        index, split = evaluation_indices(
            len(self.pool), self.pinned, np.arange(20), 0, np.random.default_rng(0))
        self.assertEqual(split, "train")
        self.assertEqual(index.tolist(), list(range(20)))

    def test_a_holdout_never_contains_a_probe_pair_member(self):
        from ..train import evaluation_indices

        index, split = evaluation_indices(
            len(self.pool), self.pinned, np.arange(20), 12, np.random.default_rng(0))
        self.assertEqual(split, "holdout")
        self.assertEqual(index.size, 12)
        self.assertFalse(set(index.tolist()) & set(self.pinned.tolist()))

    def test_an_oversized_holdout_is_refused(self):
        from ..train import evaluation_indices

        with self.assertRaises(ValueError):
            evaluation_indices(len(self.pool), np.arange(len(self.pool)), np.arange(4), 8,
                               np.random.default_rng(0))

    def test_held_out_frames_are_scored_but_not_trained_on(self):
        from ..train import train
        from .test_train import config

        cfg = config(max_updates=6, probe_every=3, batch_size=8, holdout_frames=10)
        cfg["decoder_eval"] = dict(DECODER_CFG)
        with tempfile.TemporaryDirectory() as tmp:
            result = train(cfg, self.dataset, self.pairs, tmp)
            written = set(os.listdir(tmp))
        self.assertEqual(result.eval_split, "holdout")
        self.assertEqual(result.eval_frames, 10)
        self.assertIn("decoder_eval.csv", written)
        self.assertIn("decoder_predictions.csv", written)

    def test_the_readout_is_written_at_every_probe(self):
        from ..train import train
        from .test_train import config

        cfg = config(max_updates=6, probe_every=3, batch_size=8)
        cfg["decoder_eval"] = dict(DECODER_CFG)
        with tempfile.TemporaryDirectory() as tmp:
            result = train(cfg, self.dataset, self.pairs, tmp)
            with open(os.path.join(tmp, "decoder_eval.csv")) as handle:
                rows = list(csv.DictReader(handle))
            written = set(os.listdir(tmp))
        self.assertEqual([int(row["update"]) for row in rows], [0, 3, 6])
        self.assertTrue(all(row["split"] == "train" for row in rows))
        self.assertIsNotNone(result.decoder)
        # Figures are optional -- matplotlib may be absent -- but never empty.
        for name in ("decoder_confusion.png", "decoder_accuracy.png", "decoder_examples.png"):
            if name in written:
                self.assertGreater(os.path.getsize(os.path.join(tmp, name)), 0)

    def test_the_report_says_which_frames_were_scored(self):
        from ..run import write_report
        from ..train import train
        from .test_train import config

        cfg = config(max_updates=3, probe_every=3, batch_size=8)
        cfg["decoder_eval"] = dict(DECODER_CFG)
        with tempfile.TemporaryDirectory() as tmp:
            result = train(cfg, self.dataset, self.pairs, tmp)
            path = write_report(
                os.path.join(tmp, "report.md"), cfg, self.dataset, self.pairs, result)
            text = open(path).read()
        self.assertIn("What the decoder recovered", text)
        self.assertIn("training** frames", text)
        self.assertIn("not generalisation", text)


if __name__ == "__main__":
    unittest.main()
