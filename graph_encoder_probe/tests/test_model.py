"""The encoder-to-decoder wiring, and the properties the measurement relies on.

Four things have to hold before a distance means anything: reconstruction
gradients reach both modules, an edit survives compaction into the tensors the
encoder actually consumes, measuring changes nothing, and a reloaded checkpoint
measures the same.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from .synthetic import make_dataset

try:
    import torch
except ImportError:                                        # pragma: no cover
    torch = None

MODEL_CFG = {
    "layers": 2,
    "simple_units": 64,
    "decoder_units": 32,
    "embed": 16,
    "bbox_query_dim": 4,
    "bbox_beta": 0.1,
    "reverse_edges": True,
    "act": "SiLU",
    "centroid_origin": [0.0, 0.0, 0.0],
    "centroid_scale": 5.0,
}

PROBE_KWARGS = dict(batch_size=16, zero_token_eps=1e-6)


@unittest.skipIf(torch is None, "torch is not installed")
class ProbeModel(unittest.TestCase):
    def setUp(self):
        from ..model import build_model
        from ..pairs import build_pairs
        from ..train import build_pool

        torch.manual_seed(0)
        self.dataset = make_dataset(48, seed=13)
        self.pairs = build_pairs(self.dataset.frames, per_group=4, controls=3, seed=13)
        self.pool, self.index_a, self.index_b = build_pool(self.dataset, self.pairs)
        self.model = build_model(MODEL_CFG, self.dataset.meta)

    def test_forward_emits_the_token_and_the_four_terms(self):
        from ..model import LOSS_KEYS

        out = self.model(self.pool.torch_batch(np.arange(8)))
        self.assertEqual(tuple(out.token.shape), (8, self.model.token_dim))
        self.assertEqual(sorted(k for k in out.losses if k in LOSS_KEYS), sorted(LOSS_KEYS))
        self.assertTrue(torch.isfinite(out.total))

    def test_the_measured_vector_is_the_encoder_readout(self):
        """Nothing sits between the pooled token and the number reported."""
        batch = self.pool.torch_batch(np.arange(4))
        with torch.no_grad():
            self.assertTrue(torch.equal(self.model.token(batch), self.model.encode(batch).token))

    def test_config_comes_from_the_dataset_not_the_yaml(self):
        from ..model import graph_config

        config = graph_config(MODEL_CFG, self.dataset.meta)
        self.assertEqual(config.n_cams, self.dataset.n_cams)
        self.assertEqual(config.entity_vocab, self.dataset.entity_vocab)

    def test_a_cache_from_other_vocabularies_is_refused(self):
        from ..model import graph_config

        meta = dict(self.dataset.meta)
        meta["vocab_sizes"] = dict(meta["vocab_sizes"]) | {"absolute": 3}
        with self.assertRaises(ValueError):
            graph_config(MODEL_CFG, meta)

    def test_edits_survive_into_the_tensors_the_encoder_consumes(self):
        """The packed difference must still be there after padding is stripped.

        ``compact_graph`` drops padded edge rows and re-indexes the survivors, so
        an edit that landed on a row it removes would be invisible here even
        though the packed arrays differ.
        """
        from graph import compact_graph

        for group in ("absolute", "temporal", "geometry", "assignment"):
            i = next(k for k, spec in enumerate(self.pairs.specs) if spec.group == group)
            a = compact_graph(self.pool.torch_batch([int(self.index_a[i])]))
            b = compact_graph(self.pool.torch_batch([int(self.index_b[i])]))
            field = {
                "absolute": "edge_abs",
                "temporal": "edge_temp",
                "geometry": "node_centroid",
                "assignment": "edge_abs",
            }[group]
            self.assertFalse(
                torch.equal(getattr(a, field), getattr(b, field)),
                f"{group}: the edit vanished when padding was stripped",
            )

    def test_gradients_reach_encoder_and_decoder(self):
        from ..run import check_gradients

        self.assertTrue(check_gradients(self.model, self.pool, None).startswith("ok"))

    def test_probing_does_not_change_weights(self):
        from ..run import check_probe_is_read_only

        message = check_probe_is_read_only(
            self.model, self.pool, self.pairs, self.index_a, self.index_b, None, PROBE_KWARGS
        )
        self.assertTrue(message.startswith("ok"), message)

    def test_probing_leaves_the_model_in_training_mode(self):
        from ..evaluate import probe

        self.model.train()
        probe(self.model, self.pool, self.pairs, self.index_a, self.index_b, update=0, **PROBE_KWARGS)
        self.assertTrue(self.model.training)

    def test_unchanged_pairs_stay_below_every_edit(self):
        """With no threshold, this is the property the measurement rests on."""
        from ..evaluate import probe
        from ..run import check_controls

        result = probe(
            self.model, self.pool, self.pairs, self.index_a, self.index_b, update=0, **PROBE_KWARGS
        )
        self.assertTrue(check_controls(result).startswith("ok"), check_controls(result))

    def test_edited_pairs_move_the_token_further_than_no_edit_does(self):
        """A randomly initialised encoder is expected to separate these already.

        Recorded as a test because the alternative -- an encoder whose token
        ignores the edit at initialisation -- would make every later measurement
        unreadable, and the run should fail loudly rather than report zeros.
        """
        from ..evaluate import probe

        result = probe(
            self.model, self.pool, self.pairs, self.index_a, self.index_b, update=0, **PROBE_KWARGS
        )
        changed = [m for m in result.measurements if m.group != "control"]
        controls = [m for m in result.measurements if m.group == "control"]
        self.assertGreater(min(m.rms for m in changed), max(m.rms for m in controls))

    def test_a_reloaded_checkpoint_measures_the_same(self):
        from ..evaluate import probe
        from ..model import load_checkpoint, save_checkpoint

        reference = probe(
            self.model, self.pool, self.pairs, self.index_a, self.index_b, update=0, **PROBE_KWARGS
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "checkpoint.pt")
            save_checkpoint(path, self.model, {"update": 0})
            reloaded, extra = load_checkpoint(path)
        self.assertEqual(extra["update"], 0)
        again = probe(
            reloaded, self.pool, self.pairs, self.index_a, self.index_b, update=0, **PROBE_KWARGS
        )
        got = again.lookup()
        for item in reference.measurements:
            self.assertAlmostEqual(got[item.name].rms, item.rms, places=6, msg=item.name)

    def test_distances_are_rms_and_cosine_as_defined(self):
        from ..evaluate import distances

        a = torch.tensor([[3.0, 4.0, 0.0, 0.0]])
        b = torch.tensor([[3.0, 0.0, 0.0, 0.0]])
        stats = distances(a, b)
        self.assertAlmostEqual(float(stats["rms"][0]), 2.0, places=6)      # sqrt(16 / 4)
        self.assertAlmostEqual(float(stats["cosine"][0]), 0.6, places=6)   # 9 / (5 * 3)

    def test_a_zero_token_is_flagged_not_scored(self):
        from ..evaluate import PairMeasurement

        item = PairMeasurement("x", "absolute", 0.0, 1.0, 0.0, 0.0, True)
        self.assertEqual(item.cosine_text(), "zero-token")


if __name__ == "__main__":
    unittest.main()
