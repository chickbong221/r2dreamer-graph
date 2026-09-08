"""A short run end to end, and the six plumbing checks it has to pass.

This is the plan's pre-flight without the simulator: the same training loop, the
same probe schedule and the same checks, on synthetic graphs so it costs
seconds. The full ``--smoke`` run adds the collection half.
"""

from __future__ import annotations

import csv
import os
import tempfile
import unittest

from .synthetic import make_dataset
from .test_model import MODEL_CFG

try:
    import torch
except ImportError:                                        # pragma: no cover
    torch = None


def config(**train_overrides) -> dict:
    train = {
        "lr": 1e-3,
        "batch_size": 16,
        "max_updates": 12,
        "probe_every": 4,
        "min_updates": 0,
        "plateau_checks": 8,
        "plateau_rel_improve": 0.01,
        "monitor_frames": 32,
        "device": "cpu",
        "seed": 0,
        "loss_scales": {"node": 1.0, "nodetgt": 1.0, "relabs": 1.0, "reltemp": 1.0},
    }
    train.update(train_overrides)
    return {
        "seed": 0,
        "model": dict(MODEL_CFG),
        "train": train,
        "probe": {
            "tolerance_control_factor": 5.0,
            "tolerance_rel_floor": 1e-3,
            "repeats": 2,
            "zero_token_eps": 1e-6,
            "batch_size": 16,
        },
    }


@unittest.skipIf(torch is None, "torch is not installed")
class ShortRun(unittest.TestCase):
    def setUp(self):
        from ..pairs import build_pairs

        self.dataset = make_dataset(48, seed=21)
        self.pairs = build_pairs(self.dataset.frames, per_group=3, controls=2, seed=21)

    def _train(self, cfg, tmp):
        from ..train import train

        return train(cfg, self.dataset, self.pairs, tmp)

    def test_the_baseline_is_measured_before_the_first_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._train(config(), tmp)
        self.assertEqual(result.first.update, 0)
        self.assertEqual(result.history[0]["update"], 0)
        self.assertIsNone(result.history[0]["train_loss"])

    def test_the_budget_is_not_convergence(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._train(config(), tmp)
        self.assertEqual(result.updates, 12)
        self.assertEqual(result.stop_reason, "budget")
        self.assertFalse(result.converged)

    def test_a_plateau_stops_early_and_says_so(self):
        cfg = config(plateau_rel_improve=10.0, plateau_checks=2, max_updates=100)
        with tempfile.TemporaryDirectory() as tmp:
            result = self._train(cfg, tmp)
        self.assertEqual(result.stop_reason, "plateau")
        self.assertTrue(result.converged)
        self.assertLess(result.updates, 100)

    def test_every_pair_gets_a_row_at_every_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._train(config(), tmp)
            with open(os.path.join(tmp, "probe_rows.csv")) as handle:
                rows = list(csv.DictReader(handle))
        updates = sorted({int(row["update"]) for row in rows})
        self.assertEqual(updates, [0, 4, 8, 12])
        self.assertEqual(len(rows), len(updates) * len(self.pairs))
        self.assertEqual(len(result.history), len(updates))

    def test_the_run_directory_holds_what_the_report_needs(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._train(config(), tmp)
            written = set(os.listdir(tmp))
        for name in ("probe_rows.csv", "progress.csv", "history.json",
                     "checkpoint_init.pt", "checkpoint_final.pt"):
            self.assertIn(name, written)

    def test_reconstruction_loss_falls(self):
        cfg = config(max_updates=40, probe_every=10)
        with tempfile.TemporaryDirectory() as tmp:
            result = self._train(cfg, tmp)
        self.assertLess(result.history[-1]["monitor_loss"], result.history[0]["monitor_loss"])

    def test_the_monitor_subset_contains_every_probe_frame(self):
        """The convergence number and the distances are read on the same graphs."""
        import numpy as np

        from ..train import build_pool, monitor_indices

        pool, index_a, index_b = build_pool(self.dataset, self.pairs)
        pinned = np.concatenate([index_a, index_b])
        chosen = monitor_indices(len(pool), pinned, 32, np.random.default_rng(0))
        self.assertTrue(set(pinned.tolist()).issubset(set(chosen.tolist())))

    def test_the_six_plumbing_checks_pass(self):
        from ..run import plumbing_checks

        cfg = config()
        with tempfile.TemporaryDirectory() as tmp:
            result = self._train(cfg, tmp)
            passed, lines = plumbing_checks(cfg, self.dataset, self.pairs, result)
        self.assertTrue(passed, "\n".join(lines))
        self.assertEqual(len(lines), 6)

    def test_the_report_names_the_stop_reason_and_every_pair(self):
        from ..run import write_report

        cfg = config()
        with tempfile.TemporaryDirectory() as tmp:
            result = self._train(cfg, tmp)
            path = write_report(
                os.path.join(tmp, "report.md"), cfg, self.dataset, self.pairs, result
            )
            text = open(path).read()
        self.assertIn("not convergence", text)
        for spec in self.pairs.specs:
            self.assertIn(spec.name, text)


if __name__ == "__main__":
    unittest.main()
