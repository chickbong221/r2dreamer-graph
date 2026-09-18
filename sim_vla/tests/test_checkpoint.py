"""Stage 8: checkpoints round-trip, and refuse an incompatible arm."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .common import require_torch


def meta(graph_enabled=True, **kwargs):
    from sim_vla.runtime.checkpoint import CheckpointMeta

    base = dict(graph_enabled=graph_enabled, stage="world_model",
                env_id="PickCube-v1", feature_dim=64,
                pretrained_revision="abc123", step=10)
    return CheckpointMeta(**(base | kwargs))


class TestCheckpoint(unittest.TestCase):
    def module(self, width=4):
        torch = require_torch()
        return torch.nn.Linear(width, width)

    def test_round_trip_restores_weights_and_optimizer(self):
        torch = require_torch()
        from sim_vla.runtime.checkpoint import load, save

        model = self.module()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.weight.data.fill_(3.0)
        opt.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            save(path, meta(), {"model": model}, {"opt": opt})
            self.assertTrue(path.with_suffix(".json").exists())

            restored = self.module()
            restored_opt = torch.optim.Adam(restored.parameters(), lr=1e-3)
            stored = load(path, meta(), {"model": restored}, {"opt": restored_opt})
            self.assertTrue(torch.allclose(restored.weight, model.weight))
            self.assertEqual(stored.step, 10)

    def test_a_graph_checkpoint_is_refused_by_the_baseline(self):
        require_torch()
        from sim_vla.runtime.checkpoint import load, save

        model = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.pt"
            save(path, meta(graph_enabled=True), {"model": model})
            with self.assertRaises(SystemExit) as caught:
                load(path, meta(graph_enabled=False), {"model": self.module()})
            # Turning the flag off is not a conversion: the graph has already
            # reached h and z.
            self.assertIn("baseline", str(caught.exception))

    def test_a_different_task_or_revision_is_refused(self):
        require_torch()
        from sim_vla.runtime.checkpoint import load, save

        model = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.pt"
            save(path, meta(), {"model": model})
            for bad in ({"env_id": "PlaceSphere-v1"},
                        {"pretrained_revision": "deadbeef"},
                        {"feature_dim": 128}):
                with self.subTest(**bad):
                    with self.assertRaises(SystemExit):
                        load(path, meta(**bad), {"model": self.module()})

    def test_resume_continues_from_the_stored_step(self):
        torch = require_torch()
        from sim_vla.runtime.checkpoint import load, save

        model = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.pt"
            save(path, meta(step=4242), {"model": model})
            stored = load(path, meta(step=0), {"model": self.module()})
            self.assertEqual(stored.step, 4242)


if __name__ == "__main__":
    unittest.main()
