"""Stage 8: checkpoints round-trip, and refuse an incompatible arm."""

from __future__ import annotations

import tempfile
import types
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

    def test_a_baseline_checkpoint_is_refused_by_the_graph_arm(self):
        """The direction that used to pass.

        ``stored.get(key) not in (None, "", 0)`` treated a recorded ``False``
        as "this checkpoint does not say", because ``False == 0``. So a
        baseline world model loaded into a graph run without a word, and the
        graph arm started from weights that had never seen a graph.
        """
        require_torch()
        from sim_vla.runtime.checkpoint import load, save

        model = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline.pt"
            save(path, meta(graph_enabled=False), {"model": model})
            with self.assertRaises(SystemExit) as caught:
                load(path, meta(graph_enabled=True), {"model": self.module()})
            self.assertIn("graph_enabled", str(caught.exception))

    def test_a_missing_module_is_not_silently_skipped(self):
        require_torch()
        from sim_vla.runtime.checkpoint import load, save

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partial.pt"
            save(path, meta(), {"world_model": self.module()})
            with self.assertRaises(SystemExit) as caught:
                load(path, meta(), {"world_model": self.module(),
                                    "actor": self.module()})
            self.assertIn("actor", str(caught.exception))
            # Explicitly asking for a partial restore is still allowed.
            load(path, meta(), {"world_model": self.module(),
                                "actor": self.module()}, strict=False)

    def test_normalization_identity_must_match(self):
        require_torch()
        from sim_vla.runtime.checkpoint import load, save

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "n.pt"
            save(path, meta(normalization_identity={"digest": "aaa"}),
                 {"model": self.module()})
            with self.assertRaises(SystemExit) as caught:
                load(path, meta(normalization_identity={"digest": "bbb"}),
                     {"model": self.module()})
            self.assertIn("normalization_identity", str(caught.exception))


class TestNormalizationIdentity(unittest.TestCase):
    """Absence and "disabled" are not the same answer.

    ``recorded and current and recorded != current`` compared only when both
    sides said something, so a checkpoint fitted with ``mean_std`` loaded into
    a run with normalization off without a word: the weights then read raw
    observations and emitted raw actions, having been trained on standardised
    ones.
    """

    def module(self):
        torch = require_torch()
        return torch.nn.Linear(4, 4)

    def descriptor(self, mode="mean_std", statistics="abc123"):
        return {"mode": mode, "fields": ["actions", "proprio"],
                "statistics": statistics, "dataset": {"name": "toy"}}

    def roundtrip(self, stored, wanted):
        from sim_vla.runtime.checkpoint import load, save

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.pt"
            save(path, meta(normalization_identity=stored),
                 {"model": self.module()})
            return load(path, meta(normalization_identity=wanted),
                        {"model": self.module()})

    def test_mode_is_recorded_by_the_fitted_normalizer(self):
        require_torch()
        from sim_vla.data.normalization import FieldStats, Normalizer

        stats = FieldStats(mean=[0.0], std=[1.0], low=[-1.0], high=[1.0],
                           minimum=[-1.0], maximum=[1.0], count=4)
        normalizer = Normalizer(fields={"actions": stats}, identity={"a": 1})
        descriptor = normalizer.descriptor()
        self.assertEqual(descriptor["mode"], "mean_std")
        self.assertEqual(descriptor["fields"], ["actions"])
        self.assertTrue(descriptor["statistics"])

    def test_the_fingerprint_follows_the_numbers_not_the_dataset(self):
        require_torch()
        from sim_vla.data.normalization import FieldStats, Normalizer

        def build(std):
            stats = FieldStats(mean=[0.0], std=[std], low=[-1.0], high=[1.0],
                               minimum=[-1.0], maximum=[1.0], count=4)
            # Same dataset identity either way.
            return Normalizer(fields={"actions": stats},
                              identity={"dataset": "same"})

        first, second = build(1.0), build(2.0)
        self.assertEqual(first.identity, second.identity)
        self.assertNotEqual(first.statistics_fingerprint(),
                            second.statistics_fingerprint(),
                            "statistics that differ share a fingerprint")
        self.assertEqual(first.statistics_fingerprint(),
                         build(1.0).statistics_fingerprint())

    def test_normalized_weights_are_refused_by_an_unnormalized_run(self):
        require_torch()
        with self.assertRaises(SystemExit) as caught:
            self.roundtrip(self.descriptor(), {"mode": "none"})
        self.assertIn("normalization", str(caught.exception))

    def test_normalized_weights_are_refused_by_a_run_that_says_nothing(self):
        require_torch()
        with self.assertRaises(SystemExit) as caught:
            self.roundtrip(self.descriptor(), {})
        self.assertIn("normalization", str(caught.exception))

    def test_unnormalized_weights_are_refused_by_a_normalized_run(self):
        require_torch()
        with self.assertRaises(SystemExit) as caught:
            self.roundtrip({"mode": "none"}, self.descriptor())
        self.assertIn("normalization", str(caught.exception))

    def test_a_different_mode_is_refused(self):
        require_torch()
        with self.assertRaises(SystemExit) as caught:
            self.roundtrip(self.descriptor(mode="mean_std"),
                           self.descriptor(mode="range"))
        self.assertIn("normalization", str(caught.exception))

    def test_different_statistics_are_refused(self):
        require_torch()
        with self.assertRaises(SystemExit) as caught:
            self.roundtrip(self.descriptor(statistics="aaa"),
                           self.descriptor(statistics="bbb"))
        self.assertIn("statistics", str(caught.exception))

    def test_the_same_normalization_loads(self):
        require_torch()
        restored = self.roundtrip(self.descriptor(), self.descriptor())
        self.assertEqual(restored.normalization_identity["mode"], "mean_std")

    def test_two_runs_with_no_normalization_load(self):
        require_torch()
        self.roundtrip({}, {})
        self.roundtrip({"mode": "none"}, {})

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


class TestCheckpointsAreOptional(unittest.TestCase):
    """Off by default, at every stage, and honoured when switched on.

    The stages hand their models to each other in memory, so a checkpoint is
    for resuming a long run rather than for reaching the next stage. Both
    failure directions are worth a test: a default that flips back to on
    fills a disk quietly, and a switch that is ignored when on loses a run
    that was meant to be recoverable.
    """

    def test_every_stage_defaults_to_not_writing(self):
        require_torch()
        import inspect

        from sim_vla.training import (pipeline, pretrain_world_model,
                                      train_imitation)
        from sim_vla.training.online import OnlineConfig

        self.assertFalse(OnlineConfig().save_checkpoints)
        for function in (pretrain_world_model.run, train_imitation.run):
            with self.subTest(function=function.__qualname__):
                default = inspect.signature(
                    function).parameters["save_checkpoint"].default
                self.assertFalse(default)
        self.assertFalse(
            inspect.signature(pipeline.run).parameters[
                "save_checkpoints"].default)

    def test_the_command_line_defaults_to_not_writing(self):
        require_torch()
        from sim_vla.training import pipeline, pretrain_world_model

        self.assertFalse(pretrain_world_model.parse_args([]).save_checkpoints)
        self.assertFalse(pipeline.parse_args([]).save_checkpoints)
        self.assertTrue(
            pipeline.parse_args(["--save-checkpoints"]).save_checkpoints)

    def test_online_checkpoint_writes_nothing_while_off(self):
        require_torch()
        from types import SimpleNamespace

        from sim_vla.training.online import OnlineConfig, OnlineTrainer

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            # A stand-in for the trainer: what is under test is the guard, and
            # building a real one would drag in a world model and an actor
            # that have nothing to do with whether a file gets written.
            off = SimpleNamespace(config=OnlineConfig(save_checkpoints=False),
                                  checkpoint_dir=directory, meta=meta())
            self.assertIsNone(OnlineTrainer.checkpoint(off))
            self.assertEqual(list(directory.iterdir()), [],
                             "checkpointing is off and a file was written")

            # On, but with nowhere to write: still nothing, and still no
            # half-written file.
            nowhere = SimpleNamespace(
                config=OnlineConfig(save_checkpoints=True),
                checkpoint_dir=None, meta=meta())
            self.assertIsNone(OnlineTrainer.checkpoint(nowhere))


class TestNoWriteControlPaths(unittest.TestCase):
    """The real Stage 1A ``run`` loop, with small injected components.

    What is injected is the dataset, the model and the normalizer -- building
    a real world model over a real h5 file would test h5py. What is *executed*
    is the thing under test: the training loop, the metadata it assembles, and
    the branch that decides whether anything is written.
    """

    def setUp(self):
        self.torch = require_torch()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def inject(self):
        torch = self.torch
        import numpy as np

        from sim_vla.training import pretrain_world_model as stage

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 4)
                self.feature_dim = 4
                self.graph_enabled = False

            def loss(self, batch):
                total = self.lin(batch["action"]).pow(2).mean()
                return total, {"toy": total}, {}

        class Sampler:
            lookahead = 0

            def batch(self, size):
                rng = np.random.default_rng(0)
                return {
                    "action": rng.standard_normal((size, 3, 4)).astype(np.float32),
                    "action_target": rng.standard_normal(
                        (size, 3, 4)).astype(np.float32),
                }

        class Data:
            metadata = {"env_id": "PickCube-v1",
                        "controller": {"action_dim": 4}}
            closed = False

            def __len__(self):
                return 3

            def close(self):
                Data.closed = True

        class Normalizer:
            fields = {}
            identity = {"digest": "toy"}

            def save(self, path):
                Path(path).write_text("{}", encoding="utf-8")
                return Path(path)

        self.model = Model()
        self.data = Data()
        originals = (stage.build, stage.fit_normalizer)
        stage.build = lambda cfg, *, device, model_yaml=None: (
            self.data, Sampler(), self.model,
            types.SimpleNamespace(lr=1e-3))
        stage.fit_normalizer = lambda data: Normalizer()
        self.addCleanup(lambda: setattr(stage, "build", originals[0]))
        self.addCleanup(lambda: setattr(stage, "fit_normalizer", originals[1]))
        return stage

    def cfg(self):
        return {"model": {"graph": {"enabled": False}},
                "task": {"env_id": "PickCube-v1", "dataset": "toy.h5"},
                "data": {"batch_size": 2, "seed": 0,
                         "normalization": "mean_std"}}

    def test_stage_1a_writes_nothing_by_default(self):
        stage = self.inject()
        out = self.root / "world_model.pt"
        result = stage.run(self.cfg(), steps=2, device="cpu", out=out,
                           log_every=0)
        self.assertIsNone(result.path)
        # No .pt, no sidecar .json, no normalization.json. The Hugging Face
        # cache is a separate thing and is not touched here.
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), [],
                         "a default run wrote a training checkpoint")

    def test_stage_1a_writes_everything_when_asked(self):
        stage = self.inject()
        from sim_vla.runtime.checkpoint import load

        out = self.root / "world_model.pt"
        result = stage.run(self.cfg(), steps=2, device="cpu", out=out,
                           save_checkpoint=True, log_every=0)
        self.assertEqual(result.path, out)
        written = sorted(p.name for p in self.root.iterdir())
        self.assertEqual(written, ["normalization.json", "world_model.json",
                                   "world_model.pt"])
        # And it round-trips into a fresh model.
        fresh = type(self.model)()
        restored = load(out, result.meta, {"world_model": fresh})
        self.assertEqual(restored.step, 2)
        self.assertTrue(self.torch.allclose(fresh.lin.weight,
                                            self.model.lin.weight))

    def test_a_failure_during_training_still_closes_the_dataset(self):
        stage = self.inject()

        def explode(batch):
            raise RuntimeError("boom")

        self.model.loss = explode
        with self.assertRaises(RuntimeError):
            stage.run(self.cfg(), steps=2, device="cpu",
                      out=self.root / "w.pt", log_every=0)
        self.assertTrue(self.data.closed,
                        "the dataset was left open by a failed run")

    def test_an_unknown_normalization_mode_is_refused(self):
        stage = self.inject()
        cfg = self.cfg()
        cfg["data"]["normalization"] = "minmax"
        with self.assertRaises(SystemExit) as caught:
            stage.run(cfg, steps=1, device="cpu", out=self.root / "w.pt",
                      log_every=0)
        self.assertIn("normalization", str(caught.exception))


class TestStagesHandOverInMemory(unittest.TestCase):
    """Stage 1B trains against Stage 1A's own object, not a reload of it."""

    def test_the_world_model_object_is_passed_straight_through(self):
        require_torch()
        from types import SimpleNamespace

        from sim_vla.training import pipeline, pretrain_world_model
        from sim_vla.training import train_imitation

        sentinel = SimpleNamespace(feature_dim=64)
        stage_a = pretrain_world_model.Stage1A(
            model=sentinel, data=SimpleNamespace(close=lambda: None,
                                                 metadata={}),
            sampler=SimpleNamespace(), model_cfg=SimpleNamespace(),
            normalizer=SimpleNamespace(), meta=meta(), losses={})
        received = {}

        def fake_stage_a(*args, **kwargs):
            received["save_a"] = kwargs.get("save_checkpoint")
            return stage_a

        def fake_stage_b(cfg, world_model, sampler, **kwargs):
            received["world_model"] = world_model
            received["save_b"] = kwargs.get("save_checkpoint")
            return train_imitation.Stage1B(
                actor=SimpleNamespace(), adapter=None, loaded=None,
                trainer=None, losses={})

        original = (pretrain_world_model.run, train_imitation.run)
        pretrain_world_model.run, train_imitation.run = (fake_stage_a,
                                                         fake_stage_b)
        try:
            pipeline.run({}, world_steps=1, imitation_steps=1, online_steps=0,
                         device="cpu", root=Path("."))
        finally:
            pretrain_world_model.run, train_imitation.run = original

        self.assertIs(received["world_model"], sentinel,
                      "Stage 1B did not receive Stage 1A's own model")
        self.assertFalse(received["save_a"])
        self.assertFalse(received["save_b"])


if __name__ == "__main__":
    unittest.main()
