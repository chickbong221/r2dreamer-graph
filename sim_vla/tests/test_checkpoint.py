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

    def test_a_world_model_from_before_progress_pretraining_is_refused(self):
        """The refusal says why, instead of suggesting strict=False -- which
        would start shaping from a random head."""
        require_torch()
        from sim_vla.runtime.checkpoint import load, save

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "world_model.pt"
            save(path, meta(), {"world_model": self.module()})
            with self.assertRaises(SystemExit) as caught:
                load(path, meta(), {"world_model": self.module(),
                                    "progress": self.module()},
                     explain={"progress": "Re-run Stage 1A."})
            message = str(caught.exception)
            self.assertIn("Re-run Stage 1A.", message)
            self.assertNotIn("strict=False", message)

    def test_the_progress_head_travels_with_its_world_model(self):
        torch = require_torch()
        from sim_vla.runtime.checkpoint import load, save

        world, head = self.module(), self.module()
        head.weight.data.fill_(0.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "world_model.pt"
            save(path, meta(), {"world_model": world, "progress": head})
            restored = self.module()
            load(path, meta(), {"world_model": self.module(),
                                "progress": restored})
            self.assertTrue(torch.equal(restored.weight, head.weight))
            # The other arms ask for no head, and a file with one still loads.
            load(path, meta(), {"world_model": self.module(), "progress": None})

    def test_weights_that_do_not_fit_are_refused_by_name(self):
        """Not torch's key-list RuntimeError, and not a silent random head."""
        require_torch()
        from sim_vla.runtime.checkpoint import load, save

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "world_model.pt"
            save(path, meta(), {"world_model": self.module(),
                                "progress": self.module(4)})
            with self.assertRaises(SystemExit) as caught:
                load(path, meta(), {"world_model": self.module(),
                                    "progress": self.module(5)},
                     explain={"progress": "Re-run Stage 1A."})
            message = str(caught.exception)
            self.assertIn("'progress'", message)
            self.assertIn("Re-run Stage 1A.", message)

    def test_a_required_extra_entry_is_checked_before_any_weight(self):
        torch = require_torch()
        from sim_vla.runtime.checkpoint import load, save

        identity = {"architecture": "networks.ProgressHead"}
        with tempfile.TemporaryDirectory() as tmp:
            old, new = Path(tmp) / "old.pt", Path(tmp) / "new.pt"
            written = self.module()
            written.weight.data.fill_(0.5)
            save(old, meta(), {"progress": written})
            save(new, meta(extra={"progress_head": identity}),
                 {"progress": written})
            target = self.module()
            target.weight.data.fill_(-1.0)
            with self.assertRaises(SystemExit) as caught:
                load(old, meta(), {"progress": target},
                     explain={"progress_head": "Re-run Stage 1A."},
                     require_extra={"progress_head": identity})
            self.assertIn("progress_head", str(caught.exception))
            self.assertIn("Re-run Stage 1A.", str(caught.exception))
            self.assertTrue(bool((target.weight == -1.0).all()),
                            "a refused checkpoint still wrote weights")
            load(new, meta(), {"progress": target},
                 require_extra={"progress_head": identity})
            self.assertTrue(torch.equal(target.weight, written.weight))

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

        # A real Normalizer, not a stand-in. A hand-written stub of it drifted
        # the moment `descriptor()` was added and this fixture started raising
        # AttributeError inside the code under test -- which is a bug in the
        # fixture reported as a bug in the stage. What this test injects is the
        # dataset and the model; the normalizer is cheap and real.
        from sim_vla.data.normalization import FieldStats, Normalizer

        stats = FieldStats(mean=[0.0] * 4, std=[1.0] * 4, low=[-1.0] * 4,
                           high=[1.0] * 4, minimum=[-1.0] * 4,
                           maximum=[1.0] * 4, count=12)
        normalizer = Normalizer(fields={"actions": stats},
                                identity={"dataset": "toy", "episodes": 3})

        self.model = Model()
        self.data = Data()
        self.normalizer = normalizer
        originals = (stage.build, stage.fit_normalizer)
        stage.build = lambda cfg, *, device, model_yaml=None: (
            self.data, Sampler(), self.model,
            types.SimpleNamespace(lr=1e-3))
        stage.fit_normalizer = lambda data: normalizer
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
        # The metadata has to identify the statistics, not just the dataset:
        # weights fitted under one normalization cannot be read under another.
        recorded = result.meta.normalization_identity
        self.assertEqual(recorded["mode"], "mean_std")
        self.assertTrue(recorded["statistics"])
        self.assertEqual(recorded["statistics"],
                         self.normalizer.statistics_fingerprint())
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


class TestJointProgressCheckpoint(unittest.TestCase):
    """The real Stage 1A ``run`` and ``resume`` for graph_progress, with a
    small real world model and head; only the dataset and the schedule
    potential are injected. The head is trained with the world model, saved in
    ``world_model.pt``, restored from it, and an older head is refused."""

    def setUp(self):
        self.torch = require_torch()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.init_seed = 0
        self.calls = []

    def inject(self):
        from unittest import mock

        from sim_vla.models.world_model import build_world_model
        from sim_vla.training import pretrain_world_model as stage
        from sim_vla.training import progress as progress_module

        from .common import fake_batch, obs_shapes, small_model_config
        from .test_progress import FakePotential

        torch = self.torch
        cfg, model_cfg = small_model_config(True)
        cfg["model"]["progress"]["enabled"] = True
        # Deliberately unequal: the head's loss weight is progress_model, and
        # beta is Stage 2's shaping coefficient.
        cfg["model"]["progress"]["beta"] = 0.37
        model_cfg.loss_scales.progress_model = 2.5
        cfg["data"]["normalization"] = "none"
        cfg["data"]["batch_size"] = 2
        template = fake_batch(graph_enabled=True)

        class Sampler:
            lookahead = 0

            def batch(self, size):
                return fake_batch(graph_enabled=True, batch=size)

        class Data:
            metadata = {"env_id": "PickCube-v1",
                        "controller": {"action_dim": 8}}

            def __len__(self):
                return 3

            def close(self):
                pass

        def build(_cfg, *, device, model_yaml=None):
            torch.manual_seed(self.init_seed)
            model = build_world_model(model_cfg, obs_shapes(template), 8,
                                      graph_enabled=True)
            return Data(), Sampler(), model, model_cfg

        real_step = stage.train_step

        def spy(model, optimizer, batch, *, progress=None):
            self.calls.append((optimizer, progress))
            return real_step(model, optimizer, batch, progress=progress)

        for target, name, value in (
                (stage, "build", build), (stage, "train_step", spy),
                (progress_module, "build_potential",
                 lambda _cfg, _meta, *, device=None: FakePotential())):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        return stage, cfg, model_cfg

    def test_the_run_trains_jointly_and_saves_the_head_with_the_model(self):
        torch = self.torch
        from sim_vla.training.progress import HEAD_IDENTITY

        stage, cfg, model_cfg = self.inject()
        out = self.root / "world_model.pt"
        result = stage.run(cfg, steps=2, device="cpu", out=out,
                           save_checkpoint=True, log_every=0)

        optimizer, progress = self.calls[0]
        head, _potential, weight = progress
        self.assertIs(head, result.progress_head)
        self.assertEqual(weight, 2.5, "the head's weight must be "
                         "loss_scales.progress_model, not progress.beta")
        # One optimizer, the world model's type and learning rate, over both.
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["lr"],
                         stage.world_lr(cfg, model_cfg))
        stepped = {id(p) for p in optimizer.param_groups[0]["params"]}
        self.assertLessEqual({id(p) for p in head.parameters()}, stepped)
        self.assertLessEqual({id(p) for p in result.model.parameters()},
                             stepped)

        payload = torch.load(out, map_location="cpu", weights_only=False)
        self.assertIn("progress", payload["modules"])
        self.assertEqual(sorted(payload["optimizers"]), ["world_model"])
        self.assertEqual(payload["meta"]["extra"]["progress_head"],
                         HEAD_IDENTITY)
        self.assertEqual(payload["meta"]["extra"]["progress_model_scale"], 2.5)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()),
                         ["world_model.json", "world_model.pt"])

    def test_resume_restores_the_trained_head_and_world_model(self):
        torch = self.torch
        from sim_vla.training.pretrain_world_model import build_progress_head
        from sim_vla.training.progress import HEAD_IDENTITY

        stage, cfg, model_cfg = self.inject()
        out = self.root / "world_model.pt"
        trained = stage.run(cfg, steps=2, device="cpu", out=out,
                            save_checkpoint=True, log_every=0)
        fresh_head = build_progress_head(cfg, model_cfg, trained.model,
                                         device="cpu")
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(
            fresh_head.parameters(), trained.progress_head.parameters())),
            "training never moved the head, so a restore proves nothing")

        self.init_seed = 99                 # a differently initialised model
        resumed = stage.resume(cfg, device="cpu", path=out)
        for (name, a), (_, b) in zip(trained.model.named_parameters(),
                                     resumed.model.named_parameters()):
            self.assertTrue(torch.equal(a, b), f"world model {name}")
        for (name, a), (_, b) in zip(
                trained.progress_head.named_parameters(),
                resumed.progress_head.named_parameters()):
            self.assertTrue(torch.equal(a, b), f"progress head {name}")
        self.assertIsNotNone(resumed.potential)
        self.assertEqual(resumed.meta.extra["progress_head"], HEAD_IDENTITY)

    def test_a_head_from_before_joint_pretraining_is_refused(self):
        """The old file: a detached twohot MLPHead under ``net.``, and no head
        identity in its metadata."""
        import dataclasses

        import networks
        from sim_vla.runtime.checkpoint import save

        stage, cfg, model_cfg = self.inject()
        trained = stage.run(cfg, steps=1, device="cpu", log_every=0)

        class OldHead(self.torch.nn.Module):
            def __init__(self, feature_dim):
                super().__init__()
                self.net = networks.MLPHead(model_cfg.critic, feature_dim)

        out = self.root / "world_model.pt"
        save(out, dataclasses.replace(trained.meta, extra={}),
             {"world_model": trained.model,
              "progress": OldHead(trained.model.feature_dim)})
        with self.assertRaises(SystemExit) as caught:
            stage.resume(cfg, device="cpu", path=out)
        message = str(caught.exception)
        self.assertIn("progress_head", message)
        self.assertIn("Re-run Stage 1A", message)

        # Even recorded as the new head, weights that do not fit are refused
        # by name rather than loaded half-way or left at random.
        save(out, trained.meta, {"world_model": trained.model,
                                 "progress": OldHead(
                                     trained.model.feature_dim)})
        with self.assertRaises(SystemExit) as caught:
            stage.resume(cfg, device="cpu", path=out)
        self.assertIn("'progress' weights do not fit", str(caught.exception))

    def plain_graph(self, cfg):
        import copy

        plain = copy.deepcopy(cfg)
        plain["model"]["progress"]["enabled"] = False
        return plain

    def test_a_jointly_trained_world_model_is_refused_by_plain_graph(self):
        """The progress loss reached this representation, so restoring it
        under ``graph`` with the head dropped is not a graph-arm run."""
        stage, cfg, _model_cfg = self.inject()
        out = self.root / "world_model.pt"
        stage.run(cfg, steps=1, device="cpu", out=out, save_checkpoint=True,
                  log_every=0)
        with self.assertRaises(SystemExit) as caught:
            stage.resume(self.plain_graph(cfg), device="cpu", path=out)
        message = str(caught.exception)
        self.assertIn("progress_head", message)
        self.assertIn("graph_progress", message)
        # The same file still resumes as the arm that trained it.
        stage.resume(cfg, device="cpu", path=out)

    def test_plain_graph_and_pre_joint_checkpoints_still_load_as_graph(self):
        """Neither world model was trained by a progress loss: a plain graph
        run's never had a head, and the old detached head never reached its
        world model."""
        import dataclasses

        import networks
        from sim_vla.runtime.checkpoint import save

        torch = self.torch
        stage, cfg, model_cfg = self.inject()
        plain = self.plain_graph(cfg)

        out = self.root / "world_model.pt"
        trained = stage.run(plain, steps=1, device="cpu", out=out,
                            save_checkpoint=True, log_every=0)
        self.assertIsNone(trained.progress_head)
        self.assertEqual(trained.meta.extra, {})
        self.init_seed = 99
        resumed = stage.resume(plain, device="cpu", path=out)
        self.assertIsNone(resumed.progress_head)
        for (name, a), (_, b) in zip(trained.model.named_parameters(),
                                     resumed.model.named_parameters()):
            self.assertTrue(torch.equal(a, b), f"world model {name}")
        # And the other way still refuses: no head to start shaping from.
        with self.assertRaises(SystemExit) as caught:
            stage.resume(cfg, device="cpu", path=out)
        self.assertIn("progress_head", str(caught.exception))

        class OldHead(torch.nn.Module):
            def __init__(self, feature_dim):
                super().__init__()
                self.net = networks.MLPHead(model_cfg.critic, feature_dim)

        old = self.root / "old_world_model.pt"
        save(old, dataclasses.replace(trained.meta, extra={}),
             {"world_model": trained.model,
              "progress": OldHead(trained.model.feature_dim)})
        resumed = stage.resume(plain, device="cpu", path=old)
        self.assertIsNone(resumed.progress_head)

    def test_stage_2_receives_the_exact_head_stage_1a_trained(self):
        from types import SimpleNamespace
        from unittest import mock

        from sim_vla.training import pipeline, pretrain_world_model
        from sim_vla.training import train_imitation

        stage, cfg, model_cfg = self.inject()
        stage_a = stage.run(cfg, steps=1, device="cpu", log_every=0)
        captured = {}

        class Env:
            num_envs, sim_backend, live_reconfiguration_freq = 1, "cpu", None

            def __init__(self, *args, **kwargs):
                pass

            def build(self):
                return self

            def close(self):
                pass

        def fake_online(_cfg, world_model, _actor, _critic, _sampler, _env,
                        **kwargs):
            captured.update(kwargs, world_model=world_model)
            return SimpleNamespace(env_steps=0, updates=0)

        def fake_stage_b(*args, **kwargs):
            return train_imitation.Stage1B(
                actor=SimpleNamespace(flow_steps=2), adapter=None,
                loaded=None, trainer=None, losses={})

        with mock.patch.object(pretrain_world_model, "run",
                               lambda *a, **k: stage_a), \
                mock.patch.object(train_imitation, "run", fake_stage_b), \
                mock.patch.object(pipeline, "run_online", fake_online), \
                mock.patch.object(pipeline.progress_module, "preflight",
                                  lambda *a, **k: None), \
                mock.patch("sim_vla.envs.maniskill.SimVlaEnv", Env):
            pipeline.run(cfg, world_steps=1, imitation_steps=1,
                         online_steps=10, device="cpu", root=self.root)
        self.assertIs(captured["progress_head"], stage_a.progress_head)
        self.assertIs(captured["potential"], stage_a.potential)
        self.assertIs(captured["world_model"], stage_a.model)


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
