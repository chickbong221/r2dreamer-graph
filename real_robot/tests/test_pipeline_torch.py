"""A small end-to-end run through model-assisted IQL, and the refusals around it.

Needs torch and omegaconf; skips without them. On a tiny packed dataset it runs
every training stage through its command-line entry point -- world-model
pretraining with diagnostics, the diagnostic report, one latent cache,
imagined transitions, model-assisted IQL with and without the progress branch
-- and then checks what must hold across stages:

* every episode trains, the diagnostic episodes are rows of the one cache, and
  no stage writes or needs a validation or test cache;
* imagined transitions carry only model outputs;
* generating transitions and training policies leaves the world model's
  checkpoint and weights unchanged;
* a policy run refuses imagined transitions made from another cache, and
  rollouts refuse a checkpoint whose weights changed after encoding.
"""

from __future__ import annotations

import glob
import json
import os
import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:                                            # pragma: no cover
    torch = None
try:
    import omegaconf
except ImportError:                                            # pragma: no cover
    omegaconf = None

from ..common import file_sha256, read_json, read_jsonl
from .test_models_torch import TINY, build_dataset


def sets(*items):
    out = []
    for item in items:
        out += ["--set", item]
    return out


@unittest.skipIf(torch is None or omegaconf is None, "needs torch and omegaconf")
class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ..evaluation import evaluate_world_model
        from ..models.world_model import weights_digest
        from ..training import encode_dataset, generate_rollouts, pretrain_world_model, train_model_assisted_iql

        cls.tmp = tempfile.TemporaryDirectory()
        root = cls.tmp.name
        cls.root = root
        cls.dataset = os.path.join(root, "datasets", "tiny")
        build_dataset(cls.dataset, episodes=(0, 1, 2), diagnostic=(2,), length=16)
        cls.paths = sets(f"dataset.paths.runs={root}/runs", f"dataset.paths.latents={root}/latents",
                         f"dataset.paths.rollouts={root}/rollouts")

        pretrain_world_model.main(["--run-name", "wm", *cls.paths, *sets(
            f"world_model.dataset={cls.dataset}",
            "world_model.model.overrides=" + json.dumps(TINY),
            "world_model.sequence.burn_in=2", "world_model.sequence.length=6", "world_model.sequence.batch_size=2",
            "world_model.train.steps=4", "world_model.train.log_every=2", "world_model.train.checkpoint_every=2",
            "world_model.train.snapshot_every=2", "world_model.diagnostics.every=2",
            "world_model.diagnostics.windows=4", "world_model.diagnostics.open_loop.horizon=3",
            "world_model.diagnostics.open_loop.starts_per_episode=2", "world_model.diagnostics.burn_in_probes=[0,2]",
        )])
        cls.wm_dir = os.path.join(root, "runs", "world_model", "wm")
        cls.final = os.path.join(cls.wm_dir, "final.pt")
        cls.final_sha = file_sha256(cls.final)
        cls.final_weights = weights_digest(torch.load(cls.final, map_location="cpu", weights_only=False)
                                           ["state"]["model"])

        evaluate_world_model.main(["--world-model", "wm", "--checkpoint", "best_diagnostic", *cls.paths])
        encode_dataset.main(["--world-model", "wm", "--name", "lat", "--progress", *cls.paths])
        cls.rollout_sets = sets("rollouts.count=40", "rollouts.shard_size=16", "rollouts.behavior_policy.steps=3",
                                "rollouts.behavior_policy.batch_size=8", "rollouts.behavior_policy.log_every=1",
                                "rollouts.behavior_policy.network.hidden=[16,16]")
        generate_rollouts.main(["--latents", "lat", "--name", "h1", *cls.paths, *cls.rollout_sets])
        cls.policy_sets = sets(
            "iql.network.hidden=[16,16]", "model_assisted_iql.train.steps=4", "model_assisted_iql.train.batch_size=10",
            "model_assisted_iql.train.log_every=2", "model_assisted_iql.train.diagnostics_every=2",
            "model_assisted_iql.train.checkpoint_every=2", "model_assisted_iql.train.snapshot_every=2",
            "model_assisted_iql.diagnostics.synthetic_rows=16", "model_assisted_iql.progress.head.steps=3",
            "model_assisted_iql.progress.head.batch_size=8", "model_assisted_iql.progress.head.hidden=[16,16]")
        train_model_assisted_iql.main(["--latents", "lat", "--rollouts", "h1", "--run-name", "base",
                                       *cls.paths, *cls.policy_sets])
        train_model_assisted_iql.main(["--latents", "lat", "--rollouts", "h1", "--run-name", "progress",
                                       "--progress", *cls.paths, *cls.policy_sets])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ----------------------------------------------------------- world model
    def test_world_model_run_keeps_final_snapshots_and_a_labelled_diagnostic_selection(self):
        for name in ("final.pt", "latest.pt", "step_00000002.pt", "step_00000004.pt", "best_diagnostic.pt",
                     "best_diagnostic.json"):
            self.assertTrue(os.path.isfile(os.path.join(self.wm_dir, name)), name)
        self.assertFalse(os.path.exists(os.path.join(self.wm_dir, "best.pt")))
        config = read_json(os.path.join(self.wm_dir, "config.json"))
        self.assertTrue(config["diagnostic_episodes_in_training"])
        self.assertEqual(config["training_episodes"], [0, 1, 2])
        self.assertEqual(config["diagnostic_episodes"], [2])

    def test_world_model_diagnostics_are_logged_as_diagnostic(self):
        rows = read_jsonl(os.path.join(self.wm_dir, "metrics.jsonl"))
        keys = set().union(*(row.keys() for row in rows))
        self.assertFalse([k for k in keys if k.startswith(("val/", "val_on_train/"))])
        for key in ("diagnostic/loss/model", "diagnostic/reward/mae", "diagnostic/cont/accuracy",
                    "diagnostic/one_step/reward_abs", "diagnostic/one_step/state_mse",
                    "diagnostic/burn_in_plus0/deter_relative_l2"):
            self.assertIn(key, keys)
        reports = sorted(glob.glob(os.path.join(self.wm_dir, "diagnostics", "step_*.json")))
        self.assertEqual(len(reports), 2)
        self.assertIn("not generalisation", read_json(reports[-1])["note"])
        self.assertTrue(glob.glob(os.path.join(self.wm_dir, "diagnostics_best_diagnostic_step*.json")))

    # -------------------------------------------------------------- latents
    def test_one_cache_holds_every_episode_with_diagnostic_rows_marked(self):
        cache = os.path.join(self.root, "latents", "lat")
        self.assertEqual(sorted(os.listdir(cache)), ["identity.json", "transitions.npz"])
        with np.load(os.path.join(cache, "transitions.npz")) as data:
            self.assertEqual(sorted(set(data["episode"].tolist())), [0, 1, 2])
            self.assertTrue(np.array_equal(data["diagnostic"], data["episode"] == 2))
            self.assertIn("progress_phi", data.files)
        identity = read_json(os.path.join(cache, "identity.json"))
        self.assertEqual(identity["world_model"]["sha256"], self.final_sha)
        self.assertEqual(identity["world_model"]["weights"], self.final_weights)
        self.assertEqual(identity["latent"]["inference"], "mode")
        self.assertEqual(identity["selection"]["diagnostic"], [2])
        for field in ("annotation", "reward", "action", "selection", "dataset"):
            self.assertIn(field, identity)

    def test_no_stage_writes_a_validation_or_test_artifact(self):
        names = [os.path.basename(p) for p in glob.glob(os.path.join(self.root, "**", "*"), recursive=True)]
        self.assertFalse([n for n in names if n.startswith(("val", "test", "train.npz"))])

    # ------------------------------------------------------------- rollouts
    def test_imagined_transitions_carry_only_model_outputs(self):
        out = os.path.join(self.root, "rollouts", "h1")
        meta = read_json(os.path.join(out, "rollouts.json"))
        self.assertEqual(meta["transitions"], 40)
        self.assertEqual(meta["world_model"]["sha256"], self.final_sha)
        for shard in glob.glob(os.path.join(out, "shard_*.npz")):
            with np.load(shard) as data:
                self.assertEqual(sorted(data.files),
                                 ["action", "cont", "horizon", "reward", "start_row", "z", "z_next"])
                self.assertTrue(np.all(data["horizon"] == 1))

    # ------------------------------------------------------------- policies
    def test_policy_runs_produce_final_policies_and_separate_critic_errors(self):
        for run in ("base", "progress"):
            run_dir = os.path.join(self.root, "runs", "iql", run)
            for name in ("final.pt", "latest.pt", "step_00000002.pt"):
                self.assertTrue(os.path.isfile(os.path.join(run_dir, name)), (run, name))
            self.assertFalse(os.path.exists(os.path.join(run_dir, "best.pt")))
            keys = set().union(*(row.keys() for row in read_jsonl(os.path.join(run_dir, "metrics.jsonl"))))
            for key in ("critic/td_abs_recorded", "critic/td_abs_synthetic", "diagnostic/action_mse",
                        "diagnostic/td_abs_recorded", "diagnostic/td_abs_synthetic", "diagnostic/v_mean",
                        "diagnostic/weight_mean"):
                self.assertIn(key, keys, (run, key))
            config = read_json(os.path.join(run_dir, "config.json"))
            self.assertEqual(config["policy_output"], "final.pt")
        self.assertTrue(os.path.isfile(os.path.join(self.root, "runs", "iql", "progress", "progress_head.json")))

    def test_policy_stages_leave_the_world_model_unchanged(self):
        from ..models.world_model import weights_digest

        self.assertEqual(file_sha256(self.final), self.final_sha)
        state = torch.load(self.final, map_location="cpu", weights_only=False)["state"]["model"]
        self.assertEqual(weights_digest(state), self.final_weights)

    # ------------------------------------------------------------- refusals
    def test_imagined_transitions_from_another_cache_are_refused(self):
        from ..common import IdentityError
        from ..training import encode_dataset, train_model_assisted_iql

        # latest.pt holds the same weights as final.pt, but it is another file.
        encode_dataset.main(["--world-model", "wm", "--checkpoint", "latest", "--name", "lat_latest", *self.paths])
        with self.assertRaises(IdentityError):
            train_model_assisted_iql.main(["--latents", "lat_latest", "--rollouts", "h1", "--run-name", "mixed",
                                           *self.paths, *self.policy_sets])

    def test_changed_world_model_weights_are_refused(self):
        from ..common import IdentityError
        from ..training import encode_dataset, generate_rollouts

        encode_dataset.main(["--world-model", "wm", "--checkpoint", "step_00000002", "--name", "lat_step2",
                             *self.paths])
        path = os.path.join(self.wm_dir, "step_00000002.pt")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload["state"]["model"]
        name = next(key for key, value in state.items() if torch.is_floating_point(value))
        state[name] = state[name] + 0.01
        torch.save(payload, path)
        with self.assertRaises(IdentityError):
            generate_rollouts.main(["--latents", "lat_step2", "--name", "h1_step2", *self.paths,
                                    *self.rollout_sets])

    def test_a_cache_name_is_not_reused_for_other_weights(self):
        from ..training import encode_dataset

        with self.assertRaises(SystemExit):
            encode_dataset.main(["--world-model", "wm", "--checkpoint", "best_diagnostic", "--name", "lat",
                                 "--progress", *self.paths])


if __name__ == "__main__":
    unittest.main()
