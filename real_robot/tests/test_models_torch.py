"""Model tests. These need torch and omegaconf, so they skip without them.

They are small on purpose: a few updates on a tiny packed dataset, checking
what would be silently wrong rather than merely slow -- that the world-model
step touches no policy parameters, that masked positions contribute nothing,
that the latent convention is deterministic, that IQL follows the reference
order (the actor reads the *updated* value), that imagined transitions reach
the critics and nothing else, that the progress branch cannot move the base
initialisation, and that checkpoints refuse a foreign identity.
"""

from __future__ import annotations

import copy
import os
import tempfile
import unittest

import numpy as np

from ..common import stable_hash
from ..graphs.pack import empty_frame

try:
    import torch
except ImportError:                                            # pragma: no cover
    torch = None
try:
    import omegaconf
except ImportError:                                            # pragma: no cover
    omegaconf = None

TINY = {
    "deter": 64, "hidden": 32, "units": 32, "depth": 4, "discrete": 4, "compile": False,
    "amp_dtype": "bfloat16", "rep_loss": "dreamer",
    "rssm": {"stoch": 4, "blocks": 4, "obs_layers": 1, "img_layers": 1, "dyn_layers": 1, "sem_layers": 1},
    "graph": {"enabled": True, "simple_units": 32, "semantic_dim": 32, "decoder_units": 16, "embed": 8,
              "layers": 1, "bbox_query_dim": 4},
    "encoder": {"mlp": {"layers": 1}, "cnn": {"depth": 4}},
    "decoder": {"mlp": {"layers": 1}, "cnn": {"depth": 4, "bspace": 8}},
    "reward": {"layers": 1}, "cont": {"layers": 1}, "progress": {"enabled": False},
}
IQL_CFG = {
    "expectile": 0.7, "beta": 3.0, "max_weight": 100.0, "gamma": 0.99, "polyak": 0.005, "lr": 3e-4,
    "actor_cosine_decay": False,
    "network": {"hidden": [32, 32], "layernorm": True, "activation": "SiLU",
                "log_std_min": -5.0, "log_std_max": 2.0},
}


def build_dataset(root: str, episodes=(0, 1), diagnostic=None, length: int = 14, size: int = 32,
                  state_dim: int = 16, action_dim: int = 7):
    """A packed dataset small enough to train on in a test, in the current manifest format."""
    from ..common import load_config
    from ..data.manifest import DatasetManifest
    from ..graphs.schema import GraphSpec
    from ..graphs.vocabulary import build_vocab, vocab_identity, vocab_sizes

    spec = GraphSpec.from_config(load_config("graph"))
    vocab = build_vocab(spec)
    episodes = [int(e) for e in episodes]
    diagnostic = [episodes[-1]] if diagnostic is None else [int(e) for e in diagnostic]
    rng = np.random.default_rng(0)
    identity = {
        "test": "tiny", "annotation_mode": "full_episode", "gemini_model": "none", "gemini_backend": "none",
        "prompt_version": "none", "video_fps": 15, "bins": "none", "graph": stable_hash(spec.identity()),
        "temporal_window": spec.temporal_window, "vocab": vocab_identity(vocab), "reward": {"version": "test"},
        "action_mapping": {"declaration": "test"}, "action_transform": {"indices": list(range(action_dim))},
        "selection": {"version": "test", "training": stable_hash(episodes), "diagnostic": diagnostic},
    }
    manifest = DatasetManifest.create(
        root, identity=identity, selection={"version": "test", "training": episodes, "diagnostic": diagnostic},
        shapes={"state": [state_dim], "action": [action_dim]},
        model_inputs={"cameras": list(spec.cameras), "image_size": [size, size], "resize_mode": "stretch",
                      "state_features": ["joints", "gripper", "eef_xyz", "eef_rot_sincos"], "effort_scale": 1000.0,
                      "image_keys": [f"image_{c}" for c in spec.cameras]},
        graph={"vocab_sizes": vocab_sizes(vocab), "n_max": spec.n_max, "e_max": spec.e_max,
               "n_cams": len(spec.cameras), "centroid_origin": [0.0, 0.0, 0.0], "centroid_scale": 1.0,
               "scene_frame": "robot_base", "temporal_window": spec.temporal_window},
        action={"indices": list(range(action_dim)), "names": [f"j{i}" for i in range(action_dim)],
                "representation": "absolute_joint_position", "units": {}, "low": [-1.0] * action_dim,
                "high": [1.0] * action_dim, "margin": 0.05, "gripper": {"index": 6, "state_index": 6,
                                                                        "open_value": 1.6, "closed_value": 0.6}})
    graph = empty_frame(spec)
    for episode in episodes:
        directory = manifest.episode_dir(episode)
        os.makedirs(directory, exist_ok=True)
        terminal = length - 2
        arrays = {
            "state": rng.normal(size=(length, state_dim)).astype(np.float32),
            "state_raw": rng.normal(size=(length, 13)).astype(np.float32),
            "action": rng.uniform(-0.8, 0.8, size=(length, action_dim)).astype(np.float32),
            "reward": (np.arange(length) * 0.01 - 1.0).astype(np.float32),
            "done": np.zeros(length, dtype=bool),
            "transition_valid": np.zeros(length, dtype=bool),
            "task_terminal": np.zeros(length, dtype=bool),
            "obs_valid": np.ones(length, dtype=bool),
            "graph_valid": np.ones(length, dtype=bool),
            **{key: np.repeat(value[None], length, axis=0) for key, value in graph.items()},
        }
        # One valid entity per frame so the graph decoder has something to read.
        arrays["graph_node_ent"][:, 0] = vocab.entity.ee_id
        arrays["graph_node_ent"][:, 1] = vocab.entity.encode(spec.entity("banana").key)
        arrays["graph_node_target"][:, 1] = 1
        arrays["graph_node_bbox"][:, :2] = np.array([0.1, 0.4, 0.2, 0.5], dtype=np.float16)
        arrays["graph_edge_rel"][:, 0] = vocab.relation.encode("grasp")
        arrays["graph_edge_abs"][:, 0] = vocab.absolute.encode("not-holds")
        arrays["graph_edge_src"][:, 0] = 0
        arrays["graph_edge_dst"][:, 0] = 1
        arrays["obs_valid"][terminal + 1:] = False
        arrays["task_terminal"][terminal] = True
        arrays["transition_valid"][:terminal] = True
        arrays["done"][terminal - 1] = True
        np.savez(os.path.join(directory, "arrays.npz"), **arrays)
        for camera in spec.cameras:
            np.save(os.path.join(directory, f"image_{camera}.npy"),
                    rng.integers(0, 255, size=(length, size, size, 3), dtype=np.uint8))
        manifest.record_episode(episode, {"n_frames": length, "diagnostic": episode in diagnostic})
    manifest.save()
    return manifest


@unittest.skipIf(torch is None or omegaconf is None, "needs torch and omegaconf")
class WorldModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ..data.episode_dataset import BuiltEpisodeStore
        from ..models.world_model import OfflineWorldModel, compose_model_config

        cls.tmp = tempfile.TemporaryDirectory()
        manifest = build_dataset(cls.tmp.name)
        cls.store = BuiltEpisodeStore(cls.tmp.name)
        cls.cfg = {"model": {"base": "configs/model/_base_.yaml", "preset": "configs/model/size50M_graph_simple.yaml",
                             "overrides": TINY},
                   "observation_keys": {"images": manifest.image_keys, "state": "state"}}
        cls.config = compose_model_config(cls.cfg, manifest, "cpu")
        cls.model = OfflineWorldModel(cls.config, manifest, cls.cfg["observation_keys"])
        cls.manifest = manifest

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def batch(self, burn_in=2, length=6, batch_size=2):
        from ..data.sequence_dataset import SequenceSampler
        from ..models.world_model import to_torch

        sampler = SequenceSampler(self.store, "training", burn_in, length, batch_size, seed=0)
        return to_torch(sampler.sample(), "cpu")

    def test_vocabulary_sizes_come_from_the_dataset(self):
        sizes = self.manifest.vocab_sizes
        self.assertEqual(int(self.config.graph.entity_vocab), sizes["entity_vocab"])
        self.assertEqual(int(self.config.graph.n_abs), sizes["n_abs"])

    def test_losses_cover_every_term_and_backpropagate(self):
        total, metrics, posterior = self.model.compute_losses(self.batch(), burn_in=2)
        for key in ("dyn", "rep", "graphdyn", "graphrep", "node", "relabs", "reltemp", "nodetgt", "rew", "con"):
            self.assertIn(f"loss/{key}", metrics, key)
        for key in self.manifest.image_keys + ["state"]:
            self.assertIn(f"loss/{key}", metrics)
        self.model.zero_grad(set_to_none=True)
        total.backward()
        touched = [name for name, p in self.model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
        self.assertTrue(any(name.startswith("graph_encoder") for name in touched))
        self.assertTrue(any(name.startswith("encoder") for name in touched))
        self.assertTrue(any(name.startswith("reward") for name in touched))
        self.model.zero_grad(set_to_none=True)

    def test_invalid_positions_do_not_change_the_loss(self):
        batch = self.batch()
        # The posterior is sampled, so both losses draw the same random numbers: any
        # difference can then only come from what the invalid positions hold.
        torch.manual_seed(0)
        clean = float(self.model.compute_losses(batch, burn_in=2)[0])
        noisy = {key: value.clone() for key, value in batch.items()}
        invalid = ~batch["obs_valid"]
        if not bool(invalid.any()):
            self.skipTest("this sample has no padding")
        for key in self.manifest.image_keys:
            noisy[key][invalid] = 255 - noisy[key][invalid]
        noisy["state"][invalid] = 7.0
        noisy["reward_in"][invalid] = 99.0
        torch.manual_seed(0)
        self.assertAlmostEqual(clean, float(self.model.compute_losses(noisy, burn_in=2)[0]), places=4)

    def test_no_policy_parameters_exist(self):
        names = [name for name, _ in self.model.named_parameters()]
        self.assertFalse([n for n in names if "actor" in n or "value" in n or "critic" in n])

    def test_feature_split_inverts_get_feat(self):
        batch = self.batch()
        out = self.model.encode_sequence(batch)
        stoch, sem, deter = self.model.split_feat(out["feat"])
        again = self.model.rssm.get_feat(stoch, deter, sem)
        self.assertTrue(torch.allclose(again, out["feat"], atol=1e-5))

    def test_mode_inference_is_deterministic_and_sampling_is_not(self):
        batch = self.batch()
        first = self.model.encode_sequence(batch)["feat"]
        second = self.model.encode_sequence(batch)["feat"]
        self.assertTrue(torch.allclose(first, second))
        torch.manual_seed(0)
        sampled = self.model.encode_sequence(batch, sample=True)["feat"]
        self.assertFalse(torch.allclose(first, sampled))

    def test_imagination_advances_without_observations(self):
        batch = self.batch()
        feat = self.model.encode_sequence(batch)["feat"][:, 0]
        out = self.model.imagine(feat, batch["prev_action"][:, 0].float())
        self.assertEqual(out["feat"].shape, feat.shape)
        self.assertEqual(out["reward"].shape, (feat.shape[0],))
        self.assertTrue(bool(((out["cont"] >= 0) & (out["cont"] <= 1)).all()))

    def test_diagnostics_report_every_section_and_say_what_they_measure(self):
        from ..evaluation.world_model_diagnostics import DiagnosticWindows, run_diagnostics

        cfg = {"open_loop": {"horizon": 3, "starts_per_episode": 2}, "burn_in_probes": [0, 2]}
        windows = DiagnosticWindows(self.store, self.manifest.diagnostic_episodes(), 2, 6, 2, 4, seed=1)
        report = run_diagnostics(self.model, self.store, windows, cfg, "cpu")
        self.assertTrue(report["diagnostic_episodes_in_training"])
        self.assertIn("not generalisation", report["note"])
        scalars = report["scalars"]
        self.assertIn("diagnostic/loss/model", scalars)
        self.assertIn("diagnostic/one_step/reward_abs", scalars)
        self.assertIn("diagnostic/one_step/state_mse", scalars)
        self.assertIn("diagnostic/open_loop_h3/deter_relative_l2", scalars)
        self.assertIn("diagnostic/burn_in_plus2/deter_relative_l2", scalars)
        self.assertFalse([key for key in scalars if not key.startswith("diagnostic/")])
        self.assertEqual(report["open_loop"]["samples_by_step"][0], 2)


def weights_of(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def same_weights(a, b) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@unittest.skipIf(torch is None, "needs torch")
class Iql(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.z_dim, self.a_dim = 24, 4

    def agent(self, lr=3e-4, beta=0.0, seed=0):
        from ..models.iql import IQL

        return IQL(self.z_dim, self.a_dim, {**IQL_CFG, "lr": lr}, total_steps=10, progress_beta=beta, init_seed=seed)

    def batch(self, n=32, progress=False, synthetic=False):
        from ..models.iql import potential_difference

        out = {"z": torch.randn(n, self.z_dim), "z_next": torch.randn(n, self.z_dim),
               "action": torch.rand(n, self.a_dim) * 1.6 - 0.8, "reward": torch.rand(n) - 1.0,
               "cont": torch.ones(n), "synthetic": torch.full((n,), float(synthetic))}
        if progress:
            out["phi"], out["phi_next"], out["phi_valid"] = torch.rand(n), torch.rand(n), torch.ones(n)
            out["progress_reward"] = potential_difference(out["phi"], out["phi_next"], out["cont"], 0.99)
            out["progress_valid"] = out["phi_valid"]
        return out

    @staticmethod
    def mixed(real, imagined):
        return {key: torch.cat([real[key], imagined[key]]) for key in set(real) & set(imagined)}

    def test_expectile_loss_is_asymmetric(self):
        from ..models.iql import expectile_loss

        above = expectile_loss(torch.ones(4), 0.7)
        below = expectile_loss(-torch.ones(4), 0.7)
        self.assertGreater(float(above), float(below))
        self.assertAlmostEqual(float(expectile_loss(torch.ones(4), 0.5)),
                               float(expectile_loss(-torch.ones(4), 0.5)))

    def test_one_update_moves_every_branch(self):
        agent = self.agent()
        before = weights_of(agent)
        metrics = agent.update(self.batch())
        for prefix in ("critic", "value", "actor"):
            moved = [name for name, p in agent.state_dict().items()
                     if name.startswith(prefix) and not torch.equal(p, before[name])]
            self.assertTrue(moved, prefix)
        for key in ("loss/critic", "loss/value", "loss/actor", "weight/clipped_fraction", "v/std",
                    "weight/effective_sample_fraction", "critic/td_abs_recorded"):
            self.assertIn(key, metrics)

    def test_the_actor_reads_the_updated_value(self):
        agent = self.agent(lr=1e-2)
        batch = self.batch(64)
        target_before = copy.deepcopy(agent.target_critic)
        value_before = copy.deepcopy(agent.value)
        metrics = agent.update(batch)
        with torch.no_grad():
            q_target = torch.min(*target_before(batch["z"], batch["action"]))
            updated = float((q_target - agent.value(batch["z"])).mean())
            stale = float((q_target - value_before(batch["z"])).mean())
        self.assertAlmostEqual(metrics["adv/mean"], updated, places=5)
        self.assertGreater(abs(updated - stale), 1e-4)

    def test_the_actor_update_matches_a_step_with_updated_values(self):
        agent = self.agent(lr=1e-2)
        reference = self.agent(lr=1e-2)            # same seed: identical weights, fresh optimisers
        self.assertTrue(same_weights(weights_of(agent), weights_of(reference)))
        batch = self.batch(64)
        agent.update(batch)
        # By hand, in the reference order: value step, recompute, actor step.
        z, a = batch["z"], batch["action"]
        from ..models.iql import expectile_loss

        with torch.no_grad():
            q_target = torch.min(*reference.target_critic(z, a))
        reference._step(reference.value_opt, expectile_loss(q_target - reference.value(z), reference.expectile))
        with torch.no_grad():
            weight = torch.exp(reference.beta * (q_target - reference.value(z))).clamp(max=reference.max_weight)
        reference._step(reference.actor_opt, -(weight * reference.actor.log_prob(z, a)).mean())
        self.assertTrue(same_weights(weights_of(agent.actor), weights_of(reference.actor)))

    def test_target_critic_trails_the_critic(self):
        agent = self.agent()
        agent.update(self.batch())
        distance = max(float((t - s).abs().max()) for t, s in
                       zip(agent.target_critic.parameters(), agent.critic.parameters()))
        self.assertGreater(distance, 0.0)

    def test_actions_stay_inside_the_support(self):
        agent = self.agent()
        action = agent.actor.act(torch.randn(16, self.z_dim), deterministic=False, noise_std=5.0)
        self.assertTrue(bool(((action >= -1) & (action <= 1)).all()))

    def test_imagined_transitions_change_the_critics_and_nothing_else(self):
        for beta in (0.0, 0.1):
            progress = beta > 0
            real = self.batch(16, progress=progress)
            first = self.agent(beta=beta, seed=3)
            second = self.agent(beta=beta, seed=3)
            one = self.mixed(real, self.batch(16, progress=progress, synthetic=True))
            other = self.mixed(real, self.batch(16, progress=progress, synthetic=True))
            metrics = first.update(real, one)
            second.update(real, other)
            self.assertAlmostEqual(metrics["critic/synthetic_fraction"], 0.5, places=6)
            self.assertIn("critic/td_abs_synthetic", metrics)
            learners = ["value", "actor"] + (["progress_value"] if progress else [])
            for name in learners:
                self.assertTrue(same_weights(weights_of(getattr(first, name)), weights_of(getattr(second, name))),
                                f"{name} read imagined transitions (beta={beta})")
            critics = ["critic"] + (["progress_critic"] if progress else [])
            for name in critics:
                self.assertFalse(same_weights(weights_of(getattr(first, name)), weights_of(getattr(second, name))),
                                 f"{name} ignored imagined transitions (beta={beta})")

    def test_the_progress_branch_cannot_move_the_base_initialisation(self):
        base = self.agent(beta=0.0, seed=7)
        torch.randn(1000)                              # consume the global stream in between
        progress = self.agent(beta=0.1, seed=7)
        for name in ("actor", "critic", "value", "target_critic"):
            self.assertTrue(same_weights(weights_of(getattr(base, name)), weights_of(getattr(progress, name))), name)

    def test_the_progress_advantage_reads_the_updated_progress_value(self):
        agent = self.agent(lr=1e-2, beta=0.5)
        batch = self.batch(64, progress=True)
        target = copy.deepcopy(agent.target_critic)
        progress_target = copy.deepcopy(agent.target_progress_critic)
        metrics = agent.update(batch)
        z, a = batch["z"], batch["action"]
        with torch.no_grad():
            env = torch.min(*target(z, a)) - agent.value(z)
            prog = (torch.min(*progress_target(z, a)) - agent.progress_value(z)) * batch["progress_valid"]
        self.assertAlmostEqual(metrics["adv/mean"], float((env + 0.5 * prog).mean()), places=5)
        self.assertIn("progress/influence_raw", metrics)
        self.assertIn("progress/critic_loss", metrics)

    def test_diagnostics_separate_recorded_and_imagined_errors(self):
        agent = self.agent()
        values = agent.diagnostics([self.batch(20), self.batch(12)], [self.batch(8, synthetic=True)])
        for key in ("action_mse", "td_abs_recorded", "td_abs_synthetic", "v_mean", "v_std", "advantage_mean",
                    "weight_mean", "weight_clipped_fraction", "weight_effective_sample_fraction", "action_abs/0"):
            self.assertIn(key, values)
        self.assertEqual(values["recorded_rows"], 32.0)
        self.assertEqual(values["synthetic_rows"], 8.0)

    def test_no_behaviour_cloning_update_exists(self):
        self.assertFalse(hasattr(self.agent(), "bc_update"))

    def test_potential_difference_matches_the_repository_definition(self):
        from progress import potential_shaping
        from ..models.iql import potential_difference

        potential = torch.tensor([[[0.2], [0.5], [0.9]]])
        cont = torch.tensor([[[1.0], [1.0], [1.0]]])
        repository = potential_shaping(potential, cont, 0.99)
        mine = potential_difference(potential[0, :-1, 0], potential[0, 1:, 0], cont[0, 1:, 0], 0.99)
        self.assertTrue(torch.allclose(repository[0, 1:, 0], mine))


@unittest.skipIf(torch is None, "needs torch")
class Checkpoints(unittest.TestCase):
    def test_a_different_identity_is_refused(self):
        from ..common import IdentityError
        from ..training.checkpoints import CheckpointManager, load_checkpoint

        with tempfile.TemporaryDirectory() as run:
            manager = CheckpointManager(run, {"dataset": "abc"})
            manager.save("latest", 1, {"model": {}}, {"note": "x"})
            load_checkpoint(manager.path("latest"), {"dataset": "abc"})
            with self.assertRaises(IdentityError):
                load_checkpoint(manager.path("latest"), {"dataset": "different"})

    def test_an_earlier_checkpoint_format_is_refused(self):
        from ..common import IdentityError
        from ..training.checkpoints import load_checkpoint

        with tempfile.TemporaryDirectory() as run:
            path = os.path.join(run, "best.pt")
            torch.save({"identity": {}, "state": {}}, path)
            with self.assertRaises(IdentityError):
                load_checkpoint(path)

    def test_a_selected_checkpoint_is_labelled_and_moves_in_the_declared_direction(self):
        from ..training.checkpoints import CheckpointManager, CheckpointSelection, load_checkpoint

        with tempfile.TemporaryDirectory() as run:
            manager = CheckpointManager(run, {})
            selection = CheckpointSelection(manager, "best_diagnostic", "diagnostic/loss", "min", note="trained on")
            self.assertTrue(selection.update(1, {"diagnostic/loss": 1.0}, lambda: {"model": {}}, {}))
            self.assertFalse(selection.update(2, {"diagnostic/loss": 2.0}, lambda: {"model": {}}, {}))
            self.assertTrue(selection.update(3, {"diagnostic/loss": 0.5}, lambda: {"model": {}}, {}))
            with self.assertRaises(KeyError):
                selection.update(4, {"other": 0.1}, lambda: {"model": {}}, {})
            payload = load_checkpoint(manager.path("best_diagnostic"))
            self.assertEqual(payload["selection"]["metric"], "diagnostic/loss")
            self.assertEqual(payload["step"], 3)
            for label in ("best", "final", "latest", "step_00000010"):
                with self.assertRaises(ValueError):
                    CheckpointSelection(manager, label, "diagnostic/loss", "min", note="")

    def test_snapshots_are_kept_by_step(self):
        from ..training.checkpoints import CheckpointManager, existing_checkpoints

        with tempfile.TemporaryDirectory() as run:
            manager = CheckpointManager(run, {})
            manager.snapshot(10, {"model": {}}, {})
            manager.snapshot(20, {"model": {}}, {})
            self.assertEqual(list(existing_checkpoints(run)), ["step_00000010", "step_00000020"])


if __name__ == "__main__":
    unittest.main()
