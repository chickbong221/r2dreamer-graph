import pathlib
import unittest

import gymnasium as gym
import numpy as np
import torch

import progress
from hydra import compose, initialize_config_dir
from tensordict import TensorDict

from dreamer import Dreamer
from tests.test_graph import graph_batch


def make_config(enabled):
    config_dir = str(pathlib.Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(
            config_name="configs",
            overrides=[
                "env=dmc_vision",
                "model=size50M_graph_simple",
                "device=cpu",
                f"model.graph.enabled={str(enabled).lower()}",
                "model.deter=16",
                "model.hidden=8",
                "model.discrete=4",
                "model.units=8",
                "model.depth=2",
                "model.rssm.blocks=4",
                "model.rssm.stoch=4",
                "model.graph.units=8",
                "model.graph.layers=1",
                "model.encoder.cnn.depth=2",
                "model.decoder.cnn.depth=2",
                "model.encoder.cnn.minres=1",
                "model.decoder.cnn.minres=1",
            ],
        ).model
    config.encoder.mlp_keys = "^(state|instruction)$"
    config.decoder.mlp_keys = "^state$"
    return config


SLOT_EDGES = 16


def make_pooled_config(
    progress=True, progress_scale=1.0, beta=0.05
):
    """The pooled graph-simple preset, shrunk but structurally identical."""
    config_dir = str(pathlib.Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(
            config_name="configs",
            overrides=[
                "env=dmc_vision",
                "model=size50M_graph_simple",
                "device=cpu",
                "model.deter=16",
                "model.hidden=8",
                "model.discrete=4",
                "model.units=8",
                "model.depth=2",
                "model.rssm.stoch=4",
                "model.rssm.blocks=4",
                "model.graph.units=8",
                "model.graph.embed=8",
                "model.graph.layers=1",
                "model.graph.simple_units=8",
                "model.graph.semantic_dim=8",
                "model.graph.decoder_units=8",
                "model.encoder.cnn.depth=2",
                "model.decoder.cnn.depth=2",
                "model.encoder.cnn.minres=1",
                "model.decoder.cnn.minres=1",
                f"model.progress.enabled={str(progress).lower()}",
                f"model.progress.beta={beta}",
                f"model.loss_scales.progress_model={progress_scale}",
            ],
        ).model
    config.encoder.mlp_keys = "^(state|instruction)$"
    config.decoder.mlp_keys = "^state$"
    return config


def pooled_spaces(with_uid=False):
    obs = {
        "image": gym.spaces.Box(0, 255, (16, 16, 3), np.uint8),
        "state": gym.spaces.Box(-np.inf, np.inf, (5,), np.float32),
        "instruction": gym.spaces.Box(-np.inf, np.inf, (7,), np.float32),
        "is_first": gym.spaces.Box(0, 1, (), np.bool_),
        "is_last": gym.spaces.Box(0, 1, (), np.bool_),
        "is_terminal": gym.spaces.Box(0, 1, (), np.bool_),
        "reward": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
        "graph_node_ent": gym.spaces.Box(0, 255, (POOLED_NODES,), np.uint8),
        "graph_node_bbox": gym.spaces.Box(0, 1, (POOLED_NODES, 2, 4), np.float16),
        "graph_node_centroid": gym.spaces.Box(
            -np.inf, np.inf, (POOLED_NODES, 3), np.float32
        ),
        "graph_node_target": gym.spaces.Box(0, 1, (POOLED_NODES,), np.uint8),
        "graph_edge_src": gym.spaces.Box(0, POOLED_NODES - 1, (POOLED_EDGES,), np.uint8),
        "graph_edge_dst": gym.spaces.Box(0, POOLED_NODES - 1, (POOLED_EDGES,), np.uint8),
        "graph_edge_rel": gym.spaces.Box(0, 10, (POOLED_EDGES,), np.uint8),
        "graph_edge_abs": gym.spaces.Box(0, 16, (POOLED_EDGES,), np.uint8),
        "graph_edge_temp": gym.spaces.Box(0, 5, (POOLED_EDGES,), np.uint8),
    }
    if with_uid:
        obs["graph_node_uid"] = gym.spaces.Box(0, 255, (POOLED_NODES,), np.uint8)
    return gym.spaces.Dict(obs), gym.spaces.Box(-1, 1, (3,), np.float32)


def pooled_sequence(batch=2, time=3, n_valid=3, relations=6, duplicate=False):
    """One pooled batch.

    ``relations`` is how many of the scorer's six end-effector-to-target facts
    the frame carries. Six is a complete ladder and the only case the progress
    target is defined on; fewer exercises the strict validity mask.
    """
    shape = (batch, time)
    values = {
        key: torch.zeros(*shape, POOLED_NODES, dtype=torch.uint8)
        for key in ("graph_node_ent", "graph_node_target")
    }
    for key in ("src", "dst", "rel", "abs", "temp"):
        values[f"graph_edge_{key}"] = torch.zeros(
            *shape, POOLED_EDGES, dtype=torch.uint8
        )
    values["graph_node_ent"][..., :n_valid] = torch.arange(
        1, n_valid + 1, dtype=torch.uint8
    )
    values["graph_node_target"][..., 1] = 1
    bbox = torch.zeros(*shape, POOLED_NODES, 2, 4, dtype=torch.float16)
    boxes = torch.tensor([
        [0.10, 0.40, 0.20, 0.50],
        [0.50, 0.90, 0.10, 0.30],
        [0.00, 0.20, 0.60, 0.80],
    ])[:n_valid]
    bbox[..., :n_valid, 0, :] = boxes.to(torch.float16)
    values["graph_node_bbox"] = bbox
    centroid = torch.zeros(*shape, POOLED_NODES, 3, dtype=torch.float32)
    centroid[..., :n_valid, :] = torch.tensor([
        [0.00, 0.00, 0.90],
        [0.40, 0.10, 0.75],
        [-0.30, 0.60, 0.75],
    ])[:n_valid]
    values["graph_node_centroid"] = centroid
    # The scorer's six end-effector-to-target facts, in its own relation order:
    # planar-distance, height-offset, contact-compat, grasp-compat, contact,
    # grasp. Labels here score 0.15 + 0.00 + 0.10 + 0.15 = 0.40.
    rel = torch.tensor([5, 6, 8, 7, 1, 2], dtype=torch.uint8)[:relations]
    lab = torch.tensor([3, 8, 13, 13, 1, 1], dtype=torch.uint8)[:relations]
    if duplicate:
        rel = torch.cat([rel, rel[:1]])
        lab = torch.cat([lab, lab[:1]])
    n_edge = int(rel.numel())
    values["graph_edge_src"][..., :n_edge] = 0
    values["graph_edge_dst"][..., :n_edge] = 1
    values["graph_edge_rel"][..., :n_edge] = rel
    values["graph_edge_abs"][..., :n_edge] = lab
    values.update(
        image=torch.randint(0, 256, (batch, time, 16, 16, 3), dtype=torch.uint8),
        state=torch.randn(batch, time, 5),
        instruction=torch.randn(batch, time, 7),
        is_first=torch.zeros(batch, time, 1, dtype=torch.bool),
        is_last=torch.zeros(batch, time, 1, dtype=torch.bool),
        is_terminal=torch.zeros(batch, time, 1, dtype=torch.bool),
        reward=torch.zeros(batch, time, 1),
        action=torch.randn(batch, time, 3).clamp(-1, 1),
    )
    values["is_first"][:, 0] = True
    values["is_last"][:, -1] = True
    return TensorDict(values, batch_size=(batch, time))


def spaces():
    obs = {
        "image": gym.spaces.Box(0, 255, (16, 16, 3), np.uint8),
        "state": gym.spaces.Box(-np.inf, np.inf, (5,), np.float32),
        "instruction": gym.spaces.Box(-np.inf, np.inf, (7,), np.float32),
        "is_first": gym.spaces.Box(0, 1, (), np.bool_),
        "is_last": gym.spaces.Box(0, 1, (), np.bool_),
        "is_terminal": gym.spaces.Box(0, 1, (), np.bool_),
        "reward": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
        "graph_node_ent": gym.spaces.Box(0, 65535, (8,), np.uint16),
        "graph_node_bbox": gym.spaces.Box(-np.inf, np.inf, (8, 2, 4), np.float16),
        "graph_node_target": gym.spaces.Box(0, 1, (8,), np.uint8),
        "graph_edge_src": gym.spaces.Box(0, 7, (168,), np.uint8),
        "graph_edge_dst": gym.spaces.Box(0, 7, (168,), np.uint8),
        "graph_edge_rel": gym.spaces.Box(0, 10, (168,), np.uint8),
        "graph_edge_abs": gym.spaces.Box(0, 16, (168,), np.uint8),
        "graph_edge_temp": gym.spaces.Box(0, 5, (168,), np.uint8),
    }
    return gym.spaces.Dict(obs), gym.spaces.Box(-1, 1, (3,), np.float32)


def sequence(batch=2, time=3):
    values = graph_batch(168, batch=batch, time=time, nodes=8)
    values.update(
        image=torch.randint(0, 256, (batch, time, 16, 16, 3), dtype=torch.uint8),
        state=torch.randn(batch, time, 5),
        instruction=torch.randn(batch, time, 7),
        is_first=torch.tensor([[True, False, False], [True, False, False]]).unsqueeze(-1),
        is_last=torch.zeros(batch, time, 1, dtype=torch.bool),
        is_terminal=torch.zeros(batch, time, 1, dtype=torch.bool),
        reward=torch.zeros(batch, time, 1),
        action=torch.randn(batch, time, 3).clamp(-1, 1),
    )
    values["is_last"][:, -1] = True
    return TensorDict(values, batch_size=(batch, time))


class PooledGraphSimpleTest(unittest.TestCase):
    """The pooled arm end to end: contract, losses, progress, imagination."""

    def _model(self, **kwargs):
        torch.manual_seed(0)
        return Dreamer(make_pooled_config(**kwargs), *pooled_spaces()).to("cpu")

    def test_schema_is_pooled_and_carries_no_identity(self):
        model = self._model()
        self.assertTrue(model.graph_pooled_simple)
        self.assertIn("graph_node_bbox", model.graph_keys)
        self.assertNotIn("graph_node_uid", model.graph_keys)
        self.assertFalse(hasattr(model.graph_encoder, "uid"))

    def test_a_stale_uid_key_is_rejected_rather_than_encoded(self):
        # Without the guard the key would simply be excluded from the encoder
        # and the run would look healthy while training the wrong contract.
        with self.assertRaisesRegex(ValueError, "graph_node_uid"):
            Dreamer(make_pooled_config(), *pooled_spaces(with_uid=True))

    def test_graph_keys_never_reach_the_mlp_encoder(self):
        model = Dreamer(make_pooled_config(), *pooled_spaces())
        for key in model.encoder.mlp_shapes:
            self.assertFalse(key.startswith("graph_"), key)

    def test_update_emits_the_declared_loss_set(self):
        model = self._model()
        _, metrics = model._cal_grad(
            model.preprocess(pooled_sequence()), model.rssm.initial(2)
        )
        emitted = {k[len("loss/"):] for k in metrics if k.startswith("loss/")}
        for name in ("node", "nodetgt", "relabs", "reltemp", "graphdyn",
                     "graphrep", "progress_model", "progress_value"):
            self.assertIn(name, emitted, name)
        # Retired losses must not reappear.
        for name in ("prior_progress_relabs", "slotdyn", "slotalive",
                     "prior_nodetgt", "prior_relabs", "prior_reltemp"):
            self.assertNotIn(name, emitted, name)
        for key, value in metrics.items():
            if key.startswith("loss/"):
                self.assertTrue(torch.isfinite(value), key)

    def test_reported_metrics_stay_cheap(self):
        model = self._model()
        _, metrics = model._cal_grad(
            model.preprocess(pooled_sequence()), model.rssm.initial(2)
        )
        for name in ("node_ent_acc", "node_target_acc", "relabs_acc",
                     "reltemp_acc", "node_bbox_loss",
                     "progress/valid_fraction", "progress/target_std",
                     "progress/head_mae"):
            self.assertIn(name, metrics, name)
        # IoU needs its own kernels and optimises nothing: evaluation only.
        self.assertNotIn("node_bbox_iou", metrics)

    def test_progress_critic_reads_exactly_the_policy_feature(self):
        # Under the world-model source the potential is already a function of
        # imag_feat, so appending a second view of it would hand the critic a
        # shortcut the actor does not have.
        model = self._model()
        self.assertEqual(model.progress_feat_size, model.rssm.feat_size)

    def test_progress_head_is_frozen_with_the_rest(self):
        # One set of parameters for training and imagination: the optimizer,
        # the checkpoint and clone_and_freeze all pick it up automatically.
        model = self._model()
        self.assertTrue(any(
            name.startswith("progress_head.")
            for name in model._named_params
        ))
        frozen = dict(model._frozen_progress_head.named_parameters())
        self.assertTrue(torch.equal(
            model.progress_head.last.weight, frozen["last.weight"]
        ))
        self.assertFalse(frozen["last.weight"].requires_grad)

    def test_imagined_progress_is_batched_not_stepwise(self):
        model = self._model()
        start = (
            torch.randn(4, model.rssm._stoch, model.rssm._discrete),
            torch.randn(4, model.rssm._deter),
            torch.randn(4, model.rssm.flat_sem),
        )
        feat, _, extra = model._imagine(start, 5)
        self.assertEqual(tuple(feat.shape), (4, 5, model.rssm.feat_size))
        # Produced in one pass over B * H after the rollout, not per step.
        self.assertEqual(tuple(extra["progress_reward"].shape), (4, 5, 1))
        self.assertEqual(
            tuple(extra["progress_feat"].shape), (4, 5, model.progress_feat_size)
        )
        # The critic feature is exactly imag_feat -- no relation block, no
        # masks, counts, boxes or observed labels.
        self.assertTrue(torch.equal(extra["progress_feat"], feat))

    def test_imagined_potential_stays_bounded(self):
        model = self._model()
        start = (
            torch.randn(4, model.rssm._stoch, model.rssm._discrete) * 3.0,
            torch.randn(4, model.rssm._deter) * 3.0,
            torch.randn(4, model.rssm.flat_sem) * 3.0,
        )
        _, _, extra = model._imagine(start, 5)
        potential = extra["progress_potential"]
        self.assertGreaterEqual(float(potential.min()), 0.0)
        self.assertLessEqual(float(potential.max()), 1.0)

    def test_zero_scale_switches_the_world_model_branch_off(self):
        model = Dreamer(
            make_pooled_config(progress=False, progress_scale=0.0), *pooled_spaces()
        ).to("cpu")
        self.assertFalse(model._progress_model)
        self.assertIsNone(model.progress_head)
        _, metrics = model._cal_grad(
            model.preprocess(pooled_sequence()), model.rssm.initial(2)
        )
        self.assertNotIn("loss/progress_model", metrics)

    def test_shaping_without_a_trained_head_is_refused(self):
        # A zero scale leaves the head unsupervised; shaping the actor with
        # it would be shaping on noise.
        with self.assertRaisesRegex(ValueError, "untrained"):
            Dreamer(
                make_pooled_config(progress=True, progress_scale=0.0),
                *pooled_spaces(),
            )

    def test_beta_zero_leaves_the_actor_objective_unchanged(self):
        """The control arm has to be the treatment arm minus one term."""
        model = Dreamer(
            make_pooled_config(progress=True, beta=0.0), *pooled_spaces()
        ).to("cpu")
        raw = model.preprocess(pooled_sequence())
        initial = model.rssm.initial(2)
        ema = model.return_ema.ema_vals.clone()

        torch.manual_seed(1234)
        _, with_progress = model._cal_grad(raw, initial)
        model._optimizer.zero_grad(set_to_none=True)
        model.return_ema.ema_vals.copy_(ema)

        model.progress_enabled = False
        torch.manual_seed(1234)
        _, without_progress = model._cal_grad(raw, initial)
        model.progress_enabled = True
        torch.testing.assert_close(
            with_progress["loss/policy"], without_progress["loss/policy"]
        )

    def test_beta_warms_up_on_environment_steps(self):
        config = make_pooled_config(progress=True, beta=0.2)
        model = Dreamer(config, *pooled_spaces()).to("cpu")
        self.assertEqual(
            model.progress_beta_start, float(config.progress.beta_warmup_start)
        )
        self.assertEqual(
            model.progress_beta_end, float(config.progress.beta_warmup_end)
        )
        # Fixed here so the arithmetic is independent of the preset.
        model.progress_beta_start, model.progress_beta_end = 200000.0, 300000.0
        beta_at = model._progress_beta_at
        self.assertEqual(beta_at(0), 0.0)
        self.assertEqual(beta_at(199999), 0.0)
        self.assertEqual(beta_at(200000), 0.0)
        self.assertAlmostEqual(beta_at(250000), 0.1)
        self.assertAlmostEqual(beta_at(300000), 0.2)
        self.assertAlmostEqual(beta_at(10**7), 0.2)
        # No step count means no schedule: beta applies in full.
        self.assertAlmostEqual(beta_at(None), 0.2)

    def test_acting_runs_on_the_pooled_contract(self):
        model = self._model()
        data = pooled_sequence(batch=2, time=1)
        obs = TensorDict(
            {key: value[:, 0] for key, value in data.items() if key != "action"},
            batch_size=(2,),
        )
        state = model.get_initial_state(2)
        action, _ = model.act(obs, state, eval=True)
        self.assertEqual(tuple(action.shape), (2, 3))
        self.assertTrue(torch.isfinite(action).all())


class DreamerGraphIntegrationTest(unittest.TestCase):
    def test_real_graph_preset_has_capacity_matched_latents(self):
        config_dir = str(pathlib.Path(__file__).resolve().parents[1] / "configs")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            config = compose(
                config_name="configs",
                overrides=["env=dmc_vision", "model=size50M_graph"],
            ).model
        self.assertEqual(config.graph.n_max, 8)
        self.assertEqual(config.graph.e_max, 168)
        self.assertEqual(config.rssm.stoch, 32)
        self.assertEqual(config.rssm.discrete, 32)

    def test_graph_on_update_and_act(self):
        config = make_config(True)
        obs_space, act_space = spaces()
        model = Dreamer(config, obs_space, act_space).to("cpu")
        raw = sequence()
        action, state = model.act(raw[:, 0].clone(), model.get_initial_state(2))
        self.assertEqual(action.shape, (2, 3))
        self.assertIn("sem", state)
        self.assertEqual(model.rssm.flat_stoch, 8)
        self.assertEqual(model.rssm.flat_sem, 6)
        self.assertEqual(model._loss_scales["image"], 0.5)
        self.assertEqual(model._loss_scales["state"], 1.0)
        data = model.preprocess(raw)
        initial = model.rssm.initial(2)
        posterior, metrics = model._cal_grad(data, initial)
        self.assertEqual(len(posterior), 3)
        for key in ("loss/node", "loss/nodetgt", "loss/relabs", "loss/reltemp", "loss/semdyn", "loss/semrep"):
            self.assertIn(key, metrics)
            self.assertTrue(torch.isfinite(metrics[key]))

    def test_single_switch_constructs_graph_free_dreamer(self):
        config = make_config(False)
        obs_space, act_space = spaces()
        model = Dreamer(config, obs_space, act_space).to("cpu")
        self.assertIsNone(model.graph_encoder)
        self.assertFalse(model.rssm.semantic)
        self.assertEqual(model.rssm._stoch, 32)
        self.assertEqual(model._loss_scales["image"], 1.0)
        self.assertFalse(any(name.startswith("graph_") for name in model._named_params))
        posterior, _ = model._cal_grad(model.preprocess(sequence()), model.rssm.initial(2))
        self.assertEqual(len(posterior), 2)

    def test_progress_requires_a_graph_simple_mode(self):
        # Full mode has neither a slot table to decode per-candidate relations
        # from nor a pooled g to run the fused head on.
        config = make_config(True)
        config.progress.enabled = True
        with self.assertRaisesRegex(ValueError, "graph-simple"):
            Dreamer(config, *spaces())

    def test_preprocess_is_shallow_and_non_mutating(self):
        config = make_config(True)
        obs_space, act_space = spaces()
        model = Dreamer(config, obs_space, act_space).to("cpu")
        raw = sequence()
        original = raw["image"].clone()
        processed = model.preprocess(raw)
        self.assertEqual(raw["image"].dtype, torch.uint8)
        self.assertEqual(processed["image"].dtype, torch.float32)
        torch.testing.assert_close(raw["image"], original)
        self.assertIs(processed["graph_node_ent"], raw["graph_node_ent"])


if __name__ == "__main__":
    unittest.main()
