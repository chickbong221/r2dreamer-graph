import pathlib
import unittest

import gymnasium as gym
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from tensordict import TensorDict

from dreamer import Dreamer
from test_graph import graph_batch


def make_config(enabled, graph_only=False):
    config_dir = str(pathlib.Path(__file__).resolve().parent / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(
            config_name="configs",
            overrides=[
                "env=dmc_vision",
                "model=size50M_graph",
                "device=cpu",
                f"model.graph.enabled={str(enabled).lower()}",
                f"model.graph_only_latent={str(graph_only).lower()}",
                "model.deter=16",
                "model.hidden=8",
                "model.discrete=4",
                "model.units=8",
                "model.depth=2",
                "model.rssm.blocks=4",
                "model.rssm.hybrid_stoch=2",
                "model.rssm.sem_stoch=2",
                "model.rssm.sem_discrete=3",
                "model.rssm.graph_only_stoch=2",
                "model.rssm.graph_only_discrete=4",
                "model.graph.units=8",
                "model.graph.layers=1",
                "model.graph.app_dim=8",
                "model.encoder.cnn.depth=2",
                "model.decoder.cnn.depth=2",
                "model.encoder.cnn.minres=1",
                "model.decoder.cnn.minres=1",
            ],
        ).model
    config.encoder.mlp_keys = "^(state|instruction)$"
    config.decoder.mlp_keys = "^state$"
    return config


SLOT_NODES = 8
SLOT_EDGES = 16


def make_slot_config(progress=False, beta=0.05):
    """The slot preset, shrunk to unit-test size but structurally identical."""
    config_dir = str(pathlib.Path(__file__).resolve().parent / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(
            config_name="configs",
            overrides=[
                "env=dmc_vision",
                "model=size100M_graph_slots",
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
                "model.graph.slot_dim=8",
                "model.graph.slot_heads=2",
                "model.encoder.cnn.depth=2",
                "model.decoder.cnn.depth=2",
                "model.encoder.cnn.minres=1",
                "model.decoder.cnn.minres=1",
                f"model.progress.enabled={str(progress).lower()}",
                f"model.progress.beta={beta}",
            ],
        ).model
    config.encoder.mlp_keys = "^(state|instruction)$"
    config.decoder.mlp_keys = "^state$"
    return config


def slot_spaces():
    obs = {
        "image": gym.spaces.Box(0, 255, (16, 16, 3), np.uint8),
        "state": gym.spaces.Box(-np.inf, np.inf, (5,), np.float32),
        "instruction": gym.spaces.Box(-np.inf, np.inf, (7,), np.float32),
        "is_first": gym.spaces.Box(0, 1, (), np.bool_),
        "is_last": gym.spaces.Box(0, 1, (), np.bool_),
        "is_terminal": gym.spaces.Box(0, 1, (), np.bool_),
        "reward": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
        "graph_node_ent": gym.spaces.Box(0, 255, (SLOT_NODES,), np.uint8),
        "graph_node_uid": gym.spaces.Box(0, 255, (SLOT_NODES,), np.uint8),
        "graph_node_target": gym.spaces.Box(0, 1, (SLOT_NODES,), np.uint8),
        "graph_edge_src": gym.spaces.Box(0, SLOT_NODES - 1, (SLOT_EDGES,), np.uint8),
        "graph_edge_dst": gym.spaces.Box(0, SLOT_NODES - 1, (SLOT_EDGES,), np.uint8),
        "graph_edge_rel": gym.spaces.Box(0, 10, (SLOT_EDGES,), np.uint8),
        "graph_edge_abs": gym.spaces.Box(0, 16, (SLOT_EDGES,), np.uint8),
        "graph_edge_temp": gym.spaces.Box(0, 5, (SLOT_EDGES,), np.uint8),
    }
    return gym.spaces.Dict(obs), gym.spaces.Box(-1, 1, (3,), np.float32)


def slot_sequence(batch=2, time=3, uids=(1, 2, 3)):
    shape = (batch, time)
    values = {
        key: torch.zeros(*shape, SLOT_NODES, dtype=torch.uint8)
        for key in ("graph_node_ent", "graph_node_uid", "graph_node_target")
    }
    for key in ("src", "dst", "rel", "abs", "temp"):
        values[f"graph_edge_{key}"] = torch.zeros(*shape, SLOT_EDGES, dtype=torch.uint8)
    count = len(uids)
    values["graph_node_ent"][..., :count] = torch.arange(1, count + 1, dtype=torch.uint8)
    values["graph_node_uid"][..., :count] = torch.tensor(uids, dtype=torch.uint8)
    values["graph_node_target"][..., 1] = 1
    values["graph_edge_src"][..., :2] = torch.tensor([0, 1], dtype=torch.uint8)
    values["graph_edge_dst"][..., :2] = torch.tensor([1, 2], dtype=torch.uint8)
    values["graph_edge_rel"][..., :2] = torch.tensor([1, 5], dtype=torch.uint8)
    values["graph_edge_abs"][..., :2] = torch.tensor([2, 3], dtype=torch.uint8)
    values["graph_edge_temp"][..., :2] = torch.tensor([0, 3], dtype=torch.uint8)
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


POOLED_NODES = 8
POOLED_EDGES = 16


def make_pooled_config(progress=True, prior_scale=1.0):
    """The pooled graph-simple preset, shrunk but structurally identical."""
    config_dir = str(pathlib.Path(__file__).resolve().parent / "configs")
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
                f"model.loss_scales.prior_progress_relabs={prior_scale}",
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


def pooled_sequence(batch=2, time=3, n_valid=3):
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
    # Four end-effector-to-target facts on relations the Pick scorer reads.
    values["graph_edge_src"][..., :4] = 0
    values["graph_edge_dst"][..., :4] = 1
    values["graph_edge_rel"][..., :4] = torch.tensor([5, 6, 8, 7], dtype=torch.uint8)
    values["graph_edge_abs"][..., :4] = torch.tensor([3, 8, 13, 13], dtype=torch.uint8)
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
        "graph_node_app": gym.spaces.Box(-np.inf, np.inf, (8, 2, 8), np.float16),
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
    values = graph_batch(168, batch=batch, time=time, nodes=8, app_dim=8)
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
        self.assertEqual(model.graph_schema, "simple_pooled_bbox")
        self.assertTrue(model.graph_pooled_simple)
        self.assertFalse(model.graph_slots)
        self.assertIn("graph_node_bbox", model.graph_keys)
        self.assertNotIn("graph_node_uid", model.graph_keys)
        self.assertIsNone(model.graph_encoder.uid)

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
                     "graphrep", "prior_progress_relabs", "progress_value"):
            self.assertIn(name, emitted, name)
        # No recurrent slot state exists in this arm.
        for name in ("slotdyn", "slotalive", "prior_nodetgt", "prior_relabs",
                     "prior_reltemp"):
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
                     "reltemp_acc", "prior_progress_acc", "prior_progress_facts",
                     "node_bbox_loss"):
            self.assertIn(name, metrics, name)
        # IoU needs its own kernels and optimises nothing: evaluation only.
        self.assertNotIn("node_bbox_iou", metrics)

    def test_progress_critic_reads_the_policy_feature_plus_the_relations(self):
        model = self._model()
        scorer_width = int(model.progress_scorer.relations.numel()) * 17
        self.assertEqual(
            model.progress_feat_size, model.rssm.feat_size + scorer_width
        )

    def test_progress_head_is_owned_by_the_decoder_and_frozen_with_it(self):
        # One set of parameters for training and imagination: the optimizer,
        # the checkpoint and clone_and_freeze all pick it up automatically.
        model = self._model()
        self.assertTrue(any(
            name.startswith("graph_decoder.progress_head")
            for name in model._named_params
        ))
        self.assertIn(
            "progress_head.weight", dict(model._frozen_graph_decoder.named_parameters())
        )
        self.assertTrue(torch.equal(
            model.graph_decoder.progress_head.weight,
            model._frozen_graph_decoder.progress_head.weight,
        ))

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
        # The critic feature is exactly imag_feat with the relation block
        # appended -- no masks, counts, boxes or observed labels.
        self.assertTrue(torch.equal(
            extra["progress_feat"][..., : model.rssm.feat_size], feat
        ))

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

    def test_zero_scale_switches_the_prior_branch_off(self):
        model = Dreamer(
            make_pooled_config(progress=False, prior_scale=0.0), *pooled_spaces()
        ).to("cpu")
        self.assertFalse(model._prior_progress)
        _, metrics = model._cal_grad(
            model.preprocess(pooled_sequence()), model.rssm.initial(2)
        )
        self.assertNotIn("loss/prior_progress_relabs", metrics)

    def test_shaping_without_a_trained_head_is_refused(self):
        # A zero scale leaves the fused head unsupervised; shaping the actor
        # with it would be shaping on noise.
        with self.assertRaisesRegex(ValueError, "untrained"):
            Dreamer(
                make_pooled_config(progress=True, prior_scale=0.0), *pooled_spaces()
            )

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
        config_dir = str(pathlib.Path(__file__).resolve().parent / "configs")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            config = compose(
                config_name="configs",
                overrides=["env=dmc_vision", "model=size50M_graph"],
            ).model
        self.assertEqual(config.graph.n_max, 8)
        self.assertEqual(config.graph.e_max, 168)
        self.assertEqual(config.rssm.stoch, 32)
        self.assertEqual(config.rssm.hybrid_stoch, 14)
        self.assertEqual(config.rssm.sem_stoch, 18)
        self.assertEqual(config.rssm.sem_discrete, 32)
        self.assertEqual(config.rssm.graph_only_stoch, 32)
        self.assertEqual(config.rssm.graph_only_discrete, 32)

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

    def test_graph_only_constructs_no_z_or_pixel_cnn(self):
        config = make_config(True, graph_only=True)
        obs_space, act_space = spaces()
        model = Dreamer(config, obs_space, act_space).to("cpu")
        self.assertTrue(model.graph_only)
        self.assertEqual(model.rssm.state_keys, ("deter", "sem"))
        self.assertEqual(model.rssm.flat_stoch, 0)
        self.assertEqual(model.rssm.flat_sem, 8)
        self.assertIsNone(model.rssm._obs_net)
        self.assertIsNone(model.rssm._img_net)
        self.assertIsNone(model.rssm._deter_net._dyn_in1)
        self.assertEqual(model.encoder.cnn_shapes, {})
        self.assertEqual(model.decoder.cnn_shapes, {})
        self.assertFalse(hasattr(model.encoder, "_cnn"))
        self.assertFalse(hasattr(model.decoder, "_cnn"))
        self.assertFalse(any(
            "_obs_net" in name or "_img_net" in name
            for name in model._named_params
        ))

        raw = sequence()
        action, state = model.act(raw[:, 0].clone(), model.get_initial_state(2))
        self.assertEqual(action.shape, (2, 3))
        self.assertNotIn("stoch", state)
        self.assertIn("deter", state)
        self.assertIn("sem", state)
        altered = raw[:, 0].clone()
        altered["instruction"] = altered["instruction"] + 1
        first_embed = model.encoder(model.preprocess(raw[:, 0]))
        second_embed = model.encoder(model.preprocess(altered))
        self.assertFalse(torch.allclose(first_embed, second_embed))
        posterior, metrics = model._cal_grad(
            model.preprocess(raw), model.rssm.initial(2)
        )
        self.assertEqual(len(posterior), 2)
        self.assertNotIn("loss/dyn", metrics)
        self.assertNotIn("loss/rep", metrics)
        self.assertNotIn("loss/image", metrics)
        self.assertFalse(any("semtgt" in key for key in metrics))
        for key in ("loss/state", "loss/semdyn", "loss/semrep", "loss/nodetgt"):
            self.assertIn(key, metrics)

    def test_graph_only_requires_graph(self):
        config = make_config(False, graph_only=True)
        obs_space, act_space = spaces()
        with self.assertRaisesRegex(ValueError, "requires graph.enabled"):
            Dreamer(config, obs_space, act_space)

    def test_slot_mode_carries_slots_and_supervises_both_branches(self):
        model = Dreamer(make_slot_config(), *slot_spaces()).to("cpu")
        self.assertTrue(model.graph_slots)
        # No pooled semantic state exists anywhere in this arm.
        self.assertIsNone(model.rssm._sem_obs)
        self.assertIsNone(model.rssm._sem_img)
        self.assertIsNone(model.graph_encoder.query)
        self.assertEqual(model.rssm.n_slots, SLOT_NODES)
        self.assertEqual(model.rssm.flat_sem, 8)  # the readout, not the slots
        # h reads a permutation-invariant summary of the slot set, never the
        # ordered flatten: three pooled 8-wide statistics plus occupancy.
        self.assertEqual(model.rssm._deter_net._dyn_in3[0].in_features, 3 * 8 + 1)
        self.assertEqual(model._loss_scales["image"], 1.0)

        raw = slot_sequence()
        action, state = model.act(raw[:, 0].clone(), model.get_initial_state(2))
        self.assertEqual(action.shape, (2, 3))
        self.assertIn("slot_meta", state)
        self.assertIn("slot_alive", state)
        self.assertEqual(tuple(state["slot_meta"].shape), (2, SLOT_NODES, 3))
        self.assertEqual(tuple(state["slot_alive"].shape), (2, SLOT_NODES))

        posterior, metrics = model._cal_grad(model.preprocess(raw), model.rssm.initial(2))
        self.assertEqual(len(posterior), 5)
        self.assertEqual(tuple(posterior[2].shape), (2, 3, SLOT_NODES, 8))
        self.assertEqual(tuple(posterior[4].shape), (2, 3, SLOT_NODES))
        for key in (
            "loss/slotdyn", "loss/slotalive", "loss/nodetgt", "loss/prior_nodetgt",
            "loss/relabs", "loss/reltemp", "loss/prior_progress_relabs",
            "loss/dyn", "loss/rep",
        ):
            self.assertIn(key, metrics)
            self.assertTrue(torch.isfinite(metrics[key]), key)
        # Replaced by the teacher-forced end-effector-to-target loss.
        self.assertNotIn("loss/prior_relabs", metrics)
        self.assertNotIn("loss/prior_reltemp", metrics)
        # The pooled-mode losses have no meaning here and must not appear.
        self.assertNotIn("loss/graphdyn", metrics)
        self.assertNotIn("loss/graphrep", metrics)
        self.assertNotIn("loss/progress_value", metrics)
        self.assertEqual(float(metrics["slot_overflow"]), 0.0)
        self.assertIn("presence_brier", metrics)

    def test_object_slot_permutation_does_not_move_the_heads(self):
        model = Dreamer(make_slot_config(), *slot_spaces()).to("cpu").eval()
        rssm_model = model.rssm
        stoch, deter, sem, meta, alive = rssm_model.initial(1)
        torch.manual_seed(0)
        sem = torch.randn_like(sem)
        alive = torch.ones_like(alive)
        order = [0, 4, 2, 7, 1, 6, 3, 5][:SLOT_NODES]
        with torch.no_grad():
            feat = rssm_model.get_feat(stoch, deter, sem, alive)
            other = rssm_model.get_feat(
                stoch, deter, sem[:, order], alive[:, order]
            )
            torch.testing.assert_close(feat, other, atol=1e-5, rtol=1e-5)
            for head in (model.reward, model.value):
                torch.testing.assert_close(
                    head(feat).mode(), head(other).mode(), atol=1e-5, rtol=1e-5
                )
            # The continuation head is a binary distribution, whose ``mode`` is
            # a property rather than a method; ``mean`` is what dreamer reads.
            torch.testing.assert_close(
                model.cont(feat).mean, model.cont(other).mean,
                atol=1e-5, rtol=1e-5,
            )
            torch.testing.assert_close(
                model.actor(feat).mode, model.actor(other).mode,
                atol=1e-5, rtol=1e-5,
            )

    def test_slot_births_off_keeps_imagined_occupancy_fixed(self):
        config = make_slot_config()
        config.graph.slot_births = False
        model = Dreamer(config, *slot_spaces()).to("cpu")
        self.assertFalse(model.rssm.slot_births)
        _, metrics = model._cal_grad(
            model.preprocess(slot_sequence()), model.rssm.initial(2)
        )
        # Presence is still predicted and supervised, but imagination carries
        # occupancy forward unchanged, so the rollout creates nothing.
        self.assertIn("loss/slotalive", metrics)
        self.assertEqual(float(metrics["imag_births"]), 0.0)

    def test_slot_mode_needs_the_relation_only_contract(self):
        config = make_slot_config()
        config.graph_simple = False
        with self.assertRaisesRegex(ValueError, "relation-only"):
            Dreamer(config, *slot_spaces())

    def test_progress_requires_a_graph_simple_mode(self):
        # Full mode has neither a slot table to decode per-candidate relations
        # from nor a pooled g to run the fused head on.
        config = make_config(True)
        config.progress.enabled = True
        with self.assertRaisesRegex(ValueError, "graph-simple"):
            Dreamer(config, *spaces())

    def test_progress_is_bounded_and_trains_its_own_critic(self):
        model = Dreamer(make_slot_config(progress=True), *slot_spaces()).to("cpu")
        self.assertTrue(model.progress_enabled)
        self.assertTrue(any(n.startswith("progress_value") for n in model._named_params))
        _, metrics = model._cal_grad(
            model.preprocess(slot_sequence()), model.rssm.initial(2)
        )
        self.assertIn("loss/progress_value", metrics)
        self.assertTrue(torch.isfinite(metrics["loss/progress_value"]))
        potential = float(metrics["progress_potential"])
        self.assertGreaterEqual(potential, 0.0)
        self.assertLessEqual(potential, 1.0)
        # Bounded shaping: the per-step reward can never exceed 1 - discount.
        self.assertLessEqual(float(metrics["progress_reward"]), 1.0 / model.horizon + 1e-6)

    def test_beta_zero_leaves_the_actor_objective_unchanged(self):
        model = Dreamer(make_slot_config(progress=True, beta=0.0), *slot_spaces()).to("cpu")
        raw = model.preprocess(slot_sequence())
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
        model = Dreamer(make_slot_config(progress=True, beta=0.2), *slot_spaces()).to("cpu")
        # The window comes from the preset, not from this test.
        self.assertEqual(model.progress_beta_start, 200000.0)
        self.assertEqual(model.progress_beta_end, 300000.0)
        beta_at = model._progress_beta_at
        self.assertEqual(beta_at(0), 0.0)
        self.assertEqual(beta_at(199999), 0.0)
        self.assertEqual(beta_at(200000), 0.0)
        self.assertAlmostEqual(beta_at(250000), 0.1)
        self.assertAlmostEqual(beta_at(300000), 0.2)
        self.assertAlmostEqual(beta_at(10**7), 0.2)
        # No step count means no schedule: beta applies in full.
        self.assertAlmostEqual(beta_at(None), 0.2)

    def test_the_warm_up_trains_the_progress_critic_without_steering_the_actor(self):
        model = Dreamer(make_slot_config(progress=True, beta=0.2), *slot_spaces()).to("cpu")
        # A critic built at outscale 0 predicts a constant zero, which makes the
        # progress advantage identically zero and every beta indistinguishable
        # from every other. Give the head a real readout first, or the
        # comparisons below compare zero with zero. The frozen copy shares this
        # storage, so it reads the same weights.
        with torch.no_grad():
            model.progress_value.last.weight.normal_(0.0, 0.3)
        raw = model.preprocess(slot_sequence())
        initial = model.rssm.initial(2)
        ema = model.return_ema.ema_vals.clone()
        progress_ema = model.progress_return_ema.ema_vals.clone()

        def run(step, progress=True):
            # Same batch, same seed, same normaliser state: the environment
            # advantage is identical across runs and only beta differs.
            model.return_ema.ema_vals.copy_(ema)
            model.progress_return_ema.ema_vals.copy_(progress_ema)
            model.progress_enabled = progress
            model._env_step = step
            torch.manual_seed(1234)
            _, metrics = model._cal_grad(raw, initial)
            model._optimizer.zero_grad(set_to_none=True)
            model.progress_enabled = True
            return metrics

        warming = run(0.0)  # inside the warm-up
        plateau = run(300000.0)  # after it
        without_progress = run(0.0, progress=False)

        # Warm-up: the critic trains and the actor never sees it.
        self.assertGreater(float(warming["progress_adv_abs"]), 0.0)
        self.assertEqual(float(warming["progress_beta"]), 0.0)
        self.assertEqual(float(warming["progress_influence"]), 0.0)
        self.assertIn("loss/progress_value", warming)
        self.assertTrue(torch.isfinite(warming["loss/progress_value"]))
        torch.testing.assert_close(
            warming["loss/policy"], without_progress["loss/policy"]
        )

        # Plateau: the same batch now moves the actor objective.
        self.assertAlmostEqual(float(plateau["progress_beta"]), 0.2)
        self.assertNotEqual(
            float(plateau["loss/policy"]), float(warming["loss/policy"])
        )
        # rho = beta * E|A_progress| / E|A_env|, on the environment advantage
        # as it was before the shaping term was mixed in.
        self.assertAlmostEqual(
            float(plateau["progress_influence"]),
            0.2 * float(plateau["progress_adv_abs"]) / float(plateau["env_adv_abs"]),
            places=4,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_one_cuda_batch_completes_without_nan(self):
        config = make_slot_config()
        config.device = "cuda"
        config.rssm.device = "cuda"
        for head in ("reward", "cont", "actor", "critic"):
            config[head].device = "cuda"
        model = Dreamer(config, *slot_spaces()).to("cuda")
        # Two batches with different valid-node counts: the real-edge path is
        # dynamic, so a shape assumption would fail on the second one.
        for uids in ((1, 2, 3), (1, 2, 3, 4, 5)):
            raw = slot_sequence(uids=uids).to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                _, metrics = model._cal_grad(
                    model.preprocess(raw), model.rssm.initial(2)
                )
            for key, value in metrics.items():
                if torch.is_tensor(value) and torch.is_floating_point(value):
                    self.assertTrue(torch.isfinite(value).all(), key)
            model._optimizer.zero_grad(set_to_none=True)

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
