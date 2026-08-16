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


SLOT_NODES = 6
SLOT_EDGES = 16


def make_slot_config(progress=False, beta=0.05):
    """The slot preset, shrunk to unit-test size but structurally identical."""
    config_dir = str(pathlib.Path(__file__).resolve().parent / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(
            config_name="configs",
            overrides=[
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
        "graph_edge_src": gym.spaces.Box(0, 5, (SLOT_EDGES,), np.uint8),
        "graph_edge_dst": gym.spaces.Box(0, 5, (SLOT_EDGES,), np.uint8),
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


class DreamerGraphIntegrationTest(unittest.TestCase):
    def test_real_graph_preset_has_capacity_matched_latents(self):
        config_dir = str(pathlib.Path(__file__).resolve().parent / "configs")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            config = compose(
                config_name="configs", overrides=["model=size50M_graph"]
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
        self.assertEqual(model.rssm._deter_net._dyn_in3[0].in_features, 6 * 8 + 6)
        self.assertEqual(model._loss_scales["image"], 1.0)

        raw = slot_sequence()
        action, state = model.act(raw[:, 0].clone(), model.get_initial_state(2))
        self.assertEqual(action.shape, (2, 3))
        self.assertIn("slot_meta", state)
        self.assertEqual(tuple(state["slot_meta"].shape), (2, SLOT_NODES, 3))

        posterior, metrics = model._cal_grad(model.preprocess(raw), model.rssm.initial(2))
        self.assertEqual(len(posterior), 4)
        self.assertEqual(tuple(posterior[2].shape), (2, 3, SLOT_NODES, 8))
        for key in (
            "loss/slotdyn", "loss/nodetgt", "loss/relabs", "loss/reltemp",
            "loss/prior_relabs", "loss/prior_reltemp", "loss/dyn", "loss/rep",
        ):
            self.assertIn(key, metrics)
            self.assertTrue(torch.isfinite(metrics[key]), key)
        # The pooled-mode losses have no meaning here and must not appear.
        self.assertNotIn("loss/graphdyn", metrics)
        self.assertNotIn("loss/graphrep", metrics)
        self.assertNotIn("loss/progress_value", metrics)
        self.assertEqual(float(metrics["slot_overflow"]), 0.0)

    def test_slot_mode_needs_the_relation_only_contract(self):
        config = make_slot_config()
        config.graph_simple = False
        with self.assertRaisesRegex(ValueError, "relation-only"):
            Dreamer(config, *slot_spaces())

    def test_progress_requires_slot_mode(self):
        config = make_config(True)
        config.progress.enabled = True
        with self.assertRaisesRegex(ValueError, "state_mode=slots"):
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
