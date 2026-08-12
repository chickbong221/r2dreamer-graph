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
