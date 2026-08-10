import pathlib
import unittest

import gymnasium as gym
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from tensordict import TensorDict

from dreamer import Dreamer
from test_graph import graph_batch


def make_config(enabled):
    config_dir = str(pathlib.Path(__file__).resolve().parent / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(
            config_name="configs",
            overrides=[
                "model=size50M_graph",
                "device=cpu",
                f"model.graph.enabled={str(enabled).lower()}",
                "model.deter=16",
                "model.hidden=8",
                "model.discrete=4",
                "model.units=8",
                "model.depth=2",
                "model.rssm.blocks=4",
                "model.graph.units=8",
                "model.graph.layers=1",
                "model.graph.app_dim=8",
                "model.encoder.cnn.depth=2",
                "model.decoder.cnn.depth=2",
                "model.encoder.cnn.minres=1",
                "model.decoder.cnn.minres=1",
            ],
        ).model


def spaces():
    obs = {
        "image": gym.spaces.Box(0, 255, (16, 16, 3), np.uint8),
        "is_first": gym.spaces.Box(0, 1, (), np.bool_),
        "is_last": gym.spaces.Box(0, 1, (), np.bool_),
        "is_terminal": gym.spaces.Box(0, 1, (), np.bool_),
        "reward": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
        "graph_node_ent": gym.spaces.Box(0, 65535, (10,), np.uint16),
        "graph_node_app": gym.spaces.Box(-np.inf, np.inf, (10, 2, 8), np.float16),
        "graph_node_bbox": gym.spaces.Box(-np.inf, np.inf, (10, 2, 4), np.float16),
        "graph_node_target": gym.spaces.Box(0, 1, (10,), np.uint8),
        "graph_edge_src": gym.spaces.Box(0, 9, (96,), np.uint8),
        "graph_edge_dst": gym.spaces.Box(0, 9, (96,), np.uint8),
        "graph_edge_rel": gym.spaces.Box(0, 10, (96,), np.uint8),
        "graph_edge_abs": gym.spaces.Box(0, 16, (96,), np.uint8),
        "graph_edge_temp": gym.spaces.Box(0, 5, (96,), np.uint8),
    }
    return gym.spaces.Dict(obs), gym.spaces.Box(-1, 1, (3,), np.float32)


def sequence(batch=2, time=3):
    values = graph_batch(96, batch=batch, time=time, app_dim=8)
    values.update(
        image=torch.randint(0, 256, (batch, time, 16, 16, 3), dtype=torch.uint8),
        is_first=torch.tensor([[True, False, False], [True, False, False]]).unsqueeze(-1),
        is_last=torch.zeros(batch, time, 1, dtype=torch.bool),
        is_terminal=torch.zeros(batch, time, 1, dtype=torch.bool),
        reward=torch.zeros(batch, time, 1),
        action=torch.randn(batch, time, 3).clamp(-1, 1),
    )
    values["is_last"][:, -1] = True
    return TensorDict(values, batch_size=(batch, time))


class DreamerGraphIntegrationTest(unittest.TestCase):
    def test_graph_on_update_and_act(self):
        config = make_config(True)
        obs_space, act_space = spaces()
        model = Dreamer(config, obs_space, act_space).to("cpu")
        raw = sequence()
        action, state = model.act(raw[:, 0].clone(), model.get_initial_state(2))
        self.assertEqual(action.shape, (2, 3))
        self.assertIn("sem", state)
        data = model.preprocess(raw)
        initial = model.rssm.initial(2)
        posterior, metrics = model._cal_grad(data, initial)
        self.assertEqual(len(posterior), 3)
        for key in ("loss/node", "loss/relabs", "loss/reltemp", "loss/semtgt", "loss/semdyn", "loss/semrep"):
            self.assertIn(key, metrics)
            self.assertTrue(torch.isfinite(metrics[key]))

    def test_single_switch_constructs_graph_free_dreamer(self):
        config = make_config(False)
        obs_space, act_space = spaces()
        model = Dreamer(config, obs_space, act_space).to("cpu")
        self.assertIsNone(model.graph_encoder)
        self.assertFalse(model.rssm.semantic)
        self.assertFalse(any(name.startswith("graph_") for name in model._named_params))
        posterior, _ = model._cal_grad(model.preprocess(sequence()), model.rssm.initial(2))
        self.assertEqual(len(posterior), 2)

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
