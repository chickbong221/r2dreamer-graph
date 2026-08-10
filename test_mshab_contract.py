import unittest
from types import SimpleNamespace

import numpy as np
import torch

from dreamer import _mask_terminal_graph
from envs.maniskill import ManiSkillVecEnv, _repo_path, _select_build_configs
from graph import GRAPH_KEYS


class _FakeGraph:
    obs_spec_shapes = {
        "graph_node_ent": (10,),
        "graph_node_app": (10, 2, 384),
        "graph_node_bbox": (10, 2, 4),
        "graph_node_target": (10,),
        "graph_edge_src": (270,),
        "graph_edge_dst": (270,),
        "graph_edge_rel": (270,),
        "graph_edge_abs": (270,),
        "graph_edge_temp": (270,),
    }
    obs_spec_dtypes = {
        "graph_node_ent": np.uint16,
        "graph_node_app": np.float16,
        "graph_node_bbox": np.float16,
        "graph_node_target": np.uint8,
        "graph_edge_src": np.uint8,
        "graph_edge_dst": np.uint8,
        "graph_edge_rel": np.uint8,
        "graph_edge_abs": np.uint8,
        "graph_edge_temp": np.uint8,
    }
    overflow_drops = np.zeros(2, np.float32)
    fact_drops = np.zeros(2, np.float32)
    target_missing = np.zeros(2, np.float32)
    cache_entries = 3


class MSHABContractTest(unittest.TestCase):
    def _adapter(self):
        env = ManiSkillVecEnv.__new__(ManiSkillVecEnv)
        env._num_envs = 2
        env._device = torch.device("cpu")
        env._camera_keys = {"image_head": "fetch_head", "image_hand": "fetch_hand"}
        env._graph = _FakeGraph()
        env._instruction = SimpleNamespace(table=SimpleNamespace(dim=768))
        env._instruction_obs = np.zeros((2, 768), np.float32)
        env._graph_obs = {
            key: np.zeros((2, *shape), dtype=env._graph.obs_spec_dtypes[key])
            for key, shape in env._graph.obs_spec_shapes.items()
        }
        return env

    @staticmethod
    def _obs():
        return {
            "state": torch.zeros(2, 31),
            "image_head": torch.zeros(2, 112, 112, 3, dtype=torch.uint8),
            "image_hand": torch.zeros(2, 112, 112, 3, dtype=torch.uint8),
        }

    def test_space_and_transition_match_graph_contract(self):
        env = self._adapter()
        space = env._build_observation_space(self._obs())
        self.assertEqual(space["image_head"].shape, (112, 112, 3))
        self.assertEqual(space["instruction"].shape, (768,))
        self.assertEqual(space["graph_edge_rel"].shape, (270,))
        transition = env._transition(
            self._obs(),
            np.zeros(2, np.float32),
            np.zeros(2, bool),
            np.zeros(2, bool),
            np.ones(2, bool),
        )
        self.assertEqual(tuple(transition.batch_size), (2,))
        self.assertTrue(set(GRAPH_KEYS).issubset(transition.keys()))
        self.assertEqual(transition["graph_node_app"].dtype, torch.float16)
        self.assertEqual(transition["is_first"].shape, (2, 1))

    def test_terminal_graph_token_is_zeroed(self):
        token = torch.ones(2, 3, 4)
        is_last = torch.tensor([[[False], [True], [False]], [[True], [False], [True]]])
        masked = _mask_terminal_graph(token, is_last)
        self.assertTrue(torch.equal(masked[0, 1], torch.zeros(4)))
        self.assertTrue(torch.equal(masked[1, 0], torch.zeros(4)))
        self.assertTrue(torch.equal(masked[0, 0], torch.ones(4)))

    def test_build_config_selection_keeps_whole_groups(self):
        plans = [
            SimpleNamespace(build_config_name="b", marker=1),
            SimpleNamespace(build_config_name="a", marker=2),
            SimpleNamespace(build_config_name="a", marker=3),
        ]
        selected = _select_build_configs(plans, 1)
        self.assertEqual([plan.marker for plan in selected], [2, 3])

    def test_packaged_instruction_path_resolves(self):
        self.assertTrue(_repo_path("scenegraph/configs/instructions.npz").is_file())


if __name__ == "__main__":
    unittest.main()
