"""Short MS-HAB graph-to-policy smoke test; no replay or optimizer updates."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import torch
from hydra import compose, initialize_config_dir


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dreamer import Dreamer  # noqa: E402
from envs.maniskill import ManiSkillVecEnv  # noqa: E402
from graph import graph_keys  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--build-configs", type=int, default=4)
    parser.add_argument("--mshab-task", default="prepare_groceries")
    parser.add_argument("--mshab-obj", default="all")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    overrides = [
        "env=mshab",
        "model=size50M_graph_simple",
        f"env.env_num={args.num_envs}",
        f"env.num_build_configs={args.build_configs}",
        f"env.mshab_task={args.mshab_task}",
        f"env.mshab_obj={args.mshab_obj}",
        f"device={args.device}",
        f"buffer.storage_device={args.device}",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        config = compose(config_name="configs", overrides=overrides)

    env = None
    try:
        env = ManiSkillVecEnv(config.env)
        spaces = env.observation_space.spaces
        missing = [key for key in graph_keys() if key not in spaces]
        if missing:
            raise AssertionError(f"observation space is missing graph keys: {missing}")
        stale = [k for k in ("graph_node_app", "graph_node_uid") if k in spaces]
        if stale:
            raise AssertionError(f"retired graph keys still emitted: {stale}")
        node_capacity = int(config.env.graph.n_max)
        edge_capacity = int(config.env.graph.e_max)
        if env.observation_space["graph_node_ent"].shape != (node_capacity,):
            raise AssertionError(
                f"expected graph_node_ent shape {(node_capacity,)}, got "
                f"{env.observation_space['graph_node_ent']}"
            )
        if env.observation_space["graph_edge_rel"].shape != (edge_capacity,):
            raise AssertionError(
                f"expected graph_edge_rel shape {(edge_capacity,)}, got "
                f"{env.observation_space['graph_edge_rel']}"
            )

        agent = Dreamer(config.model, env.observation_space, env.action_space).to(args.device)
        state = agent.get_initial_state(args.num_envs)
        action = torch.zeros(
            args.num_envs, env.action_space.shape[0], device=args.device
        )
        reset = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
        obs, done = env.step(action, reset)
        action, state = agent.act(obs, state)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        max_nodes = max_edges = target_frames = 0
        for _ in range(args.steps):
            obs, done = env.step(action, done)
            action, state = agent.act(obs, state)
            max_nodes = max(max_nodes, int(obs["graph_node_ent"].ne(0).sum(-1).max()))
            max_edges = max(max_edges, int(obs["graph_edge_rel"].ne(0).sum(-1).max()))
            target_frames += int(obs["graph_node_target"].any(-1).sum())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print("MS-HAB graph smoke: OK")
        print(f"  env frames: {args.num_envs * args.steps}")
        print(f"  graph+env+policy: {args.num_envs * args.steps / elapsed:.2f} frames/s")
        print(f"  vertices max: {max_nodes}/{node_capacity}")
        print(f"  real facts max: {max_edges}/{edge_capacity}")
        print(f"  target-flagged frames: {target_frames}/{args.num_envs * args.steps}")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
