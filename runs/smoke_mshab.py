"""Short MS-HAB graph-to-policy smoke test; no replay or optimizer updates."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import torch
from hydra import compose, initialize_config_dir


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dreamer import Dreamer  # noqa: E402
from envs.maniskill import ManiSkillVecEnv  # noqa: E402
from graph import graph_keys, graph_schema  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--build-configs", type=int, default=4)
    parser.add_argument("--mshab-task", default="prepare_groceries")
    parser.add_argument("--mshab-obj", default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--graph-only", action="store_true")
    parser.add_argument("--graph-simple", action="store_true")
    parser.add_argument("--graph-state-mode", default="pooled",
                        choices=("pooled", "slots"))
    args = parser.parse_args()
    if args.graph_only and args.graph_simple:
        parser.error("--graph-only and --graph-simple are mutually exclusive")

    overrides = [
        "env=mshab",
        "model=size50M_graph",
        f"env.env_num={args.num_envs}",
        f"env.num_build_configs={args.build_configs}",
        f"env.mshab_task={args.mshab_task}",
        f"env.mshab_obj={args.mshab_obj}",
        f"device={args.device}",
        f"buffer.storage_device={args.device}",
        f"model.graph_only_latent={str(args.graph_only).lower()}",
        f"model.graph_simple={str(args.graph_simple).lower()}",
        f"model.graph.state_mode={args.graph_state_mode}",
    ]
    dino_weights = os.environ.get("DINO_WEIGHTS")
    if dino_weights and not args.graph_simple:
        overrides.append(f"env.graph.dino_weights={dino_weights}")
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        config = compose(config_name="configs", overrides=overrides)

    env = None
    try:
        env = ManiSkillVecEnv(config.env)
        spaces = env.observation_space.spaces
        schema = graph_schema(args.graph_simple, args.graph_state_mode)
        missing = [key for key in graph_keys(schema) if key not in spaces]
        if missing:
            raise AssertionError(f"observation space is missing graph keys: {missing}")
        if args.graph_simple:
            # Pooled graph-simple keeps boxes and drops identity; slot mode the
            # reverse. Either way appearance is gone, and the key the *other*
            # simple schema owns must not be present.
            stale = ["graph_node_app"]
            stale.extend(
                ["graph_node_uid"] if args.graph_state_mode == "pooled"
                else ["graph_node_bbox", "graph_node_centroid"]
            )
            found = [key for key in stale if key in spaces]
            if found:
                raise AssertionError(
                    f"graph schema {schema} must not emit: {found}"
                )
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
        if args.graph_only and "stoch" in state:
            raise AssertionError("graph-only state must not contain stoch")
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

        mode = (
            "graph-only" if args.graph_only
            else "graph-simple" if args.graph_simple
            else "hybrid"
        )
        print(f"MS-HAB {mode} smoke: OK")
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
