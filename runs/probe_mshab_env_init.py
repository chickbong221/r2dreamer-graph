"""Construct the R2Dreamer MS-HAB adapter without replay, logging, or a model.

Run separate processes with and without ``--seed-before-env`` to detect whether
initializing PyTorch/CUDA before SAPIEN changes Vulkan camera initialization.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from hydra import compose, initialize_config_dir


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--build-configs", type=int, default=63)
    parser.add_argument("--obs-mode", default="rgb")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-before-env", action="store_true")
    args = parser.parse_args()

    overrides = [
        "env=mshab",
        "model=size50M_graph",
        "model.graph.enabled=false",
        f"env.obs_mode={args.obs_mode}",
        f"env.env_num={args.num_envs}",
        f"env.num_build_configs={args.build_configs}",
        "env.mshab_task=prepare_groceries",
        "env.mshab_obj=all",
        f"device={args.device}",
        f"seed={args.seed}",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        config = compose(config_name="configs", overrides=overrides)

    import mani_skill
    import mshab
    import sapien
    import torch

    print(f"mani_skill: {mani_skill.__file__}", flush=True)
    print(f"mshab: {mshab.__file__}", flush=True)
    print(f"sapien: {sapien.__file__}", flush=True)
    print(f"torch CUDA initialized before probe: {torch.cuda.is_initialized()}", flush=True)

    if args.seed_before_env:
        import tools

        tools.set_seed_everywhere(args.seed)
        print(
            "Applied tools.set_seed_everywhere before environment; "
            f"CUDA initialized={torch.cuda.is_initialized()}",
            flush=True,
        )

    from envs.maniskill import ManiSkillVecEnv

    print(
        "Creating R2Dreamer Pick/prepare_groceries env "
        f"(envs={args.num_envs}, obs={args.obs_mode}, "
        f"seed_before_env={args.seed_before_env})...",
        flush=True,
    )
    env = None
    try:
        env = ManiSkillVecEnv(config.env)
        print("R2DREAMER ENV: OK", flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
