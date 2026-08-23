"""Construct the R2Dreamer MS-HAB adapter without replay, logging, or a model.

Run separate processes with and without ``--seed-before-env`` to detect whether
initializing PyTorch/CUDA before SAPIEN changes Vulkan camera initialization.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile

from hydra import compose, initialize_config_dir


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--build-configs", type=int, default=63)
    parser.add_argument("--obs-mode", default="rgb")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-before-env", action="store_true")
    parser.add_argument(
        "--preamble",
        choices=("none", "imports", "buffer", "logger", "wandb"),
        default="none",
        help="Cumulatively reproduce training setup before constructing SAPIEN.",
    )
    args = parser.parse_args()

    overrides = [
        "env=mshab",
        "model=size50M",
        f"env.obs_mode={args.obs_mode}",
        f"env.env_num={args.num_envs}",
        f"env.num_build_configs={args.build_configs}",
        "env.mshab_task=prepare_groceries",
        "env.mshab_obj=all",
        f"device={args.device}",
        f"seed={args.seed}",
        "batch_size=8",
        "batch_length=64",
        "buffer.storage_device=cpu",
        "buffer.max_size=500000",
        f"wandb.enabled={'true' if args.preamble == 'wandb' else 'false'}",
        "wandb.name=r2d-env-init-probe",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        config = compose(config_name="configs", overrides=overrides)

    buffer_type = None
    if args.preamble != "none":
        # Match train.py's module imports before ManiSkill is imported.
        import tools as training_tools
        from buffer import Buffer
        from dreamer import Dreamer  # noqa: F401
        from envs import make_envs
        from trainer import OnlineTrainer  # noqa: F401

        buffer_type = Buffer
        print("Imported the complete R2Dreamer training stack", flush=True)
    else:
        from envs import make_envs

    import torch

    print(f"torch CUDA initialized before probe: {torch.cuda.is_initialized()}", flush=True)

    if args.seed_before_env:
        import tools as training_tools

        training_tools.set_seed_everywhere(args.seed)
        print(
            "Applied tools.set_seed_everywhere before environment; "
            f"CUDA initialized={torch.cuda.is_initialized()}",
            flush=True,
        )

    logger = None
    logger_dir = None
    if args.preamble in ("logger", "wandb"):
        logger_dir = tempfile.TemporaryDirectory(prefix="r2d-env-probe-")
        logger = training_tools.Logger(
            pathlib.Path(logger_dir.name), wandb_config=config.wandb
        )
        logger.log_hydra_config(config)
        print(
            "Constructed the logger before environment "
            f"(wandb={args.preamble == 'wandb'}); "
            f"CUDA initialized={torch.cuda.is_initialized()}",
            flush=True,
        )

    replay = None
    if args.preamble in ("buffer", "logger", "wandb"):
        replay = buffer_type(config.buffer)
        print(
            "Constructed the CPU-backed replay buffer before environment; "
            f"CUDA initialized={torch.cuda.is_initialized()}",
            flush=True,
        )

    print(
        "Creating R2Dreamer Pick/prepare_groceries env "
        f"(envs={args.num_envs}, obs={args.obs_mode}, "
        f"seed_before_env={args.seed_before_env}, preamble={args.preamble})...",
        flush=True,
    )
    env = None
    try:
        env, _, _, _ = make_envs(config.env)
        print("R2DREAMER ENV: OK", flush=True)
        import mani_skill
        import mshab
        import sapien

        print(f"mani_skill: {mani_skill.__file__}", flush=True)
        print(f"mshab: {mshab.__file__}", flush=True)
        print(f"sapien: {sapien.__file__}", flush=True)
    finally:
        if env is not None:
            env.close()
        if logger is not None:
            logger.close()
        if logger_dir is not None:
            logger_dir.cleanup()


if __name__ == "__main__":
    main()
