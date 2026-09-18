"""Stage 3: upstream's online TD-MPC2, with SmolVLA attached.

This is a launcher, not an algorithm. It builds the same objects
``sim_vla/tdmpc2/train.py`` builds -- upstream's environment, ``Buffer``,
``Logger`` and ``OnlineTrainer`` -- and then attaches the flow policy to the
agent before ``trainer.train()`` runs. Everything about the loop, the replay,
the MPPI planner, the Q ensemble, the target updates and the checkpointing is
upstream's.

The one thing it adds is a refusal: the environment is checked against the
demonstrations' contract before anything trains. Stage 1 fitted an encoder to a
particular camera set at a particular resolution and Stage 2 fitted a policy to
a particular action width; a live environment that differs in any of those
hands the encoder pictures it has never seen, and the only symptom is that the
returns are bad.

``smolvla.enabled: false`` skips the attachment entirely and this is the
upstream run.
"""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from ..vendor import TDMPC2 as VENDOR
from . import agent as build
from . import data as demo_data
from .config import build_cfg
from .stages import attach, check_environment_contract


def _prepare_environment_variables() -> None:
    # The same two upstream's train.py sets before importing torch.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("LAZY_LEGACY_OP", "0")


def run_online(cfg: Mapping[str, Any], *, steps: Optional[int] = None,
               device: str = "cuda", agent=None, actor=None, source=None,
               converter=None, node=None,
               log: Optional[Callable[[str], None]] = print) -> Dict[str, Any]:
    """Collect, update and plan, with upstream's trainer.

    ``agent``/``actor``/``source`` are the objects Stages 1 and 2 produced when
    the whole pipeline runs in one process. Given none, the caller is expected
    to have loaded them from checkpoints first.
    """
    _prepare_environment_variables()
    import torch

    world = dict(cfg.get("world_model") or {})
    smolvla = dict(cfg.get("smolvla") or {})
    data_cfg = dict(cfg.get("data") or {})

    if source is None:
        source = demo_data.open_demos(
            str(data_cfg.get("dataset") or cfg["task"]["dataset"]),
            render_size=int(world.get("render_size", 64)),
            include_state=bool(world.get("include_state", False)),
            cameras=data_cfg.get("cameras"))
    if node is None:
        node = build_cfg({**world, "device": device,
                          "num_cameras": len(source.images.cameras),
                          "proprio_dim": source.proprio_dim
                          if source.include_state else 0},
                         obs_shape=source.obs_shape(),
                         action_dim=source.action_dim,
                         episode_length=source.episode_length)
    if steps:
        node.steps = int(steps)
    node.work_dir = Path(node.work_dir)

    with VENDOR.active():
        from common.buffer import Buffer
        from common.logger import Logger, print_run
        from common.seed import set_seed
        from envs import make_envs
        from trainer.online_trainer import OnlineTrainer
        from tdmpc2 import TDMPC2 as Agent

        set_seed(int(node.seed))
        manager = multiprocessing.Manager()
        video_path = node.work_dir / "eval_video"
        if bool(node.save_video_local):
            video_path.mkdir(parents=True, exist_ok=True)
        logger = Logger(node, manager)

        # The environment writes obs_shape, action_dim, episode_length and
        # seed_steps back onto the config -- upstream's own behaviour -- so
        # everything downstream reads the live numbers.
        env = make_envs(node, int(node.num_envs))
        eval_env = make_envs(node, int(node.num_eval_envs),
                             video_path=video_path, is_eval=True, logger=logger)
        print_run(node)

        check_environment_contract(env, source, node, log=log)

        if agent is None:
            agent = Agent(node)
        if converter is None:
            converter = build.build_converter(
                source.metadata, action_dim=source.action_dim,
                smolvla=smolvla, device=device)

        if bool(smolvla.get("enabled", True)):
            if actor is None:
                raise SystemExit(
                    "smolvla.enabled is true but no actor was supplied. Run "
                    "stage 2 first, or point --imitation-checkpoint at its "
                    "output; Stage 3 does not train an adapter from scratch.")
            attach(agent, actor, converter, cfg, log=log)
        else:
            agent.attach_policy(None)
            if log:
                log("[stage3] SmolVLA disabled: this is the upstream agent, "
                    "with the Gaussian prior at all five policy sites")

        trainer = OnlineTrainer(cfg=node, env=env, eval_env=eval_env,
                                agent=agent, buffer=Buffer(node), logger=logger)
        trainer.train()

    return {"agent": agent, "actor": actor, "cfg": node,
            "policy": getattr(agent, "latent_policy", None),
            "usage": (agent.latent_policy.usage()
                      if getattr(agent, "latent_policy", None) is not None
                      else {})}
