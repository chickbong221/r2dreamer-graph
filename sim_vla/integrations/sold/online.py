"""Stage 3: upstream's online SOLD, with SmolVLA attached.

A launcher, not an algorithm. It builds the real ``SOLDModule`` and the real
Lightning ``Trainer`` from SOLD's own config groups, hands the module the
weights Stages 1 and 2 produced, attaches the flow policy, and calls
``trainer.fit``. The collection loop, the ring-buffer replay, the dynamics and
reward updates, the imagined rollout, the lambda returns, the critic's
regularized objective and the target EMA are all upstream's.

Two things are checked before anything runs:

* the environment is the one the demonstrations came from -- camera,
  resolution, action width and task id
* the imagined action carries gradient into the adapter and the action expert,
  because an actor update that silently learns nothing costs exactly as much as
  one that works

``smolvla.enabled: false`` skips the attachment and this is the upstream run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from ..vendor import SOLD as VENDOR
from . import env as sold_env
from . import model as build
from . import stages


def build_environment(run, *, seed: int = 0):
    world = dict(run.cfg["world_model"])
    env_cfg = dict(world.get("env") or {})
    suite = str(env_cfg.get("suite") or "maniskill")
    if suite != "maniskill":
        raise SystemExit(
            f"world_model.env.suite={suite!r}: this integration builds the "
            "target ManiSkill task from the dataset's own metadata. SOLD's "
            "own suites (mof, gym, dmcontrol) are reached through upstream's "
            "train_sold.py, not through here.")
    return sold_env.SoldManiSkillEnv(
        run.source.metadata, image_size=run.image_size,
        camera=run.source.images.cameras[0],
        max_episode_steps=int(env_cfg.get("max_episode_steps") or 150),
        action_repeat=int(env_cfg.get("action_repeat") or 1), seed=int(seed))


def run_online(run, *, steps: Optional[int] = None,
               log: Optional[Callable[[str], None]] = print) -> Dict[str, Any]:
    """Collect, update and imagine, with upstream's module and trainer."""
    import torch

    cfg = dict(run.cfg)
    world = dict(cfg["world_model"])
    if steps:
        world["max_steps"] = int(steps)
        cfg["world_model"] = world

    environment = build_environment(run, seed=int(cfg.get("seed", 0)))
    sold_env.check_contract(environment, run.source, log=log)

    module = build.sold_module(cfg, environment, device=run.device)
    # The weights Stages 1 and 2 produced. `sold_module` builds fresh modules
    # of the same shapes, so they are copied across rather than rebuilt -- the
    # alternative is a second construction path that can drift from the first.
    module.autoencoder.load_state_dict(run.parts["autoencoder"].state_dict())
    module.dynamics_predictor.load_state_dict(run.parts["dynamics"].state_dict())
    module.reward_predictor.load_state_dict(run.parts["reward"].state_dict())
    module.actor.load_state_dict(run.parts["actor"].state_dict())
    module.critic.load_state_dict(run.parts["critic"].state_dict())
    module.critic_target.load_state_dict(run.parts["critic_target"].state_dict())

    policy = None
    if bool(cfg["smolvla"].get("enabled", True)):
        if run.actor is None:
            raise SystemExit(
                "smolvla.enabled is true but no actor was built. Run stage 2 "
                "first, or point --imitation-checkpoint at its output; "
                "Stage 3 does not train an adapter from scratch.")
        policy = stages.attach(module, run.actor, run.converter, cfg, log=log)
    else:
        module.attach_policy(None)
        if log:
            log("[sold:3] SmolVLA disabled: this is the upstream module, with "
                "the Gaussian actor and its entropy bonus")

    with VENDOR.active():
        from lightning import Trainer
        from lightning.pytorch.loggers import TensorBoardLogger

        trainer = Trainer(
            max_steps=-1, max_epochs=-1, enable_checkpointing=False,
            devices=1, accelerator="auto",
            logger=TensorBoardLogger(save_dir=str(run.out_dir),
                                     name=str(cfg["task"]["env_id"])))
        trainer.fit(module)

    return {"module": module, "policy": policy,
            "usage": policy.usage() if policy is not None else {}}
