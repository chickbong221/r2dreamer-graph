import atexit
import pathlib
import sys
import warnings

import hydra
import torch

import tools
from buffer import Buffer
from checkpointing import CheckpointConfig, Checkpointer, run_identity
from dreamer import Dreamer
from envs import make_envs
from trainer import OnlineTrainer

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
# Conv shapes are fixed here -- one encoder shape, two decoder shapes -- so
# autotuning pays for itself and does not re-benchmark. Off under
# `deterministic_run`, which sets it back to False.
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    # Mirror stdout/stderr to a file under logdir while keeping console output.
    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)

    logger = tools.Logger(logdir, wandb_config=config.wandb)
    atexit.register(logger.close)
    # save config
    logger.log_hydra_config(config)

    # Before the envs and the agent, which are the expensive part: a run with
    # checkpointing on and no metric has to fail in seconds, not after a scene
    # build and however many steps it takes to reach the first evaluation.
    checkpoint_config = CheckpointConfig.from_mapping(
        config.get("checkpoint", None))
    checkpoint_config.validate()

    replay_buffer = Buffer(config.buffer)

    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
    )
    # Before .to(): the compiled schedule registers buffers, and they have to
    # move with the module.
    agent.attach_task_schedule(train_envs)
    agent = agent.to(config.device)

    checkpointer = None
    if checkpoint_config.enabled:
        checkpointer = Checkpointer(
            checkpoint_config, str(logdir),
            checkpoint_identity(config, train_envs))
        print(f"[checkpoint] rolling best on {checkpoint_config.metric!r} "
              f"from step {checkpoint_config.start_step} -> "
              f"{checkpointer.path}", flush=True)

    policy_trainer = OnlineTrainer(config.trainer, replay_buffer, logger,
                                   logdir, train_envs, eval_envs,
                                   checkpointer=checkpointer)
    # Nothing is written after this returns. One rolling best, claimed only by
    # an eligible evaluation that improved on the last: no final copy, no
    # milestone, nothing on cancellation. An interrupted run keeps the best it
    # had already earned, and gains no file it had not.
    policy_trainer.begin(agent)


def checkpoint_identity(config, envs):
    """What a saved model has to agree with before its weights mean anything.

    Read off the assets this run actually resolved rather than off the config,
    because the config can name a directory and the run can resolve a
    different one. Deliberately says nothing about the scene, the lighting or
    the episode count: an evaluation is allowed to change those, which is what
    lets Experiment C load Experiment B's checkpoint.
    """
    from envs.maniskill import _repo_path, task_schedule_source
    from scenegraph.adapters.graph_vocab import (
        build_absolute_vocab, build_entity_vocab, build_relation_vocab,
    )

    graph = config.model.graph
    source = task_schedule_source(
        envs, str(_repo_path(config.model.progress.schedule_dir)))
    whitelist_dir = source.whitelist_dir if source else ""
    return run_identity(
        whitelist_dir=whitelist_dir,
        schedule_path=source.schedule_path if source else "",
        schedule_label=source.label if source else "",
        n_max=int(graph.n_max), e_max=int(graph.e_max),
        n_cams=len(list(config.env.cameras or [])),
        entity_tokens=sorted(
            build_entity_vocab(whitelist_dir).token_to_id)
        if whitelist_dir else (),
        relation_tokens=sorted(build_relation_vocab().token_to_id),
        absolute_tokens=sorted(build_absolute_vocab().token_to_id),
    )


if __name__ == "__main__":
    main()
