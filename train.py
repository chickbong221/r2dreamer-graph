import atexit
import gc
import pathlib
import sys
import warnings

import hydra
import torch

import tools
from buffer import Buffer
from checkpointing import CheckpointConfig, CheckpointError, Checkpointer, load_checkpoint, run_identity
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
    validate_finetune(config, checkpoint_config)
    if config.finetune.enabled or config.finetune.only:
        validate_transfer_sources(config)

    if config.finetune.only:
        run_finetune(config, logger, logdir, 0, None)
        return

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
    # Only eligible, improved evaluations can write the rolling best.
    try:
        completed_step = policy_trainer.begin(agent)
    finally:
        train_envs.close()
        if eval_envs is not None:
            eval_envs.close()
    # A cancellation/exception leaves this path before transfer starts.
    if config.finetune.enabled:
        source_path = checkpointer.path if checkpointer is not None and checkpointer.n_saved else None
        # Free training replay, optimizer and simulator before the next stage.
        del policy_trainer, agent, replay_buffer, train_envs, eval_envs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        run_finetune(config, logger, logdir, completed_step, source_path)


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
        build_temporal_vocab,
    )

    graph = config.model.graph
    if not graph.enabled:
        return run_identity(whitelist_dir="", schedule_label="dreamerv3",
                            n_cams=len(list(config.env.cameras or [])))
    source = task_schedule_source(
        envs, str(_repo_path(config.model.progress.schedule_dir)))
    whitelist_dir = source.whitelist_dir if source else ""
    return run_identity(
        whitelist_dir=whitelist_dir,
        schedule_path=source.schedule_path if source else "",
        affordance_path=source.affordance_path if source else "",
        schedule_label=source.label if source else "",
        n_max=int(graph.n_max), e_max=int(graph.e_max),
        n_cams=len(list(config.env.cameras or [])),
        disable_object_object_relations=bool(
            config.env.graph.get("disable_object_object_relations", False)),
        protected_pick_fifo=bool(
            getattr(envs, "_is_mshab", False)
            and getattr(envs, "_mshab_subtask", "") == "pick"
            and config.env.graph.get("use_target_flag", True)),
        entity_ids=build_entity_vocab(whitelist_dir).token_to_id
        if whitelist_dir else {},
        relation_ids=build_relation_vocab().token_to_id,
        absolute_ids=build_absolute_vocab().token_to_id,
        temporal_ids=build_temporal_vocab().token_to_id,
    )


def validate_finetune(config, checkpoint_config):
    ft = config.finetune
    if not (ft.enabled or ft.only):
        return
    if not str(config.env.task).startswith("maniskill_PickSubtaskTrain"):
        raise ValueError("automatic object transfer is scoped to MS-HAB Pick")
    if not ft.objects or len(set(ft.objects)) != len(ft.objects) or int(ft.steps) <= 0:
        raise ValueError("fine-tuning needs distinct target objects and a positive budget")
    original = set(config.env.mshab_objects or [config.env.mshab_obj])
    if original.intersection(ft.objects):
        raise ValueError("transfer objects must differ from the main training object set")
    if int(ft.eval_episode_num) <= 0 or int(ft.eval_episode_num) % len(ft.objects):
        raise ValueError("transfer evaluation must allocate equal episodes per object")
    if ft.initialization not in ("pretrained", "scratch"):
        raise ValueError("finetune.initialization must be pretrained or scratch")
    if len(config.env.train_build_config_ids) != 1:
        raise ValueError("this transfer experiment keeps one fixed training scene")
    if ft.initialization == "pretrained" and not ft.checkpoint_path:
        if ft.only:
            raise ValueError("standalone transfer needs finetune.checkpoint_path")
        if not checkpoint_config.enabled or int(config.env.steps) < checkpoint_config.start_step:
            raise ValueError("automatic pretrained transfer needs a main run with eligible best-checkpoint saving")
        if int(config.env.eval_episode_num) <= 0:
            raise ValueError("automatic pretrained transfer needs main evaluations to select its checkpoint")


def run_finetune(config, logger, logdir, base_step, source_path):
    """Fresh replay/optimizer, pretrained weights or an explicitly chosen scratch arm."""
    from omegaconf import OmegaConf

    transfer = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    ft = config.finetune
    transfer.env.mshab_objects = list(ft.objects)
    transfer.env.mshab_obj = str(ft.objects[0])
    transfer.env.steps = int(ft.steps)
    transfer.env.eval_build_config_ids = list(transfer.env.train_build_config_ids)
    transfer.env.eval_num_build_configs = 1
    transfer.env.eval_episode_num = int(ft.eval_episode_num)
    transfer.env.eval_even_build_configs = False
    transfer.env.eval_panel = "objects"
    transfer.env.eval_lighting.enabled = False
    transfer.trainer.steps = int(ft.steps)
    transfer.trainer.eval_episode_num = int(ft.eval_episode_num)
    transfer.checkpoint.enabled = False
    stage_logger = tools.StageLogger(logger, "finetune", base_step)
    logger.log_hydra_config(transfer, name="finetune/config", step=base_step)
    tools.set_seed_everywhere(transfer.seed)
    train_envs, eval_envs, obs_space, act_space = make_envs(transfer.env)
    try:
        agent = Dreamer(transfer.model, obs_space, act_space)
        agent.attach_task_schedule(train_envs)
        pretrained = ft.initialization == "pretrained"
        if pretrained:
            path = str(ft.checkpoint_path or source_path or "")
            if not path:
                raise CheckpointError("main run produced no eligible best checkpoint; transfer not started")
            payload = load_checkpoint(path, checkpoint_identity(transfer, train_envs))
            agent.load_state_dict(payload["model"], strict=True)
            stage_logger.scalar("source_checkpoint_step", payload["checkpoint"]["step"])
            del payload
        agent = agent.to(transfer.device)
        stage_logger.scalar("pretrained", int(pretrained))
        stage_logger.scalar("budget", int(ft.steps))
        stage_logger.write(0)
        trainer = OnlineTrainer(transfer.trainer, Buffer(transfer.buffer), stage_logger,
                                logdir, train_envs, eval_envs, checkpointer=None)
        trainer.begin(agent)
    finally:
        train_envs.close()
        if eval_envs is not None:
            eval_envs.close()


def validate_transfer_sources(config):
    """Catch missing transfer data before spending the main training budget."""
    from mani_skill import ASSET_DIR
    from mshab.envs.planner import plan_data_from_file
    from envs.instruction import InstructionTable
    from envs.maniskill import _repo_path

    if config.finetune.checkpoint_path and not pathlib.Path(config.finetune.checkpoint_path).is_file():
        raise FileNotFoundError(config.finetune.checkpoint_path)
    root = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
    group = str(config.env.mshab_task)
    splits = {str(config.env.split), str(config.env.eval_split or config.env.split)}
    table = InstructionTable(_repo_path(config.env.instruction_table))
    for obj in config.finetune.objects:
        table.row("pick", f"actor:{obj}")
        if config.model.graph.enabled:
            directory = config.env.graph.whitelist_dir or f"scenegraph/configs/subtask_whitelists/{group}"
            path = _repo_path(directory) / f"pick_{obj}.json"
            if not path.is_file():
                raise FileNotFoundError(f"transfer whitelist missing: {path}")
        for split in splits:
            path = root / "task_plans" / group / "pick" / split / f"{obj}.json"
            plans = plan_data_from_file(path).plans
            scenes = {str(p.build_config_name) for p in plans}
            if not set(config.env.train_build_config_ids).issubset(scenes):
                raise ValueError(f"{obj} has no transfer plans in the training scene ({split})")
            spawn = root / "spawn_data" / group / "pick" / split / "spawn_data.pt"
            if not spawn.is_file():
                raise FileNotFoundError(f"transfer spawns missing: {spawn}")


if __name__ == "__main__":
    main()
