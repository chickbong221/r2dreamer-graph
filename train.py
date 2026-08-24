import atexit
import pathlib
import sys
import warnings

import hydra
import torch

import tools
from buffer import Buffer
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

    policy_trainer = OnlineTrainer(config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs)
    policy_trainer.begin(agent)
    # No checkpoint is written. Corrected relation labels have the same ids
    # and different geometric meaning, so resuming across the change would
    # train one label to mean two things. Runs are compared from metrics.


if __name__ == "__main__":
    main()
