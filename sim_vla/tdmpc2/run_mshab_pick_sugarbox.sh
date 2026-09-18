#!/bin/bash
# TD-MPC2 on MS-HAB TidyHouse pick 004_sugar_box, the Experiment B setting:
# train in one apartment, evaluate in the first 20. Build configs are a prefix of
# the sorted names and the training scene sorts first, so it is inside the panel.
# Lands in r2dreamer-graph's arm-B wandb group so the curves sit side by side.
# Run from this directory; train.py reads config.yaml from the working directory.

seeds=(9351)

# Shared knobs
steps=10_000_000     # matches arm B; TD-MPC2 plans at every step, so cut this if too slow
num_envs=64
steps_per_update=16  # env steps per gradient update; total updates = steps / this
# 2 cameras x 128x128 x 3 stacked frames = 0.29 MB/step, so 50k steps ~ 15 GB.
# The buffer falls back to CPU RAM when it does not fit on the GPU.
buffer_size=50_000
render_size=128

# Wandb: same project/entity as r2dreamer-graph (configs/configs.yaml).
use_wandb=true
wandb_entity="letuanhf-hanoi-university-of-science-and-technology"
wandb_project="RelRL"
export WANDB_API_KEY="${WANDB_API_KEY:?export your W&B key first (see run_ms.sh)}"

# tidy_house pick 004_sugar_box #
for seed in ${seeds[@]}
do
    python train.py model_size=5 steps=$steps seed=$seed buffer_size=$buffer_size exp_name=tdmpc2 \
        env_id=PickSubtaskTrain-v0 env_type=gpu num_envs=$num_envs steps_per_update=$steps_per_update \
        control_mode=pd_joint_delta_pos obs=rgb include_state=true render_size=$render_size \
        mshab_task=tidy_house mshab_obj=004_sugar_box mshab_max_episode_steps=100 \
        mshab_num_build_configs=1 mshab_eval_num_build_configs=20 \
        num_eval_envs=20 eval_episodes_per_env=1 eval_reconfiguration_frequency=0 \
        wandb=$use_wandb wandb_entity=$wandb_entity wandb_project=$wandb_project \
        wandb_group=mshab_tidy_house_pick_B wandb_name=B-sugarbox-tdmpc2-$seed
done
