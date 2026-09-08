#!/bin/bash
#SBATCH --job-name=r2d-hab-b-base
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0
#SBATCH --output=/home/%u/output/%x_%j.out
#SBATCH --error=/home/%u/output/%x_%j.err

# Experiment B -- one object, five training scenes, forty-two unseen ones.
#
# 004_sugar_box in five arrangements of one apartment for 8M steps, evaluated
# in thirty arrangements of the other two, fifteen from each. 125 training
# environments, 25 per scene; the split itself is frozen in
# configs/scenes/mshab_pick_b.json and checked against the installed task
# plans above, so a drifted manifest stops the run rather than quietly moving
# the experiment.
#
# The panel is 70 environments in one simulator: 30 unseen scenes at nominal
# light, 10 training-scene cases at nominal light, and C's 30 matched cases at
# 0.4 / 1.0 / 2.0 -- all thirty on the original single training scene, which
# five-scene training does not expand.
#
# The three parts are never pooled: eval/* is the 30 unseen scenes alone, so
# a policy that generalises to nothing reports zero rather than the 25% that
# averaging the ten training cases in would give.
#
# The checkpoint is selected on eval_scene/training/success_once, those ten
# normal-light training-scene cases -- deliberately not eval/success_once,
# because selecting on the number B reports would pick whichever checkpoint
# got luckiest on the test set. Eligibility starts at 6M of the 8M budget, so
# the selection window is the last two million steps rather than the final
# evaluation on its own.
#
# No transfer stage: the held-out object belongs to A.
#
# Everything else is the shipped default: batch 32 x 64, train_ratio 64,
# evaluation every 50k steps.
#
# Deliberately no `set -e`: a run that dies must not take the rest with it.

echo "================================="
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Arm: B, baseline (pure DreamerV3, no graph, no progress)"
echo "Budget: 8M steps, 100M model, checkpoint eligible from 6M"
echo "Selection: eval_scene/training/success_once"
echo "================================="

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dreamer

export NVIDIA_USERSPACE_VERSION=570.133.20
export NVIDIA_USERSPACE_DIR=$HOME/nvidia-userspace/NVIDIA-Linux-x86_64-${NVIDIA_USERSPACE_VERSION}

cd "$NVIDIA_USERSPACE_DIR"

ln -sf libGLX_nvidia.so.${NVIDIA_USERSPACE_VERSION} libGLX_nvidia.so.0
ln -sf libEGL_nvidia.so.${NVIDIA_USERSPACE_VERSION} libEGL_nvidia.so.0

cat > "$NVIDIA_USERSPACE_DIR/nvidia_icd_egl.json" <<EOF
{
    "file_format_version": "1.0.1",
    "ICD": {
        "library_path": "$NVIDIA_USERSPACE_DIR/libEGL_nvidia.so.0",
        "api_version": "1.3.0"
    }
}
EOF

export LD_LIBRARY_PATH=$NVIDIA_USERSPACE_DIR:${LD_LIBRARY_PATH:-}
export VK_DRIVER_FILES=$NVIDIA_USERSPACE_DIR/nvidia_icd_egl.json
export VK_ICD_FILENAMES=$NVIDIA_USERSPACE_DIR/nvidia_icd_egl.json

# Move to project directory
cd $HOME/projects/r2dreamer-graph

export WANDB_API_KEY="b1d6eed8871c7668a889ae74a621b5dbd2f3b070"
export MS_ASSET_DIR=/mnt/data/tuannl

# The selected model lands outside the log tree, so clearing a
# logdir cannot take the checkpoint every later number is read from.
CKPT_DIR=$MS_ASSET_DIR/mshab_transfer_checkpoint

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

mkdir -p $HOME/output "$CKPT_DIR"

# B's frozen five/42 scene split, checked against the installed task plans
# before the budget is spent: every named scene present, the two halves
# disjoint, and the split still the one the rule produces.
python -m scenegraph.tools.freeze_scene_split --check \
  --task tidy_house --subtask pick --obj 004_sugar_box --split train || exit 1

# Print initial GPU state
nvidia-smi

# Monitor GPU every 100 seconds in background
nvidia-smi -l 100 > $HOME/output/gpu_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!

# Generate timestamp properly
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

python train.py \
  env=mshab_pick_b \
  model=size100M \
  env.steps=8000000 \
  env.obs_mode=rgb \
  checkpoint.enabled=true \
  checkpoint.start_step=6000000 \
  checkpoint.metric=eval_scene/training/success_once \
  checkpoint.tiebreak='' \
  checkpoint.path=$CKPT_DIR/${TIMESTAMP}_B-scenes-and-lighting-baseline.pt \
  finetune.enabled=false \
  wandb.group=mshab_tidy_house_pick_B \
  wandb.name=B-scenes-and-lighting-baseline \
  logdir=$HOME/logdir/r2dreamer-graph/$TIMESTAMP/B-scenes-and-lighting-baseline

# Stop GPU monitor
kill $GPU_MONITOR_PID

echo "Job finished"
