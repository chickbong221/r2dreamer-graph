#!/bin/bash
#SBATCH --job-name=r2d-hab-a-base
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0
#SBATCH --output=/home/%u/output/%x_%j.out
#SBATCH --error=/home/%u/output/%x_%j.err

# Experiment A -- five tidy_house pick objects, one named training scene.
#
# 8M steps on the five objects, then 5M more on the held-out 008_pudding_box
# from A's best eligible checkpoint, logged under finetune/*. The transfer
# stage reports its results but saves no checkpoint of its own.
#
# The evaluation panel is 25 environments, five per object, fixed across
# evaluations. No lighting conditions here: A varies the object, B varies the
# scene, C varies the illumination inside B's panel.
#
# Selection stays on eval/success_once -- A trains and evaluates in the same
# scene, so there is no held-out half for that number to pool in. Eligibility
# starts at 6M of the 8M budget.
#
# Everything else is the shipped default: batch 32 x 64, train_ratio 64,
# evaluation every 50k steps.
#
# Deliberately no `set -e`: a run that dies must not take the rest with it.

echo "================================="
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Arm: A, baseline (pure DreamerV3, no graph, no progress)"
echo "Budget: 8M steps, 100M model, checkpoint eligible from 6M"
echo "Selection: eval/success_once"
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

# Print initial GPU state
nvidia-smi

# Monitor GPU every 100 seconds in background
nvidia-smi -l 100 > $HOME/output/gpu_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!

# Generate timestamp properly
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

python train.py \
  env=mshab_pick_a \
  model=size100M \
  env.steps=8000000 \
  env.obs_mode=rgb \
  checkpoint.enabled=true \
  checkpoint.start_step=6000000 \
  checkpoint.metric=eval/success_once \
  checkpoint.tiebreak='' \
  checkpoint.path=$CKPT_DIR/${TIMESTAMP}_A-five-objects-baseline.pt \
  finetune.enabled=true \
  finetune.steps=5000000 \
  wandb.group=mshab_tidy_house_pick_A \
  wandb.name=A-five-objects-baseline-transfer \
  logdir=$HOME/logdir/r2dreamer-graph/$TIMESTAMP/A-five-objects-baseline-transfer

# Stop GPU monitor
kill $GPU_MONITOR_PID

echo "Job finished"
