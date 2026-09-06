#!/bin/bash
#SBATCH --job-name=r2d-hab-base
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0
#SBATCH --output=/home/%u/output/%x_%j.out
#SBATCH --error=/home/%u/output/%x_%j.err

# Both MS-HAB Pick experiments on the graph-free control, A then B.
#
# `size50M` and `size50M_graph_simple` are identical on every RSSM setting
# -- deter, hidden, units, depth, discrete, act, norm -- so the plain preset
# is already the matched control and needs no graph overrides at all. It
# inherits graph.enabled=False and progress.enabled=False from the base
# preset, so both extensions are off by construction rather than by being
# switched off, and no whitelist or entity vocabulary applies.
#
# obs_mode drops to rgb: with the graph off nothing consumes segmentation.
# C still works -- it compares the policy's RGB, which this arm renders
# exactly as the graph arm does.
#
# Same experiments, panels and budgets as the graph arm: A for 10M + 5M
# transfer, B for 10M with no transfer stage.
#
# The two commands are the ones in slurm_a_baseline.sh and
# slurm_b_baseline.sh, which stay for launching a single arm.
#
# Deliberately no `set -e`: a run that dies must not cancel the one after it.

echo "================================="
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Arm: baseline (pure DreamerV3, no graph, no progress), experiments A then B"
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

echo "=== Experiment A ==="
python train.py \
  env=mshab_pick_a \
  model=size50M \
  env.obs_mode=rgb \
  checkpoint.enabled=true \
  checkpoint.metric=eval/success_once \
  checkpoint.tiebreak='' \
  checkpoint.path=$CKPT_DIR/${TIMESTAMP}_A-five-objects-baseline.pt \
  finetune.enabled=true \
  finetune.steps=5000000 \
  wandb.group=mshab_tidy_house_pick_A \
  wandb.name=A-five-objects-baseline-transfer \
  logdir=$HOME/logdir/r2dreamer-graph/$TIMESTAMP/A-five-objects-baseline-transfer

echo "=== Experiment B ==="
python train.py \
  env=mshab_pick_b \
  model=size50M \
  env.obs_mode=rgb \
  checkpoint.enabled=true \
  checkpoint.metric=eval/success_once \
  checkpoint.tiebreak='' \
  checkpoint.path=$CKPT_DIR/${TIMESTAMP}_B-scenes-and-lighting-baseline.pt \
  finetune.enabled=false \
  wandb.group=mshab_tidy_house_pick_B \
  wandb.name=B-scenes-and-lighting-baseline \
  logdir=$HOME/logdir/r2dreamer-graph/$TIMESTAMP/B-scenes-and-lighting-baseline
# Stop GPU monitor
kill $GPU_MONITOR_PID

echo "Job finished"
