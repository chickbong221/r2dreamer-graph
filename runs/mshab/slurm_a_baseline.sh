#!/bin/bash
#SBATCH --job-name=r2d-hab-a-base
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0
#SBATCH --output=/home/%u/output/%x_%j.out
#SBATCH --error=/home/%u/output/%x_%j.err

# Experiment A, graph-free control: pure DreamerV3.
#
# `size50M` and `size50M_graph_simple` are identical on every RSSM setting
# -- deter, hidden, units, depth, discrete, act, norm -- so the plain preset
# is already the matched control and needs no graph overrides at all. It
# inherits graph.enabled=False and progress.enabled=False from the base
# preset, so both extensions are off by construction rather than by being
# switched off, and no whitelist or entity vocabulary applies.
#
# obs_mode drops to rgb: with the graph off nothing consumes segmentation.
#
# Same evaluation panel as the graph arm: 25 environments, five per object,
# and the same 10M + 5M transfer budget, so the two A arms are matched.
#
# Deliberately no `set -e`: a run that dies must not take the rest with it.

echo "================================="
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Arm: A, baseline (pure DreamerV3, no graph, no progress)"
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

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

mkdir -p $HOME/output

# Print initial GPU state
nvidia-smi

# Monitor GPU every 100 seconds in background
nvidia-smi -l 100 > $HOME/output/gpu_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!

# Generate timestamp properly
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Main training only (disabled).
# python train.py \
#   env=mshab_pick_a \
#   model=size50M \
#   env.obs_mode=rgb \
#   checkpoint.enabled=true \
#   checkpoint.metric=eval/success_once \
#   checkpoint.tiebreak='' \
#   finetune.enabled=false \
#   wandb.group=mshab_tidy_house_pick_A \
#   wandb.name=A-five-objects-baseline \
#   logdir=$HOME/logdir/r2dreamer-graph/$TIMESTAMP/A-five-objects-baseline

# Train 10M, then transfer for 5M from A's best eligible checkpoint.
python train.py \
  env=mshab_pick_a \
  model=size50M \
  env.obs_mode=rgb \
  checkpoint.enabled=true \
  checkpoint.metric=eval/success_once \
  checkpoint.tiebreak='' \
  finetune.enabled=true \
  finetune.steps=5000000 \
  wandb.group=mshab_tidy_house_pick_A \
  wandb.name=A-five-objects-baseline-transfer \
  logdir=$HOME/logdir/r2dreamer-graph/$TIMESTAMP/A-five-objects-baseline-transfer

# Stop GPU monitor
kill $GPU_MONITOR_PID

echo "Job finished"
