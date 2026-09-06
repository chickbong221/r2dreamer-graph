#!/bin/bash
#SBATCH --job-name=r2d-msk-b005
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0
#SBATCH --output=/home/%u/output/%x_%j.out
#SBATCH --error=/home/%u/output/%x_%j.err

echo "================================="
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Arm: beta=0.05 (warm-up 200k-700k)"
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

# Mined per task. Each directory holds the task_all.json the builder binds.
WL=scenegraph/configs/subtask_whitelists

mkdir -p $HOME/output

# Print initial GPU state
nvidia-smi

# Monitor GPU every 100 seconds in background
nvidia-smi -l 100 > $HOME/output/gpu_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!

# Generate timestamp properly
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# python train.py \
#   env=maniskill \
#   model=size50M_graph_simple \
#   env.task=maniskill_PullCubeTool-v1 \
#   env.graph.whitelist_dir=$WL/PullCubeTool-v1 \
#   model.progress.beta=0.05 \
#   wandb.group=maniskill_PullCubeTool-v1 \
#   wandb.name=PullCubeTool-v1-beta005 \
#   logdir=$HOME/logdir/r2dreamer-graph/$TIMESTAMP/PullCubeTool-v1-beta005

python train.py \
  env=maniskill \
  model=size50M_graph_simple \
  env.task=maniskill_PickCube-v1 \
  env.graph.whitelist_dir=$WL/PickCube-v1 \
  model.progress.beta=0.05 \
  wandb.group=maniskill_PickCube-v1 \
  wandb.name=PickCube-v1-beta005 \
  logdir=$HOME/logdir/r2dreamer-graph/$TIMESTAMP/PickCube-v1-beta005

python train.py \
  env=maniskill \
  model=size50M_graph_simple \
  env.task=maniskill_PegInsertionSide-v1 \
  env.graph.whitelist_dir=$WL/PegInsertionSide-v1 \
  model.progress.beta=0.05 \
  wandb.group=maniskill_PegInsertionSide-v1 \
  wandb.name=PegInsertionSide-v1-beta005 \
  logdir=$HOME/logdir/r2dreamer-graph/$TIMESTAMP/PegInsertionSide-v1-beta005

# python train.py \
#   env=maniskill \
#   model=size50M_graph_simple \
#   env.task=maniskill_PlugCharger-v1 \
#   env.graph.whitelist_dir=$WL/PlugCharger-v1 \
#   model.progress.beta=0.05 \
#   wandb.group=maniskill_PlugCharger-v1 \
#   wandb.name=PlugCharger-v1-beta005 \
#   logdir=$HOME/logdir/r2dreamer-graph/$TIMESTAMP/PlugCharger-v1-beta005

# python train.py \
#   env=maniskill \
#   model=size50M_graph_simple \
#   env.reward_mode=sparse \
#   'env.reward_fallback=[]' \
#   env.task=maniskill_PlugCharger-v1 \
#   env.graph.whitelist_dir=$WL/PlugCharger-v1 \
#   model.progress.beta=1.0 \
#   wandb.group=maniskill_PlugCharger-v1 \
#   wandb.name=PlugCharger-v1-sparse-beta1 \
#   logdir=$HOME/logdir/r2dreamer-graph/$TIMESTAMP/PlugCharger-v1-sparse-beta1

# Stop GPU monitor
kill $GPU_MONITOR_PID

echo "Job finished"
