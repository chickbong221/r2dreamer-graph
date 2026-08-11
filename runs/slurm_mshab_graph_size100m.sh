#!/bin/bash
#SBATCH --job-name=r2d-ms-g100
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0
#SBATCH --output=/home/%u/output/%x_%j.out
#SBATCH --error=/home/%u/output/%x_%j.err

set -euo pipefail

echo "================================="
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Method: graph on"
echo "================================="

# Activate conda.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dreamer

# Use the user-space NVIDIA libraries required by this cluster.
export NVIDIA_USERSPACE_VERSION=570.133.20
export NVIDIA_USERSPACE_DIR="$HOME/nvidia-userspace/NVIDIA-Linux-x86_64-${NVIDIA_USERSPACE_VERSION}"

cd "$NVIDIA_USERSPACE_DIR"

ln -sf "libGLX_nvidia.so.${NVIDIA_USERSPACE_VERSION}" libGLX_nvidia.so.0
ln -sf "libEGL_nvidia.so.${NVIDIA_USERSPACE_VERSION}" libEGL_nvidia.so.0

cat > "$NVIDIA_USERSPACE_DIR/nvidia_icd_egl.json" <<EOF
{
    "file_format_version": "1.0.1",
    "ICD": {
        "library_path": "$NVIDIA_USERSPACE_DIR/libEGL_nvidia.so.0",
        "api_version": "1.3.0"
    }
}
EOF

export LD_LIBRARY_PATH="$NVIDIA_USERSPACE_DIR:${LD_LIBRARY_PATH:-}"
export VK_DRIVER_FILES="$NVIDIA_USERSPACE_DIR/nvidia_icd_egl.json"
export VK_ICD_FILENAMES="$NVIDIA_USERSPACE_DIR/nvidia_icd_egl.json"

cd "$HOME/projects/r2dreamer-graph"

export WANDB_API_KEY="b1d6eed8871c7668a889ae74a621b5dbd2f3b070"
export MS_ASSET_DIR=/mnt/data/tuannl
export DINO_WEIGHTS=/home/tuannl/mnt_data/checkpoints/dinov2_vits14_reg4_pretrain.pth

# Compute nodes have no outbound network. This official DINOv2 checkpoint must
# be placed at DINO_WEIGHTS before the graph job is submitted.
if [[ ! -f "$DINO_WEIGHTS" ]]; then
  echo "Missing DINOv2 checkpoint: $DINO_WEIGHTS" >&2
  echo "Download dinov2_vits14_reg4_pretrain.pth to that path before sbatch." >&2
  exit 1
fi

mkdir -p "$HOME/output" "$HOME/logdir/r2dreamer-graph"

nvidia-smi
nvidia-smi --query-gpu=name,uuid,pci.bus_id,driver_version,memory.total --format=csv,noheader

# Record GPU state every 100 seconds and always stop the monitor on exit.
nvidia-smi -l 100 > "$HOME/output/gpu_${SLURM_JOB_ID}.log" &
GPU_MONITOR_PID=$!

cleanup() {
  if kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
    kill "$GPU_MONITOR_PID"
  fi
}
trap cleanup EXIT INT TERM

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

export CUBLAS_WORKSPACE_CONFIG=:4096:8

PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 python train.py \
  deterministic_run=true \
  env=mshab \
  model=size100M_graph \
  model.graph.enabled=true \
  model.graph.n_max=12 \
  model.amp_dtype=bfloat16 \
  env.obs_mode=rgb+segmentation \
  batch_size=32 \
  batch_length=64 \
  env.env_num=126 \
  env.train_ratio=64 \
  env.mshab_task=prepare_groceries \
  env.mshab_obj=all \
  env.num_build_configs=63 \
  env.graph.dino_weights="$DINO_WEIGHTS" \
  buffer.storage_device=cpu \
  buffer.max_size=500000 \
  trainer.steps=10000000 \
  trainer.eval_every=50000 \
  trainer.update_log_every=1500 \
  device=cuda:0 \
  wandb.enabled=true \
  wandb.project=RelRL \
  wandb.entity=letuanhf-hanoi-university-of-science-and-technology \
  wandb.group=mshab-prepare-groceries-size100m-b32-s0 \
  wandb.name=graph-size100m-b32-s0 \
  logdir="$HOME/logdir/r2dreamer-graph/$TIMESTAMP/mshab-graph-size100m-b32-s0" \
  seed=0

echo "Job finished"
