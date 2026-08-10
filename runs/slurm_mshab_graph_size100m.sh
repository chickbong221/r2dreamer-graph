#!/bin/bash
#SBATCH --job-name=r2d-ms-g100
#SBATCH --partition=main
#SBATCH --nodelist=worker-0
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

vulkaninfo --summary

cd "$HOME/projects/r2dreamer-graph"

export WANDB_API_KEY="b1d6eed8871c7668a889ae74a621b5dbd2f3b070"
export MS_ASSET_DIR=/mnt/data/tuannl

mkdir -p "$HOME/output" "$HOME/logdir/r2dreamer-graph"

nvidia-smi

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

python train.py \
  env=mshab \
  model=size100M_graph \
  model.graph.enabled=true \
  model.graph.n_max=12 \
  batch_size=32 \
  batch_length=64 \
  env.env_num=32 \
  env.train_ratio=66 \
  env.mshab_task=prepare_groceries \
  env.mshab_obj=all \
  env.num_build_configs=63 \
  buffer.storage_device=cpu \
  buffer.max_size=500000 \
  trainer.steps=1010000 \
  trainer.eval_every=1000000 \
  trainer.update_log_every=5000 \
  device=cuda:0 \
  wandb.enabled=true \
  wandb.project=RelRL \
  wandb.entity=letuanhf-hanoi-university-of-science-and-technology \
  wandb.group=mshab-prepare-groceries-size100m-b32-s0 \
  wandb.name=graph-size100m-b32-s0 \
  logdir="$HOME/logdir/r2dreamer-graph/$TIMESTAMP/mshab-graph-size100m-b32-s0" \
  seed=0

echo "Job finished"
