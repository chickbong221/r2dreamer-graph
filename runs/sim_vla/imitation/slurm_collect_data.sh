#!/bin/bash
#SBATCH --job-name=r2d-svla-collect
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0
#SBATCH --output=/home/%u/output/%x_%j.out
#SBATCH --error=/home/%u/output/%x_%j.err

# 1000 successful demos each for PickCube-v1, StackCube-v1 and
# PegInsertionSide-v1. PlaceSphere-v1 already has its 1000, from
# ../server1/slurm_collect_data.sh. collect.py keeps only episodes that succeed
# within 150 steps, so --num-traj counts successes. One dataset per task,
# shared by both arms: $OUT_DIR/<EnvId>/demos.h5.
#
# No `set -e`: a task that fails or runs out of seeds (collect.py exits 1 and
# prints a top-up command) must not cancel the tasks after it.

echo "================================="
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Collecting: PickCube-v1, StackCube-v1, PegInsertionSide-v1"
echo "================================="

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

# Run from the repo root: collect.py's default config paths and its
# `from envs.maniskill import ...` resolve against it.
cd $HOME/projects/r2dreamer-graph

export MS_ASSET_DIR=/mnt/data/tuannl
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

mkdir -p $HOME/output

nvidia-smi
nvidia-smi -l 100 > $HOME/output/gpu_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# The scripted solutions' planner (mplib) segfaults under NumPy 2. If the
# workers die without a Python error, check this first.
python -c "import numpy; print('[collect] numpy', numpy.__version__)"

# Successful demos per task; worker processes (CPU sim, one env each).
NUM_TRAJ=1000
NUM_PROCS=16
# Server 1's storage, which the training scripts link data/sim_vla_demos to.
OUT_DIR=/home/tuannl/mnt_data/data/maniskill

for ENV_ID in PickCube-v1 StackCube-v1 PegInsertionSide-v1; do
  echo "--------------------------------"
  echo "[collect] $ENV_ID"
  echo "--------------------------------"
  # collect.py refuses to replace an existing demos.h5, and only once the new
  # collection is done. An earlier dataset is renamed here, never deleted.
  DATASET_DIR=$OUT_DIR/$ENV_ID
  if [ -e "$DATASET_DIR/demos.h5" ]; then
    echo "[collect] $ENV_ID: keeping the existing dataset as demos_before_$TIMESTAMP.h5/.json"
    mv "$DATASET_DIR/demos.h5" "$DATASET_DIR/demos_before_$TIMESTAMP.h5"
    if [ -e "$DATASET_DIR/demos.json" ]; then
      mv "$DATASET_DIR/demos.json" "$DATASET_DIR/demos_before_$TIMESTAMP.json"
    fi
  fi
  python -m sim_vla.data.collect \
    --env-id $ENV_ID \
    --num-traj $NUM_TRAJ \
    --num-procs $NUM_PROCS \
    --out-dir $OUT_DIR \
    --name demos
done

kill $GPU_MONITOR_PID

echo "Job finished"
