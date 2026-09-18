#!/bin/bash
#SBATCH --job-name=r2d-svla-collect
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=0
#SBATCH --output=/home/%u/output/%x_%j.out
#SBATCH --error=/home/%u/output/%x_%j.err

# Collects the sim_vla demonstration datasets for the three tasks the pipeline
# has task/schedule/whitelist coverage for: PickCube-v1, PlaceSphere-v1,
# PegInsertionSide-v1. One dataset per task, shared by every arm -- the
# collector always builds and packs the graph (sim_vla/data/collect.py), so
# the same demos.h5 trains the "dreamer" baseline (which never opens the graph
# arrays) and the "graph_progress" arm (which needs them) alike. Nothing here
# depends on --experiment; that switch is chosen at training time.
#
# Whitelist resolution is per env-id automatically: FigureGraphSource calls
# scenegraph.configs.loader.load_config(task_group=env_id), which resolves
# subtask_whitelists/<env_id>/ on its own. Running one --env-id per process
# (never mixing tasks in one call) is what keeps that resolution correct --
# see scenegraph/configs/loader.py and the note that mining several task
# groups' worth of objects into one call silently pins them to one scene.
#
# Output: data/sim_vla_demos/<EnvId>/demos.h5, matching
# sim_vla/configs/tasks/*.yaml's `task.dataset` and sim_vla/configs/base.yaml's
# `data.root` / `data.name`.
#
# Deliberately no `set -e`: a task whose seed block runs dry (collect.py exits
# 1 and prints a top-up command) must not cancel the tasks after it.

echo "================================="
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Collecting: PickCube-v1, PlaceSphere-v1, PegInsertionSide-v1"
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

# Move to project directory. Run from here, not from sim_vla/: collect.py's
# default --env-config/--model-config paths (configs/env/maniskill.yaml,
# configs/model/size50M_graph_simple.yaml) and its `from envs.maniskill import
# ...` are both resolved against the repo root, matching train.py's own cwd.
cd $HOME/projects/r2dreamer-graph

export MS_ASSET_DIR=/mnt/data/tuannl

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

mkdir -p $HOME/output

# Print initial GPU state
nvidia-smi

# Monitor GPU every 100 seconds in background
nvidia-smi -l 100 > $HOME/output/gpu_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!

# Demos to accept per task, and worker processes per collection run. Tune
# these before submitting: NUM_TRAJ=500 matches sim_vla/data/collect.py's own
# documented example; NUM_PROCS is CPU-side (collect.py forces sim_backend=cpu
# and batch size 1 per env -- the scripted solutions read an unbatched pose),
# so it scales with cores, not with the single GPU shared for rendering.
NUM_TRAJ=500
NUM_PROCS=16
OUT_DIR=data/sim_vla_demos

for ENV_ID in PickCube-v1 PlaceSphere-v1 PegInsertionSide-v1; do
  echo "--------------------------------"
  echo "[collect] $ENV_ID"
  echo "--------------------------------"
  python -m sim_vla.data.collect \
    --env-id $ENV_ID \
    --num-traj $NUM_TRAJ \
    --num-procs $NUM_PROCS \
    --out-dir $OUT_DIR \
    --name demos
done

# Stop GPU monitor
kill $GPU_MONITOR_PID

echo "Job finished"
