#!/bin/bash
#SBATCH --job-name=r2d-svla-collect
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
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
# Output: /home/tuannl/mnt_data/data/maniskill/<EnvId>/demos.h5 -- server 1's
# storage, not the repo-relative data/sim_vla_demos that sim_vla/configs/
# tasks/*.yaml's `task.dataset` and sim_vla/configs/base.yaml's `data.root` /
# `data.name` resolve to. The training scripts in this folder symlink
# data/sim_vla_demos to this directory before running, since
# sim_vla.training.pipeline has no flag to point `data.root` elsewhere.
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

# --- SmolVLA pretrained checkpoint ------------------------------------------
# sim_vla.training.pipeline's Stage 1B loads lerobot/smolvla_base from the
# Hugging Face Hub at run time (sim_vla/training/train_imitation.py ->
# sim_vla/models/pretrained.py:load_policy). If the training nodes don't have
# outbound internet, that fetch fails deep inside a training job instead of
# here, where it's cheap to retry. HF_HOME is redirected to a dedicated,
# clearly-named directory rather than each user's default ~/.cache/huggingface,
# so every training script in this folder can point HF_HOME at the exact same
# place and never touch the network.
export HF_HOME=/home/tuannl/mnt_data/mshab_transfer_checkpoint
mkdir -p "$HF_HOME"
python -m sim_vla.download_pretrained --out "$HF_HOME"
# Prints, among other things:
#   Put this in sim_vla/configs/base.yaml:
#     actor.revision: <40-char commit hash>
# Paste that hash in as a *quoted* string. resolve_revision() only skips its
# own network call when actor.revision is already a literal commit hash --
# left at "" (the checked-in default) or "main", it still calls the Hub API on
# every run even once the weights themselves are cached, which defeats the
# point of pre-downloading for an offline training node.

# Demos to accept per task, and worker processes per collection run. Tune
# these before submitting: NUM_TRAJ=500 matches sim_vla/data/collect.py's own
# documented example; NUM_PROCS is CPU-side (collect.py forces sim_backend=cpu
# and batch size 1 per env -- the scripted solutions read an unbatched pose),
# so it scales with cores, not with the single GPU shared for rendering.
NUM_TRAJ=1000
NUM_PROCS=16
# Server 1's own storage, not the repo-relative default: this mount is what
# the training scripts in this folder symlink data/sim_vla_demos to.
OUT_DIR=/home/tuannl/mnt_data/data/maniskill

for ENV_ID in PlaceSphere-v1; do
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
