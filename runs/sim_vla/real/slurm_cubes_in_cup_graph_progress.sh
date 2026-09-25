#!/bin/bash
#SBATCH --job-name=r2d-svla-real-cup-gp
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0
#SBATCH --output=/home/%u/output/%x_%j.out
#SBATCH --error=/home/%u/output/%x_%j.err

# SO-101 cubes_in_cup (task 3, both cubes into the cup), arm=graph_progress.
# World model -> imitation on the recorded episodes; no simulator, so no
# evaluation. Submit from the repository root after
# `bash runs/sim_vla/real/prepare.sh cubes_in_cup`. wandb reads WANDB_API_KEY from
# the submitting shell.

set -eo pipefail

echo "================================="
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "sim_vla: SO-101 cubes_in_cup (real data), arm=graph_progress, imitation only (1A -> 1B)"
echo "================================="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dreamer

cd "$SLURM_SUBMIT_DIR"
test -f sim_vla/__init__.py || { echo "FATAL: submit from the repository root" >&2; exit 1; }
source runs/sim_vla/real/setup.sh

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p $HOME/output

nvidia-smi
nvidia-smi -l 100 > $HOME/output/gpu_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Overridable at submit time, e.g. SEED=1 sbatch <this file>.
SEED="${SEED:-0}"

WORLD_STEPS=15000
IMITATION_STEPS=15000
WORLD_LR=1e-4
WORLD_WARMUP=1000
WORLD_FINAL_LR=1e-5
IMITATION_LR=1e-4
IMITATION_WARMUP=1000
IMITATION_FINAL_LR=2.5e-6

OUT_DIR=$HOME/logdir/r2dreamer-graph/sim_vla/real/$TIMESTAMP/cubes_in_cup/graph_progress_seed${SEED}
echo "[out] $OUT_DIR"

python -m sim_vla.training.pipeline \
  --data real \
  --task cubes_in_cup \
  --experiment graph_progress \
  --world-steps $WORLD_STEPS \
  --imitation-steps $IMITATION_STEPS \
  --world-lr $WORLD_LR \
  --imitation-lr $IMITATION_LR \
  --world-warmup-steps $WORLD_WARMUP \
  --world-final-lr $WORLD_FINAL_LR \
  --imitation-warmup-steps $IMITATION_WARMUP \
  --imitation-final-lr $IMITATION_FINAL_LR \
  --online-steps 0 \
  --seed "$SEED" \
  --device cuda \
  --save-checkpoints \
  --out "$OUT_DIR"

echo "Job finished"
