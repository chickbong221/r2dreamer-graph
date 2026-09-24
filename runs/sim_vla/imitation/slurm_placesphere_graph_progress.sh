#!/bin/bash
#SBATCH --job-name=r2d-svla-ps-gp-il
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0
#SBATCH --output=/home/%u/output/%x_%j.out
#SBATCH --error=/home/%u/output/%x_%j.err

# Fail the job when any step fails, so a crashed trainer does not exit 0.
set -eo pipefail

# sim_vla arm 3 (graph_progress): graph, plus the progress head Stage 1A
# trains with the world model (shaping is Stage 2 only, so unused here).
# Imitation only: Stage 1A world model -> Stage 1B imitation -> evaluation in
# the simulator. No online stage. Differs from slurm_placesphere_baseline.sh
# only by --experiment. Needs PlaceSphere-v1/demos.h5 from ../server1/slurm_collect_data.sh.

echo "================================="
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "sim_vla: PlaceSphere-v1, arm=graph_progress, imitation only (1A -> 1B -> eval)"
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

cd $HOME/projects/r2dreamer-graph

# Demos live on server 1's storage; point data/sim_vla_demos at them.
mkdir -p data
ln -sfn /home/tuannl/mnt_data/data/maniskill data/sim_vla_demos

export MS_ASSET_DIR=/mnt/data/tuannl
export WANDB_API_KEY="b1d6eed8871c7668a889ae74a621b5dbd2f3b070"
# SmolVLA was downloaded here by ../server1/slurm_collect_data.sh.
export HF_HOME=/home/tuannl/mnt_data/mshab_transfer_checkpoint
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

# Keep in sync with slurm_placesphere_baseline.sh.
WORLD_STEPS=25000
IMITATION_STEPS=25000
# Windows per step in both stages; the default 16 used ~34 of 80 GB.
BATCH_SIZE=32
# Peak rates: linear warmup, then cosine decay to the final rate at the
# stage's last step.
WORLD_LR=1e-4
WORLD_WARMUP=1000
WORLD_FINAL_LR=1e-5
IMITATION_LR=1e-4
IMITATION_WARMUP=1000
IMITATION_FINAL_LR=2.5e-6
# Simulator episodes after Stage 1B, on seeds no demo used (eval.seeds_start).
EVAL_EPISODES=20

OUT_DIR=$HOME/logdir/r2dreamer-graph/sim_vla/$TIMESTAMP/placesphere/graph_progress_imitation_seed${SEED}
echo "[out] $OUT_DIR"

# --save-checkpoints: world_model.pt after 1A, imitation.pt after 1B, then
# imitation_eval.json after the evaluation, all in $OUT_DIR.
python -m sim_vla.training.pipeline \
  --task placesphere \
  --experiment graph_progress \
  --world-steps $WORLD_STEPS \
  --imitation-steps $IMITATION_STEPS \
  --batch-size $BATCH_SIZE \
  --world-lr $WORLD_LR \
  --imitation-lr $IMITATION_LR \
  --world-warmup-steps $WORLD_WARMUP \
  --world-final-lr $WORLD_FINAL_LR \
  --imitation-warmup-steps $IMITATION_WARMUP \
  --imitation-final-lr $IMITATION_FINAL_LR \
  --online-steps 0 \
  --eval-episodes $EVAL_EPISODES \
  --seed "$SEED" \
  --device cuda \
  --save-checkpoints \
  --out "$OUT_DIR"

echo "Job finished"
