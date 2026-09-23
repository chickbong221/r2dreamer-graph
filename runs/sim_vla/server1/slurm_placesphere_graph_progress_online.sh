#!/bin/bash
#SBATCH --job-name=r2d-svla-ps-gp-online
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0
#SBATCH --output=/home/%u/output/%x_%j.out
#SBATCH --error=/home/%u/output/%x_%j.err

# Fail the job when any step fails, so a crashed trainer does not exit 0.
set -eo pipefail

# sim_vla arm 3 (graph_progress), Stage 2 only: restores Stage 1A + 1B weights
# and starts a fresh online run. Differs from
# slurm_placesphere_baseline_online.sh only by --experiment and RESUME_FROM.

echo "================================="
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "sim_vla: PlaceSphere-v1, arm=graph_progress (beta=0.05), stage 2 only, actor=pathwise executed chunk"
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

# Still needed after a resume: Stage 2 mixes demos into replay, and the
# normalizer is rebuilt from the dataset.
mkdir -p data
ln -sfn /home/tuannl/mnt_data/data/maniskill data/sim_vla_demos

export MS_ASSET_DIR=/mnt/data/tuannl
export WANDB_API_KEY="b1d6eed8871c7668a889ae74a621b5dbd2f3b070"
# SmolVLA was downloaded here by slurm_collect_data.sh.
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
ONLINE_STEPS="${ONLINE_STEPS:-500000}"

# Keep in sync with slurm_placesphere_baseline_online.sh.
ONLINE_WORLD_LR=2e-5
# Stage 2 rates: critic/progress 1.5x r2dreamer's 4e-5; the actor lower, so
# the pretrained expert is not overwritten. PROGRESS_LR is graph_progress only.
ACTOR_LR="${ACTOR_LR:-3e-6}"
CRITIC_LR=6e-5
PROGRESS_LR=6e-5
# Replay windows per update (base 16). 20 x 56 = 1120 imagination starts, all
# at once with microbatch 0: ~75 GB est. on the graph arm; 560 if OOM.
BATCH_SIZE=20
IMAGINATION_MICROBATCH=0
CRITIC_WARMUP=600
NUM_ENVS=16
# Stage 1B's imitation loss weighted into the actor update after warm-up.
# --return-norm below divides the RL term by the running return spread.
DEMO_ANCHOR=0.5
ANCHOR_MICROBATCH=0
# Progress shaping ramps 0 -> beta over these env steps (graph_progress only).
PROGRESS_WARMUP_START=30000
PROGRESS_WARMUP_END=100000

# Read only: the earlier --save-checkpoints run's --out directory.
RESUME_FROM=/home/tuannl/logdir/r2dreamer-graph/sim_vla/20260922_185807/placesphere/graph_progress

for stage_file in world_model.pt imitation.pt; do
  if [ ! -f "$RESUME_FROM/$stage_file" ]; then
    echo "FATAL: $RESUME_FROM/$stage_file does not exist." >&2
    exit 1
  fi
done
echo "[resume] $RESUME_FROM"
ls -la "$RESUME_FROM"

OUT_DIR=$HOME/logdir/r2dreamer-graph/sim_vla/$TIMESTAMP/placesphere/graph_progress_online_lr${ACTOR_LR}_seed${SEED}
echo "[out] $OUT_DIR"

python -m sim_vla.training.pipeline \
  --task placesphere \
  --experiment graph_progress \
  --resume-from "$RESUME_FROM" \
  --world-steps 0 \
  --imitation-steps 0 \
  --online-steps "$ONLINE_STEPS" \
  --seed "$SEED" \
  --batch-size $BATCH_SIZE \
  --online-world-lr $ONLINE_WORLD_LR \
  --actor-lr "$ACTOR_LR" \
  --critic-lr $CRITIC_LR \
  --progress-lr $PROGRESS_LR \
  --imagination-microbatch $IMAGINATION_MICROBATCH \
  --critic-warmup $CRITIC_WARMUP \
  --num-envs $NUM_ENVS \
  --demo-anchor $DEMO_ANCHOR \
  --anchor-microbatch $ANCHOR_MICROBATCH \
  --return-norm \
  --progress-warmup-start $PROGRESS_WARMUP_START \
  --progress-warmup-end $PROGRESS_WARMUP_END \
  --device cuda \
  --save-checkpoints \
  --out "$OUT_DIR"

echo "Job finished"
