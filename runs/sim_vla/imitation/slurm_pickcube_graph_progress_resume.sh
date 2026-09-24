#!/bin/bash
#SBATCH --job-name=r2d-svla-pc-gp-il-resume
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0
#SBATCH --output=/home/%u/output/%x_%j.out
#SBATCH --error=/home/%u/output/%x_%j.err

# Fail the job when any step fails, so a crashed trainer does not exit 0.
set -eo pipefail

# sim_vla arm 3 (graph_progress), PickCube: reuses only the world model --
# and the progress head saved with it -- from RESUME_FROM, retrains
# imitation with slurm_pickcube_graph_progress.sh's current
# settings, then runs the simulator evaluation. --resume-from would also
# restore RESUME_FROM's imitation.pt and skip imitation, so the world model is
# linked into a fresh OUT_DIR, which is both --resume-from and --out. Differs
# from slurm_pickcube_baseline_resume.sh only by --experiment and
# RESUME_FROM.

echo "================================="
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "sim_vla: PickCube-v1, arm=graph_progress, world model restored, imitation retrained -> eval"
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

# Imitation settings; keep in sync with slurm_pickcube_graph_progress.sh.
# The world model is restored, so none of its settings apply.
IMITATION_STEPS=25000
IMITATION_LR=1e-4
IMITATION_WARMUP=1000
IMITATION_FINAL_LR=2.5e-6
# Simulator episodes after Stage 1B, on seeds no demo used (eval.seeds_start).
EVAL_EPISODES=20

# Read only: the world model is taken from here, nothing else.
RESUME_FROM=/home/duongnm2/logdir/r2dreamer-graph/sim_vla/20260923_184843/pickcube/graph_progress_imitation_seed0

if [ ! -f "$RESUME_FROM/world_model.pt" ]; then
  echo "FATAL: $RESUME_FROM/world_model.pt does not exist." >&2
  exit 1
fi
echo "[resume] world model from $RESUME_FROM"
ls -la "$RESUME_FROM"

OUT_DIR=$HOME/logdir/r2dreamer-graph/sim_vla/$TIMESTAMP/pickcube/graph_progress_imitation_resume_seed${SEED}
mkdir -p "$OUT_DIR"
# normalization.json is left out on purpose: the statistics are refitted from
# the current dataset and must match the checkpoint's, so a PickCube dataset
# re-collected since is refused instead of silently used.
ln -s "$RESUME_FROM/world_model.pt" "$OUT_DIR/world_model.pt"
if [ -f "$RESUME_FROM/world_model.json" ]; then
  ln -s "$RESUME_FROM/world_model.json" "$OUT_DIR/world_model.json"
fi
echo "[out] $OUT_DIR"

# --save-checkpoints writes imitation.pt and then imitation_eval.json into
# $OUT_DIR, beside the linked world model.
python -m sim_vla.training.pipeline \
  --task pickcube \
  --experiment graph_progress \
  --resume-from "$OUT_DIR" \
  --world-steps 0 \
  --imitation-steps $IMITATION_STEPS \
  --imitation-lr $IMITATION_LR \
  --imitation-warmup-steps $IMITATION_WARMUP \
  --imitation-final-lr $IMITATION_FINAL_LR \
  --online-steps 0 \
  --eval-episodes $EVAL_EPISODES \
  --seed "$SEED" \
  --device cuda \
  --save-checkpoints \
  --out "$OUT_DIR"

echo "Job finished"
