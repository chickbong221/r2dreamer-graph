#!/bin/bash
#SBATCH --job-name=r2d-svla-pi-gp-online
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0
#SBATCH --output=/home/%u/output/%x_%j.out
#SBATCH --error=/home/%u/output/%x_%j.err

# Propagate training failures to Slurm.
set -eo pipefail

# Stage 2 only, graph_progress arm: restore Stage 1A + 1B from an earlier
# --save-checkpoints run and go straight to online training. Everything about
# the arm is unchanged from slurm_peginsertion_graph_progress.sh -- same
# partition, same resources, same env, same --experiment, same --online-steps
# -- so this stays step-for-step comparable to the baseline resume beside it.
#
# beta is not overridden here either: it comes from sim_vla/configs/base.yaml
# (0.05) via configs/experiments/graph_progress.yaml turning progress.enabled
# on. progress_module.warmup_for() scales the 20%/60% warm-up fractions to
# --online-steps, so resuming into a fresh 500k-step stage 2 gives the same
# warm-up window the original run was going to get.
#
# --resume-from restores world_model.pt and imitation.pt instead of training
# them (sim_vla/training/pipeline.py: stage 1A "skipped, restoring ...", stage
# 1B likewise). The arm, the env, the feature width, the pretrained revision
# and the normalization statistics must all match the checkpoint; pipeline.py
# refuses a mismatch rather than coercing it.
#
# --world-steps/--imitation-steps are 0 because nothing is being pretrained.
# They are ignored outright once both checkpoints are restored, but 0 is the
# honest value to log to wandb. The preflight below is what makes that safe:
# with imitation.pt absent and --imitation-steps 0, pipeline.run() returns
# after stage 1A and online training would be skipped *silently*, reporting
# success. A missing checkpoint must stop the run, not shrink it.

echo "================================="
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "sim_vla: PegInsertionSide-v1, arm=graph_progress (beta=0.05), stage 2 only (resumed)"
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

# Demos are still needed after a resume: stage 2 mixes demonstration batches
# into the replay (OnlineConfig.demo_fraction), and pretrain_world_model.resume
# reloads the dataset to rebuild the normalizer the checkpoint was fitted with.
mkdir -p data
ln -sfn /home/tuannl/mnt_data/data/maniskill data/sim_vla_demos

export MS_ASSET_DIR=/mnt/data/tuannl

# Not hardcoded here. The key that used to sit inline in these scripts is in
# git history and should be rotated; export it in the submitting shell
# (`export WANDB_API_KEY=...` before sbatch, which --export=ALL forwards) or
# run `wandb login` once to write ~/.netrc. Unset stops the run immediately
# rather than after the first stage.
: "${WANDB_API_KEY:?export WANDB_API_KEY before submitting (do not hardcode it in this file)}"

export HF_HOME=/home/tuannl/mnt_data/mshab_transfer_checkpoint

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

# Optional allocator aid; the main memory control is imagination microbatching.
# Reserved-but-unallocated memory alone does not establish fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p $HOME/output

# Print initial GPU state
nvidia-smi

# Monitor GPU every 100 seconds in background
nvidia-smi -l 100 > $HOME/output/gpu_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT

# Generate timestamp properly
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# The earlier run's --out directory, holding world_model.pt and imitation.pt.
RESUME_FROM=/home/tuannl/logdir/r2dreamer-graph/sim_vla/20260919_140921/peginsertion/graph_progress

for stage_file in world_model.pt imitation.pt; do
  if [ ! -f "$RESUME_FROM/$stage_file" ]; then
    echo "FATAL: $RESUME_FROM/$stage_file does not exist." >&2
    echo "  --resume-from wants the --out directory of an earlier" >&2
    echo "  --save-checkpoints run, holding both stage files. Resuming the" >&2
    echo "  actor without its own world model is refused by pipeline.py," >&2
    echo "  and resuming neither would retrain both from scratch." >&2
    kill $GPU_MONITOR_PID 2>/dev/null
    exit 1
  fi
done
echo "[resume] $RESUME_FROM"
ls -la "$RESUME_FROM"

# Unchanged from slurm_peginsertion_baseline.sh: the comparison against the
# graph_progress arm is only valid at equal online budgets, and
# progress_module.warmup_for() scales its warm-up fractions to this number.
ONLINE_STEPS=500000

# Keep replay batch 16. Accumulate gradients across all 256 imagination
# starts in groups of 16, then take ONE actor/critic optimizer step.
# Lower IMAGINATION_MICROBATCH if necessary without shrinking either batch.
BATCH_SIZE=16
IMAGINATION_BATCH=256
IMAGINATION_MICROBATCH=16
IMAG_HORIZON=15
# Same replay-timesteps / environment-step ratio as normal ManiSkill Dreamer.
TRAIN_RATIO=64
ONLINE_PRECISION=bfloat16

python -m sim_vla.training.pipeline \
  --task peginsertion \
  --experiment graph_progress \
  --resume-from $RESUME_FROM \
  --world-steps 0 \
  --imitation-steps 0 \
  --online-steps $ONLINE_STEPS \
  --batch-size $BATCH_SIZE \
  --imagination-batch $IMAGINATION_BATCH \
  --imagination-microbatch $IMAGINATION_MICROBATCH \
  --imag-horizon $IMAG_HORIZON \
  --train-ratio $TRAIN_RATIO \
  --online-precision $ONLINE_PRECISION \
  --device cuda \
  --save-checkpoints \
  --out $HOME/logdir/r2dreamer-graph/sim_vla/$TIMESTAMP/peginsertion/graph_progress

echo "Job finished"
