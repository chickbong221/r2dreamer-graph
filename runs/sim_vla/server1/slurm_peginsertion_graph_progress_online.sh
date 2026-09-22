#!/bin/bash
#SBATCH --job-name=r2d-svla-pi-gp-online
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0
#SBATCH --output=/home/%u/output/%x_%j.out
#SBATCH --error=/home/%u/output/%x_%j.err

# Fail the job when any step fails. Without this the script ran past a crashed
# trainer to its closing echo, so a failed run exited 0 and looked successful.
# -u is deliberately not set: the banner reads SLURM_JOB_ID and
# CUDA_VISIBLE_DEVICES before Slurm necessarily defines them.
set -eo pipefail

# sim_vla arm 3 (graph + progress shaping), stage 2 only: restores Stage 1A + 1B from an earlier
# --save-checkpoints run and trains online with the executed-chunk actor.
# Compared against slurm_peginsertion_baseline_online.sh, which differs by
# --experiment and its resume directory alone; every training flag here is
# identical on purpose.
#
# Every eligible replay state starts one imagined rollout: the actor is
# conditioned there once, generates one action chunk, and the first
# actor.execute of its actions (5, from sim_vla/configs/base.yaml) are stepped
# through the world model. The actor maximises that chunk's bootstrapped
# return pathwise -- through the flow sampler, the transitions and the heads --
# and the critic is fitted to the same return at the start state. The
# environment replans on the same period, so the policy optimised is the
# policy collecting. Returns stay float32; only the transformer runs under
# bfloat16 autocast.
#
# This is a NEW stage 2 run. It restores Stage 1A/1B weights and nothing else
# -- not the replay, critic, optimizer state, step counter or environment
# state -- so it is not a continuation of an interrupted online run.
# --world-steps 0 and --imitation-steps 0 are the honest values; the preflight
# below is what makes them safe, since a missing imitation.pt would otherwise
# make pipeline.run() return after Stage 1A and skip online training silently.
#
# No setting here is claimed to guarantee convergence or task success.

echo "================================="
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "sim_vla: PegInsertionSide-v1, arm=graph_progress, stage 2 only, actor=pathwise executed chunk"
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
# into the replay, the imitation anchor draws its own demonstration windows,
# and pretrain_world_model.resume reloads the dataset to rebuild the
# normalizer the checkpoint was fitted with.
mkdir -p data
ln -sfn /home/tuannl/mnt_data/data/maniskill data/sim_vla_demos

export MS_ASSET_DIR=/mnt/data/tuannl

export WANDB_API_KEY="b1d6eed8871c7668a889ae74a621b5dbd2f3b070"

# Matches slurm_collect_data.sh's HF_HOME: same cache, so this job loads the
# already-downloaded SmolVLA weights instead of reaching the network again.
export HF_HOME=/home/tuannl/mnt_data/mshab_transfer_checkpoint

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

# Optional allocator aid; the real memory controls are the microbatch settings
# below. Reserved-but-unallocated memory alone does not establish fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p $HOME/output

# Print initial GPU state
nvidia-smi

# Monitor GPU every 100 seconds in background. Stopped by an EXIT trap rather
# than a trailing kill: under set -e a failed trainer never reaches the end of
# the script, and the monitor would outlive it.
nvidia-smi -l 100 > $HOME/output/gpu_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT

# Generate timestamp properly
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Overridable from the submitting environment, e.g. SEED=1 sbatch <this file>.
# The defaults are this run's selected configuration.
SEED="${SEED:-0}"
ONLINE_STEPS="${ONLINE_STEPS:-500000}"
ACTOR_LR="${ACTOR_LR:-1e-5}"

# Fixed for this experiment, and identical in both arms: these are the update
# rule and its budgets, not the state, so a difference here would not be a
# difference between the arms being compared.
BATCH_SIZE=16
# Starts imagined together. An update has no cap on its starts -- it imagines
# every scored row of the replay batch -- so this is what bounds its memory.
IMAGINATION_MICROBATCH=32
TRAIN_RATIO=64
ONLINE_PRECISION=bfloat16
CRITIC_WARMUP=150
# Parallel online envs on the GPU backend, as the main trainer runs them.
# Updates happen between vector steps at the same train_ratio, so the update
# budget is unchanged; only collection gets faster.
NUM_ENVS=16

# base.yaml sets actor.flow_steps to 1 for both collection and imagination.
# actor.execute also comes from that shared config, so both paths use the
# same number of actions from each generated chunk. Profiling stays
# available but off -- add --profile-online for a short run; it synchronizes
# at every phase boundary and is not for a 500k-step job.

# The earlier run's --out directory, holding world_model.pt and imitation.pt.
# Read only: this run never writes into it.
RESUME_FROM=/home/tuannl/logdir/r2dreamer-graph/sim_vla/20260919_140921/peginsertion/graph_progress

for stage_file in world_model.pt imitation.pt; do
  if [ ! -f "$RESUME_FROM/$stage_file" ]; then
    echo "FATAL: $RESUME_FROM/$stage_file does not exist." >&2
    echo "  --resume-from wants the --out directory of an earlier" >&2
    echo "  --save-checkpoints run, holding both stage files. Resuming the" >&2
    echo "  actor without its own world model is refused by pipeline.py," >&2
    echo "  and resuming neither would retrain both from scratch." >&2
    exit 1
  fi
done
echo "[resume] $RESUME_FROM"
ls -la "$RESUME_FROM"

# A fresh directory per run, naming what distinguishes this one: arm, learning
# rate and seed. Two submissions cannot collide.
OUT_DIR=$HOME/logdir/r2dreamer-graph/sim_vla/$TIMESTAMP/peginsertion/graph_progress_online_lr${ACTOR_LR}_seed${SEED}
echo "[out] $OUT_DIR"

python -m sim_vla.training.pipeline \
  --task peginsertion \
  --experiment graph_progress \
  --resume-from "$RESUME_FROM" \
  --world-steps 0 \
  --imitation-steps 0 \
  --online-steps "$ONLINE_STEPS" \
  --seed "$SEED" \
  --batch-size $BATCH_SIZE \
  --imagination-microbatch $IMAGINATION_MICROBATCH \
  --train-ratio $TRAIN_RATIO \
  --online-precision $ONLINE_PRECISION \
  --critic-warmup $CRITIC_WARMUP \
  --actor-lr "$ACTOR_LR" \
  --num-envs $NUM_ENVS \
  --device cuda \
  --save-checkpoints \
  --out "$OUT_DIR"

echo "Job finished"
