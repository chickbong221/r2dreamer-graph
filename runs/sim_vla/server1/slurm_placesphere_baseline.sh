#!/bin/bash
#SBATCH --job-name=r2d-svla-ps-bl
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0
#SBATCH --output=/home/%u/output/%x_%j.out
#SBATCH --error=/home/%u/output/%x_%j.err

# sim_vla arm 1: Dreamer + SmolVLA, no graph anywhere in the pipeline
# (configs/experiments/dreamer.yaml). Compared against
# slurm_placesphere_graph_progress.sh, which differs by --experiment alone.
#
# Needs data/sim_vla_demos/PlaceSphere-v1/demos.h5 -- run slurm_collect_data.sh
# first. All three stages run in one process and hand their models over in
# memory; --save-checkpoints also writes world_model.pt, imitation.pt and
# online_latest.pt under --out, so a later job can --resume-from that directory
# and go straight to Stage 2. Anything not passed below is whatever
# sim_vla/configs/base.yaml says; sim_vla/README.md explains the objective.

echo "================================="
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "sim_vla: PlaceSphere-v1, arm=dreamer (baseline, no graph, no progress), actor=pathwise executed chunk"
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

# The demos live in server 1's own storage and the pipeline has no --data-root
# flag, so this symlink is what makes ${data.root}/<EnvId>/demos.h5 resolve.
mkdir -p data
ln -sfn /home/tuannl/mnt_data/data/maniskill data/sim_vla_demos

export MS_ASSET_DIR=/mnt/data/tuannl
export WANDB_API_KEY="b1d6eed8871c7668a889ae74a621b5dbd2f3b070"
# Same cache as slurm_collect_data.sh: SmolVLA is downloaded there already.
export HF_HOME=/home/tuannl/mnt_data/mshab_transfer_checkpoint
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p $HOME/output

nvidia-smi
nvidia-smi -l 100 > $HOME/output/gpu_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# This run's choice, not repo defaults. slurm_placesphere_graph_progress.sh has
# to match them: the comparison is only valid at equal steps and equal rates.
WORLD_STEPS=100
IMITATION_STEPS=100
ONLINE_STEPS=500000
WORLD_LR=1e-4
IMITATION_LR=1e-4

# Overridable from the submitting environment, e.g. SEED=1 sbatch <this file>.
SEED="${SEED:-0}"
ACTOR_LR="${ACTOR_LR:-6e-5}"
# Start states imagined together. An update has no cap on its starts -- it
# imagines every scored row of the replay batch -- so this bounds its memory.
IMAGINATION_MICROBATCH=32
CRITIC_WARMUP=0
NUM_ENVS=16

python -m sim_vla.training.pipeline \
  --task placesphere \
  --experiment dreamer \
  --world-steps $WORLD_STEPS \
  --imitation-steps $IMITATION_STEPS \
  --world-lr $WORLD_LR \
  --imitation-lr $IMITATION_LR \
  --online-steps $ONLINE_STEPS \
  --seed "$SEED" \
  --imagination-microbatch $IMAGINATION_MICROBATCH \
  --critic-warmup $CRITIC_WARMUP \
  --actor-lr "$ACTOR_LR" \
  --num-envs $NUM_ENVS \
  --device cuda \
  --save-checkpoints \
  --out $HOME/logdir/r2dreamer-graph/sim_vla/$TIMESTAMP/placesphere/dreamer

kill $GPU_MONITOR_PID

echo "Job finished"
