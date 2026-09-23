#!/bin/bash
#SBATCH --job-name=r2d-svla-ps-bl
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0
#SBATCH --output=/home/%u/output/%x_%j.out
#SBATCH --error=/home/%u/output/%x_%j.err

# sim_vla arm 1 (dreamer): no graph. Differs from
# slurm_placesphere_graph_progress.sh only by --experiment.
# Needs PlaceSphere-v1/demos.h5 from slurm_collect_data.sh.

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

# Demos live on server 1's storage; point data/sim_vla_demos at them.
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

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Keep in sync with slurm_placesphere_graph_progress.sh.
WORLD_STEPS=30000
IMITATION_STEPS=25000
ONLINE_STEPS=500000
# World model, Stage 1A and Stage 2.
WORLD_LR=1e-4
IMITATION_LR=1e-4

# Overridable at submit time, e.g. SEED=1 sbatch <this file>.
SEED="${SEED:-0}"
# Stage 2 rates (1.5x r2dreamer's 4e-5); PROGRESS_LR is graph_progress only.
ACTOR_LR="${ACTOR_LR:-6e-5}"
CRITIC_LR=6e-5
PROGRESS_LR=6e-5
# 896 imagination starts per update, in 2 groups (80 GB GPU); lower if OOM.
IMAGINATION_MICROBATCH=448
CRITIC_WARMUP=150
NUM_ENVS=64

python -m sim_vla.training.pipeline \
  --task placesphere \
  --experiment dreamer \
  --world-steps $WORLD_STEPS \
  --imitation-steps $IMITATION_STEPS \
  --world-lr $WORLD_LR \
  --online-world-lr $WORLD_LR \
  --imitation-lr $IMITATION_LR \
  --online-steps $ONLINE_STEPS \
  --seed "$SEED" \
  --imagination-microbatch $IMAGINATION_MICROBATCH \
  --critic-warmup $CRITIC_WARMUP \
  --actor-lr "$ACTOR_LR" \
  --critic-lr $CRITIC_LR \
  --progress-lr $PROGRESS_LR \
  --num-envs $NUM_ENVS \
  --device cuda \
  --save-checkpoints \
  --out $HOME/logdir/r2dreamer-graph/sim_vla/$TIMESTAMP/placesphere/dreamer

kill $GPU_MONITOR_PID

echo "Job finished"
