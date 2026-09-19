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
# (configs/experiments/dreamer.yaml). No graph encoder, no semantic latent, no
# graph losses, no graph-derived progress -- actor/critic state is (h, z)
# only. Compared against slurm_placesphere_graph_progress.sh, which differs by
# --experiment alone; every other flag here is identical on purpose.
#
# Needs data/sim_vla_demos/PlaceSphere-v1/demos.h5 -- run
# slurm_collect_data.sh first. That dataset does carry graphs (the collector
# always records them), but sim_vla/data/dataset.py never opens the graph
# arrays for this arm: the acceptance test corrupts and deletes the stored
# graphs and asserts a baseline's batches are byte-identical.
#
# Runs all three stages in one process (world model -> imitation -> online),
# per sim_vla/training/pipeline.py. Checkpoints are saved so a crashed run's
# world-model/imitation weights are not lost, even though the pipeline itself
# does not resume a run from them.

echo "================================="
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "sim_vla: PlaceSphere-v1, arm=dreamer (baseline, no graph, no progress)"
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

# Demos were collected by slurm_collect_data.sh into server 1's own storage,
# not the repo-relative default (sim_vla/configs/base.yaml: data.root=
# data/sim_vla_demos). sim_vla.training.pipeline has no --data-root flag, so
# this symlink is what makes ${data.root}/<EnvId>/${data.name} resolve to the
# real files.
mkdir -p data
ln -sfn /home/tuannl/mnt_data/data/maniskill data/sim_vla_demos

export MS_ASSET_DIR=/mnt/data/tuannl

export WANDB_API_KEY="b1d6eed8871c7668a889ae74a621b5dbd2f3b070"

# Matches slurm_collect_data.sh's HF_HOME: same cache, so this job loads the
# already-downloaded SmolVLA weights instead of reaching the network again.
export HF_HOME=/home/tuannl/mnt_data/mshab_transfer_checkpoint

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

mkdir -p $HOME/output

# Print initial GPU state
nvidia-smi

# Monitor GPU every 100 seconds in background
nvidia-smi -l 100 > $HOME/output/gpu_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!

# Generate timestamp properly
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Stage budgets. Tune before submitting -- these are not repo defaults, they
# are this run's choice. online.py's own OnlineConfig defaults to 1,000,000
# total_steps; progress_module.warmup_for() scales its 20%/60% warm-up
# fractions to whatever --online-steps is given here, so the baseline and the
# graph_progress run only need to agree on this number, not on an absolute
# warm-up window.
WORLD_STEPS=100000
IMITATION_STEPS=50000
ONLINE_STEPS=500000

python -m sim_vla.training.pipeline \
  --task placesphere \
  --experiment dreamer \
  --world-steps $WORLD_STEPS \
  --imitation-steps $IMITATION_STEPS \
  --online-steps $ONLINE_STEPS \
  --device cuda \
  --save-checkpoints \
  --out $HOME/logdir/r2dreamer-graph/sim_vla/$TIMESTAMP/placesphere/dreamer

# Stop GPU monitor
kill $GPU_MONITOR_PID

echo "Job finished"
