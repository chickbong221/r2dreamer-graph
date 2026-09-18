#!/bin/bash
#SBATCH --job-name=r2d-svla-ps-bl
#SBATCH --partition=A100-IML
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0
#SBATCH --output=/netscratch/ttran/tmp_iclr2026/output/%x_%j.out
#SBATCH --error=/netscratch/ttran/tmp_iclr2026/output/%x_%j.err

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

# This cluster's real home directory has too small a quota for this project;
# everything -- conda, the repo clone, data -- lives under this workspace
# instead, so HOME is overridden before anything below (including conda's own
# profile script) resolves ~ or $HOME.
export HOME=/netscratch/ttran/tmp_iclr2026

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dreamer

# This partition's nodes ship a working Vulkan ICD already -- the manual
# userspace-driver link server 1 needs (mismatched system driver vs. what
# SAPIEN/mani_skill's renderer expects) does not apply here. Left in place,
# commented, so the two setups stay easy to diff.
# export NVIDIA_USERSPACE_VERSION=570.133.20
# export NVIDIA_USERSPACE_DIR=$HOME/nvidia-userspace/NVIDIA-Linux-x86_64-${NVIDIA_USERSPACE_VERSION}

# cd "$NVIDIA_USERSPACE_DIR"

# ln -sf libGLX_nvidia.so.${NVIDIA_USERSPACE_VERSION} libGLX_nvidia.so.0
# ln -sf libEGL_nvidia.so.${NVIDIA_USERSPACE_VERSION} libEGL_nvidia.so.0

# cat > "$NVIDIA_USERSPACE_DIR/nvidia_icd_egl.json" <<EOF
# {
#     "file_format_version": "1.0.1",
#     "ICD": {
#         "library_path": "$NVIDIA_USERSPACE_DIR/libEGL_nvidia.so.0",
#         "api_version": "1.3.0"
#     }
# }
# EOF

# export LD_LIBRARY_PATH=$NVIDIA_USERSPACE_DIR:${LD_LIBRARY_PATH:-}
# export VK_DRIVER_FILES=$NVIDIA_USERSPACE_DIR/nvidia_icd_egl.json
# export VK_ICD_FILENAMES=$NVIDIA_USERSPACE_DIR/nvidia_icd_egl.json

# Proves the ICD is actually visible before the renderer needs it, instead of
# failing deep inside SAPIEN with a much less legible error.
vulkaninfo --summary

# Move to project directory
cd $HOME/projects/r2dreamer-graph

# Demos were collected by slurm_collect_data.sh into server 2's own storage
# ($HOME/data), not the repo-relative default (sim_vla/configs/base.yaml:
# data.root=data/sim_vla_demos). sim_vla.training.pipeline has no --data-root
# flag, so this symlink is what makes ${data.root}/<EnvId>/${data.name}
# resolve to the real files.
mkdir -p data
ln -sfn $HOME/data data/sim_vla_demos

# server 1's asset mount does not exist here. Left unset: mani_skill falls
# back to its own default cache under $HOME (now /netscratch/ttran/tmp_iclr2026
# above), so assets are downloaded once there. Point this at a shared
# read-only asset cache instead if this cluster has one.
# export MS_ASSET_DIR=/mnt/data/tuannl

# Matches slurm_collect_data.sh's HF_HOME: same cache, so this job loads the
# already-downloaded SmolVLA weights instead of reaching the network again.
export HF_HOME=/netscratch/ttran/tmp_iclr2026/checkpoint

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
