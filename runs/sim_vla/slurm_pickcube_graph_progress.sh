#!/bin/bash
#SBATCH --job-name=r2d-svla-pc-gp
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0
#SBATCH --output=/home/%u/output/%x_%j.out
#SBATCH --error=/home/%u/output/%x_%j.err

# sim_vla arm 3: graph + progress shaping (configs/experiments/graph_progress.
# yaml), beta=0.05 from sim_vla/configs/base.yaml's model.progress.beta --
# the same default configs/model/_base_.yaml uses for the simulator's own
# graph arm. Actor/critic state is (h, z, g); the progress head regresses onto
# SchedulePotential's observed-graph phase (scenegraph/configs/schedules/
# PickCube-v1.json + subtask_whitelists/PickCube-v1), and beta*(gamma*phi(s')
# - phi(s)) is added to the imagined advantage during stage 2 only -- it is
# never added to the environment reward that gets reported.
#
# Compared against slurm_pickcube_baseline.sh's "dreamer" arm at the same
# --world-steps/--imitation-steps/--online-steps: that isolates the graph +
# progress method's effect, since every other setting matches.
#
# Needs data/sim_vla_demos/PickCube-v1/demos.h5 -- run slurm_collect_data.sh
# first. progress_module.preflight() (sim_vla/training/progress.py) checks the
# schedule, the whitelist directory and the dataset's recorded absolute-token
# vocabulary before Stage 1A starts, and refuses the run rather than training
# a head on invented targets if any of them is missing.

echo "================================="
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "sim_vla: PickCube-v1, arm=graph_progress (beta=0.05)"
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

export MS_ASSET_DIR=/mnt/data/tuannl

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

# Must match slurm_pickcube_baseline.sh's stage budgets -- the comparison is
# only valid at equal steps. beta itself is not overridden here: it comes
# from sim_vla/configs/base.yaml (0.05) via configs/experiments/
# graph_progress.yaml turning progress.enabled on.
WORLD_STEPS=100000
IMITATION_STEPS=50000
ONLINE_STEPS=500000

python -m sim_vla.training.pipeline \
  --task pickcube \
  --experiment graph_progress \
  --world-steps $WORLD_STEPS \
  --imitation-steps $IMITATION_STEPS \
  --online-steps $ONLINE_STEPS \
  --device cuda \
  --save-checkpoints \
  --out $HOME/logdir/r2dreamer-graph/sim_vla/$TIMESTAMP/pickcube/graph_progress

# Stop GPU monitor
kill $GPU_MONITOR_PID

echo "Job finished"
