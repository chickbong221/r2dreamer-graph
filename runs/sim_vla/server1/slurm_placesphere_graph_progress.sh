#!/bin/bash
#SBATCH --job-name=r2d-svla-ps-gp
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
# PlaceSphere-v1.json + subtask_whitelists/PlaceSphere-v1), and beta*(gamma*
# phi(s') - phi(s)) is added to the imagined advantage during stage 2 only --
# it is never added to the environment reward that gets reported.
#
# Compared against slurm_placesphere_baseline.sh's "dreamer" arm at the same
# --world-steps/--imitation-steps/--online-steps: that isolates the graph +
# progress method's effect, since every other setting matches.
#
# Needs data/sim_vla_demos/PlaceSphere-v1/demos.h5 -- run
# slurm_collect_data.sh first. progress_module.preflight() (sim_vla/training/
# progress.py) checks the schedule, the whitelist directory and the dataset's
# recorded absolute-token vocabulary before Stage 1A starts, and refuses the
# run rather than training a head on invented targets if any of them is
# missing.
#
# Runs all three stages in one process (world model -> imitation -> online),
# per sim_vla/training/pipeline.py; the stages hand their models over in
# memory. --save-checkpoints additionally writes them under --out:
# world_model.pt (+ normalization.json) after Stage 1A -- carrying the
# progress head trained jointly with that world model -- imitation.pt after
# Stage 1B, and online_latest.pt, rewritten every 10k env steps in Stage 2.
# A later job can pass `--resume-from <that --out dir>` to restore Stage 1A
# (head included) and 1B from the first two and go straight to online
# training; Stage 2 itself is not resumed from online_latest.pt. A
# world_model.pt written before joint progress pretraining is refused.

echo "================================="
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "sim_vla: PlaceSphere-v1, arm=graph_progress (beta=0.05), actor=flow_reinforce"
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

# As in the PegInsertion flow_reinforce scripts. Optional allocator aid; the
# real memory controls are the microbatch settings below.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p $HOME/output

# Print initial GPU state
nvidia-smi

# Monitor GPU every 100 seconds in background
nvidia-smi -l 100 > $HOME/output/gpu_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!

# Generate timestamp properly
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Must match slurm_placesphere_baseline.sh's stage budgets -- the comparison
# is only valid at equal steps. beta itself is not overridden here: it comes
# from sim_vla/configs/base.yaml (0.05) via configs/experiments/
# graph_progress.yaml turning progress.enabled on.
WORLD_STEPS=30000
IMITATION_STEPS=25000
ONLINE_STEPS=500000
# 2x Stage 1A's default (4e-5) and 1.5x Stage 1B's (1e-4), for the shorter
# budgets above.
WORLD_LR=1e-4
IMITATION_LR=2e-4

# Stage 2: the flow_reinforce actor objective with the settings of
# slurm_peginsertion_*_online_flow_reinforce.sh, run in this same job right
# after Stage 1B, on the models it hands over in memory.
# Overridable from the submitting environment, e.g. SEED=1 sbatch <this file>.
SEED="${SEED:-0}"
ACTOR_LR="${ACTOR_LR:-1e-5}"
DEMO_ANCHOR="${DEMO_ANCHOR:-0.5}"
FLOW_NOISE_STD="${FLOW_NOISE_STD:-0.03}"
ACTOR_TRANSITION_MICROBATCH="${ACTOR_TRANSITION_MICROBATCH:-64}"
# Also Stage 1A/1B's batch: one shared setting, 16 either way.
BATCH_SIZE=16
IMAGINATION_BATCH=128
IMAGINATION_MICROBATCH=32
IMAG_HORIZON=15
TRAIN_RATIO=64
ONLINE_PRECISION=bfloat16
CRITIC_WARMUP=150
ANCHOR_WINDOWS=8
ANCHOR_WINDOW_MICROBATCH=4
ANCHOR_ROWS=64
ANCHOR_MICROBATCH=16
GRAD_REPORT_EVERY=50
# Parallel online envs on the GPU backend, as the main trainer runs them.
NUM_ENVS=128

python -m sim_vla.training.pipeline \
  --task placesphere \
  --experiment graph_progress \
  --world-steps $WORLD_STEPS \
  --imitation-steps $IMITATION_STEPS \
  --world-lr $WORLD_LR \
  --imitation-lr $IMITATION_LR \
  --online-steps $ONLINE_STEPS \
  --seed "$SEED" \
  --batch-size $BATCH_SIZE \
  --imagination-batch $IMAGINATION_BATCH \
  --imagination-microbatch $IMAGINATION_MICROBATCH \
  --imag-horizon $IMAG_HORIZON \
  --train-ratio $TRAIN_RATIO \
  --online-precision $ONLINE_PRECISION \
  --critic-warmup $CRITIC_WARMUP \
  --actor-objective flow_reinforce \
  --flow-noise-std "$FLOW_NOISE_STD" \
  --actor-transition-microbatch "$ACTOR_TRANSITION_MICROBATCH" \
  --actor-lr "$ACTOR_LR" \
  --demo-anchor "$DEMO_ANCHOR" \
  --anchor-windows $ANCHOR_WINDOWS \
  --anchor-window-microbatch $ANCHOR_WINDOW_MICROBATCH \
  --anchor-rows $ANCHOR_ROWS \
  --anchor-microbatch $ANCHOR_MICROBATCH \
  --grad-report-every $GRAD_REPORT_EVERY \
  --num-envs $NUM_ENVS \
  --eval-sampler stochastic \
  --device cuda \
  --save-checkpoints \
  --out $HOME/logdir/r2dreamer-graph/sim_vla/$TIMESTAMP/placesphere/graph_progress

# Stop GPU monitor
kill $GPU_MONITOR_PID

echo "Job finished"
