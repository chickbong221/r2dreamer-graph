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
echo "sim_vla: PlaceSphere-v1, arm=dreamer (baseline, no graph, no progress), actor=flow_reinforce"
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

# Stage budgets. Tune before submitting -- these are not repo defaults, they
# are this run's choice. online.py's own OnlineConfig defaults to 1,000,000
# total_steps; progress_module.warmup_for() scales its 20%/60% warm-up
# fractions to whatever --online-steps is given here, so the baseline and the
# graph_progress run only need to agree on this number, not on an absolute
# warm-up window.
WORLD_STEPS=30000
IMITATION_STEPS=25000
ONLINE_STEPS=500000
# 2x Stage 1A's default (4e-5) and 1.5x Stage 1B's (1e-4), for the shorter
# budgets above.
WORLD_LR=8e-5
IMITATION_LR=1.5e-4

# Stage 2: the flow_reinforce actor objective with the settings of
# slurm_peginsertion_*_online_flow_reinforce.sh, run in this same job right
# after Stage 1B -- there is no PlaceSphere checkpoint to resume from.
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
  --experiment dreamer \
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
  --out $HOME/logdir/r2dreamer-graph/sim_vla/$TIMESTAMP/placesphere/dreamer

# Stop GPU monitor
kill $GPU_MONITOR_PID

echo "Job finished"
