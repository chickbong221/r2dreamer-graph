# Shared setup for the ManiSkill sweeps. Sourced, not executed.
#
# Everything here is lifted from runs/slurm_mshab_graph_simple_size100m.sh: the
# same conda env, the same user-space NVIDIA/Vulkan shim the cluster needs
# before SAPIEN can open a renderer, the same GPU monitor.

set -uo pipefail

echo "================================="
echo "Job started on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "GPUs allocated: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "================================="

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dreamer

export NVIDIA_USERSPACE_VERSION=570.133.20
export NVIDIA_USERSPACE_DIR="$HOME/nvidia-userspace/NVIDIA-Linux-x86_64-${NVIDIA_USERSPACE_VERSION}"

cd "$NVIDIA_USERSPACE_DIR"
ln -sf "libGLX_nvidia.so.${NVIDIA_USERSPACE_VERSION}" libGLX_nvidia.so.0
ln -sf "libEGL_nvidia.so.${NVIDIA_USERSPACE_VERSION}" libEGL_nvidia.so.0

cat > "$NVIDIA_USERSPACE_DIR/nvidia_icd_egl.json" <<EOF
{
    "file_format_version": "1.0.1",
    "ICD": {
        "library_path": "$NVIDIA_USERSPACE_DIR/libEGL_nvidia.so.0",
        "api_version": "1.3.0"
    }
}
EOF

export LD_LIBRARY_PATH="$NVIDIA_USERSPACE_DIR:${LD_LIBRARY_PATH:-}"
export VK_DRIVER_FILES="$NVIDIA_USERSPACE_DIR/nvidia_icd_egl.json"
export VK_ICD_FILENAMES="$NVIDIA_USERSPACE_DIR/nvidia_icd_egl.json"

REPO="$HOME/projects/r2dreamer-graph"
cd "$REPO"

export WANDB_API_KEY="b1d6eed8871c7668a889ae74a621b5dbd2f3b070"
export MS_ASSET_DIR=/mnt/data/tuannl

mkdir -p "$HOME/output"

nvidia-smi --query-gpu=name,uuid,pci.bus_id,driver_version,memory.total --format=csv,noheader

nvidia-smi -l 100 > "$HOME/output/gpu_${SLURM_JOB_ID:-local}.log" &
GPU_MONITOR_PID=$!
cleanup() {
  if kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then kill "$GPU_MONITOR_PID"; fi
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------- knobs ---- #
# Tabletop tasks, not ReplicaCAD: no task plans, no spawn data, no scene
# builder, and the robot is a Panda with a wrist camera rather than a Fetch.
TASKS=(
  PlaceSphere-v1
  PullCubeTool-v1
  PickCube-v1
  StackCube-v1
  PegInsertionSide-v1
  PlugCharger-v1
)
MODEL=size50M_graph_simple
BATCH=32
ENV_NUM=126
STEPS=3000000
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# One task. $1 is the gym id, $2 the beta, $3 the tag that distinguishes this
# arm inside the wandb group.
run_task() {
  local task="$1" beta="$2" tag="$3"
  local name="maniskill_${task}-${MODEL}-${tag}"
  echo ""
  echo "----- ${name} -----"
  PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 python train.py \
    env=maniskill \
    model="$MODEL" \
    "env.task=maniskill_${task}" \
    model.graph.n_max=8 \
    model.graph.e_max=168 \
    model.amp_dtype=bfloat16 \
    "model.progress.beta=${beta}" \
    env.obs_mode=rgb+segmentation \
    "batch_size=${BATCH}" \
    batch_length=64 \
    "env.env_num=${ENV_NUM}" \
    env.train_ratio=64 \
    env.eval_episode_num=10 \
    env.eval_time_limit=200 \
    buffer.storage_device=cpu \
    buffer.max_size=750000 \
    "trainer.steps=${STEPS}" \
    trainer.eval_every=50000 \
    trainer.video_pred_log=false \
    trainer.video_every=1000000 \
    trainer.update_log_every=1500 \
    device=cuda:0 \
    wandb.enabled=true \
    wandb.project=RelRL \
    wandb.entity=letuanhf-hanoi-university-of-science-and-technology \
    "wandb.name=${name}" \
    "logdir=${REPO}/logdir/${TIMESTAMP}/${name}" \
    seed=0
}

# Sequential, and one task failing must not cancel the rest -- an overnight
# sweep that stops at the first bad task wastes the whole allocation. Failures
# are collected and re-reported at the end.
run_all() {
  local beta="$1" tag="$2"
  local failed=()
  local status
  for task in "${TASKS[@]}"; do
    run_task "$task" "$beta" "$tag"
    status=$?
    if [ "$status" -ne 0 ]; then
      echo "!! ${task} exited ${status}"
      failed+=("$task")
    fi
  done
  if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED: ${failed[*]}"
    return 1
  fi
  echo "All ${#TASKS[@]} tasks finished"
}
