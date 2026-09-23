#!/bin/bash
#SBATCH --job-name=r2d-svla-ps-eval
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0
#SBATCH --output=/home/%u/output/%x_%j.out
#SBATCH --error=/home/%u/output/%x_%j.err

# Fail the job when a setup step fails. The two evaluations below run even if
# the other one fails, and the job exits nonzero if either did.
set -eo pipefail

# Evaluates the two 2026-09-22 PlaceSphere checkpoints (dreamer and
# graph_progress) for 20 simulator episodes each, without training anything:
# world_model.pt and imitation.pt are restored, Stage 2 is skipped. Prints a
# results table at the end; per-episode results are in
# $OUT_ROOT/<arm>/imitation_eval.json.

echo "================================="
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "sim_vla: PlaceSphere-v1, evaluate the 2026-09-22 dreamer and graph_progress checkpoints"
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

# Still needed when nothing trains: the normalizer and the env's settings are
# rebuilt from the dataset.
mkdir -p data
ln -sfn /home/tuannl/mnt_data/data/maniskill data/sim_vla_demos

export MS_ASSET_DIR=/mnt/data/tuannl
export WANDB_API_KEY="b1d6eed8871c7668a889ae74a621b5dbd2f3b070"
# SmolVLA was downloaded here by ../server1/slurm_collect_data.sh.
export HF_HOME=/home/tuannl/mnt_data/mshab_transfer_checkpoint
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p $HOME/output

nvidia-smi
nvidia-smi -l 100 > $HOME/output/gpu_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Overridable at submit time, e.g. SEED=1 sbatch <this file>.
SEED="${SEED:-0}"
# Simulator episodes per checkpoint, on seeds no demo used (eval.seeds_start).
EVAL_EPISODES=20

# Read only: the --out directories of the 2026-09-22 --save-checkpoints runs.
DREAMER_FROM=/home/tuannl/logdir/r2dreamer-graph/sim_vla/20260922_190405/placesphere/dreamer
GRAPH_PROGRESS_FROM=/home/tuannl/logdir/r2dreamer-graph/sim_vla/20260922_185807/placesphere/graph_progress

OUT_ROOT=$HOME/logdir/r2dreamer-graph/sim_vla/$TIMESTAMP/placesphere/eval_20260922_seed${SEED}
echo "[out] $OUT_ROOT"

evaluate() {
  local arm=$1 from=$2
  echo "--------------------------------"
  echo "[eval] $arm <- $from"
  echo "--------------------------------"
  for stage_file in world_model.pt imitation.pt; do
    if [ ! -f "$from/$stage_file" ]; then
      echo "FATAL: $from/$stage_file does not exist." >&2
      return 1
    fi
  done
  # Both stages are restored, so --save-checkpoints writes only
  # imitation_eval.json here.
  python -m sim_vla.training.pipeline \
    --task placesphere \
    --experiment "$arm" \
    --resume-from "$from" \
    --world-steps 0 \
    --imitation-steps 0 \
    --online-steps 0 \
    --eval-episodes $EVAL_EPISODES \
    --seed "$SEED" \
    --device cuda \
    --save-checkpoints \
    --out "$OUT_ROOT/$arm"
}

STATUS=0
evaluate dreamer "$DREAMER_FROM" || STATUS=1
evaluate graph_progress "$GRAPH_PROGRESS_FROM" || STATUS=1

echo "================================="
echo "PlaceSphere-v1, $EVAL_EPISODES episodes per checkpoint"
python - "$OUT_ROOT" <<'EOF'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print(f"{'arm':<16}{'success':>9}{'at end':>9}{'return':>9}{'steps to success':>18}")
for arm in ("dreamer", "graph_progress"):
    path = root / arm / "imitation_eval.json"
    if not path.is_file():
        print(f"{arm:<16}  no result: {path} was not written")
        continue
    result = json.loads(path.read_text(encoding="utf-8"))
    median = result.get("steps_to_success_median")
    median = "-" if median is None else f"{median:.0f}"
    print(f"{arm:<16}{result['success_rate']:>9.2f}"
          f"{result['success_at_end_rate']:>9.2f}"
          f"{result['env_return_mean']:>9.1f}{median:>18}")
print("success: at any step; at end: still successful at the last step;")
print("steps to success: median over the episodes that succeeded.")
EOF
echo "================================="

echo "Job finished"
exit $STATUS
