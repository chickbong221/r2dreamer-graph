#!/usr/bin/env bash
#SBATCH --job-name=mshab-h100-diag
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/%u/output/%x_%j.out
#SBATCH --error=/home/%u/output/%x_%j.err

set -Eeuo pipefail

MODE="${1:-all}"
case "$MODE" in
  vulkan|smoke-small|smoke-scale|all) ;;
  *)
    echo "Usage: sbatch $0 [vulkan|smoke-small|smoke-scale|all]" >&2
    exit 2
    ;;
esac

echo "================================="
echo "MS-HAB H100 diagnostic"
echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-interactive}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Mode: $MODE"
echo "================================="

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dreamer

REPO_DIR="${REPO_DIR:-$HOME/projects/r2dreamer-graph}"
NVIDIA_USERSPACE_VERSION="${NVIDIA_USERSPACE_VERSION:-570.133.20}"
NVIDIA_USERSPACE_DIR="${NVIDIA_USERSPACE_DIR:-$HOME/nvidia-userspace/NVIDIA-Linux-x86_64-${NVIDIA_USERSPACE_VERSION}}"
export MS_ASSET_DIR="${MS_ASSET_DIR:-/mnt/data/tuannl}"
export DINO_WEIGHTS="${DINO_WEIGHTS:-/home/tuannl/mnt_data/checkpoints/dinov2_vits14_reg4_pretrain.pth}"

EGL_DRIVER="$NVIDIA_USERSPACE_DIR/libEGL_nvidia.so.${NVIDIA_USERSPACE_VERSION}"
if [[ ! -d "$REPO_DIR" ]]; then
  echo "Repository not found: $REPO_DIR" >&2
  exit 1
fi
if [[ ! -f "$EGL_DRIVER" ]]; then
  echo "NVIDIA EGL driver not found: $EGL_DRIVER" >&2
  exit 1
fi
if [[ ! -f "$DINO_WEIGHTS" ]]; then
  echo "DINOv2 checkpoint not found: $DINO_WEIGHTS" >&2
  exit 1
fi
if ! command -v vulkaninfo >/dev/null 2>&1; then
  echo "vulkaninfo is required. Install it before submitting:" >&2
  echo "  conda install -n dreamer -c conda-forge vulkan-tools" >&2
  exit 1
fi

mkdir -p "$HOME/output"
DIAG_DIR="${DIAG_DIR:-$HOME/output/mshab_h100_diag_${SLURM_JOB_ID:-manual}}"
mkdir -p "$DIAG_DIR"
ICD_FILE="$DIAG_DIR/nvidia_icd_egl.json"

cat > "$ICD_FILE" <<EOF
{
  "file_format_version": "1.0.1",
  "ICD": {
    "library_path": "$EGL_DRIVER",
    "api_version": "1.3.0"
  }
}
EOF

export LD_LIBRARY_PATH="$NVIDIA_USERSPACE_DIR:${LD_LIBRARY_PATH:-}"
export VK_DRIVER_FILES="$ICD_FILE"
export VK_ICD_FILENAMES="$ICD_FILE"
export VK_LOADER_DEBUG="${VK_LOADER_DEBUG:-error,warn}"

cd "$REPO_DIR"

echo "Diagnostic output: $DIAG_DIR"
nvidia-smi -L
nvidia-smi
echo "Git commit: $(git rev-parse HEAD)"

python - <<'PY'
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

print(f"Python: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")
for package in (
    "torch",
    "torchrl",
    "tensordict",
    "mani-skill",
    "sapien",
    "gymnasium",
    "numpy",
    "wandb",
):
    try:
        print(f"{package}: {version(package)}")
    except PackageNotFoundError:
        print(f"{package}: NOT INSTALLED")
PY

check_vulkan() {
  echo "--- Vulkan summary ---"
  if ! timeout --foreground 30s vulkaninfo --summary 2>&1 | tee "$DIAG_DIR/vulkan-summary.txt"; then
    echo "FAIL: vulkaninfo --summary failed or timed out." >&2
    return 1
  fi

  echo "--- Vulkan queue capabilities ---"
  if ! timeout --foreground 60s vulkaninfo > "$DIAG_DIR/vulkan-full.txt" 2>&1; then
    echo "FAIL: full vulkaninfo failed or timed out." >&2
    return 1
  fi
  grep -i -A8 "queueFlags" "$DIAG_DIR/vulkan-full.txt" | tee "$DIAG_DIR/vulkan-queues.txt" || true
  if ! grep -qiE "GRAPHICS|VK_QUEUE_GRAPHICS_BIT" "$DIAG_DIR/vulkan-queues.txt"; then
    echo "FAIL: no graphics-capable Vulkan queue was reported." >&2
    return 1
  fi
  echo "PASS: Vulkan exposes a graphics-capable queue."
}

run_smoke() {
  local label="$1"
  local timeout_seconds="$2"
  local num_envs="$3"
  local build_configs="$4"
  local log_file="$DIAG_DIR/${label}.txt"

  echo "--- $label: envs=$num_envs build_configs=$build_configs ---"
  set +e
  timeout --foreground "${timeout_seconds}s" \
    python -X faulthandler runs/smoke_mshab.py \
      --num-envs "$num_envs" \
      --build-configs "$build_configs" \
      --steps 1 \
      --device cuda:0 \
      2>&1 | tee "$log_file"
  local status="${PIPESTATUS[0]}"
  set -e

  if [[ "$status" -eq 124 ]]; then
    echo "FAIL: $label timed out after ${timeout_seconds}s." >&2
    return 1
  fi
  if [[ "$status" -ne 0 ]]; then
    echo "FAIL: $label exited with status $status." >&2
    return "$status"
  fi
  echo "PASS: $label completed."
}

check_vulkan

case "$MODE" in
  vulkan)
    ;;
  smoke-small)
    run_smoke smoke-small 300 1 1
    ;;
  smoke-scale)
    run_smoke smoke-scale 600 32 63
    ;;
  all)
    run_smoke smoke-small 300 1 1
    run_smoke smoke-scale 600 32 63
    ;;
esac

echo "All requested diagnostics passed. Logs: $DIAG_DIR"
