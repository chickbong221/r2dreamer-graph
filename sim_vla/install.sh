#!/usr/bin/env bash
#   bash sim_vla/install.sh                    # new conda env "dreamer"
#   ENV_NAME=simvla bash sim_vla/install.sh
#   RECREATE=1 bash sim_vla/install.sh         # replace an existing env
#   TORCH_CUDA=cu126 bash sim_vla/install.sh
#   bash sim_vla/install.sh --verify           # check the active env

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQUIREMENTS="$REPO_ROOT/sim_vla/requirements.txt"
ENV_NAME="${ENV_NAME:-dreamer}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"
PYTHON_VERSION="3.11"

verify() {
  cd "$REPO_ROOT"
  python - <<'PY'
import importlib
import os
import platform
import site
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

failed, incomplete = [], []


def say(status, what, detail=""):
    if status == "FAIL":
        failed.append(what)
    elif status == "INCOMPLETE":
        incomplete.append(what)
    print(f"[verify] {status:<10} {what}" + (f": {detail}" if detail else ""),
          flush=True)


say("ok" if sys.version_info[:2] == (3, 11) else "FAIL", "python",
    platform.python_version())

pinned = []
for line in Path("sim_vla/requirements.txt").read_text().splitlines():
    spec = line.split("#", 1)[0].strip()
    if "==" not in spec:
        continue
    name, want = (part.strip() for part in spec.split("==", 1))
    name = name.split("[", 1)[0]
    try:
        have = version(name)
    except PackageNotFoundError:
        say("FAIL", name, f"not installed (requirements.txt pins {want})")
        continue
    if have.split("+", 1)[0] != want:
        say("FAIL", name, f"{have} installed, requirements.txt pins {want}")
    else:
        pinned.append(f"{name} {have}")
say("ok", "pins", ", ".join(pinned))

check = subprocess.run([sys.executable, "-m", "pip", "check"],
                       capture_output=True, text=True)
lines = [line for line in (check.stdout + check.stderr).splitlines()
         if line.strip() and "No broken requirements" not in line]
expected = [line for line in lines
            if line.startswith("rerun-sdk ") and "numpy" in line]
other = [line for line in lines if line not in expected]
if other:
    say("FAIL", "pip check", "; ".join(other))
else:
    say("ok", "pip check", "only rerun-sdk's numpy>=2, as expected"
        if expected else "no broken requirements")

user_site = Path(site.getusersitepackages())
if site.ENABLE_USER_SITE and user_site.is_dir() and any(user_site.iterdir()):
    print(f"[verify] warning    {user_site} has packages, and they shadow "
          "this env's. Empty it, or export PYTHONNOUSERSITE=1.", flush=True)

modules = [
    "sim_vla.training.pipeline", "sim_vla.training.train_imitation",
    "sim_vla.training.progress", "sim_vla.models.smolvla_actor",
    "sim_vla.models.latent_adapter", "sim_vla.envs.maniskill",
    "sim_vla.evaluation.policy", "envs.maniskill",
    "scenegraph.figures.graph_source", "scenegraph.core.schedule",
    "h5py", "wandb", "cv2", "mani_skill.envs",
    "lerobot.policies.smolvla.modeling_smolvla",
]
broken = 0
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        broken += 1
        say("FAIL", f"import {name}", f"{type(exc).__name__}: {exc}")
if not broken:
    say("ok", "imports", f"{len(modules)} modules")

try:
    import torch

    if torch.cuda.is_available():
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        (x @ x).sum().item()
        say("ok", "cuda", f"torch {torch.__version__}, CUDA "
            f"{torch.version.cuda}, {torch.cuda.get_device_name(0)}")
    else:
        say("INCOMPLETE", "cuda",
            "no GPU visible here; training needs one. Run "
            "`bash sim_vla/install.sh --verify` on a GPU node")
except Exception as exc:
    say("FAIL", "cuda", f"{type(exc).__name__}: {exc}")

try:
    import gymnasium as gym
    import numpy as np
    import mani_skill.envs

    env = gym.make("StackCube-v1", obs_mode="rgb+segmentation",
                   control_mode="pd_joint_pos", render_mode="rgb_array",
                   sensor_configs=dict(width=112, height=112),
                   sim_backend="cpu", num_envs=1,
                   reward_mode="normalized_dense", max_episode_steps=150)
    try:
        env.reset(seed=0)
        obs, *_ = env.step(np.zeros(env.action_space.shape, np.float32))
        rgb = obs["sensor_data"]["base_camera"]["rgb"]
    finally:
        env.close()
    say("ok", "simulator", f"StackCube-v1 rendered {tuple(rgb.shape)}")
except Exception as exc:
    say("FAIL", "simulator", f"{type(exc).__name__}: {exc}. SAPIEN renders "
        "with Vulkan: see 'Vulkan' in sim_vla/README.md")

try:
    import yaml
    from huggingface_hub import constants

    from sim_vla.models.pretrained import load_policy, model_facts

    actor = yaml.safe_load(Path("sim_vla/configs/base.yaml").read_text())["actor"]
    loaded = load_policy(str(actor["pretrained"]), str(actor["revision"]))
    facts = model_facts(loaded)
    say("ok", "smolvla", f"{loaded.repo_id}@{loaded.revision[:12]}, lerobot "
        f"{loaded.lerobot_version}, chunk {facts['chunk_size']}, cached in "
        f"{constants.HF_HUB_CACHE}")
except Exception as exc:
    say("FAIL", "smolvla", f"{type(exc).__name__}: {exc}")

print("[verify] HF_HOME=" + os.environ.get("HF_HOME", "(unset: ~/.cache/huggingface)")
      + " WANDB_API_KEY=" + ("set" if os.environ.get("WANDB_API_KEY") else "unset"),
      flush=True)
if failed:
    print(f"[verify] FAILED: {', '.join(failed)}", flush=True)
    sys.exit(1)
if incomplete:
    print(f"[verify] INCOMPLETE: {', '.join(incomplete)} could not be checked "
          "here", flush=True)
    sys.exit(2)
print("[verify] passed", flush=True)
PY
}

if [ "${1:-}" = "--verify" ]; then
  verify
  exit $?
fi

if ! command -v conda >/dev/null 2>&1; then
  for base in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3"; do
    if [ -f "$base/etc/profile.d/conda.sh" ]; then
      source "$base/etc/profile.d/conda.sh"
      break
    fi
  done
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found: put it on PATH or install miniconda/miniforge" >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

env_exists() {
  [ -d "$(conda info --base)/envs/$1" ] && return 0
  conda env list | awk -v name="$1" \
    '$1 == name || $NF ~ ("/" name "$") {found = 1} END {exit !found}'
}

if env_exists "$ENV_NAME"; then
  if [ "${RECREATE:-0}" = "1" ]; then
    echo ">>> removing the existing conda env '$ENV_NAME' (RECREATE=1)"
    conda env remove -y -n "$ENV_NAME"
  else
    echo "conda env '$ENV_NAME' already exists and is left untouched." >&2
    echo "  RECREATE=1 bash sim_vla/install.sh           replaces it" >&2
    echo "  ENV_NAME=<new> bash sim_vla/install.sh       builds another" >&2
    exit 1
  fi
fi

echo ">>> creating conda env '$ENV_NAME' (python $PYTHON_VERSION)"
conda create -y -n "$ENV_NAME" -c conda-forge --override-channels \
  "python=$PYTHON_VERSION" pip
set +u
conda activate "$ENV_NAME"
set -u

echo ">>> installing sim_vla/requirements.txt (torch from the $TORCH_CUDA index)"
python -m pip install --upgrade pip
WITHOUT_NUMPY="$(mktemp)"
trap 'rm -f "$WITHOUT_NUMPY"' EXIT
grep -vE '^numpy==' "$REQUIREMENTS" > "$WITHOUT_NUMPY"
python -m pip install -r "$WITHOUT_NUMPY" \
  --extra-index-url "https://download.pytorch.org/whl/$TORCH_CUDA"

NUMPY="$(grep -E '^numpy==' "$REQUIREMENTS" | awk '{print $1}')"
echo ">>> installing $NUMPY; pip reports rerun-sdk's numpy>=2, which is expected"
python -m pip install "$NUMPY"

# sapien pulls in opencv-python; keep the headless cv2 build on disk.
HEADLESS="$(grep -E '^opencv-python-headless==' "$REQUIREMENTS" | awk '{print $1}')"
python -m pip install --force-reinstall --no-deps "$HEADLESS"

echo ">>> verifying the env"
verify
echo ">>> done: conda activate $ENV_NAME, then see sim_vla/README.md"
