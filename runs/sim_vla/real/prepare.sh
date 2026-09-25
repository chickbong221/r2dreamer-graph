#!/bin/bash
# Download the SO-101 episodes and convert one task, on the login node (it
# needs internet). Run from the repository root; reruns skip what is current.
#   bash runs/sim_vla/real/prepare.sh stackcube
set -eo pipefail
TASK="${1:?usage: bash runs/sim_vla/real/prepare.sh <stackcube|cubes_in_cup>}"
test -f sim_vla/__init__.py || { echo "run from the repository root" >&2; exit 1; }

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dreamer
source runs/sim_vla/real/setup.sh

python -m sim_vla.data.prepare_real --task "$TASK" --lerobot "$SO101_DATA/so101-multitask"
