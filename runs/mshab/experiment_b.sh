#!/bin/bash
# Experiment B -- one object, one training scene, evaluated on all 63.
# GRAPH_BRANCH=false selects DreamerV3; FINETUNE=true adds 3M transfer steps.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

WL=scenegraph/configs/subtask_whitelists/tidy_house
UNION=$WL/pick_all.json

# Same training/evaluation/transfer workflow for either model branch.
case "${GRAPH_BRANCH:-true}" in
  true) MODEL=size50M_graph_simple ;;
  false) MODEL=size50M ;;
  *) echo "GRAPH_BRANCH must be true or false" >&2; exit 2 ;;
esac

require() { [ -n "${2:-}" ] || { echo "BLOCKED: $1" >&2; exit 1; }; }

if [[ "${GRAPH_BRANCH:-true}" == true ]]; then
  [ -f "$UNION" ] || { echo "BLOCKED: no $UNION" >&2; exit 1; }
  python tests/probes/validate_task_assets.py \
    --task tidy_house --disable-object-object-relations --targets \
    002_master_chef_can 003_cracker_box 004_sugar_box 005_tomato_soup_can \
    007_tuna_fish_can 008_pudding_box 009_gelatin_box 010_potted_meat_can \
    024_bowl || { echo "BLOCKED: the tidy_house assets do not validate" >&2; exit 1; }
  require "set ENTITY_VOCAB from audit_graph_capacity" "${ENTITY_VOCAB:-}"
fi
ENTITY_VOCAB=${ENTITY_VOCAB:-19}

TS=$(date +%Y%m%d_%H%M%S)

# The explicit panel allocates 63 B environments plus 30 C environments.
python train.py env=mshab_pick_b model="$MODEL" \
    env.graph.whitelist_dir="$WL" model.graph.entity_vocab="$ENTITY_VOCAB" \
    model.graph.n_max=8 model.graph.e_max=168 \
    checkpoint.enabled=true checkpoint.metric=eval/success_once checkpoint.tiebreak='' \
    finetune.enabled="${FINETUNE:-false}" \
    wandb.group=mshab_tidy_house_pick_B wandb.name="B-scenes-and-lighting-$MODEL" \
    logdir="$REPO_ROOT/logdir/$TS/experiment_B" "$@"
