#!/bin/bash
# Experiment B -- one object, one training scene, evaluated on all 63.
#
# READINESS -- this script refuses to run until each is resolved. None is a
# value to guess at:
#
#   1. The tidy_house Pick assets must validate. The nine-object schema-v9
#      collection is mined; what a run still needs is every bin the graph
#      builder demands, which is what validate_task_assets checks below.
#   2. entity_vocab comes from the mined asset. Capacity: 8 nodes, 168 edges.
#      EE + target + rest site are protected; five context rows use FIFO.
#      With object_object_spatial=false, six physical nodes need at most
#      6*6 + C(6,2)*6 + 3 = 129 edges with OO restored, or 39 EE-only.
#        python -m scenegraph.tools.audit_graph_capacity #          --whitelist-dir scenegraph/configs/subtask_whitelists/tidy_house
#   3. Checkpoints use eval/success_once (higher is better), with no tiebreak.
#      Best-only saving still starts at 8M environment steps.
#   4. The terminal rungs must be checked against the real bins:
#        python -m scenegraph.tools.check_terminal_rungs --asset ... #          --schedule ... --sites ... --tolerance <pick_cfg.ee_rest_thresh>
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

WL=scenegraph/configs/subtask_whitelists/tidy_house
UNION=$WL/pick_all.json

require() { [ -n "${2:-}" ] || { echo "BLOCKED: $1" >&2; exit 1; }; }

[ -f "$UNION" ] || { echo "BLOCKED: no $UNION -- mine the assets first" >&2; exit 1; }
# Every gate a run hits at construction, checked in a second rather than
# after a scene build: the nine targets, no unresolved member, every required
# bin calibrated, every surface with a plane, and a schedule that compiles.
python tests/probes/validate_task_assets.py --task tidy_house --disable-object-object-relations --targets   002_master_chef_can 003_cracker_box 004_sugar_box 005_tomato_soup_can   007_tuna_fish_can 008_pudding_box 009_gelatin_box 010_potted_meat_can   024_bowl || { echo "BLOCKED: the tidy_house assets do not validate" >&2; exit 1; }

require "set ENTITY_VOCAB from audit_graph_capacity" "${ENTITY_VOCAB:-}"

TS=$(date +%Y%m%d_%H%M%S)

# eval_episode_num is 63 and eval_even_build_configs is true, so MS-HAB checks
# divisibility and hands one configuration to each evaluation sub-scene.
python train.py env=mshab_pick_b model=size50M_graph_simple \
    env.graph.whitelist_dir="$WL" model.graph.entity_vocab="$ENTITY_VOCAB" \
    model.graph.n_max=8 model.graph.e_max=168 \
    checkpoint.enabled=true checkpoint.metric=eval/success_once checkpoint.tiebreak='' \
    wandb.group=mshab_tidy_house_pick_B wandb.name=B-one-object-scene-generalisation \
    logdir="$REPO_ROOT/logdir/$TS/experiment_B" "$@"
