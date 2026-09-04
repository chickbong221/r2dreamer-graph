#!/bin/bash
# Experiment B -- one object, one training scene, evaluated on all 63.
#
# READINESS -- this script refuses to run until each is resolved. None is a
# value to guess at:
#
#   1. The tidy_house Pick assets must be re-mined from a schema-v9 collection
#      of all nine objects. The committed ones carry _bins_migrated_pre_anchor
#      and GraphBuilder refuses them.
#   2. model.graph.entity_vocab / n_max / e_max come from the mined asset and
#      a per-configuration capacity audit, never from a remembered number:
#        python -m scenegraph.tools.audit_graph_capacity #          --whitelist-dir scenegraph/configs/subtask_whitelists/tidy_house
#   3. checkpoint.metric is unset. Which evaluation selects the best model is
#      an experiment decision.
#   4. The terminal rungs must be checked against the real bins:
#        python -m scenegraph.tools.check_terminal_rungs --asset ... #          --schedule ... --sites ... --tolerance <pick_cfg.ee_rest_thresh>
set -euo pipefail

WL=scenegraph/configs/subtask_whitelists/tidy_house
UNION=$WL/pick_all.json

require() { [ -n "${2:-}" ] || { echo "BLOCKED: $1" >&2; exit 1; }; }

[ -f "$UNION" ] || { echo "BLOCKED: no $UNION -- mine the assets first" >&2; exit 1; }
python - <<'EOF' || exit 1
import json, sys
d = json.load(open("scenegraph/configs/subtask_whitelists/tidy_house/pick_all.json"))
if d.get("_bins_migrated_pre_anchor"):
    sys.exit("BLOCKED: the union asset is key-migrated, not re-mined")
EOF

require "set ENTITY_VOCAB from audit_graph_capacity" "${ENTITY_VOCAB:-}"
require "set N_MAX from the capacity audit"          "${N_MAX:-}"
require "set E_MAX from the capacity audit"          "${E_MAX:-}"
require "set CKPT_METRIC (e.g. eval/success_once)"   "${CKPT_METRIC:-}"

TS=$(date +%Y%m%d_%H%M%S)

# eval_episode_num is 63 and eval_even_build_configs is true, so MS-HAB checks
# divisibility and hands one configuration to each evaluation sub-scene.
python train.py   env=mshab_pick_b   model=size50M_graph_simple   env.graph.whitelist_dir=$WL   model.graph.entity_vocab=$ENTITY_VOCAB   model.graph.n_max=$N_MAX   model.graph.e_max=$E_MAX   checkpoint.enabled=true   checkpoint.metric=$CKPT_METRIC   checkpoint.tiebreak=${CKPT_TIEBREAK:-}   wandb.group=mshab_tidy_house_pick_B   wandb.name=B-one-object-scene-generalisation   logdir=$HOME/logdir/r2dreamer-graph/$TS/experiment_B
