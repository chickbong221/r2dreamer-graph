#!/usr/bin/env bash
# Validation only: no training, asset mining or promotion of collected data.
# Usage: bash runs/mshab/validate.sh [--probe-only]
set -euo pipefail

MODE=${1:-full}
if [[ $# -gt 1 || ( "$MODE" != full && "$MODE" != --probe-only ) ]]; then
    echo "Usage: bash runs/mshab/validate.sh [--probe-only]" >&2
    exit 2
fi

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/logdir"
# Unique directory inside the repository's persistent logdir, never /tmp.
MSHAB_CHECK_DIR=$(mktemp -d "$REPO_ROOT/logdir/mshab-validation.XXXXXX")
printf 'Results: %s\n' "$MSHAB_CHECK_DIR"
trap 'rc=$?; printf "Validation EXIT=%s; results: %s\n" "$rc" "$MSHAB_CHECK_DIR"' EXIT

run_check() {
    local name="$1"
    shift
    set +e
    "$@" 2>&1 | tee "$MSHAB_CHECK_DIR/$name.log"
    local rc=${PIPESTATUS[0]}
    set -e
    printf '%s EXIT=%s\n' "$name" "$rc" | tee -a "$MSHAB_CHECK_DIR/status.txt"
    return "$rc"
}

if [[ "$MODE" == full ]]; then
    run_check tests python -m unittest discover -s tests -t .
    run_check assets python tests/probes/validate_task_assets.py \
        --task tidy_house --disable-object-object-relations --targets \
        002_master_chef_can 003_cracker_box 004_sugar_box \
        005_tomato_soup_can 007_tuna_fish_can 008_pudding_box \
        009_gelatin_box 010_potted_meat_can 024_bowl
    run_check terminal_rungs python -m scenegraph.tools.check_terminal_rungs \
        --asset scenegraph/configs/subtask_whitelists/tidy_house/pick_all.json \
        --schedule scenegraph/configs/schedules/tidy_house/pick.json \
        --tolerance 0.05
else
    run_check probe_tests python -m unittest tests.test_potential_probe -v
fi

# Exercise the approved V1 training capacities, including overflow checks.
# The collector wrapper's incidental files stay in this validation directory.
run_check potential python tests/probes/probe_policy_potential.py \
    --whitelist-dir scenegraph/configs/subtask_whitelists/tidy_house \
    --affordance scenegraph/configs/affordances/tidy_house.json \
    --thresholds scenegraph/configs/thresholds.yaml \
    --asset-dir "$MSHAB_CHECK_DIR/probe_data" \
    --ckpt-root "${MSHAB_CKPT_ROOT:-/root/projects/ReLDreamer/mshab_checkpoints}" \
    --task tidy_house --subtask pick --obj 004_sugar_box --algo rl \
    --build-config v3_sc0_staging_00.scene_instance.json \
    --num-envs 4 --max-episode-steps 200 --max-total-steps 4000 \
    --n-max 12 --e-max 384 \
    --disable-object-object-relations \
    --out "$MSHAB_CHECK_DIR/pick_trace.json"

run_check capacity python -m scenegraph.tools.audit_graph_capacity \
    --whitelist-dir scenegraph/configs/subtask_whitelists/tidy_house \
    --subtask pick \
    --occupancy-json "$MSHAB_CHECK_DIR/pick_trace.occupancy.json"

printf '\nThis is a one-scene observed-capacity check, not a 63-scene bound.\n'
printf 'V1 keeps the existing mined support-plane proxy; exact-surface refinement is deferred.\n'
