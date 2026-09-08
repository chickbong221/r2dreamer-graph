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

# Each launcher parses, and carries exactly one active `python train.py`.
# Reading only: nothing here executes a launcher or starts training.
check_launchers() {
    local rc=0 script expected active
    # One command per single-arm launcher; two per merged one, B then A.
    for entry in runs/mshab/slurm_a_beta005.sh:1 runs/mshab/slurm_a_baseline.sh:1 \
                 runs/mshab/slurm_b_beta005.sh:1 runs/mshab/slurm_b_baseline.sh:1 \
                 runs/mshab/slurm_beta005.sh:2 runs/mshab/slurm_baseline.sh:2; do
        script=${entry%:*}
        expected=${entry##*:}
        if ! bash -n "$script"; then
            printf '%s: bash syntax error\n' "$script"
            rc=1
            continue
        fi
        active=$(grep -c '^python train\.py' "$script" || true)
        if [[ "$active" -ne "$expected" ]]; then
            printf '%s: %s active training command(s), expected %s\n' \
                "$script" "$active" "$expected"
            rc=1
            continue
        fi
        printf '%s: syntax ok, %s active training command(s)\n' "$script" "$active"
    done
    return "$rc"
}

# The settings the two experiments are actually specified with. Read from the
# files, never executed: a launcher that has drifted from the agreed budget,
# model or selection metric is the failure this catches, and it has to be
# caught before a node spends eight million steps on it.
#
# Its own function so that the grep pattern above does not share a paragraph
# with the overrides quoted below: the dead-reference test reads these files
# looking for training commands, and a pattern that matches one is not one.
check_launcher_settings() {
    local rc=0 script setting order
    for script in runs/mshab/slurm_a_beta005.sh runs/mshab/slurm_a_baseline.sh \
                  runs/mshab/slurm_b_beta005.sh runs/mshab/slurm_b_baseline.sh \
                  runs/mshab/slurm_beta005.sh runs/mshab/slurm_baseline.sh; do
        for setting in 'env.steps=8000000' 'checkpoint.start_step=6000000' \
                       'model=size100M'; do
            if ! grep -qF -- "$setting" "$script"; then
                printf '%s: missing %s\n' "$script" "$setting"
                rc=1
            fi
        done
    done
    for script in runs/mshab/slurm_b_beta005.sh runs/mshab/slurm_b_baseline.sh; do
        if ! grep -qF -- 'checkpoint.metric=eval_scene/training/success_once' "$script"; then
            printf '%s: B must select on the training-scene cases\n' "$script"
            rc=1
        fi
    done
    for script in runs/mshab/slurm_a_beta005.sh runs/mshab/slurm_a_baseline.sh; do
        if ! grep -qF -- 'checkpoint.metric=eval/success_once' "$script"; then
            printf '%s: A must select on eval/success_once\n' "$script"
            rc=1
        fi
    done
    for script in runs/mshab/slurm_a_beta005.sh runs/mshab/slurm_b_beta005.sh \
                  runs/mshab/slurm_beta005.sh; do
        if ! grep -qF -- 'model.progress.beta=0.1' "$script"; then
            printf '%s: graph arm must set the agreed progress beta\n' "$script"
            rc=1
        fi
    done
    # B before A in both merged launchers.
    for script in runs/mshab/slurm_beta005.sh runs/mshab/slurm_baseline.sh; do
        order=$(grep -o 'env=mshab_pick_[ab]' "$script" | tr '\n' ' ')
        if [[ "$order" != "env=mshab_pick_b env=mshab_pick_a " ]]; then
                printf '%s: experiment order is [%s], expected B then A\n' \
                "$script" "$order"
            rc=1
        fi
    done
    if [[ "$rc" -eq 0 ]]; then
        printf 'launcher settings ok: 8M budget, 6M eligibility, '
        printf '100M models, per-experiment metric, B then A\n'
    fi
    return "$rc"
}

# Structure only: the counts, the disjointness and the pinned lighting scene.
# No simulator and no dataset, so it runs anywhere. `freeze_scene_split
# --check` in the full run below is what compares it against installed plans.
check_scene_manifest() {
    python - <<'PYEOF'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("scene_manifest",
                                              "envs/scene_manifest.py")
module = importlib.util.module_from_spec(spec)
sys.modules["scene_manifest"] = module
spec.loader.exec_module(module)
split = module.load_manifest("configs/scenes/mshab_pick_b.json")
counts = module.counts(split)
assert counts == {"train": 5, "held_out": 30, "evaluation": 35}, counts
assert not set(split.train) & set(split.held_out)
assert split.lighting_scene == "v3_sc0_staging_00.scene_instance.json"
print(f"scene manifest: {counts}, lighting on {split.lighting_scene}")
PYEOF
}

if [[ "$MODE" == full ]]; then
    run_check launchers check_launchers
    run_check launcher_settings check_launcher_settings
    run_check scene_manifest check_scene_manifest
    run_check tests python -m unittest discover -s tests -t .
    # Needs the installed MS-HAB task plans; writes nothing.
    run_check scene_split python -m scenegraph.tools.freeze_scene_split --check \
        --task tidy_house --subtask pick --obj 004_sugar_box --split train
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

# Exercise the training capacities and protected-node/FIFO packing.
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
    --n-max 8 --e-max 168 \
    --disable-object-object-relations \
    --out "$MSHAB_CHECK_DIR/pick_trace.json"

run_check capacity python -m scenegraph.tools.audit_graph_capacity \
    --whitelist-dir scenegraph/configs/subtask_whitelists/tidy_house \
    --subtask pick \
    --occupancy-json "$MSHAB_CHECK_DIR/pick_trace.occupancy.json"

printf '\nThis is a one-scene probe; FIFO bounds context on every scene.\n'
printf 'V1 keeps the existing mined support-plane proxy; exact-surface refinement is deferred.\n'
