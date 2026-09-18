#!/bin/bash
# The sim_vla test suite, in stages, stopping at the first failure.
#
# Staged because the failures are ordered: a broken sequence loader makes every
# later stage fail in a way that says nothing about the later stage. Running
# them in dependency order means the first red stage is the one to read.
#
#   bash sim_vla/run_tests.sh              # every stage
#   bash sim_vla/run_tests.sh 1 3          # stages 1 through 3
#   SIM_VLA_DEMOS=data/sim_vla_demos bash sim_vla/run_tests.sh
#
# This is testing, not training. Nothing here runs a full pretraining or an RL
# session; the heaviest stages load the pretrained checkpoint, take a few
# gradient steps on a handful of demonstrations, step the simulator for a few
# episodes, and write and reload a checkpoint. Budget minutes, not hours.
#
# A stage whose module does not exist yet is reported as PENDING and does not
# pass. That is deliberate: a suite that silently skips what has not been
# written reports green for work that has not happened.

set -u
FIRST="${1:-1}"
LAST="${2:-9}"
DEMOS="${SIM_VLA_DEMOS:-data/sim_vla_demos}"
PY="${PYTHON:-python}"

cd "$(dirname "$0")/.." || exit 1

# stage | description | what it runs
STAGES=(
  "1|data contract, sequence alignment, graph isolation|$PY -m unittest tests.test_sim_vla_data tests.test_sim_vla_pipeline"
  "2|collected datasets on this machine|$PY -m sim_vla.data.audit --root $DEMOS --graph"
  "3|conditional world model, both arms|$PY -m unittest sim_vla.tests.test_world_model"
  "4|pretrained SmolVLA loads, adapter takes gradient|$PY -m unittest sim_vla.tests.test_pretrained sim_vla.tests.test_adapter"
  "5|imitation trainer overfits a few demonstrations|$PY -m unittest sim_vla.tests.test_imitation"
  "6|simulator integration: env, replay, final observations|$PY -m unittest sim_vla.tests.test_env"
  "7|flow sampler and imagination gradients|$PY -m unittest sim_vla.tests.test_imagination"
  "8|critics, online loop, checkpoint write and resume|$PY -m unittest sim_vla.tests.test_online sim_vla.tests.test_checkpoint"
  "9|progress variant and its graph dependency|$PY -m unittest sim_vla.tests.test_progress"
)

pending=()
ran=0

for entry in "${STAGES[@]}"; do
  num="${entry%%|*}"
  rest="${entry#*|}"
  desc="${rest%%|*}"
  cmd="${rest#*|}"
  if [ "$num" -lt "$FIRST" ] || [ "$num" -gt "$LAST" ]; then continue; fi

  # A unittest target that does not import yet is pending, not failing. Checked
  # by import rather than by file path so a module that exists but cannot be
  # imported still counts as a real failure.
  module=$(echo "$cmd" | grep -o 'sim_vla\.tests\.[a-z_]*' | head -1)
  if [ -n "$module" ] && ! $PY -c "import importlib,sys; sys.exit(0 if importlib.util.find_spec('$module') else 1)" 2>/dev/null; then
    echo "=== stage $num: $desc"
    echo "--- PENDING: $module is not implemented yet"
    pending+=("$num:$desc")
    continue
  fi

  echo "=== stage $num: $desc"
  echo "--- $cmd"
  if ! eval "$cmd"; then
    echo
    echo "!!! stage $num failed: $desc"
    echo "!!! later stages were not run; fix this one first"
    exit 1
  fi
  ran=$((ran + 1))
  echo
done

echo "=== $ran stage(s) passed"
if [ ${#pending[@]} -gt 0 ]; then
  echo "=== ${#pending[@]} stage(s) pending, not passed:"
  for item in "${pending[@]}"; do echo "    $item"; done
  exit 2
fi
