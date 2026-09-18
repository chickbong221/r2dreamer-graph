#!/bin/bash
# The sim_vla test suite, in stages, stopping at the first failure.
#
#   bash sim_vla/run_tests.sh              # every stage
#   bash sim_vla/run_tests.sh 1 4          # stages 1 through 4
#   SIM_VLA_DEMOS=data/sim_vla_demos bash sim_vla/run_tests.sh
#
# Staged because the failures are ordered: a broken sequence loader makes every
# later stage fail in a way that says nothing about the later stage.
#
# This is testing, not training. The heaviest stages load the pretrained
# checkpoint, take a few hundred optimiser steps on tiny tensors, step the
# simulator for a handful of transitions, and write and reload a checkpoint.
# Budget minutes. Full pretraining and RL are separate entry points --
# sim_vla/training/pretrain_world_model.py and sim_vla/training/online.py --
# and nothing here invokes them.
#
# Three outcomes per stage, and only the first is success:
#
#   passed      something ran and nothing failed
#   INCOMPLETE  every test skipped -- a missing dependency, weights or dataset
#   FAILED      a test failed
#
# A stage that skipped everything exits non-zero. A suite that reported green
# because lerobot was not installed would be worse than no suite.

set -u
FIRST="${1:-1}"
LAST="${2:-9}"
DEMOS="${SIM_VLA_DEMOS:-data/sim_vla_demos}"
PY="${PYTHON:-python}"

cd "$(dirname "$0")/.." || exit 1
STAGE="$PY -m sim_vla.tests.run_stage"

STAGES=(
  "1|data contract, sequence alignment, graph isolation|$PY -m unittest tests.test_sim_vla_data tests.test_sim_vla_pipeline"
  "2|collected datasets, against their own metadata|$PY -m sim_vla.data.audit --root $DEMOS --graph"
  "3|world model builds and trains in both arms|$STAGE sim_vla.tests.test_world_model"
  "4|real pretrained SmolVLA, adapter, gradient flow|$STAGE sim_vla.tests.test_pretrained sim_vla.tests.test_adapter"
  "5|flow-matching imitation, chunk masking, overfit|$STAGE sim_vla.tests.test_imitation"
  "6|simulator integration: env, replay, final observation|$STAGE sim_vla.tests.test_env"
  "7|flow sampler gradients and latent imagination|$STAGE sim_vla.tests.test_imagination"
  "8|critics, actor update, checkpoint write and resume|$STAGE sim_vla.tests.test_online sim_vla.tests.test_checkpoint"
  "9|progress shaping and its graph dependency|$STAGE sim_vla.tests.test_progress"
)

passed=0
incomplete=()

for entry in "${STAGES[@]}"; do
  num="${entry%%|*}"
  rest="${entry#*|}"
  desc="${rest%%|*}"
  cmd="${rest#*|}"
  if [ "$num" -lt "$FIRST" ] || [ "$num" -gt "$LAST" ]; then continue; fi

  echo "=== stage $num: $desc"
  echo "--- $cmd"
  eval "$cmd"
  status=$?
  if [ $status -eq 2 ]; then
    echo "--- stage $num INCOMPLETE (nothing verified)"
    incomplete+=("$num:$desc")
  elif [ $status -ne 0 ]; then
    echo
    echo "!!! stage $num FAILED: $desc"
    echo "!!! later stages were not run; fix this one first"
    exit 1
  else
    passed=$((passed + 1))
  fi
  echo
done

echo "=== $passed stage(s) passed"
if [ ${#incomplete[@]} -gt 0 ]; then
  echo "=== ${#incomplete[@]} stage(s) INCOMPLETE -- not verified, not passed:"
  for item in "${incomplete[@]}"; do echo "    $item"; done
  echo "=== install the missing dependency or collect the missing data, then re-run"
  exit 2
fi
