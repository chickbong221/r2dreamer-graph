#!/bin/bash
# The TD-MPC2 and SOLD SmolVLA integration suites, in stages.
#
#   bash sim_vla/integrations/run_tests.sh          # every stage
#   bash sim_vla/integrations/run_tests.sh 1 4      # stages 1 through 4
#   PYTHON=python3.11 bash sim_vla/integrations/run_tests.sh
#
# Staged because the failures are ordered: a broken window loader makes every
# later stage fail in a way that says nothing about the later stage. TD-MPC2
# comes before SOLD because it was implemented first and shares the utilities.
#
# Three outcomes per stage, and only the first is success:
#
#   passed      every required module ran and nothing failed
#   INCOMPLETE  a required module skipped, or nothing ran        (exit 2)
#   FAILED      something failed                                 (exit 1)
#
# Stages 5 and 8 are REQUIRED: they are the real pretrained checkpoint, the
# real simulator and upstream's own SOLDModule. A suite that reported green
# because lerobot or Lightning was missing would be worse than no suite, so a
# machine without them reports INCOMPLETE and names what it lacked.
#
# This is testing, not training. Budget minutes. The staged training entry
# points are separate and nothing here invokes them:
#
#   python -m sim_vla.integrations.tdmpc2.run pipeline
#   python -m sim_vla.integrations.sold.run  pipeline

set -u
FIRST="${1:-1}"
LAST="${2:-9}"
PY="${PYTHON:-python}"

cd "$(dirname "$0")/../.." || exit 1
STAGE="$PY -m sim_vla.tests.run_stage"

STAGES=(
  "1|vendored imports, parameter counting, action units, chunk masks|$STAGE sim_vla.integrations.tests.test_shared --require"
  "2|TD-MPC2 with the integration disabled, and its sizing|$STAGE sim_vla.integrations.tests.test_tdmpc2_native --require sim_vla.integrations.tests.test_tdmpc2_native"
  "3|TD-MPC2 conditioning, imitation, the five policy sites|$STAGE sim_vla.integrations.tests.test_tdmpc2_smolvla --require sim_vla.integrations.tests.test_tdmpc2_smolvla"
  "4|SOLD with the integration disabled, and its sizing|$STAGE sim_vla.integrations.tests.test_sold_native --require"
  "5|SOLD slot adapter, imitation, imagined-return gradients|$STAGE sim_vla.integrations.tests.test_sold_smolvla --require"
  "6|upstream SOLDModule with the flow actor attached|$STAGE sim_vla.integrations.tests.test_sold_online --require sim_vla.integrations.tests.test_sold_online"
  "7|the real SmolVLA checkpoint, dataset and simulator|$STAGE sim_vla.integrations.tests.test_real --require sim_vla.integrations.tests.test_real"
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
    echo "--- stage $num INCOMPLETE (a required module skipped)"
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
