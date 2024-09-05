#!/bin/bash

# This is a simple script used to test the repo in the CI pipeline.

set -eu

# Move to repo root.
scriptdir=$(dirname "$(readlink -f "$0")")
cd "$scriptdir/.."

set -x

experiment_id="sqlite-opt-demo"
storage_db="mlos_bench.sqlite"

trial_id_start=$(echo "SELECT MAX(trial_id) FROM trial WHERE exp_id='$experiment_id';" | sqlite3 "$storage_db")

TRIAL_CONFIG_REPEAT_COUNT=2 MAX_SUGGESTIONS=2 LOG_LEVEL=INFO ./run_sqlite_demo.sh

trial_success_count=$(echo "SELECT COUNT(*) FROM trial WHERE exp_id='$experiment_id' AND status='SUCCEEDED' AND trial_id > $trial_id_start;" | sqlite3 "$storage_db")

# Validate that the number of trials increased as expected.
if [ $trial_success_count -ne 4 ]; then
    echo "ERROR: Unexpected number of trials succeeded: $trial_success_count"
    exit 1
fi

echo "OK"
exit 0
