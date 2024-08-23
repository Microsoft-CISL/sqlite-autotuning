#!/bin/bash

set -euo pipefail

scriptdir=$(dirname "$(readlink -f "$0")")
cd "$scriptdir/"

TRIAL_CONFIG_REPEAT_CONFIG=${MAX_SUGGESTIONS:-3}
MAX_SUGGESTIONS=${MAX_SUGGESTIONS:-100}
LOG_LEVEL=${LOG_LEVEL:-INFO}

set -x
conda run -n mlos pip install -U -r requirements.txt
conda run -n mlos \
    mlos_bench --config config/cli/local-sqlite-opt.jsonc \
        --globals config/experiments/sqlite-sync-journal-pagesize-caching-experiment.jsonc \
        --trial-config-repeat-count $TRIAL_CONFIG_REPEAT_CONFIG \
        --max_suggestions $MAX_SUGGESTIONS \
        --log-level $LOG_LEVEL
