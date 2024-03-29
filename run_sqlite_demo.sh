#!/bin/bash

set -euo pipefail

scriptdir=$(dirname "$(readlink -f "$0")")
cd "$scriptdir/"

set -x
pip install -U -r requirements.txt
conda run -n mlos \
    mlos_bench --config config/cli/local-sqlite-opt.jsonc \
        --globals config/experiments/sqlite-sync-journal-pagesize-caching-experiment.jsonc \
        --trial-config-repeat-count 3 \
        --max-iterations 100 \
#        --log-level DEBUG
