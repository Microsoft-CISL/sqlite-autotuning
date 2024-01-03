#!/bin/bash -x

mlos_bench --config config/cli/local-sqlite-opt.jsonc \
    --globals config/experiments/sqlite-sync-journal-pagesize-caching-experiment.jsonc \
    --max-iterations 100
