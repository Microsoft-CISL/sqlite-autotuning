#!/bin/bash

# This is a simple script used to test the repo in the CI pipeline.

set -eu

set -x

# Move to repo root.
scriptdir=$(dirname "$(readlink -f "$0")")
cd "$scriptdir/.."

TRIAL_CONFIG_REPEAT_COUNT=2 MAX_SUGGESTIONS=2 LOG_LEVEL=INFO ./run_sqlite_demo.sh

# TODO: Validate that the number of trials increased as expected.

# Run the jupyter notebooks.
find . -maxdepth 2 -name '*.ipynb' -print0 | xargs -t -0 -P0 -n1 conda run -n mlos jupyter execute
