#!/bin/bash

# This is a simple script used to test the repo in the CI pipeline.

set -eu

# Move to repo root.
scriptdir=$(dirname "$(readlink -f "$0")")
cd "$scriptdir/.."

set -x

# Run the jupyter notebooks.
find . -maxdepth 2 -name '*.ipynb' -print0 | xargs -t -0 -P0 -n1 conda run -n mlos jupyter execute

echo "OK"
exit 0
