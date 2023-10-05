#!/bin/bash

set -eu
set -x

OUT_DIR="$1"    # should be a local dir, not smb/cifs, for the results to get saved

scriptdir=$(dirname "$(readlink -f "$0")")
cd "$scriptdir"
source ../common.sh

OUT_CUMULATIVE="$OUT_DIR/benchbase-${BENCHBASE_BENCHMARK}-system-metrics.csv"

# TODO: Add support for combining multiple metrics from several benchmarks together.

# Parse the summary results file and save the details as a CSV (with a header).

time_file=$(ls -tr "$OUT_DIR"/results/exec-${BENCHBASE_BENCHMARK}.time.out | tail -n1)
if [ -z "$time_file" ]; then
    echo "ERROR: Failed to find a time.out file to use." >&2
    exit 1
fi

echo "metric,value" > $OUT_CUMULATIVE
cat "$time_file" \
    | grep ': ' \
    | sed -r -e 's/^\s*//' -e 's/: /,/' \
    >> "$OUT_CUMULATIVE"
cat "$OUT_CUMULATIVE"
