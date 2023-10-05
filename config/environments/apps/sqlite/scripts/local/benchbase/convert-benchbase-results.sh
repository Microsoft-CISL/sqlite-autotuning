#!/bin/bash

set -eu
set -x

OUT_DIR="$1"    # should be a local dir, not smb/cifs, for the results to get saved

scriptdir=$(dirname "$(readlink -f "$0")")
cd "$scriptdir"
source ../../common.sh

OUT_CUMULATIVE="$OUT_DIR/benchbase-${BENCHBASE_BENCHMARK}-metrics.csv"

# TODO: Add support for combining multiple metrics from several benchmarks together.

# Parse the summary results file and save the details as a CSV (with a header).

summary_file=$(ls -tr "$OUT_DIR"/results/${BENCHBASE_BENCHMARK}_*.summary.json | tail -n1)
if [ -z "$summary_file" ]; then
    echo "ERROR: Failed to find a summary.json file to use." >&2
    exit 1
fi

echo "metric,value" > $OUT_CUMULATIVE
cat "$summary_file" \
    | jq -r -e '.["Latency Distribution"] | to_entries | .[] | .key + "," + (.value | tostring)' \
    >> "$OUT_CUMULATIVE"
cat "$summary_file" \
    | jq -r -e 'to_entries | .[] | select(.key == "Latency Distribution" | not) | .key + "," + (.value | tostring)' \
    | sed -r -e 's/^([^,]+,[^,]+).*$/\1/' \
    >> "$OUT_CUMULATIVE"

cat "$OUT_CUMULATIVE"
