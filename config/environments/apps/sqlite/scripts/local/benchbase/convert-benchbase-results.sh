#!/bin/bash
##
## Copyright (c) Microsoft Corporation.
## Licensed under the MIT License.
##

set -eu
set -x

OUT_DIR="$1"    # should be a local dir, not smb/cifs, for the results to get saved

scriptdir=$(dirname "$(readlink -f "$0")")
cd "$scriptdir"
source ../../common.sh

OUT_CUMULATIVE="$OUT_DIR/benchbase-${BENCHMARK}-metrics.csv"

# TODO: Expand/generalize for additional metrics.
# TODO: Add support for combining multiple metrics from several benchmarks together.

# Parse the summary results file and save the details as a CSV (with a header).
echo "metric,value" > $OUT_CUMULATIVE
summary_file=$(ls -tr "$OUT_DIR"/results/*.summary.json | tail -n1)
cat "$summary_file" \
    | jq -r '.["Latency Distribution"] | to_entries | .[] | .key + "," + (.value | tostring)' \
    >> "$OUT_CUMULATIVE"
cat "$summary_file" \
    | jq -r 'to_entries | .[] | select(.key == "Latency Distribution" | not) | .key + "," + (.value | tostring)' | sed -r -e 's/^([^,]+,[^,]+).*$/\1/' \
    >> "$OUT_CUMULATIVE"
