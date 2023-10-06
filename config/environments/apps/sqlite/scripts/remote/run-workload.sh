#!/bin/bash

# Run the benchmark workload using benchbase.

set -eu

scriptdir=$(dirname "$(readlink -f "$0")")
cd "$scriptdir/"

source ../common.sh

set -x

TRIAL_DIR="${1:-$DB_DIR/trial}"

check_docker
../build-image.sh

# Make sure some bind mount sources exist.
mkdir -p "$TRIAL_DIR/results"
chgrp $(id -g) "$TRIAL_DIR/results"
chmod g+w "$TRIAL_DIR/results"
if [ ! -f "$TRIAL_DIR/$BENCHBASE_CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $TRIAL_DIR/$BENCHBASE_CONFIG_FILE" >&2
    echo "Trying sample config instead for now." >&2
    cp ../local/benchbase/config/sample_${BENCHBASE_BENCHMARK}_config.xml "$TRIAL_DIR/$BENCHBASE_CONFIG_FILE"
fi

BENCHBASE_ARGS='--exec=true'
if [ "$USE_PRELOADED_DB" != 'true' ]; then
    BENCHBASE_ARGS=" --create=true --load=true"
fi
# else assume we already created it in prepare-workload.sh

# Note: overriding the entrypoint in order to run with "time" for some basic resource stats.
docker run --rm \
    -i --log-driver=none -a STDIN -a STDOUT -a STDERR --rm \
    --network=host \
    -v "$DB_DIR/$DB_FILE:/benchbase/profiles/sqlite/$DB_FILE" \
    -v "$TRIAL_DIR/results:/benchbase/results" \
    -v "$TRIAL_DIR/$BENCHBASE_CONFIG_FILE:/benchbase/config/sqlite/$BENCHBASE_CONFIG_FILE" \
    --user containeruser:$(id -g) \
    --env BENCHBASE_PROFILE=sqlite \
    --entrypoint /usr/bin/time \
    $BENCHBASE_IMAGE \
    -v -o /benchbase/results/exec-${BENCHBASE_BENCHMARK}.time.out \
    /benchbase/entrypoint.sh \
    -b $BENCHBASE_BENCHMARK -c "/benchbase/config/sqlite/$BENCHBASE_CONFIG_FILE" \
    $BENCHBASE_ARGS \
    -s 1 -im 1000 \
    -d /benchbase/results \
    -jh /benchbase/results/exec-${BENCHBASE_BENCHMARK}.json \
    2>&1 | tee "$TRIAL_DIR/results/exec-${BENCHBASE_BENCHMARK}.log"

if [ ! -s "$DB_DIR/$DB_FILE" ]; then
    echo "ERROR: db file is empty: $DB_DIR/$DB_FILE" >&2
    exit 1
fi
if [ ! -s "$TRIAL_DIR/results/exec-${BENCHBASE_BENCHMARK}.json" ]; then
    echo "ERROR: results file is empty: $TRIAL_DIR/results/exec-${BENCHBASE_BENCHMARK}.json" >&2
    exit 1
fi

exit 0
