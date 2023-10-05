#!/bin/bash

# Run the benchmark workload using benchbase.

set -eu

scriptdir=$(dirname "$(readlink -f "$0")")
cd "$scriptdir/"

source ../common.sh

set -x

OUT_DIR="${1:-$TARGET_DIR}"

check_docker
../build-image.sh

# Make sure some bind mount sources exist.
mkdir -p "$OUT_DIR/results"
chgrp $(id -g) "$OUT_DIR/results"
chmod g+w "$OUT_DIR/results"
if [ ! -f "$OUT_DIR/config/$BENCHBASE_CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $OUT_DIR/config/$BENCHBASE_CONFIG_FILE" >&2
    echo "Trying sample config instead for now." >&2
    mkdir -p "$OUT_DIR/config"
    cp ../local/benchbase/config/${BENCHBASE_CONFIG_FILE} "$OUT_DIR/config/$BENCHBASE_CONFIG_FILE"
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
    -v "$TARGET_DIR/$DB_FILE:/benchbase/profiles/sqlite/$DB_FILE" \
    -v "$OUT_DIR/results:/benchbase/results" \
    -v "$OUT_DIR/config/$BENCHBASE_CONFIG_FILE:/benchbase/config/sqlite/$BENCHBASE_CONFIG_FILE" \
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
    2>&1 | tee "$OUT_DIR/results/exec-${BENCHBASE_BENCHMARK}.log"

if [ ! -s "$TARGET_DIR/$DB_FILE" ]; then
    echo "ERROR: db file is empty: $TARGET_DIR/$DB_FILE" >&2
    exit 1
fi
if [ ! -s "$OUT_DIR/results/exec-${BENCHBASE_BENCHMARK}.json" ]; then
    echo "ERROR: results file is empty: $OUT_DIR/results/exec-${BENCHBASE_BENCHMARK}.json" >&2
    exit 1
fi

exit 0
