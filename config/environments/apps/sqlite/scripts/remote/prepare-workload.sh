#!/bin/bash

# Optionally move pre-loaded database files into place to use.

set -eu

scriptdir=$(dirname "$(readlink -f "$0")")
cd "$scriptdir/"

source ../common.sh

set -x

# The temporary dir for this trial.
# If not provided, default to the target dir for local script debugging.
OUT_DIR="${1:-$TARGET_DIR}"

mkdir -p "$TARGET_DIR"
chgrp $(id -g) "$TARGET_DIR"
chmod g+w "$TARGET_DIR"
if [ "$USE_PRELOADED_DB" == 'true' ]; then
    echo "INFO: Using pre-loaded database files."
    mkdir -p "$BACKUP_DIR"
    if [ ! -s "$BACKUP_DIR/$DB_FILE" ]; then
        echo "INFO: Pre-loaded database file not found. Creating it now."

        check_docker

        # Make sure some bind mount sources exist.
        mkdir -p "$OUT_DIR/results"
        chgrp $(id -g) "$OUT_DIR/results"
        chmod g+w "$OUT_DIR/results"
        touch "$TARGET_DIR/$DB_FILE"
        chgrp $(id -g) "$TARGET_DIR/$DB_FILE"
        chmod g+w "$TARGET_DIR/$DB_FILE"
        if [ ! -f "$OUT_DIR/config/$BENCHBASE_CONFIG_FILE" ]; then
            echo "ERROR: Config file not found: $OUT_DIR/config/$BENCHBASE_CONFIG_FILE" >&2
            echo "Trying sample config instead for now." >&2
            mkdir -p "$OUT_DIR/config"
            cp ../local/benchbase/config/${BENCHBASE_CONFIG_FILE} "$OUT_DIR/config/$BENCHBASE_CONFIG_FILE"
            # Use async writes for (untimed) preloading.
            sed -i -r -e "s|(<url>jdbc:sqlite:.*)${BENCHBASE_BENCHMARK}.db[?]|\1${BENCHBASE_BENCHMARK}.db?synchronous=off\&amp;|" "$OUT_DIR/config/$BENCHBASE_CONFIG_FILE"
            sed -i -r -e "s|(<url>jdbc:sqlite:.*)${BENCHBASE_BENCHMARK}.db<|\1${BENCHBASE_BENCHMARK}.db?synchronous=off<|" "$OUT_DIR/config/$BENCHBASE_CONFIG_FILE"
        fi

        docker run --rm \
            --network=host \
            -v "$TARGET_DIR/$DB_FILE:/benchbase/profiles/sqlite/$DB_FILE" \
            -v "$OUT_DIR/results:/benchbase/results" \
            -v "$OUT_DIR/config/$BENCHBASE_CONFIG_FILE:/benchbase/config/sqlite/$BENCHBASE_CONFIG_FILE" \
            --user containeruser:$(id -g) \
            --env BENCHBASE_PROFILE=sqlite \
            --workdir /benchbase \
            $BENCHBASE_IMAGE \
            -b $BENCHBASE_BENCHMARK -c "/benchbase/config/sqlite/$BENCHBASE_CONFIG_FILE" \
            --create=true --load=true \
            -d /benchbase \
            -jh /benchbase/results/exec-${BENCHBASE_BENCHMARK}.json

        cp "$TARGET_DIR/$DB_FILE" "$BACKUP_DIR/$DB_FILE"
    else
        echo "INFO: Pre-loaded database file found, copying into place."
        cp "$BACKUP_DIR/$DB_FILE" "$TARGET_DIR/"
    fi
else
    echo "INFO: Using empty database files."
    rm -f "$TARGET_DIR/$DB_FILE"
    touch "$TARGET_DIR/$DB_FILE"
fi

if [ ! -e "$TARGET_DIR/$DB_FILE" ]; then
    echo "ERROR: db file is empty: $TARGET_DIR/$DB_FILE" >&2
    exit 1
fi

chgrp $(id -g) "$TARGET_DIR/$DB_FILE"
chmod g+w "$TARGET_DIR/$DB_FILE"

exit 0
