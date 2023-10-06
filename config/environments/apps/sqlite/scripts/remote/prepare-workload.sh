#!/bin/bash

# Optionally move pre-loaded database files into place to use.

set -eu

scriptdir=$(dirname "$(readlink -f "$0")")
cd "$scriptdir/"

source ../common.sh

set -x

# The temporary dir for this trial.
# If not provided, default to the db dir for local script debugging.
TRIAL_DIR="${1:-$DB_DIR/trial}"
if [ -z "$TRIAL_DIR" ]; then
    echo "ERROR: No trial dir provided." >&2
fi

mkdir -p "$DB_DIR"
chgrp $(id -g) "$DB_DIR"
chmod g+w "$DB_DIR"
if [ "$USE_PRELOADED_DB" == 'true' ]; then
    echo "INFO: Using pre-loaded database files."
    mkdir -p "$DB_BAK_DIR"
    if [ ! -s "$DB_BAK_DIR/$DB_FILE" ]; then
        echo "INFO: Pre-loaded database file not found. Creating it now."

        check_docker
        ../build-image.sh

        # Make sure some bind mount sources exist.
        mkdir -p "$TRIAL_DIR/results"
        chgrp $(id -g) "$TRIAL_DIR/results"
        chmod g+w "$TRIAL_DIR/results"
        touch "$DB_DIR/$DB_FILE"
        chgrp $(id -g) "$DB_DIR/$DB_FILE"
        chmod g+w "$DB_DIR/$DB_FILE"
        if [ ! -f "$TRIAL_DIR/$BENCHBASE_CONFIG_FILE" ]; then
            echo "ERROR: Config file not found: $TRIAL_DIR/$BENCHBASE_CONFIG_FILE" >&2
            echo "Trying sample config instead for now." >&2
            cp ../local/benchbase/config/sample_${BENCHBASE_BENCHMARK}_config.xml "$TRIAL_DIR/$BENCHBASE_CONFIG_FILE"
            # Use async writes for (untimed) preloading.
            sed -i -r \
                -e "s|(<url>jdbc:sqlite:.*)${BENCHBASE_BENCHMARK}.db[?]|\1${BENCHBASE_BENCHMARK}.db?synchronous=off\&amp;|" \
                -e "s|(<url>jdbc:sqlite:.*)${BENCHBASE_BENCHMARK}.db<|\1${BENCHBASE_BENCHMARK}.db?synchronous=off<|" \
                "$TRIAL_DIR/$BENCHBASE_CONFIG_FILE"
        fi

        docker run --rm \
            -i --log-driver=none -a STDIN -a STDOUT -a STDERR --rm \
            --network=host \
            -v "$DB_DIR/$DB_FILE:/benchbase/profiles/sqlite/$DB_FILE" \
            -v "$TRIAL_DIR/results:/benchbase/results" \
            -v "$TRIAL_DIR/$BENCHBASE_CONFIG_FILE:/benchbase/config/sqlite/$BENCHBASE_CONFIG_FILE" \
            --user containeruser:$(id -g) \
            --env BENCHBASE_PROFILE=sqlite \
            $BENCHBASE_IMAGE \
            -b $BENCHBASE_BENCHMARK -c "/benchbase/config/sqlite/$BENCHBASE_CONFIG_FILE" \
            --create=true --load=true \
            -d /benchbase/results \
            -jh /benchbase/results/exec-${BENCHBASE_BENCHMARK}.json

        cp "$DB_DIR/$DB_FILE" "$DB_BAK_DIR/$DB_FILE"
    else
        echo "INFO: Pre-loaded database file found, copying into place."
        cp "$DB_BAK_DIR/$DB_FILE" "$DB_DIR/"
    fi
else
    echo "INFO: Using empty database files."
    rm -f "$DB_DIR/$DB_FILE"
    touch "$DB_DIR/$DB_FILE"
fi

if [ ! -e "$DB_DIR/$DB_FILE" ]; then
    echo "ERROR: db file is empty: $DB_DIR/$DB_FILE" >&2
    exit 1
fi

chgrp $(id -g) "$DB_DIR/$DB_FILE"
chmod g+w "$DB_DIR/$DB_FILE"

exit 0
