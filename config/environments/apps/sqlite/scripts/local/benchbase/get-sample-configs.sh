#!/bin/bash

# Grab the sample configs from the container image.
# Can then be edited locally or copies can be made/adapted.

set -eu

set -x

scriptdir=$(dirname "$(readlink -f "$0")")
cd "$scriptdir/"

source ../../common.sh

BENCHBASE_BENCHMARKS=${BENCHBASE_BENCHMARKS:-tpcc twitter wikipedia ycsb}

mkdir -p config/
# Always need the plugin.xml file.
if [ ! -f "./config/plugin.xml" ] || [ "${FORCE:-false}" == 'true' ]; then
    ./copy-file-from-container-image.sh \
            "$BENCHBASE_IMAGE" \
            "/benchbase/config/sqlite/plugin.xml" \
            "./config/"
fi
for benchmark in $BENCHBASE_BENCHMARKS; do
    if [ ! -f "./config/sample_${benchmark}_config.xml" ] || [ "${FORCE:-false}" == 'true' ]; then
        ./copy-file-from-container-image.sh \
            "$BENCHBASE_IMAGE" \
            "/benchbase/config/sqlite/sample_${benchmark}_config.xml" \
            "./config/"
            # FIXME: Adjust the default path to the database file in the container so
            # that we can remap it to a bind mount.
            #sed -i "s|:${benchmark}.db|:/benchbase/db/${benchmark}.db|" "./config/sample_${benchmark}_config.xml"
    fi
done
