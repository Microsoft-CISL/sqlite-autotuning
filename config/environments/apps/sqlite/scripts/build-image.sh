#!/bin/bash

set -eu

scriptdir=$(dirname "$(readlink -f "$0")")
cd "$scriptdir/"

source ./common.sh

build_args="--build-arg http_proxy=${http_proxy:-} --build-arg https_proxy=${https_proxy:-} --build-arg no_proxy=${no_proxy:-}"
if [ "${NO_CACHE:-false}" == 'true' ]; then
    build_args+='--no-cache --pull'
fi

tmpdir=$(mktemp -d) # empty context dir

set -x
docker build $build_args -f ./Dockerfile -t "$BENCHBASE_IMAGE" "$tmpdir"

rm -rf "$tmpdir"
