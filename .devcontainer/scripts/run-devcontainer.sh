#!/bin/bash
##
## Copyright (c) Microsoft Corporation.
## Licensed under the MIT License.
##

# Quick hacky script to start a devcontainer in a non-vscode shell for testing.

set -eu

# Move to repo root.
scriptdir=$(dirname "$(readlink -f "$0")")
repo_root=$(readlink -f "$scriptdir/../..")
repo_name=$(basename "$repo_root")
cd "$repo_root"

container_name="$repo_name.$(stat -c%i "$repo_root/")"

if ! docker exec $container_name true 2>/dev/null; then
    FORCE="${FORCE:-}" ./.devcontainer/scripts/create-devcontainer.sh
fi

set -x

DOCKER_EXEC_ARGS="${DOCKER_EXEC_ARGS:-}"
if [ "$1" == "-it" ]; then
    shift
    DOCKER_EXEC_ARGS="${DOCKER_EXEC_ARGS} -it"
elif [ "$1" == "-i" ]; then
    shift
    DOCKER_EXEC_ARGS="${DOCKER_EXEC_ARGS} -i"
fi

docker exec $DOCKER_EXEC_ARGS "$container_name" $*

