#!/bin/bash
##
## Copyright (c) Microsoft Corporation.
## Licensed under the MIT License.
##

set -x

set -eu
scriptdir=$(dirname "$(readlink -f "$0")")
repo_root=$(readlink -f "$scriptdir/../..")
repo_name=$(basename "$repo_root")
cd "$scriptdir/"

container_name="$repo_name.$(stat -c%i "$repo_root/")"

DEVCONTAINER_IMAGE="devcontainer-cli:uid-$(id -u)"
MLOS_DEVCONTAINER_IMAGE="mloscore.azurecr.io/mlos-devcontainer:latest"
MLOS_AUTOTUNING_IMAGE="mlos-devcontainer:$container_name"

# Get the helper container that has the devcontainer CLI tool for building the devcontainer.
NO_CACHE="${NO_CACHE:-}" ./build-devcontainer-cli.sh

if [ "${NO_CACHE:-}" == 'true' ]; then
    # Also update the cache of the base image.
    tmpdir=$(mktemp -d)
    docker --config="$tmpdir" pull $MLOS_DEVCONTAINER_IMAGE || true
    rmdir "$tmpdir"
fi

DOCKER_GID=$(stat -c'%g' /var/run/docker.sock)
# Make this work inside a devcontainer as well.
if [ -w /var/run/docker-host.sock ]; then
    DOCKER_GID=$(stat -c'%g' /var/run/docker-host.sock)
fi

# Build the devcontainer image.
rootdir="$repo_root"

# Run the initialize command on the host first.
# Note: command should already pull the cached image if possible.
pwd
devcontainer_json=$(cat "$rootdir/.devcontainer/devcontainer.json" | sed -e 's|^\s*//.*||' -e 's|/\*|\n&|g;s|*/|&\n|g' | sed -e '/\/\*/,/*\//d')
initializeCommand=$(echo "$devcontainer_json" | docker run -i --rm $DEVCONTAINER_IMAGE jq -e -r '.initializeCommand[]')
if [ -z "$initializeCommand" ]; then
    echo "No initializeCommand found in devcontainer.json" >&2
    exit 1
else
    eval "pushd "$rootdir/"; $initializeCommand; popd"
fi

# Make this work inside a devcontainer as well.
if [ -n "${LOCAL_WORKSPACE_FOLDER:-}" ]; then
    rootdir="$LOCAL_WORKSPACE_FOLDER"
fi

docker run -i --rm \
    --user $(id -u):$DOCKER_GID \
    -v "$rootdir":/src \
    -v /var/run/docker.sock:/var/run/docker.sock \
    --env DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-1} \
    --env BUILDKIT_INLINE_CACHE=1 \
    --env http_proxy=${http_proxy:-} \
    --env https_proxy=${https_proxy:-} \
    --env no_proxy=${no_proxy:-} \
    $DEVCONTAINER_IMAGE \
    devcontainer build --workspace-folder /src \
        --image-name $MLOS_AUTOTUNING_IMAGE
