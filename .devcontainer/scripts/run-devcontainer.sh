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

test -d MLOS || git clone --single-branch https://github.com/microsoft/MLOS.git
cd MLOS && git pull && cd ..

# Be sure to use the host workspace folder if available.
workspace_root=${LOCAL_WORKSPACE_FOLDER:-$repo_root}

if [ -e /var/run/docker-host.sock ]; then
    docker_gid=$(stat -c%g /var/run/docker-host.sock)
else
    docker_gid=$(stat -c%g /var/run/docker.sock)
fi

set -x
mkdir -p "/tmp/$container_name/dc/shellhistory"
docker run -it --rm \
    --name "$container_name" \
    --user vscode:$docker_gid \
    -v "$HOME/.azure":/dc/azure \
    -v "/tmp/$container_name/dc/shellhistory:/dc/shellhistory" \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$workspace_root":"/workspaces/$repo_name" \
    --workdir "/workspaces/$repo_name" \
    --env CONTAINER_WORKSPACE_FOLDER="/workspaces/$repo_name" \
    --env LOCAL_WORKSPACE_FOLDER="$workspace_root" \
    --env http_proxy="${http_proxy:-}" \
    --env https_proxy="${https_proxy:-}" \
    --env no_proxy="${no_proxy:-}" \
    mloscore.azurecr.io/mlos-devcontainer:latest \
    $*
