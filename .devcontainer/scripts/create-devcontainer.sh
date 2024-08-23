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
GIT_SSH_COMMAND='ssh -oBatchMode=yes' GIT_TERMINAL_PROMPT=0 git -C MLOS pull

.devcontainer/build/build-devcontainer.sh

# Be sure to use the host workspace folder if available.
workspace_root=${LOCAL_WORKSPACE_FOLDER:-$repo_root}

if [ -e /var/run/docker-host.sock ]; then
    docker_gid=$(stat -c%g /var/run/docker-host.sock)
else
    docker_gid=$(stat -c%g /var/run/docker.sock)
fi

if [ "${FORCE:-}" == 'true' ]; then
    docker rm --force $container_name
fi

# Use explicit docker create since the `devcontainer create` command
# has some issues both with nested containers and in its ability to
# start in the background.
set -x
mkdir -p "/tmp/$container_name/dc/shellhistory"
docker create \
    --name "$container_name" \
    --user $(id -u):$docker_gid \
    -v "$HOME/.azure":/dc/azure \
    -v "/tmp/$container_name/dc/shellhistory:/dc/shellhistory" \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$workspace_root":"/workspaces/$repo_name" \
    --workdir "/workspaces/$repo_name" \
    --env containerWorkspaceFolder="/workspaces/$repo_name" \
    --env CONTAINER_WORKSPACE_FOLDER="/workspaces/$repo_name" \
    --env localWorkspaceFolder="$workspace_root" \
    --env LOCAL_WORKSPACE_FOLDER="$workspace_root" \
    --env LOCAL_USER_NAME="$USER" \
    --env http_proxy="${http_proxy:-}" \
    --env https_proxy="${https_proxy:-}" \
    --env no_proxy="${no_proxy:-}" \
    mlos-devcontainer:$container_name \
    /bin/bash -c 'echo "Container starting"; trap "exit 0" 15; while sleep 1 & wait $!; do :; done'

docker start $container_name

# Change the UID of the vscode user to match the current user.
docker exec -u root $container_name groupmod --non-unique --gid $(id -g) vscode
docker exec -u root $container_name usermod --non-unique --uid $(id -u) --gid $(id -g) vscode
docker exec -u root $container_name chown -R $(id -u):$(id -g) /home/vscode

# Run the devcontainer startup scripts in the container we started.
docker run --rm -i \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$workspace_root":"$repo_root" \
    --workdir "$repo_root" \
    --user $(id -u):$docker_gid \
    devcontainer-cli:uid-$(id -u) \
    devcontainer set-up --container-id $container_name
