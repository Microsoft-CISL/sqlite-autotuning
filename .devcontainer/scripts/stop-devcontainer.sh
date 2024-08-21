#!/bin/bash
##
## Copyright (c) Microsoft Corporation.
## Licensed under the MIT License.
##

# Quick hacky script to stop a devcontainer in a non-vscode shell for testing.

set -eu

# Move to repo root.
scriptdir=$(dirname "$(readlink -f "$0")")
repo_root=$(readlink -f "$scriptdir/../..")
repo_name=$(basename "$repo_root")
cd "$repo_root"

container_name="$repo_name.$(stat -c%i "$repo_root/")"

set -x

docker stop "$container_name"
docker rm "$container_name"
