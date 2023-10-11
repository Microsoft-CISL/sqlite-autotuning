#!/bin/bash

# Quick helper script to help copy files out of a container image.

set -eu

set -x

image="${1:-}"
src="${2:-}"
dst="${3:-./}"

if [ -z "$image" ] || [ -z "$src" ]; then
    echo "usage: $0 <image>:<src> [<dst>]" >&2
    exit 1
fi

id=$(docker create "$image")
docker cp "$id:$src" "$dst" || true
docker rm "$id"
