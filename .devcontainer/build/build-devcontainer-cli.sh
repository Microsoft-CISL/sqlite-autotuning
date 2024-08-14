#!/bin/bash
##
## Copyright (c) Microsoft Corporation.
## Licensed under the MIT License.
##

set -x

set -eu

# Actual building happens on the MLOS CI/CD pipeline, so we just need to pull the image here.
docker pull mloscore.azurecr.io/devcontainer-cli:latest
