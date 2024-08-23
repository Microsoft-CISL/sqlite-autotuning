# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

FROM mloscore.azurecr.io/devcontainer-cli:latest

# Adjust the uid/gid of the node user to match the host user
USER root
ARG NODE_UID=1000
ARG NODE_GID=1000
RUN groupmod --non-unique --gid $NODE_GID node \
    && usermod --non-unique --uid $NODE_UID --gid $NODE_GID node \
    && chown -R $NODE_UID:$NODE_GID /home/node

USER node
WORKDIR /src

CMD "/bin/bash"
