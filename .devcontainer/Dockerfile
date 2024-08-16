FROM mloscore.azurecr.io/mlos-devcontainer:latest

# Local tweaks to the upstream image.
USER vscode

# Make sure that github.com is in the known_hosts file.
RUN mkdir -p /home/vscode/.ssh \
    && ( \
        grep -q ^github.com /home/vscode/.ssh/known_hosts \
        || ssh-keyscan github.com >> /home/vscode/.ssh/known_hosts \
    )
