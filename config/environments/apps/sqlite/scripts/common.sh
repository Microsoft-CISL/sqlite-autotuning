BENCHBASE_IMAGE="benchbase-sqlite-with-stats:latest"
BENCHBASE_BENCHMARK="${BENCHBASE_BENCHMARK:-tpcc}"
BENCHBASE_CONFIG_FILE="${BENCHBASE_CONFIG_FILE:-config_${BENCHBASE_BENCHMARK}.xml}"

USE_PRELOADED_DB="${USE_PRELOADED_DB:-true}"

repo_root=$(git rev-parse --show-toplevel || true)
if [ -n "$repo_root" ]; then
    tmp_dir="$repo_root/tmp"
fi
if [ -n "${LOCAL_WORKSPACE_FOLDER:-}" ]; then
    # When executing inside a devcontainer we need to use the host's path instead.
    tmp_dir="$LOCAL_WORKSPACE_FOLDER/tmp"
fi

DB_DIR="${DB_DIR:-$tmp_dir/benchbase}"
DB_BAK_DIR="${DB_BAK_DIR:-$DB_DIR.bak}"

DB_FILE="${DB_FILE:-$BENCHBASE_BENCHMARK.db}"

check_root() {
    if [ $EUID != 0 ]; then
        echo "ERROR: This script expects to be executed with root privileges." >&2
        exit 1
    fi
}

check_docker() {
    if ! hash docker 2>/dev/null; then
        check_root

        # Taken from https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository
        distro=$(lsb_release -is | tr '[:upper:]' '[:lower:]')

        # Remove any older versions
        apt-get remove docker docker-engine docker.io containerd runc || true

        # Allow apt to use a repo over HTTPS
        apt-get update
        apt-get -y install \
            ca-certificates \
            curl \
            gnupg \
            lsb-release

        # Add Docker's official GPG key
        mkdir -p /etc/apt/keyrings
        curl -fsSL https://download.docker.com/linux/$distro/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

        # Set up the repo
        echo \
            "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/$distro \
            $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

        # Install latest version of Docker Engine and related
        apt-get update
        apt-get -y install docker-ce docker-ce-cli containerd.io
        adduser $USER docker
        for user in mlos azureuser admin; do
            if id $user >/dev/null 2>&1; then
                adduser $user docker
            fi
        done
    fi
}

translate_devcontainer_dir() {
    path="$1"
    # Attempt to translate a devcontainer path to a host path.
    if [ -z "${CONTAINER_WORKSPACE_FOLDER:-}" ] || [ -z "${LOCAL_WORKSPACE_FOLDER:-}" ]; then
        echo "$path"
    else
        echo "$path" | sed -r "s|^$CONTAINER_WORKSPACE_FOLDER|$LOCAL_WORKSPACE_FOLDER|"
    fi
}
