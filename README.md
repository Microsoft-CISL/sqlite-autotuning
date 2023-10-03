# MLOS Autotuning for Sqlite Repo

This repo is a fork of the [mlos-autotuning-template](https://msgsl.visualstudio.com/MLOS/_git/mlos-autotuning-template) repo.

It is meant as an demo/example for tuning sqlite.

## Prerequisites

- `git`
- `docker`
- `vscode`
- Azure
  - Subscription ID
  - Resource Group Name

## Getting Started

1. Clone this repository.
2. Open this repository in VSCode.
3. Reopen in a devcontainer.

    > For additional dev environment details, see the devcontainer [README.md](.devcontainer/README.md)

4. Reopen the workspace.

    - Browse to the [`mlos-autouning.code-workspace`](./mlos-autotuning.code-workspace) file and follow the prompt in the lower right to reopen.

5. Activate the conda environment in the integrated terminal:

    ```sh
    conda activate mlos
    ```

6. Login to the Azure CLI:

    ```sh
    az login
    ```

7. Stash some relevant auth info (e.g., subscription ID, resource group, etc.):

    ```sh
    ./MLOS/scripts/generate-azure-credentials-config.sh > global_azure_config.json
    ```

8. Run the `mlos_bench` tool.

    For instance, to run the Redis example from the upstream MLOS repo (pulled locally):

    ```sh
    mlos_bench --config "./config/cli/azure-sqlite-opt.jsonc" --globals "./config/experiments/experiment-sqlite-bench.jsonc" --max_iterations 10
    ```

## See Also

For other examples, please see one of the following:

- [sqlite-autotuning](https://dev.azure.com/msgsl/MLOS/_git/sqlite-autotuning)
