# MLOS Autotuning for Sqlite Repo

This repo is a fork of the [mlos-autotuning-template](https://msgsl.visualstudio.com/MLOS/_git/mlos-autotuning-template) repo.

It is meant as an demo/example for tuning sqlite.

## Prerequisites

### Codespaces

- *Nothing!*

  Just open the project in your browser via Github CodeSpaces using the Code drop down at the top of this page :)

  In that case you can skip to step 5 below.

### Local

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

6. Run the `mlos_bench` tool.

    For instance, to run the sqlite example from the upstream MLOS repo (pulled locally):

    ```sh
    mlos_bench --config "./config/cli/local-sqlite-opt.jsonc" --globals "./config/experiments/experiment-sqlite-tpcc.jsonc" --max_iterations 10
    ```
