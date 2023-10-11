# MLOS Autotuning for Sqlite Repo

This repo is a fork of the [mlos-autotuning-template](https://msgsl.visualstudio.com/MLOS/_git/mlos-autotuning-template) repo.

It is meant as an demo/example for tuning sqlite.

## Prerequisites

### Codespaces

- *None!*

  Just open the project in your browser via Github Codespaces using the **<> Code** drop down at the top of this page :-)

  In that case you can skip to step 5 below.

  <!-- markdownlint-disable-next-line MD033 -->
  <img src="./doc/images/github-open-in-codespace.png" style="width:500px" />

  > Note: you can also re-open a codespace in your local VSCode instance once created.

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

4. Reopen the workspace (if prompted).

    <!-- markdownlint-disable-next-line MD033 -->
    <img src="./doc/images/codespace-open-workspace.png" style="width:500px" />

    > Note: you can trigger the prompt by browse to the [`mlos-autouning.code-workspace`](./mlos-autotuning.code-workspace) file and follow the prompt in the lower right to reopen.


5. Activate the conda environment in the integrated terminal (lower panel):

    ```sh
    conda activate mlos
    ```

    <!-- markdownlint-disable-next-line MD033 -->
    <img src="./doc/images/codespace-terminal.png" style="width:500px" />

6. Run the `mlos_bench` tool.

    For instance, to run the sqlite example from the upstream MLOS repo (pulled locally):

    ```sh
    # Run the oneshot benchmark.
    mlos_bench --config "./config/cli/local-sqlite-bench.jsonc" --globals "./config/experiments/sqlite-sync-experiment.jsonc"
    ```

    ```sh
    # Run the optimization loop.
    mlos_bench --config "./config/cli/local-sqlite-opt.jsonc" --globals "./config/experiments/sqlite-sync-journal-pagesize-caching-experiment.jsonc" --max-iterations 10000 2>&1 | tee local-sqlite-opt.log
    ```
