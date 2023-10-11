# MLOS Autotuning for Sqlite Repo

This repo is a fork of the [mlos-autotuning-template](https://msgsl.visualstudio.com/MLOS/_git/mlos-autotuning-template) repo.
<!-- TODO: Open source that template repo and update the link. -->

It is meant as a basic demo example for tuning a local [`sqlite`](https://www.sqlite.org/) instance running via [`benchbase`](https://github.com/cmu-db/benchbase) and analyzing the results.

There are two items in this example:

1. The [`mlos_storage_demo.ipynb`](./mlos_storage_demo.ipynb) notebook to help explore some existing data collected using the [`mlos_bench`](https://github.com/microsoft/MLOS) tool.

1. Some configs and example commands to use `mlos_bench` to autotune a `sqlite` workload (see below).

## Prerequisites

### Codespaces

- Just a [Github Account](https://github.com/account) :-)

### Local

- `git`
- `docker`
- `vscode`
- Azure
  - Subscription ID
  - Resource Group Name

## Prior to Class

1. Create a github account if you do not already have one - [Github Account](https://github.com/account)
1. Open the [project](https://github.com/Microsoft-CISL/sqlite-autotuning/tree/main) in your browser.  Navigate to the green **<> Code** drop down at the top of page and select the green **Create codespace on main** button.
    <!-- markdownlint-disable-next-line MD033 -->
    <img src="./doc/images/github-open-in-codespace.png" style="width:500px" />

1. Reopen the workspace (if prompted).
    <!-- markdownlint-disable-next-line MD033 -->
    <img src="./doc/images/codespace-open-workspace.png" style="width:500px" />

   > Note: you can trigger the prompt by browse to the [`mlos-autouning.code-workspace`](./mlos-autotuning.code-workspace) file and follow the prompt in the lower right to reopen.

1. **That's it!**  If you run into any issues, please reach out to the teaching team and we can assist prior to class starting.

## Start of Class

1. Open the codespace created above

1. Make sure the MLOS dependencies are up to date.

    > To be executed in the integrated terminal at the bottom of the VSCode window:

    ```sh
    # Pull the latest MLOS code.
    git -C MLOS pull
    ```

1. Make sure the `mlos_bench.sqlite` data is available.

    > To be executed in the integrated terminal at the bottom of the VSCode window:

    ```sh
    # Download the previously generated results database.
    test -f mlos_bench.sqlite || wget -Nc https://adumlosdemostorage.blob.core.windows.net/adu-mlos-db-example/adu_notebook_db/mlos_bench.sqlite
    ```

1. Activate the conda environment in the integrated terminal (lower panel):

    ```sh
    conda activate mlos
    ```

    <!-- markdownlint-disable-next-line MD033 -->
    <img src="./doc/images/codespace-terminal.png" style="width:500px" />

1. Make sure the TPC-C database is preloaded.

    > Note: this is an optimization.  If not present, the scripts below will generate it the first time it's needed.

    ```sh
    mkdir -p workdir/benchbase/db.bak
    wget -Nc -O workdir/benchbase/db.bak/tpcc.db https://adumlosdemostorage.blob.core.windows.net/adu-mlos-db-example/adu_notebook_db/tpcc.db
    ```

1. Run the `mlos_bench` tool as a one-shot benchmark.

    For instance, to run the sqlite example from the upstream MLOS repo (pulled locally):

    > To be executed in the integrated terminal at the bottom of the VSCode window:

    ```sh
    # Run the one-shot benchmark.
    # This will run a single experiment trial and output the results to the local results database.
    mlos_bench --config "./config/cli/local-sqlite-bench.jsonc" --globals "./config/experiments/sqlite-sync-experiment.jsonc"
    ```

    This should take a few minutes to run and does the following:

    - Loads the CLI config [`./config/cli/local-sqlite-bench.jsonc`](./config/cli/local-sqlite-bench.jsonc)
        - The [`config/experiments/sqlite-sync-experiment.jsonc`](./config/experiments/sqlite-sync-experiment.jsonc) further customizes that config with the experiment specific parameters (e.g., telling it which tunable parameters to use for the experiment, the experiment name, etc.).

            Alternatively, Other config files from the [`config/experiments/`](./config/experiments/) directory can be referenced with the `--globals` argument as well in order to customize the experiment.
    - The CLI config also references and loads the root environment config [`./config/environments/apps/sqlite/sqlite-local-benchbase.jsonc`](./config/environments/apps/sqlite/sqlite-local-benchbase.jsonc).

        - In that config the `setup` section lists commands used to
          1. Prepare a config for the `sqlite` instance based on the tunable parameters specified in the experiment config,
          1. Load or restores a previously loaded copy of a `tpcc.db` `sqlite` instance using a `benchbase` `docker` image.
        - Next, the `run` section lists commands used to
          1. execute a TPC-C workload against that `sqlite` instance
          1. assemble the results into a file that is read in the `read_results_file` config section in order to store them into the `mlos_bench` results database.

1. Run the `mlos_bench` tool as an optimization loop.

    ```sh
    # Run the optimization loop by referencing a different config file
    # that specifies an optimizer and objective target.
    mlos_bench --config "./config/cli/local-sqlite-opt.jsonc" --globals "./config/experiments/sqlite-sync-journal-pagesize-caching-experiment.jsonc" --max-iterations 150
    ```

    The command above will run the optimization loop for 150 iterations, which should take about 30 minutes since each trial should takes about 12 seconds to run.

    > Note: a 10 second run is not very long evaluation period.  It's used here to keep the demo short, but in practice you would want to run for longer to get more accurate results.

    To do this, it follows the procedure outlined above, but instead of running a single trial, it runs an optimization loop that runs multiple trials, each time updating the tunable parameters based on the results of the previous trial, balancing exploration and exploitation to find the optimal set of parameters.

    <!-- TODO: Add an image depicting what's going on here -->

    > Note: while that's executing you can try exploring other previously collected data using the [`mlos_storage_demo.ipynb`](./mlos_storage_demo.ipynb) notebook.

1. Use the [`mlos_sqlite_demo.ipynb`](./mlos_sqlite_demo.ipynb) notebook to analyze the results.
