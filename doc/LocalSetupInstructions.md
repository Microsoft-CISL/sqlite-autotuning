# Local Setup Instructions

These instructions are for setting up your local development environment if you are not using GitHub Codespaces.

## Using a Local Dev Container

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. Install [Visual Studio Code](https://code.visualstudio.com/).
3. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/Microsoft-CISL/sqlite-autotuning.git
   cd sqlite-autotuning
   ```
4. Open the cloned repository in Visual Studio Code.
5. Install the `@recommended` Extensions, especially the [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) for Visual Studio Code.
6. Open the project in a Dev Container:

   - Press `F1` to open the Command Palette.
   - Type `Remote-Containers: Reopen in Container` and select it.

   This will build the Docker container as specified in the `.devcontainer` folder and open your project inside that container.

## Using a Local Python Environment via Conda

1. Install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/Microsoft-CISL/sqlite-autotuning.git
   cd sqlite-autotuning
   ```

3. Clone the MLOS repository as a submodule:

   ```bash
   git clone https://github.com/microsoft/MLOS.git MLOS
   ```

4. Create a new conda environment:

   ```bash
   conda env create -f MLOS/conda-envs/environment.yml
   conda activate mlos
   ```

5. Install the required local dependencies:

   ```bash
   pip install -U -r requirements.txt
   ```

6. Get the remote data files:

   ```bash
   test -f mlos_bench.sqlite || wget -q -Nc https://mlospublic.z13.web.core.windows.net/sqlite-autotuning/mlos_bench.sqlite
   mkdir -p workdir/benchbase/db.bak && wget -q -c -O workdir/benchbase/db.bak/tpcc.db https://mlospublic.z13.web.core.windows.net/sqlite-autotuning/tpcc.db
   ```

7. Continue with the instructions in the README to set up and run the project.
