# MLOS-Autotuning dev environment

This repo contains some scripts and configs to help do autotuning for a target system (e.g., MySQL).

To do so, it makes use of the [MLOS](https://github.com/microsoft/MLOS) project.

We generally expect most of the development for autotuning different target systems to happen in their own repository and *just* use the MLOS project as a dependency.

However, on occasion it's necessary to make changes to MLOS itself to support a new target system's features.

The layout in this repo is intended to help with that.

## Layout

To begin, we reuse the devcontainer published from the MLOS repo.

Next, we add a copy of the `MLOS` repo itself to this one during the devcontainer startup and make sure it is pip (editable) installed.

> ~~MLOS is not yet in pypi, so this is another part of why we currently do it this way.~~

## General Usage

In general, the workflow for using this repo is listed in the [README.md](../README.md) in the root of the directory and simply involves opening this repo in VSCode and then reopening it in a devcontainer when prompted.

All development for the target system should happen in the [`config/`](../config/) directory.

## Local development

To develop on MLOS locally, you need to adjust the default MLOS clone to include your remote fork and checkout a different branch of it.

```sh
# Move into the MLOS repo cloned directory:
cd MLOS
# Add your fork as a remote:
git remote add <username> git@github.com:<username>/MLOS.git
git remote fetch
# Create a new branch:
git checkout -b <branch> main
git push -u <username> <branch>
```
