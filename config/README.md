# [`config`](./) tree details

In order to make some configs reusable and easier to read, the `mlos_bench` config tree is broken up into several sections.

## [`cli`](./cli/)

The [`cli`](./cli/) directory contains config files that can be fed to the `--config` argument of `mlos_bench`.

Those configs reference other required configs such as the root `--environment` config, the `--storage` driver to use for retaining results, `--logging-level`, `--config-path`, etc.

> Note: the `config_path` (search path for referenced config files) automatically includes the upstream `mlos_bench` config examples so some portions can be reused from that directly.

## [`experiments`](./experiments/)

Typically some number of `--global` configs are also provided in the `mlos_bench` cli invocation.

These files define basic `"key": "value"` mappings that can be replaced in the various other config files in order to make it easier to reuse configs across many different experiments.

For instance, one may replace the `experiment_id` or `optimization_target` or `VmSize` so that other configs can reference them (via `required_args`) as `$experiment_id`, `$optimization_target`, `$VmSize` without needing to be edited and duplicated throughout the `config` tree.

## [`environments`](./environments/)

An `Environment` can represent something like a local script execution, fileshare for sharing intermediaries, or a remote execution environment like a VM.

This is where `Tunables` get assigned, benchmarks run, and results collected.

`Environment` configs usually start with a "root" config which is a `CompositeEnvironment` that defines a tree of other `Environments`.

## [`optimizers`](./optimizers/)

An `Optimizer` config can be optionally referenced in the `cli` config.

Each `Optimizer` config can optionally control additional features about an experiment like which backend optimizer to use (e.g., FLAML, SMAC3, etc.), the target objective(s) to optimize, and what direction to optimize them (e.g., minimize or maximize).

Some of these settings can also be set in the `--globals` config files in order to more flexibly control such settings from the [`experiments`](./experiments/) configs.

## [`services`](./services/)

A `Service` is an abstraction for providing some functionality that an Environment might need.

For instance, `remote_exec`, `local_exec`, `upload`, `download`, etc.

Some Services require additional configuration, such as a fileshare name, whereas others only require to be registered as usable by the `Environment(s)` at that level.

## [`storage`](./storage/)

Results from `Trials` executed by `mlos_bench` for an `Experiment` are persisted to `Storage`.

By default a local `sqlite` database is used, but other `SQL` based storage can be used by providing a `--storage` config file argument to `mlos_bench`.
