#!/usr/bin/env python3

"""
A script to prepare the server prior to running the benchmark.

In the case of sqlite, there isn't really a "server", so this script is more
about preparing the database file and benchbase configs.
"""

# TODO:
# - [ ] rewrite the benchbase config file
#   - [ ] add the database file path
#   - [ ] add sqlite connection params
#   - [ ] set the number of clients
#   - [ ] set the scale factor
#   - [ ] set the duration
