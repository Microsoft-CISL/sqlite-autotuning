#!/usr/bin/env python3

"""
A script to prepare the server prior to running the benchmark.

In the case of sqlite, there isn't really a "server", so this script is more
about preparing the database file and benchbase configs.
"""

import logging
import sys
import json
import urllib.parse

from urllib.parse import ParseResult
from bs4 import BeautifulSoup

# Future work, also support adjusting the following:
#   - [ ] set the number of clients
#   - [ ] set the scale factor
#   - [ ] set the duration

logging.getLogger().setLevel(logging.INFO)


def write_new_config_file(input_file: str, tunables_file: str, output_file: str):
    """
    Read the input file, modifies it with the parameters in the provided
    tunables file, and writes it to the output file.

    Parameters
    ----------
    input_file : str
        Source benchbase config file.
    tunables_file : str
        Input tunables file.
    output_file : str
        Destination benchbase config file.
    """
    with open(input_file, "r+t", encoding='utf-8') as f:
        config = BeautifulSoup(f, "xml")
    url: ParseResult = urllib.parse.urlparse(config.parameters.url.string)
    # Returns a dict of the query string elements but each value is a list.
    query_string_elems = urllib.parse.parse_qs(url.query)

    with open(tunables_file, "r+t") as f:
        tunables = json.load(f)

    for key, val in tunables.items():
        # Replace the query string element with the new value (as a list).
        query_string_elems[key] = [val]

    # Rebuild the query string.
    url = url._replace(query=urllib.parse.urlencode(query_string_elems, doseq=True))
    logging.info("Updating url from %s to %s", config.parameters.url.string, url.geturl())
    config.parameters.url.string = url.geturl()

    # Write the new config out.
    logging.info("Writing new config to %s", output_file)
    with open(output_file, "w+t", encoding='utf-8') as f:
        f.write(str(config))


def usage() -> None:
    """Print the usage message."""
    print("Usage: prepare_server_config.py <input_file> <tunables_file> <output_file>")
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        usage()

    # remove the script name
    sys.argv.pop(0)
    (input_file, tunables_file, output_file) = sys.argv
    write_new_config_file(input_file, tunables_file, output_file)
