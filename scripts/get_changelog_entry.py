#!/usr/bin/env python3

"""
This script will print the changelog entry for the given version. If the 
requested version is not found, it raise an error and print a list of available 
versions. You can use the exit code to determine if the script ran 
successfully.

Usage:
  python scripts/get_changelog_entry.py VERSION

Example:
  python scripts/get_changelog_entry.py 0.6.1
"""

import re
import sys
from datetime import datetime
from pathlib import Path

from packaging.version import parse

# Get requested version (first command line argument)
version_request = parse(sys.argv[1])

CHANGELOG = Path(__file__).parent.parent / "CHANGELOG.md"
with CHANGELOG.open() as f:
    text = f.read()

# Split changelog by horizontal rule
entries = text.split("\n---\n")
entries = [entry.strip() for entry in entries[1:]]

# Map version number to entries
mapping = {}
for entry in entries:
    expressions = [
        r"## \[(?P<version>.*)\]\((?P<link>.*)\) - <small>(?P<date>.*)</small>",
        r"## (?P<version>.*) - <small>(?P<date>.*)</small>",
    ]

    version = None
    for matcher in expressions:
        match = re.search(matcher, entry)
        if match:
            version = match.group("version")
            date = match.group("date")
            
            # Validate the date
            def is_valid_date(date_string):
                try:
                    datetime.strptime(date_string, "%Y-%m-%d")
                    return True
                except ValueError:
                    return False
                
            if not is_valid_date(date):
                raise ValueError(f"Invalid date: {date}")

            break

    if version is None:
        raise ValueError(f"Could not parse version from entry: {entry}")

    mapping[version] = entry

# Print the requested version to stdout
try:
    print(mapping[str(version_request)])
except KeyError as exc:
    s = f"Requested version '{version_request}' not found in changelog, available versions are: "
    s += ", ".join([k for k in mapping.keys()])
    raise KeyError(s) from exc
