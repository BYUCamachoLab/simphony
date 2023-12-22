#!/usr/bin/env python3
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)
"""Automates the release process.

This script is intended to be run from the root of the repository.
Addtionally, the environment in which simphony is installed must be
activated.
"""

from subprocess import Popen, PIPE
from pathlib import Path


CWD = Path(__file__).parent.absolute()
HOME = CWD.parent
print(f"Directory: {HOME}")


def execute(args):
    """Execute a command in the shell.

    Parameters
    ----------
    args : list of str
        The command to execute.

    Returns
    -------
    str
        The output of the command.

    Raises
    ------
    Exception
        If the command returns a non-zero exit code.
    """
    proc = Popen(args, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    exitcode = proc.returncode
    if exitcode:
        raise Exception(f"{err.decode()}")
    return out.decode()


# Check that the working directory is clean
result = execute(["git", "status", "--porcelain"])
if result:
    raise Exception("working directory is not clean")

SIMPHONY_VERSION = execute(
    ["python3", "-c", "import simphony; print(simphony.__version__)"]
).strip()
print(f"Simphony version: {SIMPHONY_VERSION}")

RELEASE_TEXT = execute(["python3", "get_changelog_entry.py", SIMPHONY_VERSION]).strip()
bar = max([len(line) for line in RELEASE_TEXT.splitlines()])
print("Release text:")
print(f"{'-'*bar}")
print(f"{RELEASE_TEXT}")
print(f"{'-'*bar}")

TAG_NAME = f"v{SIMPHONY_VERSION}"
print(f"Tag name: {TAG_NAME}")

# Check that the tag does not already exist
result = execute(["git", "tag", "-l", TAG_NAME])
if result:
    raise Exception(f"Error: tag {TAG_NAME} already exists")

# Tag the repository and push the tag
execute(["git", "tag", TAG_NAME])
print(f"Tag {TAG_NAME} created")

execute(["git", "push", "origin", TAG_NAME])
print(f"Tag {TAG_NAME} pushed")
