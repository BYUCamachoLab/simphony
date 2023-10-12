# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)
"""Utilities for loading and caching data files."""

import importlib.resources
from pathlib import Path

import simphony.libraries

SOURCE_DATA_PATH = "siepic/source_data"


def _resolve_source_filepath(filename: str) -> Path:
    """Gets the absolute path to the source data files relative to ``source_data/``.

    Parameters
    ----------
    filename : str
        The name of the file to be found.

    Returns
    -------
    filepath : str
        The absolute path to the file.
    """
    filepath = Path(SOURCE_DATA_PATH) / filename
    try:  # python >= 3.9
        return importlib.resources.files(simphony.libraries) / filepath
    except AttributeError:  # fall back to method deprecated in 3.11.
        ctx = importlib.resources.path(simphony, "libraries")
        with ctx as path:
            return path / filepath
