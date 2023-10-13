# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)
"""Utilities for loading and caching data files."""

import importlib.resources
from functools import lru_cache
from pathlib import Path
from typing import Union

import numpy as np

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


@lru_cache()
def _load_txt_cached(path: Union[Path, str]) -> np.ndarray:
    """Loads a text file from the source_data directory and caches it.

    Parameters
    ----------
    filename : str
        The name of the file to be loaded.

    Returns
    -------
    content : str
        The contents of the file.
    """
    return np.loadtxt(path)
