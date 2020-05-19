# -*- coding: utf-8 -*-
#
# Copyright © Spyder Project Contributors
# Licensed under the terms of the MIT License
# (see spyder/__init__.py for details)

"""Simphony Photonic Simulator.

This module implements a free and open source photonic integrated
circuit (PIC) simulation engine. It is speedy and easily extensible.
"""

import io
import os
import sys

import setuptools

from simphony import __version__, __website_url__  # analysis:ignore

# ==============================================================================
# Constants
# ==============================================================================
NAME = "simphony"
LIBNAME = "simphony"

# ==============================================================================
# Auxiliary functions
# ==============================================================================
extra_files = []
data_files_ext = [
    ".sparam",
    ".dat",
    ".txt",
    ".npy",
    ".npz",
]


def package_data_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            name, ext = os.path.splitext(filename)
            if ext in data_files_ext:
                paths.append(os.path.join("..", path, filename))
    return paths


extra_files += package_data_files("simphony/library")
extra_files += ["*.ini"]

# ==============================================================================
# Use README for long description
# ==============================================================================
with io.open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

# ==============================================================================
# Setup arguments
# ==============================================================================
setup_args = dict(
    name=NAME,
    version=__version__,
    description="Simphony: A Simulator for Photonic circuits",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    # download_url=__website_url__ + "",
    author="Sequoia Ploeg",
    author_email="sequoia.ploeg@ieee.org",
    url=__website_url__,
    license="GPLv3+",
    keywords="photonics simulation circuits science",
    platforms=["Windows", "Linux", "Mac OS-X"],
    packages=setuptools.find_packages(),
    package_data={"": extra_files},
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.6",
)

install_requires = [
    "scipy>=1.2.1",
    "numpy",
    "parsimonious>=0.8.1",
]

extras_require = {
    "test": ["pytest",],
}

if "setuptools" in sys.modules:
    setup_args["install_requires"] = install_requires
    setup_args["extras_require"] = extras_require

    # setup_args.pop('scripts', None)


# ==============================================================================
# Main setup
# ==============================================================================
setuptools.setup(**setup_args)
