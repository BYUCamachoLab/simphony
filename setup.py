# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""Simphony Photonic Simulator.

This module implements a free and open source photonic integrated
circuit (PIC) simulation engine. It is speedy and easily extensible.
"""

import io
import sys

import setuptools

from simphony import __version__, __website_url__  # analysis:ignore

# ==============================================================================
# Constants
# ==============================================================================
NAME = "simphony"
LIBNAME = "simphony"

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
    author="AustP",
    author_email="austp17@gmail.com",
    url=__website_url__,
    license="MIT",
    keywords="photonics simulation circuits science",
    platforms=["Windows", "Linux", "Mac OS-X"],
    packages=setuptools.find_packages(),
    package_data={
        "": ["*.ini"],
        "simphony.libraries.siepic": ["source_data/*", "source_data/*/*"],
        "simphony.tests": ["mzi.json"],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
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
    "scipy>=1.5.4",
    "numpy>=1.19.5",
    "parsimonious>=0.8.1",
]

extras_require = {
    "test": ["pytest"],
}

if "setuptools" in sys.modules:
    setup_args["install_requires"] = install_requires
    setup_args["extras_require"] = extras_require

    # setup_args.pop('scripts', None)


# ==============================================================================
# Main setup
# ==============================================================================
setuptools.setup(**setup_args)
