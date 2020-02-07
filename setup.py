# -*- coding: utf-8 -*-
#
# Copyright © Spyder Project Contributors
# Licensed under the terms of the MIT License
# (see spyder/__init__.py for details)

"""
Simphony Photonic Simulator

This module implements a free and open source
photonic integrated circuit (PIC) simulation engine. 
It is speedy and easily extensible.
"""

import io
# import re
import os
import sys

import setuptools

# verstr = "unknown"
# try:
#     verstrline = open('simphony/_version.py', "rt").read()
# except EnvironmentError:
#     # No version file.
#     raise RuntimeError("Unable to find version in simphony/_version.py")
# else:
#     VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
#     mo = re.search(VSRE, verstrline, re.M)
#     if mo:
#         verstr = mo.group(1)
#     else:
#         raise RuntimeError("unable to find version in simphony/_version.py")

#==============================================================================
# Constants
#==============================================================================
NAME = 'simphony'
LIBNAME = 'simphony'
from simphony import __version__, __website_url__  #analysis:ignore

#==============================================================================
# Auxiliary functions
#==============================================================================
extra_files = []
data_files_ext = [
    '.sparam',
    '.dat',
    '.txt',
    '.npy',
    '.npz',
]

def package_data_files(directory):
    paths =[]
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            name, ext = os.path.splitext(filename)
            if ext in data_files_ext:
                paths.append(os.path.join('..', path, filename))
    return paths

extra_files += package_data_files('simphony/DeviceLibrary')
extra_files += ['*.ini']

#==============================================================================
# Use Readme for long description
#==============================================================================
with io.open('README.md', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

# with open("README.md", "r") as fh:
#     long_description = fh.read()

#==============================================================================
# Setup arguments
#==============================================================================
setup_args = dict(
    name=NAME,
    version=__version__,
    description='Simphony: A Simulator for Photonic circuits',
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    # download_url=__website_url__ + "",
    author='Sequoia Ploeg',
    author_email='sequoia.ploeg@ieee.org',
    url=__website_url__,
    license='MIT',
    keywords='photonics simulation circuits science',
    platforms=["Linux"], #["Windows", "Linux", "Mac OS-X"] support coming
    packages=setuptools.find_packages(),
    package_data={
        '': extra_files,
    },
    classifiers=['License :: OSI Approved :: MIT License',
                #  'Operating System :: MacOS',
                #  'Operating System :: Microsoft :: Windows',
                 'Operating System :: POSIX :: Linux',
                #  'Operating System :: OS Independent',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.4',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Development Status :: 2 - Pre-Alpha',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                #  'Intended Audience :: Developers',
                 'Topic :: Scientific/Engineering'],
)

install_requires = [
    'scipy>=1.2.1',
    'numpy',
    'numba',
]

if 'setuptools' in sys.modules:
    setup_args['install_requires'] = install_requires
    # setup_args['extras_require'] = extras_require

    # setup_args['entry_points'] = {
    #     'gui_scripts': [
    #         '{} = spyder.app.start:main'.format(
    #             'spyder3' if PY3 else 'spyder')
    #     ]
    # }

    # setup_args.pop('scripts', None)


#==============================================================================
# Main setup
#==============================================================================
setuptools.setup(**setup_args)