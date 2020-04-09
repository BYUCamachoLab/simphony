# -*- coding: utf-8 -*-
#
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
cli_start
=========

This package contains the base classes for defining models.
"""

import sys
import simphony

from simphony.app.cli_options import get_options

def main():
    options = get_options()
    print(options)

if __name__ == '__main__':
    main()