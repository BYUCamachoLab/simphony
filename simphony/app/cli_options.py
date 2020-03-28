# -*- coding: utf-8 -*-
#
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
cli_options
=================

This file contains the simphony command line options.
"""

import argparse

def get_options(argv=None):
    """Convert options into commands.
    
    Returns
    -------
    commands, message
    """
    parser = argparse.ArgumentParser(
        prog='simphony',
        usage="%(prog)s [options] files",
    )
    parser.add_argument(
        '-lf', 
        '--lumerical', 
        dest='lumerical_file',
        action='store',
        default='',
        help='A Lumerical Script File (.lsf) netlist to run Simphony on.',
        metavar='',
    )
    options = parser.parse_args(argv)
    return options