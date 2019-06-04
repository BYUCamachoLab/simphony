"""Simphony Photonic Simulator

This module implements a free and open source photonic integrated circuit (PIC)
simulation engine. It is speedy and easily extensible.
"""

name = "simphony"
from simphony._version import __version__
__author__ = 'Sequoia Ploeg, Hyrum Gunther'

__all__ = [
    'netlist',
    'simulation',
    'models',
]

from . import *
