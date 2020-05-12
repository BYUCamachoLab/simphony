"""
simphony.library.gdsfactory
==============================

This package contains parameterized models of PIC components from https://github.com/gdsfactory

"""

from simphony.library.gdsfactory.load import load
from simphony.library.gdsfactory.mmi1x2 import mmi1x2

__all__ = ["load", "mmi1x2"]
