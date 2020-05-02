# -*- coding: utf-8 -*-
#
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
simphony.plugins.siepic
=======================
"""

from simphony.plugins.siepic.parser import load_spi
from simphony.plugins.siepic.builders import build_circuit

def load(filename, libraries=['simphony.library.siepic']):
    """
    Loads a spice file as exported by SiEPIC Tools in KLayout.

    Parameters
    ----------
    filename : str
        The .spi file to be loaded.
    libraries : list of str, optional
        The libraries containing the component models used in the spice file.
        By default, this is the SiEPIC model library.
    
    Returns
    -------
    built : dict
        A dictionary of constructed Python objects, with the following keys:
            - `circuits`: dictionary of circuit names to their corresponding
              instantiated Subcircuit objects.
            - `subcircuits`: instantiated Subcircuit objects for all 
              subcircuits found in the spice data.
            - `analyses`: instantiated Simulation objects for all network
              analyzers found in the spice data.
    """
    # Loading the main file includes any other files linked to it internally.
    data = load_spi(filename)

    # Building the circuit returns the subcircuits, the circuits, and the 
    # simulation objects.
    built = build_circuit(data, 'simphony.library.siepic')
    return built