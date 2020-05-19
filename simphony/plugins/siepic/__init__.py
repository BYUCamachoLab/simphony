# -*- coding: utf-8 -*-
# Copyright © 2019-2020 Simphony Project Contributors and others (see AUTHORS.txt).
# The resources, libraries, and some source files under other terms (see NOTICE.txt).
#
# This file is part of Simphony.
#
# Simphony is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Simphony is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Simphony. If not, see <https://www.gnu.org/licenses/>.

"""
simphony.plugins.siepic
=======================
"""

from simphony.plugins.siepic.builders import build_circuit
from simphony.plugins.siepic.parser import load_spi


def load(filename, libraries=["simphony.library.siepic"]):
    """Loads a spice file as exported by SiEPIC Tools in KLayout.

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
    built = build_circuit(data, "simphony.library.siepic")
    return built
