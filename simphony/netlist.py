# -*- coding: utf-8 -*-
#
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

from collections import OrderedDict
import keyword
import logging


_module_logger = logging.getLogger(__name__)

class Node:
    """
    This implements a node in the circuit. It stores references to the pins 
    that are connected to the node.
    """
    pass

class Netlist:
    """
    This implements a base class for a netlist.
    """
    pass

class Subcircuit:
    """
    This implements a subcircuit that can be constructed and reused throughout
    the circuit.

    Attributes
    ----------
    elements : list of elements
    connections : netlist
    pins : the new pins to use as the pins of the subcircuit
    """
    pass

class Circuit:
    """
    This class implements a cicuit netlist.
    
    To get the corresponding Spice netlist use:
       
       ```
       circuit = Circuit()
       ...
       str(circuit)
       ```

    Attributes
    ----------
    elements : list
    subcircuits : list
    connections : netlist
    """
    pass