# -*- coding: utf-8 -*-
#
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

from collections import OrderedDict
# import keyword
import copy
import logging

_module_logger = logging.getLogger(__name__)


class DeviceModel:
    def __init__(self, name, model_type, **parameters):
        """
        Defines a device model.

        Parameters
        ----------
        name : str
        model_type : str
        parameters : dict
        """
        self._name = str(name)
        self._model_type = str(model_type)

        self._parameters = {}
        for key, value in parameters.items():
            if key.endswith('_'):
                key = key[:-1]
            self._parameters[key] = value

    def clone(self):
        """
        Creates a deep copied clone of the model.

        Objects in memory are stored separate from the original and can be
        safely modified without changing the original.

        Returns
        -------
        copy : DeviceModel
            An independent copy of the original DeviceModel.
        """
        return copy.deepcopy(self)

    @property
    def name(self):
        return self._name

    @property
    def model_type(self):
        return self._model_type

    @property
    def parameters(self):
        return self._parameters.keys()

    def __getitem__(self, name):
        return self._parameters[name]

    def __getattr__(self, name):

        try:
            return self._parameters[name]
        except KeyError:
            if name.endswith('_'):
                return self._parameters[name[:-1]]

    # def __repr__(self):
    #     return str(self.__class__) + ' ' + self.name

    # def __str__(self):
    #     return ".model {0._name} {0._model_type} ({1})".format(self, join_dict(self._parameters))


class PinDefinition:
    """
    This class defines a pin of some element.
    """
    def __init__(self, position, name=None, alias=None, optional=False):
        """
        Initializes a pin.

        Parameters
        ----------
        position :
        name : str, optional
        alias : str, optional
        """
        self._position = position
        self._name = name
        self._alias = alias
        # self._optional = optional

    def clone(self):
        return copy.deepcopy(self)

    @property
    def position(self):
        return self._position

    @property
    def name(self):
        return self._name

    @property
    def alias(self):
        return self._alias

    # @property
    # def optional(self):
    #     return self._optional


class Pin(PinDefinition):
    """
    This class implements a pin of an element. It stores a reference to the 
    element, the name of the pin, and the node.
    """
    _logger = _module_logger.getChild('Pin')

    def __init__(self, element, pin_definition, node):
        """
        Initializes a pin to an element and connects it to a node.

        Parameters
        ----------
        element
        pin_definition
        node
        """
        super().__init__(pin_definition.position, pin_definition.name, pin_definition.alias)

        self._element = element
        self._node = node

        node.connect(self)

    @property
    def element(self):
        return self._element

    @property
    def node(self):
        return self._node

    def __repr__(self):
        return "Pin {} of {} on node {}".format(self._name, self._element.name, self._node)

    def disconnect(self):
        self._node.disconnect(self)
        self._node = None


class Node:
    """
    This implements a node in the circuit. It stores references to the pins 
    that are connected to the node.
    """
    pass

class Netlist:
    """
    This implements a base class for a netlist.

    In its simplest form, a netlist consists of a list of the electronic 
    components in a circuit and a list of the nodes they are connected to.

    Attributes
    ----------
    _nets?
    """
    def __init__(self):
        self._nodes = {}
        self._subcircuits = OrderedDict()
        self._elements = OrderedDict()
        self._models = {}

    @property
    def nodes(self):
        return self._nodes.values()

    @property
    def node_names(self):
        return self._nodes.keys()

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
    
    def to_spice():
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
    _logger = _module_logger.getChild('Circuit')

    _logger.info('Circuit called.')
    pass