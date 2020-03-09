# -*- coding: utf-8 -*-
#
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

from collections import OrderedDict
# import keyword
import copy
import logging
import uuid

from simphony.elements import Element

_module_logger = logging.getLogger(__name__)


# class DeviceModel:
#     def __init__(self, name, model_type, **parameters):
#         """
#         Defines a device model.

#         Parameters
#         ----------
#         name : str
#         model_type : str
#         parameters : dict
#         """
#         self._name = str(name)
#         self._model_type = str(model_type)

#         self._parameters = {}
#         for key, value in parameters.items():
#             if key.endswith('_'):
#                 key = key[:-1]
#             self._parameters[key] = value

#     def clone(self):
#         """
#         Creates a deep copied clone of the model.

#         Objects in memory are stored separate from the original and can be
#         safely modified without changing the original.

#         Returns
#         -------
#         copy : DeviceModel
#             An independent copy of the original DeviceModel.
#         """
#         return copy.deepcopy(self)

#     @property
#     def name(self):
#         return self._name

#     @property
#     def model_type(self):
#         return self._model_type

#     @property
#     def parameters(self):
#         return self._parameters.keys()

#     def __getitem__(self, name):
#         return self._parameters[name]

#     def __getattr__(self, name):

#         try:
#             return self._parameters[name]
#         except KeyError:
#             if name.endswith('_'):
#                 return self._parameters[name[:-1]]

#     # def __repr__(self):
#     #     return str(self.__class__) + ' ' + self.name

#     # def __str__(self):
#     #     return ".model {0._name} {0._model_type} ({1})".format(self, join_dict(self._parameters))


# class PinDefinition:
#     """
#     This class defines a pin of some element.
#     """
#     def __init__(self, position, name=None, alias=None, optional=False):
#         """
#         Initializes a pin.

#         Parameters
#         ----------
#         position :
#         name : str, optional
#         alias : str, optional
#         """
#         self._position = position
#         self._name = name
#         self._alias = alias
#         # self._optional = optional

#     def clone(self):
#         return copy.deepcopy(self)

#     @property
#     def position(self):
#         return self._position

#     @property
#     def name(self):
#         return self._name

#     @property
#     def alias(self):
#         return self._alias

#     # @property
#     # def optional(self):
#     #     return self._optional


# class Pin(PinDefinition):
#     """
#     This class implements a pin of an element. It stores a reference to the 
#     element, the name of the pin, and the node.
#     """
#     _logger = _module_logger.getChild('Pin')

#     def __init__(self, element, pin_definition, node):
#         """
#         Initializes a pin to an element and connects it to a node.

#         Parameters
#         ----------
#         element
#         pin_definition
#         node
#         """
#         super().__init__(pin_definition.position, pin_definition.name, pin_definition.alias)

#         self._element = element
#         self._node = node

#         node.connect(self)

#     @property
#     def element(self):
#         return self._element

#     @property
#     def node(self):
#         return self._node

#     def __repr__(self):
#         return "Pin {} of {} on node {}".format(self._name, self._element.name, self._node)

#     def disconnect(self):
#         self._node.disconnect(self)
#         self._node = None


# class Node:
#     """
#     This implements a node in the circuit. It stores references to the pins 
#     that are connected to the node.
#     """
#     pass

# class Netlist:
#     """
#     This implements a base class for a netlist.

#     In its simplest form, a netlist consists of a list of the electronic 
#     components in a circuit and a list of the nodes they are connected to.

#     Attributes
#     ----------
#     _nets?
#     """
#     def __init__(self):
#         self._nodes = {}
#         self._subcircuits = OrderedDict()
#         self._elements = OrderedDict()
#         self._models = {}

#     @property
#     def nodes(self):
#         return self._nodes.values()

#     @property
#     def node_names(self):
#         return self._nodes.keys()


class NetGenerator:
    def __init__(self):
        self.num = 1

    def __next__(self):
        ret = self.num
        self.num += 1
        return ret


class Subcircuit:
    """
    This implements a subcircuit that can be constructed and reused throughout
    the circuit.

    Attributes
    ----------
    elements : list of elements
    connections : netlist
    pins : the new pins to use as the pins of the subcircuit
    nets
    """
    _logger = _module_logger.getChild('Subcircuit')

    def __init__(self, blockname=None):
        self.blockname = blockname
        self.netid = NetGenerator()
        # self.nets = OrderedDict()
        self.nets = {}
        self.blocks = OrderedDict()
        self.labels = {}

    def __getitem__(self, name):
        """
        Allows for access to blocks within the subcircuit by name or index,
        similar to a dictionary or list.
        """
        if type(name) is str:
            try:
                return self.blocks[name]
            except KeyError:
                raise KeyError('name "{}" not in subcircuit.'.format(name))
        elif type(name) is int:
            return list(self.blocks.items())[name][1]
        else:
            raise KeyError('name "{}" not in subcircuit.'.format(name))

    def add(self, blocks):
        """
        Adds elements to a subcircuit.

        If any one item fails to be added to the circuit, none of the items
        are added.

        Parameters
        ----------
        blocks : list of tuples
            A list of elements to be added. Tuples are of the form 
            (`name`, `block`), where `name` is a unique 
            string identifying the element in the subcircuit and 
            `block` can be an instance of some element (i.e. a subclass of
            `simphony.elements.Element`) or another subcircuit.
        
        Raises
        ------
        TypeError
            If `blocks` is not a list.
        """
        if type(blocks) is not list:
            raise TypeError('list expected, received {}'.format(type(blocks)))
        elements = {}
        for item in blocks:
            name, element = item
            if name is None:
                while True:
                    name = element.__class__.__name__ + "_" + str(uuid.uuid4())[:8]
                    if name not in self.blocks.keys():
                        break
                elements[name] = element
            elif name not in self.blocks.keys():
                self._logger.debug('Inserting "{}": {}'.format(name, element))
                elements[name] = element
            elif name in self.blocks.keys():
                raise KeyError('name "{}" is already defined.'.format(name))
            else:
                raise NotImplementedError('I thought this point was unreachable. \
                    If you got here, it\'s time to open a new issue on GitHub. \
                        Lemme know how you did it.')
        self.blocks.update(elements)

    def connect(self, element1, node1, element2, node2):
        """
        Connect two elements with a net.

        Netlists are unique to and stored by a Subcircuit object. This means 
        net identifiers (numbers, by default) can be reused between separate
        subcircuits but must be unique within each.

        Parameters
        ----------
        element1 : 
        node1 : 
        element2 :
        node2 : 
        """
        # Handle the first element first
        e1, n1, e2, n2 = None, node1, None, node2
        if issubclass(type(element1), Element):
            for name, element in self.blocks.items():
                if element1 is element:
                    e1 = name
        elif type(element1) is str:
            e1 = element1
        else:
            raise TypeError('element1 should be string or Element, not "{}"'.format(type(element1)))
        _ = self[e1]._node_idx_by_name(n1)
        
        if issubclass(type(element2), Element):
            for name, element in self.blocks.items():
                if element2 is element:
                    e2 = name
        elif type(element2) is str:
            e2 = element2
        else:
            raise TypeError('element1 should be string or Element, not "{}"'.format(type(element1)))
        _ = self[e2]._node_idx_by_name(n2)

        self.nets[next(self.netid)] = (e1, n1, e2, n2)

    def connect_many(self, conns):
        """
        A convenience function for connecting many nets at once.

        Parameters
        ----------
        conns : list of tuple
            A list of tuples, each formed as a tuple of arguments in the same
            order as that accepted by `connect`.
        """
        for c in conns:
            self.connect(*c)

    def to_spice(self):
        """
        Perhaps this shouldn't be built into Subcircuit, maybe an adapter
        class or some translator instantiated with a Subcircuit that iterates
        through and creates a netlist.
        """
        out = ""
        for item in self.blocks.keys():
            out += str(type(self.blocks[item])) + '\n'
        return out

# class Circuit:
#     """
#     This class implements a cicuit netlist.
    
#     To get the corresponding Spice netlist use:
       
#        ```
#        circuit = Circuit()
#        ...
#        str(circuit)
#        ```

#     Attributes
#     ----------
#     elements : list
#     subcircuits : list
#     connections : netlist
#     """
#     _logger = _module_logger.getChild('Circuit')

#     _logger.info('Circuit called.')
#     pass