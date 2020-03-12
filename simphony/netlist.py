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


class NetGenerator:
    def __init__(self):
        self.num = 1

    def __next__(self):
        ret = self.num
        self.num += 1
        return ret


class Netlist:
    def __init__(self):
        self.netid = NetGenerator()
        self.nets = {}


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

    def __init__(self, name=None):
        self.name = name
        self.netid = NetGenerator()
        self.nets = {}
        self._blocks = []
        self.labels = {}

    def __getitem__(self, item):
        """
        Allows for access to blocks within the subcircuit by name or index,
        similar to a dictionary or list.
        """
        if type(item) is str:
            try:
                return self.blocks[item]
            except KeyError:
                raise KeyError('name "{}" not in subcircuit.'.format(item))
        elif type(item) is int:
            return self._blocks[item]
        else:
            raise KeyError('"{}" not in subcircuit.'.format(item))

    @property
    def blocks(self):
        return {obj.name: obj for obj in self._blocks}

    @property
    def nodes(self):
        nodes = [(block.name, node) for block in self._blocks for node in block.nodes]
        print(nodes)
        print(self.nets.values())
        for net in self.nets.values():
            e1, n1, e2, n2 = net
            nodes.remove((e1, n1))
            nodes.remove((e2, n2))
        return nodes


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
        tmp = []
        for block in blocks:
            if block.name not in self.blocks.keys():
                tmp.append(block)
            else:
                raise ValueError('name "{}" is already in circuit (names must be unique)')
        self._blocks += tmp


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
        e1, n1, e2, n2 = None, node1, None, node2

        # Handle the first element first
        # If the element is an object:
        if issubclass(type(element1), Element):
            for name, element in self._blocks.items():
                if element1 is element:
                    e1 = name
        # Else if the element is a string name:
        elif type(element1) is str:
            e1 = element1
        # Otherwise it's a TypeError
        else:
            raise TypeError('element1 should be string or Element, not "{}"'.format(type(element1)))
        # Ensure that the specified node exists on the specified element
        _ = self[e1]._node_idx_by_name(n1)
        
        # Handle the second element next
        # If the element is an object:
        if issubclass(type(element2), Element):
            for name, element in self._blocks.items():
                if element2 is element:
                    e2 = name
        # Else if the element is a string name:
        elif type(element2) is str:
            e2 = element2
        # Otherwise it's a TypeError
        else:
            raise TypeError('element1 should be string or Element, not "{}"'.format(type(element1)))
        # Ensure that the specified node exists on the specified element
        _ = self[e2]._node_idx_by_name(n2)

        self.nets[next(self.netid)] = (e1, n1, e2, n2)
        # uid = next(self.netid)
        # self.nets[(e1, n1)] = uid
        # self.nets[(e2, n2)] = uid

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
        for item in self._blocks.keys():
            out += str(type(self._blocks[item])) + '\n'
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