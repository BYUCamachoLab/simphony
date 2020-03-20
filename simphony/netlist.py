# -*- coding: utf-8 -*-
#
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

from collections import OrderedDict
import itertools
import logging
import uuid

from simphony.elements import Model

_module_logger = logging.getLogger(__name__)


class Pin:
    _logger = _module_logger.getChild('Pin')
    
    def __init__(self, pinlist, name):
        """
        Parameters
        ----------
        pinlist : simphony.elements.PinList
        name : str
        """
        self._pinlist = pinlist
        self._name = name

    def __repr__(self):
        o = ".".join([self.__module__, type(self).__name__])
        return "<'{}' {} object from '{}' at {}>".format(self.name, o, self.element.name, hex(id(self)))

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        # TODO: Make sure that there is no pin with the same name in _pinlist
        self._name = value

    @property
    def element(self):
        return self._pinlist.element


class PinList:
    """
    Notes
    -----
    If renaming pins, the assigned value must be a string.

    Examples
    --------
    >>> pins = PinList('n1', 'n2', 'n3')
    >>> pins.n2 = 'out1'
    """
    _logger = _module_logger.getChild('PinList')
    _pins = []

    def __init__(self, element, *args):
        """
        Parameters
        ----------
        element : simphony.elements.Element
        args : tuple of str
        """
        self.element = element
        self._pins = [Pin(self, arg) for arg in args]

    def __getitem__(self, item):
        return self._pins[item]

    def __setitem__(self, key, value):
        self._pins[key].name = value

    def __getattr__(self, name):
        for pin in self._pins:
            if name == pin.name:
                return pin
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name not in [pin.name for pin in self._pins]:
            return super().__setattr__(name, value)
        if value in [pin.name for pin in self._pins]:
            error = "'{}' already exists in PinList".format(value)
            raise AttributeError(error)
        for pin in self._pins:
            if name == pin.name:
                pin.name = value
                return

    def __str__(self):
        _pins = str(tuple([pin.name for pin in self._pins]))
        return "{} (Pins: {})".format(self.__class__.__name__, _pins)

    def __len__(self):
        return len(self._pins)

    @property
    def pins(self):
        self._logger.warn("'pins' property called")
        return self._pins

    @pins.setter
    def pins(self, names):
        if type(names) is not tuple:
            err = "expected type 'tuple' but got '{}'".format(type(names))
            raise TypeError(err)
        if len(names) != len(self._pins):
            err = "number of new pins does not match number of existing pins ({} != {})".format(len(names), len(self._pins))
            raise ValueError(err)
        for idx, pin in enumerate(self._pins):
            pin.name = names[idx]

    def index(self, item):
        """
        Returns
        -------
        idx : int
            The index of the pin passed in.
        """
        return self._pins.index(item)

    @property
    def names(self):
        return tuple([pin.name for pin in self._pins])


class Element:
    def __init__(self, model, name=None):
        self.name = name if name else model.__class__.__name__ + "_" + str(uuid.uuid4())[:8]
        self.model = model
        self._pins = PinList(self, *model.pins)

    @property
    def pins(self):
        return self._pins

    @pins.setter
    def pins(self, value):
        self._pins.pins = value

    def __str__(self):
        pins = str(tuple([pin.name for pin in self.pins]))
        return "Element (Name: '{}', Model: '{}', Pins: {})".format(self.name, self.model.__class__.__name__, pins)

class ElementList:
    """
    Maintains an ordered dict. If an update to a key is attempted, the update
    fails. Keys must be deleted before being used.
    """
    def __init__(self):
        self.elements = OrderedDict()

    def __getitem__(self, item):
        """
        Allows for access to blocks within the subcircuit by name or index,
        similar to a dictionary or list.
        """
        if type(item) is str:
            try:
                return self.elements[item]
            except KeyError:
                raise KeyError('name "{}" not in subcircuit.'.format(item))
        elif type(item) is int:
            return list(self.elements.values())[item]
        else:
            raise KeyError('"{}" not in subcircuit.'.format(item))

    def __setitem__(self, key, value):
        if key in self.elements.keys():
            raise KeyError("Key '{}' already exists.".format(key))
        self.elements[key] = value

    def __len__(self):
        return len(self.elements)

    def __str__(self):
        if len(self) > 0:
            val = '{'
            for k, v in self.elements.items():
                val += "{}: {}, ".format(k, v)
            val = val[:-2] + '}'
            return val
        else:
            return "{}"


class Netlist:
    def __init__(self):
        self.nets = []

    def __str__(self):
        val = ''
        o = ".".join([self.__module__, type(self).__name__])
        val += "<{} object at {}>".format(o, hex(id(self)))
        for item in self.nets:
            val += '\n  {}'.format(str(item))
        return val

    def add(self, pin1, pin2):
        for pin in itertools.chain(*self.nets):
            culprit = None
            if pin is pin1:
                culprit = pin1
            if pin is pin2:
                culprit = pin2
            if culprit:
                err = "Netlist already contains connection for {}".format(culprit)
                raise ValueError(err)
        self.nets.append((pin1, pin2))


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
        self.elements = ElementList()
        self.netlist = Netlist()

    # TODO: Do we want to be able to access elements as dictionary items from
    # the subcircuit directly? Or make them pass through `elements` first?

    # def __repr__(self):
    #     val = ''
    #     o = ".".join([self.__module__, type(self).__name__])
    #     val += "<{} object at {}>".format(o, hex(id(self)))
    #     for item in self._blocks:
    #         val += '\n  {}'.format(str(item))
    #     return val

    @property
    def pins(self):
        all_pins = set([pin for element in self.elements for pin in element.pins])
        con_pins = set(itertools.chain(*self.netlist.nets))
        ext_pins = all_pins ^ con_pins
        return ext_pins

    def add(self, elements):
        """
        Adds elements to a subcircuit.

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
        if type(elements) is not list:
            raise TypeError('list expected, received {}'.format(type(elements)))
        
        new_elements = []
        for item in elements:
            e = Element(*item) if type(item) is tuple else Element(item)
            self.elements[e.name] = e
            new_elements.append(e)
        return new_elements

    def connect(self, element1, pin1, element2, pin2):
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
        e1 = self._get_element(element1)
        p1 = self._get_pin(e1, pin1)
        e2 = self._get_element(element2)
        p2 = self._get_pin(e2, pin2)
        self.netlist.add(p1, p2)

    def _get_element(self, element):
        if issubclass(type(element), Element):
            return element
        elif type(element) is str:
            return self.elements[element]
        else:
            raise TypeError('element should be string or Element, not "{}"'.format(type(element)))

    def _get_pin(self, element, pin):
        """
        Parameters
        ----------
        element : Element
        pin : str
        """
        if type(pin) is Pin:
            return pin
        elif type(pin) is str:
            return getattr(element.pins, pin)
        else:
            err = "expected type 'Pin' or 'str', got {}".format(type(pin))
            raise TypeError(err)

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
            self._logger.debug("Handling connection: {}".format(c))
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

    @property
    def model(self):
        wl_bounds = None
        pins = None
        def s_parameters(self, a, b):
            return a + b
        
        klass = type(self.name, (Model,), {})
        klass.wl_bounds = wl_bounds
        klass.pins = pins
        klass.s_parameters = s_parameters

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