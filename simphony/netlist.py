# -*- coding: utf-8 -*-
#
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
simphony.netlist
================

This package contains the base classes for defining circuits.
"""

from collections import OrderedDict
import copy
import itertools
import logging
import uuid

from simphony.elements import Model

_module_logger = logging.getLogger(__name__)


class Pin:
    """
    A class representing a pin on a unique element instance. 

    Note that these are not the pins defined in Models, but are created from
    the names defined there.
    """
    _logger = _module_logger.getChild('Pin')
    
    def __init__(self, pinlist, name):
        """
        Creates a new pin.

        Parameters
        ----------
        pinlist : simphony.elements.PinList
            The `PinList` this pin resides in.
        name : str
            The name of the pin.
        """
        self._pinlist = pinlist
        self._name = name

    def __repr__(self):
        o = ".".join([self.__module__, type(self).__name__])
        try:
            return "<'{}' {} object from '{}' at {}>".format(self.name, o, self.element.name, hex(id(self)))
        except AttributeError:
            return "<'{}' {} object from at {}>".format(self.name, o, hex(id(self)))

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        # TODO: Make sure that there is no pin with the same name in _pinlist
        self._name = value

    @property
    def element(self):
        """
        Returns the element to which this pin belongs by tracing the path to
        PinList, which ought to hold a reference to an `Element`.
        """
        return self._pinlist.element

    @property
    def index(self):
        return self._pinlist.index(self)


class PinList:
    """
    A list of pins belonging to an `Element`, indexed the same way the 
    s-parameters of a `Model` are indexed.

    `PinList` maintains unique `Pin` names within its list. Pins can also be 
    accessed by index instead of name.

    Attributes
    ----------
    element : simphony.elements.Element
        The `Element` the `PinList` belongs to.
    pins : list of simphony.element.Pin
        A list of `Pin` objects, indexed in the same order as the s-parameters
        of the `Model` it represents.

    Notes
    -----
    If renaming pins, the assigned value must be a string.

    Examples
    --------
    >>> pinlist = PinList(None, 'n1', 'n2', 'n3')
    >>> pinlist.pins
    [<'n1' simphony.netlist.Pin object from at 0x7f1098ef39b0>, <'n2' simphony.netlist.Pin object from at 0x7f1098ef3c88>, <'n3' simphony.netlist.Pin object from at 0x7f108aec6160>]
    >>> pinlist.n2 = 'out1'
    >>> pinlist.pins
    [<'n1' simphony.netlist.Pin object from at 0x7f1098ef39b0>, <'out1' simphony.netlist.Pin object from at 0x7f1098ef3c88>, <'n3' simphony.netlist.Pin object from at 0x7f108aec6160>]
    >>> pinlist.pins = ('out', 'in', 'mix')
    >>> pinlist.pins = ('n1')
    """
    _logger = _module_logger.getChild('PinList')
    _pins = []

    def __init__(self, element, *args):
        """
        Parameters
        ----------
        element : simphony.elements.Element
            The element this PinList defines the pins for.
        args : str or Pin
            Number of unnamed arguments is not limited; each corresponds to a
            new `Pin` in the `PinList`. If str, Pin is created. If Pin, its
            `pinlist` attribute is updated to point to this `PinList`.
        """
        self.element = element
        self._pins = [self._normalize(pin) for pin in args]

    def _normalize(self, pin):
        """
        Takes a pin argument (string or Pin) and creates a `Pin` object.

        Parameters
        ----------
        pin : str or Pin
            The pin to be normalized to a `Pin`.
        """
        if type(pin) is Pin:
            pin._pinlist = self
            return pin
        if type(pin) is str:
            return Pin(self, pin)
        err = "expected type 'str' or 'Pin', got {}".format(type(pin))
        raise TypeError(err)

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
        self._logger.debug("'pins' property called")
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

    def index(self, pin):
        """
        Given a `Pin` object, returns its index or position in the `PinList`.

        Parameters
        ----------
        pin : simphony.netlist.Pin
            The pin object to be found in the PinList.

        Returns
        -------
        idx : int
            The index of the pin passed in.
        """
        return self._pins.index(pin)

    @property
    def names(self):
        """
        Get the names of the pins in the `PinList`, in order.

        Returns
        -------
        names : tuple of str
            The formal names of each pin in the pinlist.
        """
        return tuple([pin.name for pin in self._pins])


class Element:
    """
    Represents an instantiation of some model in a circuit.

    Unites a `Model` with a `PinList` to allow unique instances to be 
    instantiated within a `Subcircuit`.

    Attributes
    ----------
    name : str
        The read-only name of the element, unique within each `Subcircuit`. 
        If not specified on instantiation, it is autogenerated.
    model : simphony.elements.Model
        A reference to a `Model` instance (NOTe: it must be an instance, not a 
        class reference).
    pins : simphony.netlist.PinList
        A PinList, generated automatically from the model, with pins renameable
        after instantiation.
    """
    def __init__(self, model, name=None):
        self.model = model
        self._name = name if name else self.generate_name()
        self._pins = PinList(self, *model.pins)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        err = "'name' attribute is read-only."
        raise ValueError(err)

    @property
    def pins(self):
        return self._pins

    @pins.setter
    def pins(self, value):
        self._pins.pins = value

    @property
    def wl_bounds(self):
        return self.model.wl_bounds

    def __str__(self):
        pins = str(tuple([pin.name for pin in self.pins]))
        return "Element (Name: '{}', Model: '{}', Pins: {})".format(self.name, self.model.__class__.__name__, pins)

    def _generate_name(self) -> str:
        """
        Generates a new name for the `Element` based on the `Model` class name
        and a randomly generated string.

        Returns
        -------
        name : str
            A new random name for the Element.
        """
        return self.model.__class__.__name__ + "_" + str(uuid.uuid4())[:8]

class ElementList:
    """
    Maintains an ordered dict. If an update to an existing key is attempted, 
    the update fails. Keys must be deleted before being used.

    Dictionary is a mapping of names (type: `str`) to elements or blocks (type:
    `Element` or `Subcircuit`).
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

    def __delitem__(self, key):
        del self.elements[key]

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

    def __iter__(self):
        yield from self.elements.values()

    def keys(self):
        """
        Returns the keys of the `ElementList` as a list of strings.

        Returns
        -------
        keys : list of str
            The keys (or, names of `Element` instances) of `ElementList`.
        """
        return self.elements.keys()


class Netlist:
    """
    Maintains a list of all connections, or "nets", in a circuit.

    Attributes
    ----------
    nets : list of list
        Nets is a list of connections, stored as a list of two `Pins`.
    """
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
        self.nets.append([pin1, pin2])

    def clone(self):
        ret = Netlist()
        ret.nets = [[p[0], p[1]] for p in self.nets]
        return ret

    def __len__(self):
        return len(self.nets)

    def __iter__(self):
        yield from self.nets


class Subcircuit:
    """
    This implements a subcircuit that can be constructed and reused throughout
    the circuit.

    Attributes
    ----------
    name : str
        A formal name for the Subcircuit (`None` allowed).
    elements : list of elements
    connections : netlist
    pins : the new pins to use as the pins of the subcircuit
    nets
    """
    _logger = _module_logger.getChild('Subcircuit')

    def __init__(self, name=None):
        self.name = name if name else str(uuid.uuid4())
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

    @property
    def wl_bounds(self):
        """
        Returns a tuple of the valid wavelength range.
        """
        min_wl = []
        max_wl = []
        for element in self.elements:
            min_wl.append(element.wl_bounds[0])
            max_wl.append(element.wl_bounds[1])
        return (min(min_wl), max(max_wl))

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
            # TODO: Find some way to guarantee that the automatically generated
            # name does not already exist in the ElementList.
            if type(item) is tuple:
                model, name = item 
            else:
                model, name = item, None
            if issubclass(type(model), Subcircuit):
                self.elements[name if name else model.name] = model
            elif issubclass(type(model), Model):
                e = Element(model, name)
                self.elements[e.name] = e
            # new_elements.append(e)
        # return new_elements

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
            if issubclass(type(element), Element):
                return getattr(element.pins, pin)
            elif issubclass(type(element), Subcircuit):
                for opin in element.pins:
                    if opin.name == pin:
                        return opin
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