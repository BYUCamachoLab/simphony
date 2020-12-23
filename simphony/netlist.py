# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
simphony.netlist
================

This package contains the base classes for defining circuits.
"""

import copy
import itertools
import logging
import uuid

from typing import Optional, Union
from simphony.elements import Model

_module_logger = logging.getLogger(__name__)


class Pin:
    """A class representing a pin on a unique element instance.

    Note that these are not the pins defined in Models, but are created from
    the names defined there.

    Parameters
    ----------
    pinlist : simphony.elements.PinList
        The ``PinList`` this pin resides in.
    name : str
        The name of the pin.

    Notes
    -----
    A Pin can only exist in one PinList at a time. Moving a Pin into another
    PinList will result in the automatic and silent change of the ``pinlist``
    reference.
    """

    _logger = _module_logger.getChild("Pin")

    def __init__(self, pinlist, name: Optional[str]) -> None:
        self.pinlist = pinlist
        self.name = name

    def __repr__(self):
        try:
            return "<Pin '{}' at {}>".format(self.name, self.element)
        except:
            return "<Pin '{}' at {}>".format(self.name, hex(id(self)))

    @property
    def element(self):
        """Returns the element to which this pin belongs by tracing the path to
        PinList, which ought to hold a reference to an ``Element``."""
        return self.pinlist.element

    @property
    def index(self) -> int:
        return self.pinlist.index(self)


class PinList:
    """A list of pins belonging to an ``Element``, indexed the same way the
    s-parameters of a ``Model`` are indexed.

    ``PinList`` maintains unique ``Pin`` names within its list. Pins can also be
    accessed by index instead of name.

    Parameters
    ----------
    element : simphony.elements.Element
        The element this PinList defines the pins for.
    pins : str or Pin
        Number of unnamed arguments is not limited; each corresponds to a
        new ``Pin`` in the ``PinList``. If str, Pin is created. If Pin, its
        ``pinlist`` attribute is updated to point to this ``PinList``.

    Attributes
    ----------
    element : simphony.elements.Element
        The ``Element`` the ``PinList`` belongs to.
    pins : list of simphony.element.Pin
        A list of ``Pin`` objects, indexed in the same order as the s-parameters
        of the ``Model`` it represents.

    Notes
    -----
    If renaming pins, the assigned value must be a string.

    .. note::
       If a PinList contains two Pins with the same string name and access by
       string value is attempted, a ``LookupError`` is raised complaining that the
       name is ambiguous.

    Warning
    -------
    Adding two PinLists together will change the pinlist reference of the pins
    they contain to point to the new result. This is because Pins can only
    be referenced by one PinList at a time. Inserting them into a new PinList
    automatically and silently changes their references.

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

    _logger = _module_logger.getChild("PinList")

    def __init__(self, element: None, *pins) -> None:
        self.element = element
        self.pins = []

        for pin in pins:
            self.append(pin)

    def __getitem__(self, item: Union[str, Pin, int]) -> Pin:
        if type(item) is str:
            ret = None
            for pin in self.pins:
                if pin.name == item:
                    if ret is None:
                        ret = pin
                    else:
                        raise LookupError(
                            "Name '{}' is ambiguous; multiple pins with that name exist!".format(
                                item
                            )
                        )
            return ret
        elif type(item) is int:
            return self.pins[item]
        elif type(item) is Pin:
            if item in self.pins:
                return item
        err = "'{}' not in PinList.".format(item)
        raise KeyError(err)

    def __setitem__(self, key, value):
        key = self[key]
        if type(value) is str:
            key.name = value
            return
        elif type(value) is Pin:
            value = self._normalize(value)
            idx = self.index(key)
            self.pins[idx] = value
            return
        err = "'{}' not in PinList.".format(key)
        raise KeyError(err)

    def __len__(self) -> int:
        return len(self.pins)

    def __add__(self, other):
        pinlist = PinList(self.element)
        pinlist.pins = self.pins + other.pins
        for pin in pinlist:
            pin.pinlist = pinlist
        return pinlist

    def __repr__(self):
        return str(self.pins)

    def _normalize(self, pin: Union[str, Pin]) -> Pin:
        if type(pin) is Pin:
            pin.pinlist = self
            return pin
        if type(pin) is str:
            return Pin(self, pin)
        err = "expected type 'str' or 'Pin', got {}".format(type(pin))
        raise TypeError(err)

    def contains(self, pin: Pin) -> bool:
        """
        Parameters
        ----------
        pin : str or Pin
            The pin to verify is in the list.

        Returns
        -------
        bool
            True if the pin is in the list.
        """
        if type(pin) is str:
            if pin.name in [pin.name for pin in self.pins]:
                return True
        elif type(pin) is Pin:
            if pin in self.pins:
                return True
        return False

    def append(self, pin: Union[str, Pin]) -> None:
        """Takes a pin argument (string or Pin) and creates a ``Pin`` object.

        Parameters
        ----------
        pin : str or Pin
            The pin to be normalized to a ``Pin``.
        """
        pin = self._normalize(pin)
        if self.contains(pin):
            raise ValueError("name '{}' is not unique in PinList")
        self.pins.append(pin)

    def remove(self, *pins) -> None:
        """Removes a pin from the pinlist by name or value.

        Parameters
        ----------
        pins : str or Pin
            Variable length argument list; the pins to be removed from the
            circuit.
        """
        for pin in pins:
            self.pins.remove(self[pin])

    def pop(self, idx=-1):
        """Removes a pin from the pinlist by index (or, the last inserted pin
        by default).

        Parameters
        ----------
        idx : int, optional
            The index of the pin to remove from the list. If none, removes the
            last item in the list.
        """
        return self.pins.pop(idx)

    def rename_pin(self, current_name, new_name):
        self[current_name].name = new_name

    def rename_pins(self, *names):
        if len(names) != len(self.pins):
            err = "number of new pins does not match number of existing pins ({} != {})".format(
                len(names), len(self.pins)
            )
            raise ValueError(err)
        for pin, name in zip(self.pins, names):
            pin.name = name

    def index(self, pin: Pin) -> int:
        """Given a ``Pin`` object, returns its index or position in the
        ``PinList``.

        Parameters
        ----------
        pin : simphony.netlist.Pin
            The pin object to be found in the PinList.

        Returns
        -------
        idx : int
            The index of the pin passed in.
        """
        return self.pins.index(pin)

    @property
    def pinnames(self):
        """Get the names of the pins in the ``PinList``, in order.

        Returns
        -------
        names : tuple of str
            The formal names of each pin in the pinlist.
        """
        return tuple([pin.name for pin in self.pins])


class Element:
    """Represents an instantiation of some model in a circuit.

    Unites a ``Model`` with a ``PinList`` to allow unique instances to be
    instantiated within a ``Subcircuit``.

    Parameters
    ----------
    model : simphony.elements.Model
        The model this element represents.
    name : str, optional
        Unique string identifying the element, autogenerated if not specified.

    Attributes
    ----------
    name : str
        The read-only name of the element, unique within each ``Subcircuit``.
        If not specified on instantiation, it is autogenerated.
    model : simphony.elements.Model
        A reference to a ``Model`` instance (NOTe: it must be an instance, not a
        class reference).
    pins : simphony.netlist.PinList
        A PinList, generated automatically from the model, with pins renameable
        after instantiation.

    Notes
    -----
    Deep copying doesn't have the full effect on this object. Since models are
    supposed to be universal throughout a simulation (thereby reducing cache
    and comparison time), when this object is deep copied, the ``model``
    attribute remains as a reference to the same former model object.
    """

    # FIXME: Do we want `name` to be read-only/

    def __init__(self, model, name=None):
        self.model = model
        self.name = name if name else self._generate_name()
        self.pinlist = PinList(self, *model.pins)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "model":
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __repr__(self):
        return "<Element '{}' at {}>".format(self.name, hex(id(self)))

    @property
    def pins(self):
        return self.pinlist

    @pins.setter
    def pins(self, value):
        if type(value) is tuple:
            self.pinlist.rename_pins(*value)

    @property
    def wl_bounds(self):
        return self.model.wl_bounds

    def _generate_name(self) -> str:
        """Generates a new name for the ``Element`` based on the ``Model``
        class name and a randomly generated string.

        Returns
        -------
        name : str
            A new random name for the Element.
        """
        return self.model.__class__.__name__ + "_" + str(uuid.uuid4())[:8]


class ElementList:
    """Maintains an ordered dict. If an update to an existing key is attempted,
    the update fails. Keys must be deleted before being used.

    Dictionary is a mapping of names (type: ``str``) to elements or blocks (type:
    ``Element`` or ``Subcircuit``).

    Allows for access to blocks within the subcircuit by name or index,
    similar to a dictionary or list.
    """

    def __init__(self):
        self.elements = []

    def __getitem__(self, item):
        if type(item) is str:
            for element in self.elements:
                if element.name == item:
                    return element
        elif type(item) is int:
            return self.elements[item]
        elif type(item) is Element:
            if item in self.elements:
                return item
        err = '"{}" not in ElementList.'.format(item)
        raise KeyError(err)

    def __len__(self):
        return len(self.elements)

    def __iter__(self):
        yield from self.elements

    def append(self, element):
        if element.name in [element.name for element in self.elements]:
            raise KeyError("Key '{}' already exists.".format(element.name))
        self.elements.append(element)

    def remove(self, name):
        element = self[name]
        self.elements.remove(element)

    def pop(self, idx):
        return self.elements.pop(idx)

    def keys(self):
        """Returns the keys of the ``ElementList`` as a list of strings.

        Returns
        -------
        keys : list of str
            The keys (or, names of ``Element`` instances) of ``ElementList``.
        """
        return self.elements.keys()


class Netlist:
    """Maintains a list of all connections, or "nets", in a circuit.

    Attributes
    ----------
    nets : list of list
        Nets is a list of connections, stored as a list of two ``Pins``.
    """

    def __init__(self):
        self.nets = []

    def __getitem__(self, idx):
        return self.nets[idx]

    def __iter__(self):
        yield from self.nets

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
    """This implements a subcircuit that can be constructed and reused
    throughout the circuit.

    Parameters
    ----------
    name : str
        A name for identifying the subcircuit.

    Attributes
    ----------
    name : str
        A formal name for the Subcircuit (``None`` allowed).
    elements : list of elements
    connections : netlist
    pins : the new pins to use as the pins of the subcircuit
    nets
    """

    _logger = _module_logger.getChild("Subcircuit")

    def __init__(self, name=None):
        self.name = name if name else str(uuid.uuid4())
        self.elements = ElementList()
        self.netlist = Netlist()
        self.settings = {}

    @property
    def pins(self):
        all_pins = set([pin for element in self.elements for pin in element.pins])
        con_pins = set(itertools.chain(*self.netlist.nets))
        ext_pins = all_pins ^ con_pins
        return ext_pins

    @property
    def wl_bounds(self):
        """Returns a tuple of the valid wavelength range."""
        min_wl = []
        max_wl = []
        for element in self.elements:
            minn, maxx = element.wl_bounds
            # min_wl.append(element.wl_bounds[0])
            # max_wl.append(element.wl_bounds[1])
            if minn is not None:
                min_wl.append(minn)
            if maxx is not None:
                max_wl.append(maxx)
        return (min(min_wl), max(max_wl))

    def add(self, elements):
        """Adds elements to a subcircuit.

        Parameters
        ----------
        elements : list of tuples
            A list of elements to be added. Tuples are of the form
            (``name``, ``block``), where ``name`` is a unique
            string identifying the element in the subcircuit and
            ``block`` can be an instance of some element (i.e. a subclass of
            ``simphony.elements.Element``) or another subcircuit.

        Returns
        -------
        added : list
            A list of object references to elements added to the subcircuit.
            Insertion order is preserved (order of the list is the same as the
            order elements were added).

        Raises
        ------
        TypeError
            If ``blocks`` is not a list.
        """
        if type(elements) is not list:
            raise TypeError("list expected, received {}".format(type(elements)))

        added = []
        for item in elements:
            # TODO: Find some way to guarantee that the automatically generated
            # name does not already exist in the ElementList.
            if type(item) is tuple:
                model, name = item
            else:
                model, name = item, None

            if issubclass(type(model), Subcircuit):
                model.name = name if name else model.name
                self.elements.append(model)
                added.append(model)
            elif issubclass(type(model), Model):
                self.elements.append(Element(model, name))
                added.append(self.elements[-1])
        return added

    def connect(self, element1, pin1, element2, pin2):
        """Connect two elements with a net.

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
            raise TypeError(
                'element should be string or Element, not "{}"'.format(type(element))
            )

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
                # return getattr(element.pins, pin)
                return element.pins[pin]
            elif issubclass(type(element), Subcircuit):
                for opin in element.pins:
                    if opin.name == pin:
                        return opin
        else:
            err = "expected type 'Pin' or 'str', got {}".format(type(pin))
            raise TypeError(err)

    def connect_many(self, conns):
        """A convenience function for connecting many nets at once.

        Parameters
        ----------
        conns : list of tuple
            A list of tuples, each formed as a tuple of arguments in the same
            order as that accepted by :py:func:`connect`.
        """
        for c in conns:
            self._logger.debug("Handling connection: {}".format(c))
            self.connect(*c)

    def to_spice(self):
        """Perhaps this shouldn't be built into Subcircuit, maybe an adapter
        class or some translator instantiated with a Subcircuit that iterates
        through and creates a netlist."""
        out = ""
        for item in self._blocks.keys():
            out += str(type(self._blocks[item])) + "\n"
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
