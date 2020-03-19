# -*- coding: utf-8 -*-
#
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import copy
import logging
import uuid

from scipy.interpolate import interp1d

# FIXME: Is interpolating in frequency better than in wavelength?

_module_logger = logging.getLogger(__name__)


# def rename_keys(d, keys):
#     return OrderedDict([(keys.get(k, k), v) for k, v in d.items()])


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
        return "<'{}' {} object at {}>".format(self.name, o, hex(id(self)))

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        # TODO: Make sure that there is no pin with the same name in _pinlist
        self._name = value


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
    pins = []

    def __init__(self, element, *args):
        """
        Parameters
        ----------
        element : simphony.elements.Element
        args : tuple of str
        """
        self.element = element
        self.pins = [Pin(self, arg) for arg in args]

    def __getitem__(self, item):
        return self.pins[item]

    def __setitem__(self, key, value):
        self.pins[key].name = value

    def __getattr__(self, name):
        for pin in self.pins:
            if name == pin.name:
                return pin
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name not in [pin.name for pin in self.pins]:
            return super().__setattr__(name, value)
        if value in [pin.name for pin in self.pins]:
            raise AttributeError("'{}' already exists in PinList".format(value))
        for pin in self.pins:
            if name == pin.name:
                pin.name = value
                return

    def __repr__(self):
        val = ''
        o = ".".join([self.__module__, type(self).__name__])
        val += "<{} object at {}>".format(o, hex(id(self)))
        for idx, pin in enumerate(self.pins):
            val += "\n  {}".format(pin)
        return val

    def __len__(self):
        return len(self.pins)

    def index(self, item):
        """
        Returns
        -------
        idx : int
            The index of the pin passed in.
        """
        return self.pins.index(item)
        

class Element:
    """
    The basic element type describing the model for a component with scattering
    parameters.

    Any class that inherits from Element or its subclasses must declare the
    attributes of an element, see Attributes. Following the general EAFP coding
    style of Python, errors will only be raised when an unimplemented function
    is called, not when the class instance is created.

    Attributes
    ----------
    nodes : tuple of str
        A tuple of all the node names for the element.
    wl_bounds : tuple of float
        A tuple of the valid wavelength bounds for the element in the order
        (lower, upper).

    Notes
    -----
    If you extended the element with attributes you don't want included
    in an equality comparison, you can add the name of the attribute (as a 
    string) to the base object's `_ignored_` list and it won't be used.
    """
    _logger = _module_logger.getChild('Element')

    pins = None
    wl_bounds = (None, None)
    ignore = None

    _ignored_ = ['name', 'pins', 'wl_bounds']

    def __init__(self, name=None):
        """
        Parameters
        ----------
        name : str
        """
        self.name = name if name else self.__class__.__name__ + "_" + str(uuid.uuid4())[:8]

        if self.pins:
            self.pins = PinList(self, *self.pins)
        if self.ignore:
            self._ignored_ += self.ignore

    def __eq__(self, other):
        if type(self) is type(other):
            not_ignored = set([attr for attr in self.__dict__ if attr not in self._ignored_])
            not_ignored = set([attr for attr in other.__dict__ if attr not in other._ignored_])
            sdict = {k : v for k, v in self.__dict__.items() if k not in self._ignored_}
            odict = {k : v for k, v in other.__dict__.items() if k not in other._ignored_}
            return sdict == odict
        else:
            return False

    def __ne__(self, other: 'Element'):
        return not self.__eq__(other)

    def __hash__(self):
        return super().__hash__()

    def _monte_carlo_(self, *args, **kwargs):
        """Implements the monte carlo routine for the given Element.

        Raises
        ------
        NotImplementedError
            If the function hasn't been implemented by a child class.
        """
        raise NotImplementedError

    def rename_pins(self, *pins):
        for i, pin in enumerate(pins):
            self.pins[i].name = pin

    def s_parameters(self, start: float, end: float, num: int):
        """
        Returns scattering parameters for the element with its given 
        parameters.

        Parameters
        ----------
        start : float
            The lower wavelength bound for the simulation.
        end : float
            The upper wavelength bound for the simulation.
        num : int
            The number of points to interpolate between `start` and `end`.

        Returns
        -------
        f, s : float, array
            The frequency range and corresponding scattering parameters.
        
        Raises
        ------
        NotImplementedError
            Raised if the subclassing element doesn't implement this function.
        """
        raise NotImplementedError

    @staticmethod
    def interpolate(resampled, sampled, s_parameters):
        """Returns the result of a cubic interpolation for a given frequency range.

        Parameters
        ----------
        output_freq : np.array
            The desired frequency range for a given input to be interpolated to.
        input_freq : np.array
            A frequency array, indexed matching the given s_parameters.
        s_parameters : np.array
            S-parameters for each frequency given in input_freq.

        Returns
        -------
        result : np.array
            The values of the interpolated function (fitted to the input 
            s-parameters) evaluated at the `output_freq` frequencies.
        """
        func = interp1d(sampled, s_parameters, kind='cubic', axis=0)
        return func(resampled)

# class PElement(Element):
#     """
#     Parameterized Elements have scattering parameters that are calculated on
#     the fly, usually based on instance attributes.
#     """
#     pass

# class SElement(Element):
#     """
#     Static Elements have pre-simulated scattering parameters, often loaded 
#     from a file and independent of instance attributes.
#     """
#     pass
