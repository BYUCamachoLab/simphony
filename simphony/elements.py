# -*- coding: utf-8 -*-
#
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

from collections import OrderedDict
import copy
import logging
import uuid

from scipy.interpolate import interp1d

# FIXME: Is interpolating in frequency better than in wavelength?

_module_logger = logging.getLogger(__name__)


def rename_keys(d, keys):
    return OrderedDict([(keys.get(k, k), v) for k, v in d.items()])


class Pin:
    _logger = _module_logger.getChild('Pin')
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        o = ".".join([self.__module__, type(self).__name__])
        return "<'{}': {} object at {}>".format(self.name, o, hex(id(self)))


class PinList:
    """
    Notes
    -----
    If renaming pins, the assigned value must be a string.

    Examples
    --------
    pins = PinList('n1', 'n2', 'n3')
    pins.n2 = 'out1'
    """
    _logger = _module_logger.getChild('PinList')
    pins = OrderedDict()

    def __init__(self, *args):
        self.pins = OrderedDict({arg: Pin(arg) for arg in args})

    def __getitem__(self, item):
        return list(self.pins.values())[item]

    def __setitem__(self, key, value):
        if type(key) is str:
            idx = self.pins.index(key)
            self.pins[idx] = value
        elif type(key) is int:
            self.pins[key] = value
        else:
            raise TypeError

    def __getattr__(self, name):
        for key, pin in self.pins.items():
            if name == pin.name:
                return pin
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name not in self.pins.keys():
            return super().__setattr__(name, value)
        if value in self.pins.keys():
            raise AttributeError("'{}' already exists in PinList".format(value))
        self.pins[name].name = value
        self.pins = rename_keys(self.pins, {name: value})

    def __str__(self):
        val = ''
        o = ".".join([self.__module__, type(self).__name__])
        val += "<{} object at {}>".format(o, hex(id(self)))
        for idx, pin in enumerate(self.pins.values()):
            val += "\n - {}: {}".format(idx, pin)
        return val

    def __len__(self):
        return len(self.pins)
        

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
    ignored = ['ignored']

    _ignored_ = ['name', 'pins', 'wl_bounds']

    def __init__(self, name: str=None):
        if name:
            self.name = name
        else:
            self.name = self.__class__.__name__ + "_" + str(uuid.uuid4())[:8]

        self.pins = copy.deepcopy(self.pins)
        self._ignored_ += self.ignored

    # def __getattr__(self, name):
    #     self._logger.debug("Getting attribute '{}'".format(name))
    #     return super().__getattribute__(name)

    def __eq__(self, other: 'Element'):
        # TODO: What if the two instances have different class variable values?
        # As in, it was instantiated with the default but one of them was later
        # changed. These are variables that subclasses implement but that are
        # not required by the parent Element class.
        # parent_attr = dir(Element)
        # child_attr = dir(self)
        # if type(self) is type(other):
        #     not_ignored = [attr for attr in dir(self) if attr not in dir(Element)]
        #     sdict = {k : getattr(self, k) for k in not_ignored}
        #     odict = {k : getattr(other, k) for k in not_ignored}
        #     return sdict == odict
        # else:
        #     return False
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
        raise NotImplementedError

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
