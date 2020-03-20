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


class Model:
    """
    The basic element type describing the model for a component with scattering
    parameters.

    Any class that inherits from Model or its subclasses must declare the
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
    _logger = _module_logger.getChild('Model')

    pins = None
    wl_range = (None, None)

    def _monte_carlo_(self, *args, **kwargs):
        """Implements the monte carlo routine for the given Element.

        Raises
        ------
        NotImplementedError
            If the function hasn't been implemented by a child class.
        """
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

class PModel(Model):
    """
    Parameterized Elements have scattering parameters that are calculated on
    the fly, usually based on instance attributes.
    """
    pass

class SModel(Model):
    """
    Static Elements have pre-simulated scattering parameters, often loaded 
    from a file and independent of instance attributes.
    """
    pass

class EModel(Model):
    """
    Extended models are those that are made up of other models. For instance,
    a subcircuit can be converted into a model and its monte carlo methods
    are preserved.
    """
    pass
