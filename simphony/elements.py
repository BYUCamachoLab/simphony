# -*- coding: utf-8 -*-
#
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
simphony.elements
=================

This package contains the base classes for defining models.
"""

import copy
import logging
import uuid

from scipy.interpolate import interp1d

_module_logger = logging.getLogger(__name__)


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
    pins : tuple of str
        A tuple of all the default pin names of the device.
    freq_range : tuple of float
        A tuple of the valid frequency bounds for the element in the order
        (lower, upper). Can be made (-infty, infty) be setting to (None, None).

    Notes
    -----
    If you extended the element with attributes you don't want included
    in an equality comparison, you can add the name of the attribute (as a 
    string) to the base object's `_ignored_` list and it won't be used.
    """
    _logger = _module_logger.getChild('Model')

    pins = None
    freq_range = (None, None)

    def s_parameters(self, freq):
        """
        Returns scattering parameters for the element with its given 
        parameters.

        Parameters
        ----------
        freq : np.ndarray
            The frequency range to get scattering parameters for.

        Returns
        -------
        s : np.ndarray
            The scattering parameters corresponding to the frequency range.
            Its shape should be:
                (the number of frequency point x ports x ports)
            If the scattering parameters are requested for only a single 
            frequency, for example, and the device has 4 ports, the shape
            returned by `s_parameters` would be (1, 4, 4).
        
        Raises
        ------
        NotImplementedError
            Raised if the subclassing element doesn't implement this function.
        """
        raise NotImplementedError

    def monte_carlo_s_parameters(self, freq):
        """
        Implements the monte carlo routine for the given Model.

        If no monte carlo routine is defined, ideal s-parameters for the given
        frequency range are returned.

        Parameters
        ----------
        freq : np.ndarray
            The frequency range to generate monte carlo s-parameters over.

        Returns
        -------
        s : np.ndarray
            The scattering parameters corresponding to the frequency range.
            Its shape should be:
                (the number of frequency points x ports x ports)
            If the scattering parameters are requested for only a single 
            frequency, for example, and the device has 4 ports, the shape
            returned by `monte_carlo_s_parameters` would be (1, 4, 4).
        """
        return self.s_parameters(freq)
    
    def regenerate_monte_carlo_parameters(self):
        """
        Regenerates parameters used to generate monte carlo s-matrices.

        If a monte carlo method is not implemented for a given model, this
        method does nothing. However, it can optionally be implemented so that
        parameters are regenerated once per circuit simulation. This ensures
        correlation between all components of the same type that reference 
        this model in a circuit. For example, the effective index of a 
        waveguide should not be different for each waveguide in a small 
        circuit; they will be more or less consistent within a single small
        circuit.

        The MonteCarloSweepSimulation calls this function once per run over
        the circuit.

        Notes
        -----
        This function should not accept any parameters, but may act on instance
        or class attributes.
        """
        pass

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
