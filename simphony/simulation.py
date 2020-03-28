# -*- coding: utf-8 -*-
#
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
simphony.simulation
===================

This package contains the base classes for running simulations.
"""

import copy
import logging
import uuid

import numpy as np
from scipy.constants import c
from scipy.interpolate import interp1d

from simphony.connect import connect_s, innerconnect_s
from simphony.elements import Model
from simphony.netlist import Subcircuit, ElementList, Element, PinList

_module_logger = logging.getLogger(__name__)


def freq2wl(freq):
    """Convenience function for converting from frequency to wavelength.

    Parameters
    ----------
    freq : float
        The frequency in SI units (Hz).

    Returns
    -------
    wl : float
        The wavelength in SI units (m).
    """
    return c/freq

def wl2freq(wl):
    """Convenience function for converting from wavelength to frequency.

    Parameters
    ----------
    wl : float
        The wavelength in SI units (m).

    Returns
    -------
    freq : float
        The frequency in SI units (Hz).
    """
    return c/wl

class SimulationResult:
    """
    A simulated block of a circuit; can represent either elements or entire
    subcircuits.

    It is used by Simulation in order to store s-parameters of recursively
    included subcircuits and elements while cascading all blocks into one final
    component representing the circuit as a whole.

    Attributes
    ----------
    pins : simphony.netlist.PinList
        An ordered tuple of the nodes of the component.
    f : np.array
        A numpy array of the frequency values in its simulation.
    s : np.array
        A numpy array of the s-parameter matrix for the given frequency range.
    """
    _logger = _module_logger.getChild('SimulationResult')

    def __init__(self, freq=None, s=None, pinlist=None):
        """
        Instantiates an object from a Component if provided; empty, if not.

        Parameters
        ----------
        component : Component, optional
            A component to initialize the data members of the object.
        """
        self._pinlist = None

        self.f = freq
        self.s = s
        self.pinlist = pinlist

    @property
    def pinlist(self):
        return self._pinlist

    @pinlist.setter
    def pinlist(self, pinlist):
        self._logger.debug('pinlist property set')
        if pinlist:
            pinlist.element = self
            self._pinlist = pinlist
            assert self.pinlist.element == self

class SweepSimulationResult(SimulationResult):
    def __init__(self, freq, s, pins):
        super().__init__(freq, s, pins)

    def data(self, inp, outp, dB=False):
        """
        Parameters
        ----------
        inp : Pin
            Input pin.
        outp : Pin
            Output pin.
        """
        freq = self.f
        s = abs(self.s[:, inp.index, outp.index])**2
        if dB:
            s = np.log10(s)
        return freq, s


class Simulation:
    """
    Once a simulation is run, it is completely decoupled from the circuit 
    which created it. Its pins, while bearing the same name, are unique
    objects.

    Attributes
    ----------
    circuit : simphony.netlist.Subcircuit
        A simulation is instantiated with a completed circuit.
    """
    def __init__(self, circuit: Subcircuit):
        self.circuit = copy.deepcopy(circuit)


class SweepSimulation(Simulation):
    """
    A swept simulation.

    Attributes
    ----------
    start : float
        The starting simulation frequency (in Hz).
    stop : float
        The ending simulation frequency (in Hz).
    num : int
        The number of sampled points between `start` and `stop`.
    """
    def __init__(self, circuit: Subcircuit, start: float=1.5e-6, stop: float=1.6e-6, num: int=2000, mode='wl'):
        """
        Parameters
        ----------
        circuit : Subcircuit
            The circuit to be simulated.
        start : float
            The start wavelength (in meters) or frequency (in Hz).
        stop : float
            The stop wavelength (in meters) or frequency (in Hz).
        num : int
            The number of sampled points.
        mode : str
            Defines sweep range mode; either 'wl' for wavelength (m) or 
            'freq' for frequency (Hz).
        """
        super().__init__(circuit)
        if start > stop:
            raise ValueError("simulation 'start' value must be less than 'stop' value.")
        if mode == 'wl':
            self.start = wl2freq(stop)
            self.stop = wl2freq(start)
        elif mode == 'freq':
            self.start = start
            self.stop = stop
        else:
            err = "mode '{}' is not one of 'freq' or 'wl'".format(mode)
            raise ValueError(err)
        self.num = num

    def _cache_elements(self):
        # cache = {}
        self.cache = self._cache_elements_helper(self.circuit, {})
        
    def _cache_elements_helper(self, circuit, cache: dict):
        """
        Recursively caches all circuit in the subcircuit.

        Parameters
        ----------
        circuit : list of simphony.netlist.ElementList
            The elements to be cached.
        cache : simphony.simulation.Cache
            A cache for containing simulated models.

        Returns
        -------
        cache : dict
            The updated cache.
        """
        # For every item in the circuit
        for item in circuit.elements:

            # If it's an Element type, cache it.
            if issubclass(type(item), Element):
                self._cache_elements_element_helper(item, cache)
            
            # If it's a subcircuit, recursively call this function.
            elif type(item) is Subcircuit:
                self._cache_elements_helper(item, cache)
            
            # If it's something else--
            # well, ya got trouble, right here in River City.
            else:
                raise TypeError('Invalid object in circuit (type "{}")'.format(type(item)))

        return cache

    def _cache_elements_element_helper(self, element: Element, cache: dict):
        # Caching items base case: if matching object in cache, return.
        model = element.model
        if model in cache:
            return cache
        
        # Ensure that models have required attributes.
        try:
            lower, upper = model.freq_range
        except TypeError:
            raise NotImplementedError('Does the model "{}" define a valid frequency range?'.format(type(model).__name__))
        
        # Ensure that models are valid with current simulation parameters.
        if lower > self.start or upper < self.stop:
            raise ValueError('Simulation frequencies ({} - {}) out of valid bounds for "{}"'.format(self.start, self.stop, type(model).__name__))

        # Cache the element's s-matrix using the simulation parameters
        cache[model] = model.s_parameters(self.start, self.stop, self.num)
        return cache

    def simulate(self):
        self._cache_elements()
        sim = self._simulate_helper(self.circuit)
        sim = SweepSimulationResult(sim.f, sim.s, sim.pinlist)
        return sim

    def _simulate_helper(self, circuit):
        elements = {}
        netlist = circuit.netlist

        # For every item in the circuit
        for item in circuit.elements:

            # If it's an Element type, simulate it.
            if issubclass(type(item), Element):
                elements[item.name] = self._create_simulated_result(item, netlist)
            
            # If it's a subcircuit, recursively call this function.
            elif type(item) is Subcircuit:
                elements[item.name] = self._simulate_helper(item)
            
            # If it's something else--
            # well, ya got trouble, right here in River City.
            else:
                err = 'Invalid object in circuit (type "{}")'.format(type(item))
                raise TypeError(err)

        # Connect all the elements together and return a super element.
        built = self.connect_circuit(elements, netlist) 
        # TODO: Remove assertion
        assert type(built) is SimulationResult
        return built

    def _create_simulated_result(self, element, netlist):
        f, s = self.cache[element.model]
        sim = SimulationResult(f, s, element.pins)
        return sim

    @staticmethod
    def connect_circuit(elements, netlist) -> SimulationResult:
        """
        Connects the s-matrices of a photonic circuit given its Netlist
        and returns a single 'SimulatedComponent' object containing the frequency
        array, the assembled s-matrix, and a list of the external nets (negative 
        integers).

        Parameters
        ----------
        component_list : List[SimulatedComponent]
            A list of the components to be connected.
        net_count : int
            The total number of internal nets in the component list.

        Returns
        -------
        SimulatedComponent
            After the circuit has been fully connected, the result is a single 
            ComponentSimulation with fields f (frequency), s (s-matrix), and nets 
            (external ports: negative numbers, as strings).
        """
        _logger = _module_logger.getChild('SweepSimulation.connect_circuit')

        # FIXME: What if there are no items in the netlist (only one element
        # in the circuit)?
        for net in netlist:
            p1, p2 = net
            if p1.element == p2.element:
                _logger.debug('Internal connection')
                combined = SimulationResult()
                combined.f = p1.element.f
                combined.s = innerconnect_s(p1.element.s, p1.index, p2.index)
                pinlist = p1.pinlist
                pinlist.remove(p1)
                pinlist.remove(p2)
                combined.pinlist = pinlist
            else:
                _logger.debug('External connection')
                combined = SimulationResult()
                combined.f = p1.element.f
                combined.s = connect_s(p1.element.s, p1.index, p2.element.s, p2.index)
                pinlist = p1.pinlist + p2.pinlist
                pinlist.remove(p1)
                pinlist.remove(p2)
                combined.pinlist = pinlist
        return combined


class SinglePortSweepSimulation(SweepSimulation):
    def __init__(self, circuit, start=1.5e-6, stop=1.6e-6, num=2000):
        super().__init__(circuit, start, stop, num)


class MonteCarloSweepSimulation(SweepSimulation):
    pass
#     """A simulator that models manufacturing variability by altering the
#     width, thickness, and length of waveguides instantiated from a 
#     `ebeam_wg_integral_1550` from the default DeviceLibrary.
#     """
#     def __init__(self, netlist: Netlist, start_freq: float=1.88e+14, stop_freq: float=1.99e+14, num: int=2000):
#         """Initializes the MonteCarloSimulation with a Netlist and runs a 
#         single simulation for the "ideal," pre-modified model.

#         Parameters
#         ----------
#         netlist : Netlist
#             The netlist to be simulated.
#         start_freq : float
#             The starting (lower) value for the frequency range to be analyzed.
#         stop_freq : float
#             The ending (upper) value for the frequency range to be analyzed.
#         num : int
#             The number of points to be used between start_freq and stop_freq.
#         """
#         super().__init__(netlist, start_freq=start_freq, stop_freq=stop_freq, num=num)

#     def monte_carlo_sim(self, num_sims: int=10, 
#         mu_width: float=0.5, sigma_width: float=0.005, 
#         mu_thickness: float=0.22, sigma_thickness: float=0.002, 
#         mu_length: float=1.0, sigma_length: float=0):
#         """Runs a Monte Carlo simulation on the netlist and stores the results
#         in an attribute called `results`.

#         Parameters
#         ----------
#         num_sims : int, optional
#             The number of varied simulations to perform.
#         mu_width : float, optional
#             The mean width to use for the waveguide.
#         sigma_width : float, optional
#             The standard deviation to use for altering the waveguide width.
#         mu_thickness : float, optional
#             The mean thickness to use for the waveguide.
#         sigma_thickness : float, optional
#             The standard deviation to use for altering the waveguide thickness.
#         mu_length : float, optional
#             The mean length of the waveguide (as a decimal of the actual 
#             length, i.e. 50% -> 0.5).
#         sigma_length : float, optional
#             The standard deviation to use for altering the waveguide length.

#         Returns
#         -------
#         time : int
#             The amount of time it took, in seconds, to complete the simulation.
#         """
#         start = time.time()

#         # Randomly generate variation in the waveguides.
#         random_width = np.random.normal(mu_width, sigma_width, num_sims)
#         random_thickness = np.random.normal(mu_thickness, sigma_thickness, num_sims)
#         random_deltaLength = np.random.normal(mu_length, sigma_length, num_sims)

#         # Create an array for holding the results
#         results_shape = np.append(np.asarray([num_sims]), self.s_parameters().shape)
#         self.results = np.zeros([dim for dim in results_shape], dtype='complex128')

#         # Run simulations with varied width and thickness
#         for sim in range(num_sims):
#             modified_netlist = copy.deepcopy(self.netlist)
#             for component in modified_netlist.components:
#                 if component.model.component_type == "ann_wg_integral":
#                     component.extras['width'] = random_width[sim]
#                     component.extras['thickness'] = random_thickness[sim]
#                     # TODO: Implement length monte carlo using random_deltaLength[sim]
#             self.results[sim, :, :, :] = Simulation(modified_netlist, self.start_freq, self.stop_freq, self.num).s_parameters()
            
#         stop = time.time()
#         return (stop - start)


class MultiInputSweepSimulation(SweepSimulation):
    pass
#     """A simulator that models sweeping multiple inputs simultaneously by 
#     performing algebraic operations on the simulated, cascaded s-parameter
#     matrix.
#     """
#     def __init__(self, netlist):
#         """Initializes the MultiInputSimulation with a Netlist and runs a 
#         single simulation for the "ideal," pre-modified model.

#         Parameters
#         ----------
#         netlist : Netlist
#             The netlist to be simulated.
#         """
#         super().__init__(netlist)

#     def multi_input_simulation(self, inputs: list=[]):
#         """Given a list of ports to use as inputs, calculates the response
#         of the circuit for all ports. Results are stored as an attribute and
#         can be accessed by retrieving `.simulated_matrix` from the simulation
#         object.

#         Parameters
#         ----------
#         inputs : list
#             A 0-indexed list of the ports to be used as inputs.
#         """
#         active = [0] * len(self.external_ports)
#         for val in inputs:
#             active[val] = 1
#         self.simulated_matrix = self._measure_s_matrix(active)

#     def _measure_s_matrix(self, inputs):
#         """Performs the algebra for simulating multiple inputs.

#         Parameters
#         ----------
#         inputs : list
#             A list with length equal to the number of rows/columns of the 
#             s-parameter matrix (corresponds to the number of external ports). 
#             Port indices with a '0' are considered "off," where ports indices
#             that store a '1' correspond to an active laser input.
#         """
#         num_ports = len(inputs)
#         inputs = np.array(inputs)
#         out = np.zeros([len(self.freq_array), num_ports], dtype='complex128')
#         for i in range(len(self.freq_array)):
#             out[i, :] = np.dot(np.reshape(self.s_parameters()[i, :, :], [num_ports, num_ports]), inputs.T)
#         return out

#     def export_s_matrix(self):
#         """Returns the matrix result of the multi-input simulation.

#         Returns
#         -------
#         frequency, matrix: np.array, np.ndarray
#         """
#         return self.freq_array, self.simulated_matrix


# class MonteCarloSimulation(Simulation):

                

    
# class MultiInputSimulation(Simulation):

