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

# import copy
# import time
# from datetime import timedelta
# from typing import List, Callable
import logging
import uuid

import numpy as np
from scipy.constants import c
from scipy.interpolate import interp1d
from simphony.connect import connect_s, innerconnect_s

from simphony.elements import Element
from simphony.netlist import Subcircuit

_module_logger = logging.getLogger(__name__)


def freq2wl(freq):
    """
    Parameters
    ----------
    freq : float
        The frequency in SI units.

    Returns
    -------
    wl : float
        The wavelength in SI units.
    """
    return c/freq

def wl2freq(wl):
    """
    Parameters
    ----------
    wl : float
        The wavelength in SI units.

    Returns
    -------
    freq : float
        The frequency in SI units.
    """
    return c/wl

class SimulatedBlock:
    """
    A simulated block of a circuit; can represent either elements or entire
    subcircuits.

    It is used by Simulation in order to store s-parameters of recursively
    included subcircuits and elements while cascading all blocks into one final
    component representing the circuit as a whole.

    Attributes
    ----------
    nodes : tuple of str
        An ordered tuple of the nodes of the component.
    f : np.array
        A numpy array of the frequency values in its simulation.
    s : np.array
        A numpy array of the s-parameter matrix for the given frequency range.
    """
    def __init__(self, freq=None, s=None, nodes=None):
        """
        Instantiates an object from a Component if provided; empty, if not.

        Parameters
        ----------
        component : Component, optional
            A component to initialize the data members of the object.
        """
        self.f = freq
        self.s = s
        self.nodes = nodes

class Cache:
    """
    A cache for simulated scattering parameters and recording which elements
    they correspond to.
    """
    _logger = _module_logger.getChild('Cache')

    def __init__(self):
        # _elements_id_mapping is a mapping of Element instances to string id's
        # implemented using a dict. Every element that's been cached is 
        # contained within _elements_id_mapping, regardless of whether an 
        # identical object is included.
        
        # _elements_id_mapping = {element: uid}
        self._elements_id_mapping = {}
        # _id_cache_mapping = {uid: parameters}
        self._id_cache_mapping = {}

    def __setitem__(self, key, value):
        self._logger.debug('Entering __setitem__')
        # Check to see if an identical item is already in the cache
        for k, v in self._elements_id_mapping.items():
            # If it is, then set the new item's uid to the existing item's uid
            if key == k:
                self._elements_id_mapping[key] = v
                return
        
        # Otherwise, the element has not yet been saved to the cache.
        # Create a unique id for the parameters and save which id the element
        # should use to access the cached value.
        uid = uuid.uuid4()
        self._id_cache_mapping[uid] = value
        self._elements_id_mapping[key] = uid

    def __getitem__(self, key):
        self._logger.debug('Entering __getitem__')
        if key in self._elements_id_mapping.keys():
            return self._id_cache_mapping[self._elements_id_mapping[key]]
        else:
            raise KeyError
        # return self.__dict__[key]

    def contains(self, element) -> bool:
        for key in self._elements_id_mapping.keys():
            if element is key:
                return True
            elif element == key:
                self._elements_id_mapping[element] = self._elements_id_mapping[key]
                return True
            else:
                return False

    def keys(self):
        return self._elements_id_mapping.keys()


class Simulation:
    """
    Attributes
    ----------
    circuit : simphony.netlist.subcircuit
        A simulation is instantiated with a completed circuit.
    """
    def __init__(self, circuit: Subcircuit):
        self.circuit = circuit


class SweepSimulation(Simulation):
    """
    Attributes
    ----------
    start : float
    stop : float
    num : float
    cache : dict
    """
    def __init__(self, circuit: Subcircuit, start: float=1.5e-6, stop: float=1.6e-6, num: int=2000):
        super().__init__(circuit)
        self.start = start
        self.stop = stop
        self.num = num

    def _cache_elements(self):
        cache = Cache()
        self.cache = self._cache_elements_helper(self.circuit, cache)
        
    def _cache_elements_helper(self, blocks: Subcircuit, cache: Cache):
        """
        Recursively caches all blocks in the subcircuit.

        Parameters
        ----------
        blocks : list
        cache : dict

        Returns
        -------
        cache : dict
            The updated cache.
        """
        # For every item in the circuit
        for block in blocks:

            # If it's an Element type, cache it.
            if issubclass(type(block), Element):
                self._cache_elements_element_helper(block, cache)
            
            # If it's a subcircuit, recursively call this function.
            elif type(block) is Subcircuit:
                self._cache_elements_helper(block, cache)
            
            # If it's something else--
            # well, ya got trouble, right here in River City.
            else:
                raise TypeError('Invalid object in circuit (type "{}")'.format(type(block)))

        return cache

    def _cache_elements_element_helper(self, element: Element, cache: Cache):
        # Caching items base case: if matching object in cache, return.
        if cache.contains(element):
            return cache
        
        # Ensure that models have required attributes.
        try:
            lower, upper = element.wl_bounds
        except TypeError:
            raise NotImplementedError('Does the model "{}" define a valid frequency range?'.format(type(element).__name__))
        
        # Ensure that models are valid with current simulation parameters.
        if lower < self.start or upper > self.stop:
            raise ValueError('Simulation frequencies ({}-{}) out of valid bounds for "{}"'.format(self.start, self.stop, type(element).__name__))

        # Cache the element's s-matrix using the simulation parameters
        cache[element] = element.s_parameters(self.start, self.stop, self.num)
        return cache

    def simulate(self):
        self._cache_elements()
        self._simulate_helper(self.circuit)

    def _simulate_helper(self, blocks: Subcircuit):
        elements = []
        # For every item in the circuit
        for block in blocks:

            # If it's an Element type, cache it.
            if issubclass(type(block), Element):
                f, s = self.cache[block]
                sim = SimulatedBlock(f, s, block.nodes)
                elements.append(sim)
            
            # If it's a subcircuit, recursively call this function.
            elif type(block) is Subcircuit:
                elements.append(self._simulate_helper(block))
            
            # If it's something else--
            # well, ya got trouble, right here in River City.
            else:
                raise TypeError('Invalid object in circuit (type "{}")'.format(type(block)))

        # Connect all the elements together and return a super element.
        self.connect_circuit(elements, blocks.nets)
        print(elements)

    @staticmethod
    def connect_circuit(components, nets) -> SimulatedBlock:
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
        # component_list = copy.deepcopy(components)
        for n in range(len(nets)):
            logging.debug("Entering pass {} of {}".format(n, len(nets)))
            # ca, ia, cb, ib = match_ports(n, component_list)
            for net in nets.values():
                e1, n1, e2, n2 = net
                
                print(net)

        #         # If pin occurances are in the same component:
        #         if e1 == e2:
        #             component_list[ca].s = innerconnect_s(component_list[ca].s, ia, ib)
        #             del component_list[ca].nets[ia]
        #             if ia < ib:
        #                 del component_list[ca].nets[ib-1]
        #             else:
        #                 del component_list[ca].nets[ib]

        #         # If pin occurances are in different components:
        #         else:
        #             combination = SimulatedBlock()
        #             combination.f = component_list[0].f
        #             combination.s = connect_s(component_list[ca].s, ia, component_list[cb].s, ib)
        #             del component_list[ca].nets[ia]
        #             del component_list[cb].nets[ib]
        #             combination.nets = component_list[ca].nets + component_list[cb].nets
        #             del component_list[ca]
        #             if ca < cb:
        #                 del component_list[cb-1]
        #             else:
        #                 del component_list[cb]
        #             component_list.append(combination)

        # assert len(component_list) == 1
        # return component_list[0]

    # def match_ports(net_id, component_list) -> list:
    #     """
    #     Finds the components connected together by the specified net_id (string) in
    #     a list of components provided by the caller (even if the component is 
    #     connected to itself).

    #     Parameters
    #     ----------
    #     net_id : int
    #         The net id or name to which the components being searched for are 
    #         connected.
    #     component_list : list[SimulatedComponent]
    #         The complete list of components to be searched.

    #     Returns
    #     -------
    #     [comp1, netidx1, comp2, netidx2]
    #         A list (length 4) of integers with the following meanings:
    #         - comp1: Index of the first component in the list with a matching 
    #             net id.
    #         - netidx1: Index of the net in the ordered net list of 'comp1' 
    #             (corresponds to its column or row in the s-parameter matrix).
    #         - comp2: Index of the second component in the list with a matching 
    #             net id.
    #         - netidx1: Index of the net in the ordered net list of 'comp2' 
    #             (corresponds to its column or row in the s-parameter matrix).
    #     """
    #     logging.debug("Entering match_ports(), searching for net: " + str(net_id))
    #     filtered_comps = [component for component in component_list if net_id in component.nets]
    #     comp_idx = [component_list.index(component) for component in filtered_comps]
    #     net_idx = []
    #     for comp in filtered_comps:
    #         net_idx += [i for i, x in enumerate(comp.nets) if x == net_id]
    #     if len(comp_idx) == 1:
    #         comp_idx += comp_idx
        
    #     return [comp_idx[0], net_idx[0], comp_idx[1], net_idx[1]]


class SinglePortSweepSimulation(SweepSimulation):
    def __init__(self, circuit, start=1.5e-6, stop=1.6e-6, num=2000):
        super().__init__(circuit, start, stop, num)



#
#
# SIMULATED COMPONENTS
#
# Classes and functions for storing the results of each component that is 
# simulated.

# #
# #
# # BASE SIMULATION STRUCTURE
# #
# # The main class and functions pertaining to operations that any type of 
# # simulation will invariably need.
# #
# #


# class Simulation:
#     """The main simulation class providing methods for running simulations,
#     tracking frequency ranges, and matrix rearranging (according to port
#     numbering).

#     All simulators can inherit from Simulation to get basic functionality that
#     doesn't need reimplementing.
#     """
#     def __init__(self, netlist: Netlist, start_freq: float=1.88e+14, stop_freq: float=1.99e+14, num: int=2000):
#         """Creates and automatically runs a simulation by cascading the netlist.

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
#         self.netlist: Netlist = netlist
#         self.start_freq: float = start_freq
#         self.stop_freq: float = stop_freq
#         self.num: int = num
#         self.cache: dict = {}

#         logging.info("Starting simulation...")
#         start = time.time()
#         self._cache_models()
#         self._cascade()
#         stop = time.time()
#         logging.info("Simulation complete.")
#         logging.info("Total simulation time: " + str(timedelta(seconds=(stop - start))))

#     @property
#     def freq_array(self):
#         """Returns the linearly spaced frequency array from start_freq to stop_freq with num points."""
#         logging.debug("Entering freq_array()")
#         return np.linspace(self.start_freq, self.stop_freq, self.num)

#     def _cache_models(self):
#         """Caches all models marked as `cachable=True` to avoid constant I/O 
#         operations and redundant calculations"""
#         logging.debug("Entering _cache_models()")
#         for component in self.netlist.components:
#             if component.model.component_type not in self.cache and component.model.cachable:
#                 freq, s_parameters = interpolate(self.freq_array, *component.get_s_parameters())
#                 self.cache[component.model.component_type] = (freq, s_parameters)

#     def _cascade(self):
#         """Cascades all components together into a single SimulatedComponent.

#         This is essentially the function that performs the full simulation.
#         """
#         logging.debug("Entering _cascade()")
#         extras = {}
#         extras['start_freq'] = self.start_freq
#         extras['stop_freq'] = self.stop_freq
#         extras['num'] = self.num
#         component_list = [component2simulated(component, self.cache, extras) for component in self.netlist.components]
#         self.combined = rearrange(connect_circuit(component_list, self.netlist.net_count))

#     def s_parameters(self):
#         """Returns the s-parameters of the cascaded, simulated netlist.

#         Returns
#         -------
#         np.ndarray
#             The simulated s-matrix.

#         # TODO: Implement slicing for the s_parameters matrix.
#         def __getitem__(self, name):
#             print(name, type(name), len(name))
#             # return self._parameters[name]
#         """
#         return self.combined.s

#     @property
#     def external_ports(self):
#         """Returns a list of the external port numbers.

#         Returns
#         -------
#         List[int]
#             The external nets of the simulated netlist. These are positive
#             integers, corresponding to rows/columns of the netlist.
#         """
#         return self.combined.nets

#     # @property
#     # def external_components(self):
#     #     # return [component for component in self.netlist.components if (any(int(x) < 0 for x in component.nets))]
#     #     externals = []
#     #     for component in self.netlist.components:
#     #         if (any(int(x) < 0 for x in component.nets)):
#     #             externals.append(component)
#     #     return externals

# def rearrange_order(ports: List[int]):
#     """Determines what order the ports should be placed in after simulation.
    
#     Ports are usually passed in as a scrambled list of negative integers.
#     This function returns a list containing indices corresponding to the
#     order in which the input list should be reordered to be sorted.

#     Parameters
#     ----------
#     ports : List[int]
#         A list of ports to be sorted.

#     Returns
#     -------
#     List[int]
#         Indices corresponding to how ports should be reordered.

#     Examples
#     --------
#     >>> list_a = [-3 -5 -2 -1 -4]
#     >>> list_b = rearrange_order(list_a)
#     >>> list_b
#     [3, 2, 0, 4, 1]
#     """
#     logging.debug("Entering rearrange_order()")
#     reordered = copy.deepcopy(ports)
#     reordered.sort(reverse = True)
#     logging.debug("Order: " + str(reordered))
#     return [ports.index(i) for i in reordered]

# def rearrange(component: SimulatedComponent) -> SimulatedComponent:
#     """Rearranges the s-matrix of the simulated component according to its 
#     port ordering.

#     Parameters
#     ----------
#     component : SimulatedComponent
#         A component that has external ports and a calculated s-parameter
#         matrix.

#     Returns
#     -------
#     SimulatedComponent
#         A single component with its reordered s-matrix and new port list.
#     """
#     concatenate_order = rearrange_order(component.nets)
#     s_params = rearrange_matrix(component.s, concatenate_order)
#     reordered_nets = [(-x - 1) for x in component.nets]
#     reordered_nets.sort()
#     return SimulatedComponent(nets=reordered_nets, freq=component.f, s_parameters=s_params)

# def rearrange_matrix(s_matrix, concatenate_order: List[int]) -> np.ndarray:
#     """Reorders a matrix given a list mapping indices to columns.

#     S-matrices are indexed in the following manner:
#     matrix[frequency, input, output].

#     Parameters
#     ----------
#     s_matrix : np.ndarray
#         The s-matrix to be rearranged.
#     concatenate_order : List[int]
#         The index-to-column mapping. See `rearrange`.

#     Returns
#     -------
#     np.ndarray
#         The reordered s-matrix.
#     """
#     port_count = len(concatenate_order)
#     reordered_s = np.zeros(s_matrix.shape, dtype=complex)
#     for i in range(port_count):
#         for j in range(port_count):
#             x = concatenate_order[i]
#             y = concatenate_order[j]
#             reordered_s[:, i, j] = s_matrix[:, x, y]
#     return reordered_s







# #
# #
# # OTHER SIMULATORS
# #
# # Other simulators, as they are coded, can be found in this section. They 
# # mostly inherit from the default simulator.
# #
# #


# class MonteCarloSimulation(Simulation):
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
                

    
# class MultiInputSimulation(Simulation):
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
