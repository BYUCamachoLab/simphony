
import copy
import time
from typing import List, Callable
import logging

import numpy as np
from scipy.interpolate import interp1d
from simphony.core import ComponentInstance, ComponentModel, Netlist
from simphony.core.connect import connect_s, innerconnect_s


def interpolate(output_freq, input_freq, s_parameters):
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
    output_freq : np.array
        The output frequency range that was passed in as a parameter.
    result : np.array
        The values of the interpolated function (fitted to the input 
        s-parameters) evaluated at the `output_freq` frequencies.
    """
    func = interp1d(input_freq, s_parameters, kind='cubic', axis=0)
    return [output_freq, func(output_freq)]

class SimulatedComponent:
    """
    This class is a simplified version of a Component in that it only contains
    an ordered list of nets, the frequency array, and the s-parameter matrix. 
    It can be initialized with or without a Component model, allowing its 
    attributes to be set after object creation.

    It is used by Simulation in order to store cached s-parameters of
    various objects and also to cascade all components into one final 
    component representing the circuit as a whole.

    Attributes
    ----------
    nets : List[int]
        An ordered list of the nets connected to the Component
    f : np.array
        A numpy array of the frequency values in its simulation.
    s : np.array
        A numpy array of the s-parameter matrix for the given frequency range.
    """
    nets: list
    f: np.array
    s: np.array

    def __init__(self, nets=[], freq=None, s_parameters=None):
        """
        Instantiates an object from a Component if provided; empty, if not.

        Parameters
        ----------
        component : Component, optional
            A component to initialize the data members of the object.
        """
        self.nets = nets
        self.f = freq
        self.s = s_parameters

class Simulation:
    """The main simulation class providing methods for running simulations,
    tracking frequency ranges, and matrix rearranging (according to port
    numbering).

    All simulators can inherit from Simulation to get basic functionality that
    doesn't need reimplementing.
    """
    def __init__(self, netlist: Netlist, start_freq: float=1.88e+14, stop_freq: float=1.99e+14, num: int=2000):
        """Creates and automatically runs a simulation by cascading the netlist.

        Parameters
        ----------
        netlist : Netlist
            The netlist to be simulated.
        start_freq : float
            The starting (lower) value for the frequency range to be analyzed.
        stop_freq : float
            The ending (upper) value for the frequency range to be analyzed.
        num : int
            The number of points to be used between start_freq and stop_freq.
        """
        self.netlist = netlist
        self.start_freq = start_freq
        self.stop_freq = stop_freq
        self.num = num
        self._cascade()

    @property
    def freq_array(self):
        """Returns the linearly spaced frequency array from start_freq to stop_freq with num points."""
        logging.debug("Entering freq_array()")
        return np.linspace(self.start_freq, self.stop_freq, self.num)

    def _cache_models(self):
        """Caches all models marked as `cachable` to avoid constant I/O 
        operations and in accordance with the DRY principle."""
        logging.debug("Entering _cache_models()")
        self._cached = {}
        for component in self.netlist.components:
            if component.model.component_type not in self._cached and component.model.cachable:
                freq, s_parameters = interpolate(self.freq_array, *component.get_s_parameters())
                self._cached[component.model.component_type] = (freq, s_parameters)

    def _component_converter(self, component: ComponentInstance) -> SimulatedComponent:
        """Converts a component into the simplified SimulatedComponent model.

        This disregards what model a component is or any attributes it has
        in favor of tracking just its nets, frequency range, and s-parameters.
        S-parameters should already be the final, interpolated, calculated 
        values.

        Parameters
        ----------
        component : ComponentInstance, optional
            The component to instantiate a SimulatedComponent from.
        """
        logging.debug("Entering _component_converter()")
        if component.model.component_type in self._cached:
            return SimulatedComponent(component.nets, *self._cached[component.model.component_type])
        else:
            component.extras['start_freq'] = self.start_freq
            component.extras['stop_freq'] = self.stop_freq
            component.extras['num'] = self.num
            return SimulatedComponent(component.nets, *component.get_s_parameters())
        pass

    def _cascade(self):
        """Cascades all components together into a single SimulatedComponent.

        This is essentially the function that performs the full simulation.
        """
        logging.debug("Entering _cascade()")
        self._cache_models()
        component_list = [self._component_converter(component) for component in self.netlist.components]
        self.combined = self._rearrange(connect_circuit(component_list, self.netlist.net_count))

    @staticmethod
    def _rearrange_order(ports: List[int]):
        """Determines what order the ports should be placed in after simulation.
        
        Ports are usually passed in as a scrambled list of negative integers.
        This function returns a list containing indices corresponding to the
        order in which the input list should be reordered to be sorted.

        Parameters
        ----------
        ports : List[int]
            A list of ports to be sorted.

        Returns
        -------
        List[int]
            Indices corresponding to how ports should be reordered.

        Examples
        --------
        >>> list_a = [-3 -5 -2 -1 -4]
        >>> list_b = _rearrange_order(list_a)
        >>> list_b
        [3, 2, 0, 4, 1]
        """
        logging.debug("Entering _rearrange_order()")
        reordered = copy.deepcopy(ports)
        reordered.sort(reverse = True)
        logging.debug("Order: " + str(reordered))
        return [ports.index(i) for i in reordered]

    @classmethod
    def _rearrange(cls, component: SimulatedComponent) -> SimulatedComponent:
        """Rearranges the s-matrix of the simulated component according to its 
        port ordering.

        Parameters
        ----------
        component : SimulatedComponent
            A component that has external ports and a calculated s-parameter
            matrix.

        Returns
        -------
        SimulatedComponent
            A single component with its reordered s-matrix and new port list.
        """
        logging.debug("Entering _rearrange()")
        concatenate_order = cls._rearrange_order(component.nets)
        logging.debug(component.s)
        s_params = cls._rearrange_matrix(component.s, concatenate_order)
        logging.debug(s_params)
        reordered_nets = [(-x - 1) for x in component.nets]
        reordered_nets.sort()
        return SimulatedComponent(nets=reordered_nets, freq=component.f, s_parameters=s_params)

    @staticmethod
    def _rearrange_matrix(s_matrix, concatenate_order: List[int]) -> np.ndarray:
        """Reorders a matrix given a list mapping indices to columns.

        S-matrices are indexed in the following manner:
        matrix[frequency, input, output].

        Parameters
        ----------
        s_matrix : np.ndarray
            The s-matrix to be rearranged.
        concatenate_order : List[int]
            The index-to-column mapping. See `_rearrange`.

        Returns
        -------
        np.ndarray
            The reordered s-matrix.
        """
        logging.debug("Entering _rearrange_matrix()")
        port_count = len(concatenate_order)
        reordered_s = np.zeros(s_matrix.shape, dtype=complex)
        for i in range(port_count):
            for j in range(port_count):
                x = concatenate_order[i]
                y = concatenate_order[j]
                logging.debug("(" + str(i) + ", " + str(j) + ") takes on the value of (" + str(x) + ", " + str(y) + ")")
                reordered_s[:, i, j] = s_matrix[:, x, y]
        return reordered_s

    def s_parameters(self):
        """Returns the s-parameters of the cascaded, simulated netlist.

        Returns
        -------
        np.ndarray
            The simulated s-matrix.
        """
        return self.combined.s

    def external_ports(self):
        """Returns a list of the external port numbers.

        Returns
        -------
        List[int]
            The external nets of the simulated netlist. These are positive
            integers, corresponding to rows/columns of the netlist.
        """
        return self.combined.nets

    def external_components(self):
        # return [component for component in self.netlist.components if (any(int(x) < 0 for x in component.nets))]
        externals = []
        for component in self.netlist.components:
            if (any(int(x) < 0 for x in component.nets)):
                externals.append(component)
        return externals


def match_ports(net_id: int, component_list: List[SimulatedComponent]) -> list:
    """
    Finds the components connected together by the specified net_id (string) in
    a list of components provided by the caller (even if the component is 
    connected to itself).

    Parameters
    ----------
    net_id : int
        The net id or name to which the components being searched for are 
        connected.
    component_list : list[SimulatedComponent]
        The complete list of components to be searched.

    Returns
    -------
    [comp1, netidx1, comp2, netidx2]
        A list (length 4) of integers with the following meanings:
        - comp1: Index of the first component in the list with a matching 
            net id.
        - netidx1: Index of the net in the ordered net list of 'comp1' 
            (corresponds to its column or row in the s-parameter matrix).
        - comp2: Index of the second component in the list with a matching 
            net id.
        - netidx1: Index of the net in the ordered net list of 'comp2' 
            (corresponds to its column or row in the s-parameter matrix).
    """
    logging.debug("Entering match_ports(), searching for net: " + str(net_id))
    filtered_comps = [component for component in component_list if net_id in component.nets]
    comp_idx = [component_list.index(component) for component in filtered_comps]
    net_idx = []
    for comp in filtered_comps:
        net_idx += [i for i, x in enumerate(comp.nets) if x == net_id]
    if len(comp_idx) == 1:
        comp_idx += comp_idx
    
    return [comp_idx[0], net_idx[0], comp_idx[1], net_idx[1]]

def connect_circuit(components: List[SimulatedComponent], net_count: int) -> SimulatedComponent:
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
    component_list = copy.deepcopy(components)
    for n in range(0, net_count):
        ca, ia, cb, ib = match_ports(n, component_list)

        #if pin occurances are in the same Cell
        if ca == cb:
            component_list[ca].s = innerconnect_s(component_list[ca].s, ia, ib)
            del component_list[ca].nets[ia]
            if ia < ib:
                del component_list[ca].nets[ib-1]
            else:
                del component_list[ca].nets[ib]

        #if pin occurances are in different Cells
        else:
            combination = SimulatedComponent()
            combination.f = component_list[0].f
            combination.s = connect_s(component_list[ca].s, ia, component_list[cb].s, ib)
            del component_list[ca].nets[ia]
            del component_list[cb].nets[ib]
            combination.nets = component_list[ca].nets + component_list[cb].nets
            del component_list[ca]
            if ca < cb:
                del component_list[cb-1]
            else:
                del component_list[cb]
            component_list.append(combination)

    assert len(component_list) == 1
    return component_list[0]


class MathPrefixes:
    TERA = 1e12
    NANO = 1e-9
    c = 299792458


class MonteCarloSimulation(Simulation):
    """A simulator that models manufacturing variability by altering the
    width, thickness, and length of waveguides instantiated from a 
    `ebeam_wg_integral_1550` from the default DeviceLibrary.
    """
    def __init__(self, netlist: Netlist, start_freq: float=1.88e+14, stop_freq: float=1.99e+14, num: int=2000):
        """Initializes the MonteCarloSimulation with a Netlist and runs a 
        single simulation for the "ideal," pre-modified model.

        Parameters
        ----------
        netlist : Netlist
            The netlist to be simulated.
        start_freq : float
            The starting (lower) value for the frequency range to be analyzed.
        stop_freq : float
            The ending (upper) value for the frequency range to be analyzed.
        num : int
            The number of points to be used between start_freq and stop_freq.
        """
        super().__init__(netlist, start_freq=start_freq, stop_freq=stop_freq, num=num)

    def monte_carlo_sim(self, num_sims: int=10, 
        mu_width: float=0.5, sigma_width: float=0.005, 
        mu_thickness: float=0.22, sigma_thickness: float=0.002, 
        mu_length: float=1.0, sigma_length: float=0, 
        timed=False, printer=None):
        """Runs a Monte Carlo simulation on the netlist and stores the results
        in an attribute called `results`.

        Parameters
        ----------
        num_sims : int, optional
            The number of varied simulations to perform.
        mu_width : float, optional
            The mean width to use for the waveguide.
        sigma_width : float, optional
            The standard deviation to use for altering the waveguide width.
        mu_thickness : float, optional
            The mean thickness to use for the waveguide.
        sigma_thickness : float, optional
            The standard deviation to use for altering the waveguide thickness.
        mu_length : float, optional
            The mean length of the waveguide (as a decimal of the actual 
            length, i.e. 50% -> 0.5).
        sigma_length : float, optional
            The standard deviation to use for altering the waveguide length.
        timed : bool, optional
            True if the simulation should be timed; False otherwise.
        printer : Callable, optional
            A function for returning messages to a user or program, such
            as the simulation time.
        """
        if timed:
            start = time.time()

        # Randomly generate variation in the waveguides.
        random_width = np.random.normal(mu_width, sigma_width, num_sims)
        random_thickness = np.random.normal(mu_thickness, sigma_thickness, num_sims)
        random_deltaLength = np.random.normal(mu_length, sigma_length, num_sims)

        # Create an array for holding the results
        results_shape = np.append(np.asarray([num_sims]), self.s_parameters().shape)
        self.results = np.zeros([dim for dim in results_shape], dtype='complex128')

        # Run simulations with varied width and thickness
        for sim in range(num_sims):
            modified_netlist = copy.deepcopy(self.netlist)
            for component in modified_netlist.components:
                if component.model.component_type == "ebeam_wg_integral_1550":
                    component.extras['width'] = random_width[sim]
                    component.extras['thickness'] = random_thickness[sim]
                    # TODO: Implement length monte carlo using random_deltaLength[sim]
            self.results[sim, :, :, :] = Simulation(modified_netlist, self.start_freq, self.stop_freq, self.num).s_parameters()
            
        if timed:
            stop = time.time()
            if printer:
                printer('Total simulation time: ' + str(stop-start) + ' seconds')

        printer("Simulation complete.")

    
# class MultiInputSimulation(Simulation):
#     def __init__(self, netlist):
#         super().__init__(netlist)

#     def multi_input_simulation(self, inputs: list=[]):
#         """
#         Parameters
#         ----------
#         inputs : list
#             A 0-indexed list of the ports to be used as inputs.
#         """
#         active = [0] * len(self.ports)
#         for val in inputs:
#             active[val] = 1
#         self.simulated_matrix = self._measure_s_matrix(active)

#     def _measure_s_matrix(self, inputs):
#         num_ports = len(inputs)
#         inputs = np.array(inputs)
#         out = np.zeros([len(self.frequency), num_ports], dtype='complex128')
#         for i in range(len(self.frequency)):
#             out[i, :] = np.dot(np.reshape(self.s_matrix[i, :, :], [num_ports, num_ports]), inputs.T)
#         return out

#     def plot(self, output_port):
#         plt.figure()
#         plt.plot(*self.get_magnitude_by_frequency_thz(output_port))
#         plt.title('Multi-Input Simulation')
#         plt.xlabel('Frequency (THz)')
#         plt.ylabel('Gain (dB)')
#         plt.draw()
#         plt.show()

#     def get_magnitude_by_frequency_thz(self, output_port):
#         """
#         Parameters
#         ----------
#         output_port : int
#             Gets the values at that output port (0-indexed).
#         """
#         freq = np.divide(self.frequency, MathPrefixes.TERA)
#         mag = np.power(np.absolute(self.simulated_matrix[:, output_port]), 2)
#         return freq, mag

#     def get_magnitude_by_wavelength_nm(self, output_port):
#         """
#         Parameters
#         ----------
#         output_port : int
#             Gets the values at that output port (0-indexed).
#         """
#         wl = self.frequencyToWavelength(self.frequency) / MathPrefixes.NANO
#         mag = np.power(np.absolute(self.simulated_matrix[:, output_port]), 2)
#         return wl, mag

#     def export_s_matrix(self):
#         """Returns the matrix result of the multi-input simulation.

#         Returns
#         -------
#         frequency, matrix: np.array, np.ndarray
#         """
#         return self.frequency, self.simulated_matrix
