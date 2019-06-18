
import copy
import time
from typing import List

import numpy as np
from scipy.interpolate import interp1d
from simphony.core import ComponentInstance, ComponentModel, Netlist
from simphony.core.connect import connect_s, innerconnect_s


def interpolate(output_freq, input_freq, s_parameters):
    func = interp1d(input_freq, s_parameters, kind='cubic', axis=0)
    return [output_freq, func(output_freq)]

class SimulatedComponent:
    """
    This class is a simplified version of a Component in that it only contains
    an ordered list of nets, the frequency array, and the s-parameter matrix. 
    It can be initialized with or without a Component model, allowing its 
    attributes to be set after object creation.

    Attributes
    ----------
    nets : list(int)
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
    def __init__(self, netlist: Netlist, start_freq: float=1.88e+14, stop_freq: float=1.99e+14, num: int=2000):
        self.netlist = netlist
        self.start_freq = start_freq
        self.stop_freq = stop_freq
        self.num = num
        self._cascade()

    @property
    def freq_array(self):
        return np.linspace(self.start_freq, self.stop_freq, self.num)

    def cache_models(self):
        self._cached = {}
        for component in self.netlist.components:
            if component.model.component_type not in self._cached and component.model.cachable:
                freq, s_parameters = interpolate(self.freq_array, *component.get_s_parameters())
                self._cached[component.model.component_type] = (freq, s_parameters)

    def _component_converter(self, component: ComponentInstance) -> SimulatedComponent:
        if component.model.component_type in self._cached:
            return SimulatedComponent(component.nets, *self._cached[component.model.component_type])
        else:
            component.extras['start_freq'] = self.start_freq
            component.extras['stop_freq'] = self.stop_freq
            component.extras['num'] = self.num
            return SimulatedComponent(component.nets, *component.get_s_parameters())
        pass

    def _cascade(self):
        self.cache_models()
        component_list = [self._component_converter(component) for component in self.netlist.components]
        self.combined = self._rearrange(connect_circuit(component_list, self.netlist.net_count))

    @staticmethod
    def _rearrange_order(ports: list):
        reordered = copy.deepcopy(ports)
        reordered.sort(reverse = True)
        return [ports.index(i) for i in reordered]

    @classmethod
    def _rearrange(cls, component: SimulatedComponent) -> SimulatedComponent:
        """Rearranges the s-matrix of the simulated component according to its port ordering.

        Returns
        -------
        SimulatedComponent
            A single component (representing the entire circuit) with its 
            reordered s-matrix and port list.
        """
        concatenate_order = cls._rearrange_order(component.nets)
        reordered_nets = [(-x - 1) for x in component.nets]
        reordered_nets.sort()
        return SimulatedComponent(nets=reordered_nets, freq=component.f, s_parameters=cls._rearrange_matrix(component.s, concatenate_order))

    @staticmethod
    def _rearrange_matrix(s_matrix, concatenate_order):
        port_count = len(concatenate_order)
        reordered_s = np.zeros(s_matrix.shape, dtype=complex)
        for i in range(port_count):
            for j in range(port_count):
                x = concatenate_order[i]
                y = concatenate_order[j]
                reordered_s[:, i, j] = s_matrix[:, x, y]
        return reordered_s

    def s_parameters(self):
        pass

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
    filtered_comps = [component for component in component_list if net_id in component.nets]
    comp_idx = [component_list.index(component) for component in filtered_comps]
    net_idx = []
    for comp in filtered_comps:
        net_idx += [i for i, x in enumerate(comp.nets) if x == net_id]
    if len(comp_idx) == 1:
        comp_idx += comp_idx
    
    return [comp_idx[0], net_idx[0], comp_idx[1], net_idx[1]]
    
    # def get_sparameters(self):
    #     """
    #     Gets the s-parameters matrix from a passed in ObjectModelNetlist by 
    #     connecting all components.

    #     Parameters
    #     ----------
    #     netlist: ObjectModelNetlist
    #         The netlist to be connected and have parameters extracted from.

    #     Returns
    #     -------
    #     s, f, externals, edge_components: np.array, np.array, list(str)
    #         A tuple in the following order: 
    #         ([s-matrix], [frequency array], [external port list], [edge components])
    #         - s-matrix: The s-parameter matrix of the combined component.
    #         - frequency array: The corresponding frequency array, indexed the same
    #             as the s-matrix.
    #         - external port list: Strings of negative numbers representing the 
    #             ports of the combined component. They are indexed in the same order
    #             as the columns/rows of the s-matrix.
    #         - edge components: list of Component objects, which are the external
    #             components.
    #     """
    #     pass
    #     # combined, edge_components = self.connect_circuit()
    #     # f = combined.f
    #     # s = combined.s
    #     # externals = combined.nets
    #     # return (s, f, externals, edge_components)

def connect_circuit(components: List[SimulatedComponent], net_count: int) -> SimulatedComponent:
    """
    Connects the s-matrices of a photonic circuit given its ObjectModelNetlist
    and returns a single 'ComponentSimulation' object containing the frequency
    array, the assembled s-matrix, and a list of the external nets (strings of
    negative numbers).

    Parameters
    ----------
    component_list : List[SimulatedComponent]
    net_count : int

    Returns
    -------
    SimulatedComponent
        After the circuit has been fully connected, the result is a single 
        ComponentSimulation with fields f (frequency), s (s-matrix), and nets 
        (external ports: negative numbers, as strings).
    list
        A list of Component objects that contain an external port.
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

    return component_list[0]



# class MathPrefixes:
#     TERA = 1e12
#     NANO = 1e-9
#     c = 299792458




#     def frequencyToWavelength(self, frequency):
#         return MathPrefixes.c / frequency

#     def getMagnitudeByFrequencyTHz(self, fromPort, toPort):
#         print("From", fromPort, "to", toPort)
#         freq = np.divide(self.frequency, MathPrefixes.TERA)
#         mag = abs(self.s_matrix[:,fromPort,toPort])**2
#         return freq, mag
    
#     def getMagnitudeByWavelengthNm(self, fromPort, toPort):
#         wl = self.frequencyToWavelength(self.frequency) / MathPrefixes.NANO
#         mag = abs(self.s_matrix[:,fromPort,toPort])**2
#         return wl, mag

#     def getPhaseByFrequencyTHz(self, fromPort, toPort):
#         freq = np.divide(self.frequency, MathPrefixes.TERA)
#         phase = np.rad2deg(np.unwrap(np.angle(self.s_matrix[:,fromPort,toPort])))
#         return freq, phase

#     def getPhaseByWavelengthNm(self, fromPort, toPort):
#         wl = self.frequencyToWavelength(self.frequency) / MathPrefixes.NANO
#         phase = np.rad2deg(np.unwrap(np.angle(self.s_matrix[:,fromPort,toPort])))
#         print(wl, phase)
#         return wl, phase

#     def exportSMatrix(self):
#         return self.s_matrix, self.frequency

# import matplotlib.pyplot as plt
# from scipy.io import savemat

# class MonteCarloSimulation:
#     DEF_NUM_SIMS = 10
#     DEF_MU_WIDTH = 0.5
#     DEF_SIGMA_WIDTH = 0.005
#     DEF_MU_THICKNESS = 0.22
#     DEF_SIGMA_THICKNESS = 0.002
#     DEF_MU_LENGTH = 0
#     DEF_SIGMA_LENGTH = 0
#     DEF_DPIN = 1
#     DEF_DPOUT = 0
#     DEF_SAVEDATA = True
#     DEF_DISPTIME = True
#     DEF_FILENAME = "monte_carlo.mat"

#     def __init__(self, netlist):
#         self.netlist = netlist
#         # Run simulation with mean width and thickness
#         self.mean_s, self.frequency, self.ports, _ = nl.get_sparameters(self.netlist) 

#     def monte_carlo_sim(self, 
#                         num_sims=DEF_NUM_SIMS, 
#                         mu_width=DEF_MU_WIDTH, 
#                         sigma_width=DEF_SIGMA_WIDTH, 
#                         mu_thickness=DEF_MU_THICKNESS,
#                         sigma_thickness=DEF_SIGMA_THICKNESS, 
#                         mu_length=DEF_MU_LENGTH, 
#                         sigma_length=DEF_SIGMA_LENGTH, 
#                         saveData=False, 
#                         filename=None, 
#                         dispTime=False, 
#                         printer=None):
#         # Timer
#         start = time.time()

#         # Random distributions
#         random_width = np.random.normal(mu_width, sigma_width, num_sims)
#         random_thickness = np.random.normal(mu_thickness, sigma_thickness, num_sims)
#         random_deltaLength = np.random.normal(mu_length, sigma_length, num_sims)

#         results_shape = np.append(np.asarray([num_sims]), self.mean_s.shape)
#         results = np.zeros([dim for dim in results_shape], dtype='complex128')

#         # Run simulations with varied width and thickness
#         for sim in range(num_sims):
#             modified_netlist = copy.deepcopy(self.netlist)
#             for component in modified_netlist.component_list:
#                 if component.__class__.__name__ == "ebeam_wg_integral_1550":
#                     component.width = random_width[sim]
#                     component.height = random_thickness[sim]
#                     # TODO: Implement length monte carlo using random_deltaLength[sim]
#             results[sim, :, :, :] = nl.get_sparameters(modified_netlist)[0]
#             if ((sim % 10) == 0) and dispTime:
#                 print(sim)

#         # rearrange matrix so matrix indices line up with proper port numbers
#         self.ports = [int(i) for i in self.ports]
#         rp = copy.deepcopy(self.ports)
#         rp.sort(reverse=True)
#         concatenate_order = [self.ports.index(i) for i in rp]
#         temp_res = copy.deepcopy(results)
#         temp_mean = copy.deepcopy(self.mean_s)
#         re_res = np.zeros(results_shape, dtype=complex)
#         re_mean = np.zeros(self.mean_s.shape, dtype=complex)
#         i=0
#         for idx in concatenate_order:
#             re_res[:,:,i,:]  = temp_res[:,:,idx,:]
#             re_mean[:,i,:] = temp_mean[:,idx,:]
#             i += 1
#         temp_res = copy.deepcopy(re_res)
#         temp_mean = copy.deepcopy(re_mean)
#         i=0
#         for idx in concatenate_order:
#             re_res[:,:,:,i] = temp_res[:,:,:,idx]
#             re_mean[:,:,i] = temp_mean[:,:,idx]
#             i+= 1    
#         self.results = copy.deepcopy(re_res)
#         self.mean_s = copy.deepcopy(re_mean)

#         # print elapsed time if dispTime is True
#         stop = time.time()
#         if dispTime and printer:
#             printer('Total simulation time: ' + str(stop-start) + ' seconds')

#         # save MC simulation results to matlab file
#         if saveData == True:
#             savemat(filename, {'freq':self.frequency, 'results':self.results, 'mean':self.mean_s})

#         printer("Simulation complete.")

#     def plot(self, num_sims, dpin, dpout):
#         plt.figure(1)
#         for i in range(num_sims):
#             plt.plot(self.frequency, 10*np.log10(abs(self.results[i, :, dpin, dpout])**2), 'b', alpha=0.1)
#         plt.plot(self.frequency,  10*np.log10(abs(self.mean_s[:, dpin, dpout])**2), 'k', linewidth=0.5)
#         title = 'Monte Carlo Simulation (' + str(num_sims) + ' Runs)'
#         plt.title(title)
#         plt.xlabel('Frequency (Hz)')
#         plt.ylabel('Gain (dB)')
#         plt.draw()
#         plt.show()

    
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
