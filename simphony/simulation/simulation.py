
import copy

import numpy as np
from scipy.interpolate import interp1d
from typing import List

from simphony.core import Netlist, ComponentModel, ComponentInstance
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
    def __init__(self, netlist: Netlist=None, start_freq: float=1.88e+14, stop_freq: float=1.99e+14, points: int=2000):
        self.netlist = netlist
        self.start_freq = start_freq
        self.stop_freq = stop_freq
        self.points = points

    @property
    def freq_array(self):
        return np.linspace(self.start_freq, self.stop_freq, self.points)

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
            return SimulatedComponent(component.nets, *component.model.get_s_parameters())
        pass

    def cascade(self):
        self.cache_models()
        component_list = [self._component_converter(component) for component in self.netlist.components]
        self.combined = connect_circuit(component_list, self.netlist.net_count)
        self._rearrange(self.combined)

    @staticmethod
    def _rearrange_order(ports: list):
        reordered = copy.deepcopy(ports)
        reordered.sort(reverse = True)
        return [ports.index(i) for i in reordered]

    @classmethod
    def _rearrange(cls, component: SimulatedComponent):
        concatenate_order = cls._rearrange_order(component.nets)

        s_matrix1 = cls._rearrange_row_then_column(component, concatenate_order)
        # s_matrix2 = cls._rearrange_in_place(component, concatenate_order)

        import matplotlib.pyplot as plt
        plt.plot(component.f, s_matrix1[:, 1, 2])
        # plt.plot(component.f, s_matrix2[:, 1, 2])
        plt.show()

        # assert np.array_equal(s_matrix1, s_matrix2)

    @staticmethod
    def _rearrange_row_then_column(component, concatenate_order):

        print("Ports:", len(concatenate_order), "Shape:", component.s.shape)
        new_s = copy.deepcopy(component.s)
        reordered_s = np.zeros(component.s.shape, dtype=complex)

        i = 0
        for idx in concatenate_order:
            reordered_s[:,i,:] = new_s[:,idx,:]
            i += 1
        new_s = copy.deepcopy(reordered_s)
        i = 0
        for idx in concatenate_order:
            reordered_s[:,:,i] = new_s[:,:,idx]
            i += 1
        
        return copy.deepcopy(reordered_s)

    @staticmethod
    def _rearrange_in_place(component: SimulatedComponent, concatenate_order):
        port_count = len(component.nets)
        new_s = copy.deepcopy(component.s)
        reordered_s = np.zeros(component.s.shape, dtype=complex)
        print("Ports:", port_count, "Shape:", component.s.shape)
        for i in range(port_count):
            for j in range(port_count):
                x = concatenate_order[i]
                y = concatenate_order[j]
                print("row/col:", i, j, "replaced by", x, y)
                reordered_s[:, i, j] = new_s[:, x, y]
                assert np.array_equal(reordered_s[:, i, j], new_s[:, x, y])
        return copy.deepcopy(reordered_s)

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

    for comp in component_list:
        print(comp, comp.nets, comp.s.shape)
    assert 0

    return component_list[0]