
import copy

import numpy as np
from scipy.interpolate import interp1d

from simphony.core import Netlist


def interpolate(output_freq, input_freq, s_parameters):
    func = interp1d(input_freq, s_parameters, kind='cubic', axis=0)
    return [output_freq, func(output_freq)]

# class CachedComponent:
#     def __init__(self, component_type, s_parameters):
#         self.component_type = component_type
#         self.s_parameters = s_parameters

class ComponentSimulation:
    """
    This class is a simplified version of a Component in that it only contains
    an ordered list of nets, the frequency array, and the s-parameter matrix. 
    It can be initialized with or without a Component model, allowing its 
    attributes to be set after object creation.

    Attributes
    ----------
    nets : list(str)
        An ordered list of the nets connected to the Component
    f : np.array
        A numpy array of the frequency values in its simulation.
    s : np.array
        A numpy array of the s-parameter matrix for the given frequency range.
    """
    nets: list
    f: np.array
    s: np.array

    def __init__(self, component=None):
        """
        Instantiates an object from a Component if provided; empty, if not.

        Parameters
        ----------
        component : Component, optional
            A component to initialize the data members of the object.
        """
        if component:
            self.nets = copy.deepcopy(component.nets)
            self.f, self.s = component.get_s_parameters()

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
        self.cached = {}
        for component in self.netlist.components:
            if component.model.component_type not in self.cached and component.model.cachable:
                freq, s_parameters = interpolate(self.freq_array, *component.get_s_parameters())
                self.cached[component.model.component_type] = (freq, s_parameters)

    # def _match_ports(self, net_id: str, component_list: list) -> list:
    #     """
    #     Finds the components connected together by the specified net_id (string) in
    #     a list of components provided by the caller (even if the component is 
    #     connected to itself).

    #     Parameters
    #     ----------
    #     net_id : str
    #         The net id or name to which the components being searched for are 
    #         connected.
    #     component_list : list
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
    #     filtered_comps = [component for component in component_list if net_id in component.nets]
    #     comp_idx = [component_list.index(component) for component in filtered_comps]
    #     net_idx = []
    #     for comp in filtered_comps:
    #         net_idx += [i for i, x in enumerate(comp.nets) if x == net_id]
    #     if len(comp_idx) == 1:
    #         comp_idx += comp_idx
        
    #     return [comp_idx[0], net_idx[0], comp_idx[1], net_idx[1]]

    # def connect_circuit(self):
    #     """
    #     Connects the s-matrices of a photonic circuit given its ObjectModelNetlist
    #     and returns a single 'ComponentSimulation' object containing the frequency
    #     array, the assembled s-matrix, and a list of the external nets (strings of
    #     negative numbers).

    #     Returns
    #     -------
    #     ComponentSimulation
    #         After the circuit has been fully connected, the result is a single 
    #         ComponentSimulation with fields f (frequency), s (s-matrix), and nets 
    #         (external ports: negative numbers, as strings).
    #     list
    #         A list of Component objects that contain an external port.
    #     """
    #     if netlist.net_count == 0:
    #         return

    #     component_list = [ComponentSimulation(component) for component in netlist.component_list]
    #     for n in range(0, netlist.net_count + 1):
    #         ca, ia, cb, ib = _match_ports(str(n), component_list)

    #         #if pin occurances are in the same Cell
    #         if ca == cb:
    #             component_list[ca].s = rf.innerconnect_s(component_list[ca].s, ia, ib)
    #             del component_list[ca].nets[ia]
    #             if ia < ib:
    #                 del component_list[ca].nets[ib-1]
    #             else:
    #                 del component_list[ca].nets[ib]

    #         #if pin occurances are in different Cells
    #         else:
    #             combination = ComponentSimulation()
    #             combination.f = component_list[0].f
    #             combination.s = rf.connect_s(component_list[ca].s, ia, component_list[cb].s, ib)
    #             del component_list[ca].nets[ia]
    #             del component_list[cb].nets[ib]
    #             combination.nets = component_list[ca].nets + component_list[cb].nets
    #             del component_list[ca]
    #             if ca < cb:
    #                 del component_list[cb-1]
    #             else:
    #                 del component_list[cb]
    #             component_list.append(combination)

    #     return component_list[0], netlist.get_external_components()
    
    def get_sparameters(self):
        """
        Gets the s-parameters matrix from a passed in ObjectModelNetlist by 
        connecting all components.

        Parameters
        ----------
        netlist: ObjectModelNetlist
            The netlist to be connected and have parameters extracted from.

        Returns
        -------
        s, f, externals, edge_components: np.array, np.array, list(str)
            A tuple in the following order: 
            ([s-matrix], [frequency array], [external port list], [edge components])
            - s-matrix: The s-parameter matrix of the combined component.
            - frequency array: The corresponding frequency array, indexed the same
                as the s-matrix.
            - external port list: Strings of negative numbers representing the 
                ports of the combined component. They are indexed in the same order
                as the columns/rows of the s-matrix.
            - edge components: list of Component objects, which are the external
                components.
        """
        pass
        # combined, edge_components = self.connect_circuit()
        # f = combined.f
        # s = combined.s
        # externals = combined.nets
        # return (s, f, externals, edge_components)
