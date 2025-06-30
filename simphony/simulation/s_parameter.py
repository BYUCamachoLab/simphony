# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from simphony.circuit import Circuit
from .simulation import Simulation, SimulationResult
from .steady_state import SteadyStateSimulation
from simphony.circuit import Circuit
from jax.typing import ArrayLike
from copy import deepcopy
import networkx as nx
from simphony.utils import graph_to_netlist
import sax

from functools import partial

class SParameterSimulationResult(SimulationResult):
    def __init__(self):
        ### TODO: This object needs to store at least 
        ### The Scattering parameter dictionary and the specified ports
        ...

class SParameterSimulation(Simulation):
    def __init__(
            self, 
            circuit: Circuit, 
            ports=None, 
            # settings: dict = None
        ):
        """s_parameter_simulation
        Calculates the S-parameters for a given set of ports in an optical
        circuit. 
        
        By default, the exposed ports and settings are taken from 
        the provided netlist, but may be overwritten with keyword arguments.
        """
        self.circuit = circuit
        
        if ports is None:
            self.ports = self.circuit.netlist['ports']
        
        # if settings is not None:
        #     self.update_settings(settings)

        self._identify_component_types()
        self._build_s_parameter_graph()
        self._validate_s_parameter_graph()
        self._initialize_steady_state_simulation()
        self.reset_settings(use_default_settings=True)


    def run(
        self, 
        wl: ArrayLike, 
        settings: dict = None, 
        use_default_settings: bool = True
    )->SParameterSimulationResult:
        s_parameter_simulation_result = SParameterSimulationResult()
        if settings is not None:
            self.reset_settings(use_default_settings=use_default_settings)
            self.add_settings(settings)

        self._instantiate_components(self.settings)
        steady_state_simulation_result = self.steady_state_simulation.run(self.settings)
        sax_circuit, sax_circuit_info = self._generate_sax_circuit(wl, steady_state_simulation_result)
        s_parameter_simulation_result.sax_circuit = sax_circuit
        s_parameter_simulation_result.sax_circuit_info = sax_circuit_info
        s_parameter_simulation_result.s_parameters = sax_circuit(wl=wl)
        return s_parameter_simulation_result

    def _clear_settings(self):
        for instance in self.circuit.graph.nodes:
                self.settings[instance] = {}

    def reset_settings(self, use_default_settings: bool = True):
        """
        Reset settings to their defaults (specified in Circuit) or clear all settings
        """
        if use_default_settings:
            self.settings = deepcopy(self.circuit.default_settings)
        else:
            self._clear_settings()

    def add_settings(self, settings: dict):
        """
        Update the current settings with additional settings.
        """
        for instance, instance_settings in settings.items():
            self.settings[instance].update(instance_settings)

    def _identify_component_types(self):
        self.all_components = set()
        self.electrical_components = set()
        self.optical_components = set()
        self.logic_components = set()
        
        graph = self.circuit.graph
        models = self.circuit.models
        for node, attr in graph.nodes(data=True):
            model = attr["component"]
            component = models[model]
            
            self.all_components.add(node)
            if component.electrical_ports:
                self.electrical_components.add(node)
            if component.logic_ports:
                self.logic_components.add(node)
            if component.optical_ports:
                self.optical_components.add(node)
        

    def _build_s_parameter_graph(self):
        non_optical_components = self.all_components - self.optical_components
        optical_only_graph = deepcopy(self.circuit.graph)
        optical_only_graph.remove_nodes_from(non_optical_components)
        
        # For now, we only consider the optical connections
        # While admittedly an edge case, if one optical section were
        # connected to a photodiode that was connected to a phase modulator
        # of another optical section, that connection would not be considered.
        edges_to_remove = []
        for edge in optical_only_graph.edges:
            src = edge[0]
            src_port = optical_only_graph.edges[edge]["src_port"]
            if not src_port in optical_only_graph.nodes[src]['optical ports']:
                edges_to_remove.append(edge)
        optical_only_graph.remove_edges_from(edges_to_remove)

        # Nodes with an exposed port are considered "entry nodes"
        entry_nodes = set()
        for attr in self.ports.values():
            node = attr.split(',')[0]
            entry_nodes.add(node)
        
        weakly_connected_components = nx.weakly_connected_components(optical_only_graph)
        s_parameter_graph_nodes = None
        for subnetwork in weakly_connected_components:
            if entry_nodes.issubset(subnetwork):
                s_parameter_graph_nodes = subnetwork
        
        if s_parameter_graph_nodes is None:
            raise ValueError("S-parameter graph could not be generated. All exposed ports must be weakly connected through optical components")

        self.s_parameter_circuit = deepcopy(self.circuit)
        nodes_to_remove = set(self.circuit.graph.nodes) - set(s_parameter_graph_nodes)
        self.s_parameter_circuit.remove_components(nodes_to_remove)

        # self.s_parameter_graph = deepcopy(optical_only_graph)
        # nodes_to_remove = set(self.s_parameter_graph.nodes) - set(s_parameter_graph_nodes)
        # self.s_parameter_graph.remove_nodes_from(nodes_to_remove)

        self.hybrid_components = set(self.s_parameter_circuit.graph.nodes)&(self.electrical_components|self.logic_components)

    def _validate_s_parameter_graph(self):
        # Signal source nodes are sources of non-optical signals
        source_nodes = set()
        s_parameter_graph_nodes = set(self.s_parameter_circuit.graph.nodes)
        potential_source_nodes = s_parameter_graph_nodes & self.optical_components & (self.electrical_components | self.logic_components)
        for node in potential_source_nodes:
            out_edges = self.circuit.graph.out_edges(node, data=True)
            for src, dst, attr in out_edges:
                src_port = attr['src_port']
                if src_port not in self.circuit.graph.nodes[src]['optical ports']:
                        source_nodes.add(node)
        
        # We do not allow any of the s_parameter_nodes to function as
        # signal sources that feed back into the s_parameter_nodes
        # Such simulations should be performed in the time-domain
        non_s_parameter_graph = deepcopy(self.circuit.graph)
        non_s_parameter_graph.remove_edges_from(self.s_parameter_circuit.graph.edges())
        for source_node in source_nodes:
            descendants = nx.descendants(non_s_parameter_graph, source_node)
            if len(descendants&s_parameter_graph_nodes) > 0:
                raise ValueError("Invalid S-parameter SubCircuit: Time-domain Simulation Required")

    def _initialize_steady_state_simulation(self):
        steady_state_circuit = deepcopy(self.circuit)
        steady_state_circuit.remove_components(self.s_parameter_circuit.graph.nodes-self.hybrid_components)
        self.steady_state_simulation = SteadyStateSimulation(steady_state_circuit)
        # steady_state_graph.remove_nodes_from(self.s_parameter_graph.nodes)
        # self.steady_state_simulation = SteadyStateSimulation(self.steady_state_graph)

    ### I am going to put this in the base class
    # def _instantiate_components(self):
    #     self.components = {}
    #     for component_name in self.circuit.graph.nodes:
    #         model_name = self.circuit.netlist['instances'][component_name]['component']
    #         model = self.circuit.models[model_name]
    #         settings = self.settings[component_name]
    #         self.components[component_name] = model(**settings)

    # def _calculate_steady_states(self):
    #     for component in self.steady_state_order:
    #         pass
    
    def _generate_sax_circuit(self, wl, steady_state_simulation_result):
        """
        """
        # I will assume that the only connections between the s-parameter portion of the circuit
        # and the steady-state portion of the circuit are electrical or optical (this might change)
        # in the future if more connection types become supported
        
        # These components will need to have their s_parameter methods completed with the steady state inputs
        incomplete_components = self.hybrid_components
        component_inputs = {component: {} for component in self.s_parameter_circuit.graph.nodes}
        for incomplete_component in incomplete_components:
            component_inputs[incomplete_component] = steady_state_simulation_result.component_inputs[incomplete_component]
        
        sax_models = {}
        for component, inputs in component_inputs.items():
            # model_name = self.circuit.netlist['instances'][component]['component']
            sax_models[component] = partial(self.components[component].s_parameters, inputs)

        # Each instance should correspond to a unique model at this point
        # Since they might have been changed but the steady state inputs
        # Therefore, we need the netlist to reflect this change

        instances = {key: key for key in self.s_parameter_circuit.netlist['instances']}
        self.s_parameter_circuit.netlist['instances'] = instances

        return sax.circuit(self.s_parameter_circuit.netlist, sax_models)
        """
        ### TODO: MATTHEW! Keep in mind that I defined the sax models to use SI units
        ### and to assume that wl is given in terms of meters, not microns
        ### To see what I mean, stop here in debug mode and run sax_models['splitter'](1.55e-6)
        ### Notice that I am using 1.55e-6 instead of 1.55 but that is what it expects
        pass
        
        ### TODO: complete the function graph_to_netlist in simphony.utils
        ### turn self.s_parameter_graph into a netlist
        ### use the sax_models dictionary above and the netlist you just generated to create
        ### a sax.circuit and return the resulting scattering dictionary
        ### Alternatively, it might be more efficient to make an "self.s_parameter_circuit"
        ### and use the remove_compoenents method from that circuit object
        ### that might get you the netlist you need for free

        sax_netlist = graph_to_netlist(self.s_parameter_circuit.graph) ## You might change this to use the alternative approach
        circuit = sax.circuit(sax_netlist, sax_models)
        ### TODO: It is probably possible for the user to keep the s-parameter graph elements parameterized
        ### and simply return a sax circuit, maybe I will do that later, don't do that yet, 
        ### For now just return the s-parameter dict
        return circuit(wl)
        """