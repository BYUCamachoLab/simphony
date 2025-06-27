# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from simphony.circuit import Circuit
from simphony.circuit import Circuit
from jax.typing import ArrayLike
from copy import deepcopy
import networkx as nx

class SParameterSimulation:
    def __init__(
            self, 
            ckt: Circuit, 
            ports=None, 
            # settings: dict = None
        ):
        """s_parameter_simulation
        Calculates the S-parameters for a given set of ports in an optical
        circuit. 
        
        By default, the exposed ports and settings are taken from 
        the provided netlist, but may be overwritten with keyword arguments.
        """
        self.circuit = ckt
        
        if ports is None:
            self.ports = self.circuit.netlist['ports']
        
        # if settings is not None:
        #     self.update_settings(settings)

        self._identify_component_types()
        self._build_s_parameter_graph()
        self._validate_s_parameter_graph()
        self._determine_steady_state_order()
        self.reset_settings(use_default_settings=True)


    def run(
        self, 
        wl: ArrayLike, 
        settings: dict = None, 
        use_default_settings: bool = True
    ):
        if settings is not None:
            self.reset_settings(use_default_settings=use_default_settings)
            self.add_settings(settings)

        self._instantiate_components()
        self._calculate_steady_states()
        self._calculate_scattering_matrix()

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
        optical_only_graph = self.circuit.graph.copy()
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

        self.s_parameter_graph = optical_only_graph.copy()
        nodes_to_remove = set(self.s_parameter_graph.nodes) - set(s_parameter_graph_nodes)
        self.s_parameter_graph.remove_nodes_from(nodes_to_remove)

    def _validate_s_parameter_graph(self):
        # Signal source nodes are sources of non-optical signals
        source_nodes = set()
        s_parameter_graph_nodes = set(self.s_parameter_graph.nodes)
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
        non_s_parameter_graph = self.circuit.graph.copy()
        non_s_parameter_graph.remove_edges_from(self.s_parameter_graph.edges())
        for source_node in source_nodes:
            descendants = nx.descendants(non_s_parameter_graph, source_node)
            if len(descendants&s_parameter_graph_nodes) > 0:
                raise ValueError("Invalid S-parameter SubCircuit: Time-domain Simulation Required")

    def _determine_steady_state_order(self):
        """
        Voltage signals at electrical ports are assumed to be constant
        for SParameterSimulations, but they are not known a priori, unless
        the voltage source is not dependent on an input signal.

        Since steady-state connections are assumemd to be uni-directional, this function is
        able to find the order in which electrical component voltages must
        be calculated to find the proper steady state.
        """
        self.steady_state_order = []
        graph = self.circuit.graph.copy()
        graph.remove_nodes_from(self.s_parameter_graph.nodes)
        while graph.number_of_nodes() > 0:
            root_nodes = [n for n in graph.nodes if graph.in_degree(n)==0]
            if len(root_nodes) == 0:
                break
            self.steady_state_order += root_nodes
            graph.remove_nodes_from(root_nodes)

        if graph.number_of_nodes() > 0:
            raise ValueError(
                "Failed to determine steady state order. " \
                "Hint: Steady state cannot be determined for circular connections."
            )
    
    def _instantiate_components(self):
        self.components = {}
        for component_name in self.circuit.graph.nodes:
            model_name = self.circuit.netlist['instances'][component_name]['component']
            model = self.circuit.models[model_name]
            settings = self.settings[component_name]
            self.components[component_name] = model(**settings)

    
    def _calculate_steady_states(self):
        for component in self.steady_state_order:
            pass
    
    def _calculate_scattering_matrix(self):
        pass