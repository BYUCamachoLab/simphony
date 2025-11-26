"""Frequency-domain (S-parameter) simulation utilities.

This module trims user circuits down to their optical subgraphs, stitches in
steady-state operating points for hybrid components, and produces callable SAX
models that can be evaluated over wavelength grids.
"""

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
    """Return object for :meth:`SParameterSimulation.run`.

    Attributes
    ----------
    sax_circuit:
        Callable SAX circuit that can be re-evaluated at arbitrary
        wavelengths after ``run`` completes.
    sax_circuit_info:
        Metadata describing port ordering and symbol mapping as produced by
        :func:`sax.circuit`.
    s_parameters:
        Convenience scattering dictionary evaluated at the wavelength(s)
        supplied to ``run``.
    """

    def __init__(self):
        pass

class SParameterSimulation(Simulation):
    """Build scattering-parameter models for optical subcircuits."""

    def __init__(
            self, 
            circuit: Circuit, 
            ports=None, 
            # settings: dict = None
        ):
        """Prepare the simulation harness.

        Parameters
        ----------
        circuit:
            :class:`Circuit` containing optical, electrical, and logic
            components.  Only the optical subgraph connected to the exposed
            ports will be retained for the S-parameter solve.
        ports:
            Optional mapping of exposed port aliases.  Defaults to the ports
            declared in the input netlist.
        """
        self.circuit = circuit
        
        if ports is None:
            ports = self.circuit.netlist['ports']
        
        # if settings is not None:
        #     self.update_settings(settings)

        self._identify_component_types()
        self._build_s_parameter_circuit(ports)
        self._validate_s_parameter_graph()
        self._initialize_steady_state_simulation()
        self.reset_settings(use_default_settings=True)


    def run(
        self, 
        settings: dict = None, 
        wl: ArrayLike=1.55e-6, 
        # use_default_settings: bool = True
    ) -> SParameterSimulationResult:
        """Compute scattering parameters at the supplied wavelength(s)."""
        s_parameter_simulation_result = SParameterSimulationResult()
        use_default_settings = True
        self.reset_settings(use_default_settings=use_default_settings)
        self.add_settings(settings)
        # s_parameter_result = SParameterSimulationResult()

        self._instantiate_components(self.settings)
        steady_state_simulation_result = self.steady_state_simulation.run(self.settings)
        sax_circuit, sax_circuit_info = self._generate_sax_circuit(wl, steady_state_simulation_result)
        s_parameter_simulation_result.sax_circuit = sax_circuit
        s_parameter_simulation_result.sax_circuit_info = sax_circuit_info
        s_parameter_simulation_result.s_parameters = sax_circuit(wl=wl)
        return s_parameter_simulation_result

    def _identify_component_types(self):
        """Classify circuit nodes by their available port types."""
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
        

    def _build_s_parameter_circuit(self, ports: dict):
        """Carve out an optical-only subnetwork reachable from ``ports``."""
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
        for attr in ports.values():
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
        self.s_parameter_circuit.netlist['ports'] = ports

    def _validate_s_parameter_graph(self):
        """Ensure the extracted optical subgraph has no hidden drive sources."""
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
        """Instantiate a steady-state solver for hybrid components."""
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
        """Convert the prepared subcircuit into a callable SAX model."""
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

        # Legacy notes retained for future work:
        # - SAX models expect wavelengths in meters, not microns.
        # - ``simphony.utils.graph_to_netlist`` can be used to regenerate a
        #   netlist from the optical graph directly if we want to keep the
        #   network parameterized instead of instantiating per-instance models.