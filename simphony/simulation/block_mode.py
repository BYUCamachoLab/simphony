from simphony.simulation.simulation import Simulation, SimulationResult, SimulationParameters, SimulationMode
from simphony.circuit.circuit import Circuit
import networkx as nx
from copy import deepcopy
from flax import struct
from dataclasses import field
import jax
import jax.numpy as jnp
from simphony.libraries.ideal.s_parameters import SParameterSax


_S_PARAMETER_META_SETTINGS = {
    "port_directionality",
    "vector_fitting_parameters",
    "delay_compensation",
}

@struct.dataclass
class BlockModeSimulationParameters(SimulationParameters):
    simulation_mode: SimulationMode = field(default_factory=lambda:SimulationMode.BLOCK_MODE)    
    directed: bool = True
    dt: float = 1e-14
    num_time_steps: int = 1000
    spectral_range: tuple[float, float] = (1.5e-6, 1.6e-6)
    center_wavelength: float = 1.55e-6
    optical_baseband_wavelengths: jax.Array = field(default_factory=lambda:jnp.array([1.55e-6]))
    store_component_inputs: bool = True
    store_component_outputs: bool = True
    store_port_outputs: bool = True
    tracked_components: tuple[str, ...] = field(default_factory=tuple)
    use_speed_up: bool = False
    
    
        
    # def __init__(
    #     self,
    #     **kwargs,
    # ):
    #     super().__init__(**kwargs)

class BlockModeSimulationResult(SimulationResult):
    def __init__(self, circuit):
        self.circuit = deepcopy(circuit)
        self.component_inputs = {}
        self.component_outputs = {}
        self.port_outputs = {}
    
    def _collect_component_inputs(self, component, output_cache)->dict:
        inputs = {}
        # TODO: Determine if STEADYSTATESIMULATION needs this change too
        # input_components = nx.ancestors(self.circuit.graph, component)
        input_components = [u for u, v in self.circuit.graph.in_edges(component)]
        for input_component in input_components:
            input_edges = self.circuit.graph.get_edge_data(input_component, component)
            for edge_number, edge in input_edges.items():
                inputs[edge['dst_port']] = output_cache[input_component][edge['src_port']]
        return inputs

class BlockModeSimulation(Simulation):
    def __init__(
        self, 
        circuit: Circuit, 
        settings = None,
        simulation_parameters = None,
        ports = None,
        # circuit: Circuit,
        # ports = None
    ):

        if settings is None:
            settings = {}
        if simulation_parameters is None:
            simulation_parameters = BlockModeSimulationParameters()

        self.simulation_parameters = simulation_parameters
        self.circuit = circuit
        self.flat_circuit = circuit.flatten()
        self.settings = settings

        if ports is None:
            ports = self.flat_circuit.netlist['ports']

        
        self.ports = ports

    def run(
        self,
    )->BlockModeSimulationResult:
        self._add_directionality_setting_to_s_parameter_components(self.flat_circuit, self.settings)
        instantiated_circuit = self.flat_circuit.instantiate(self.settings, self.simulation_parameters)
        # instantiated_circuit.display()
        simulation_result = BlockModeSimulationResult(instantiated_circuit)
        output_cache = {}
        remaining_successors = self._count_remaining_successors(instantiated_circuit)
        external_port_sources = self._group_external_ports_by_instance()
        tracked_components = set(self.simulation_parameters.tracked_components)


        self.block_mode_order = self._determine_block_mode_order_nx_method(instantiated_circuit)
        
        # self._instantiate_components(self.settings)
        for instance_name in self.block_mode_order:
            inputs = simulation_result._collect_component_inputs(instance_name, output_cache)
            if self._should_store_component(instance_name, self.simulation_parameters.store_component_inputs, tracked_components):
                simulation_result.component_inputs[instance_name] = inputs
            component = instantiated_circuit.instantiated_flat_netlist['instances'][instance_name]['model']
            outputs = component._block_mode_response(inputs, self.simulation_parameters)
            output_cache[instance_name] = outputs

            if self._should_store_component(instance_name, self.simulation_parameters.store_component_outputs, tracked_components):
                simulation_result.component_outputs[instance_name] = outputs

            if self.simulation_parameters.store_port_outputs:
                for circuit_port, instance_port in external_port_sources.get(instance_name, {}).items():
                    if instance_port in outputs:
                        simulation_result.port_outputs[circuit_port] = outputs[instance_port]

            for predecessor in instantiated_circuit.graph.predecessors(instance_name):
                remaining_successors[predecessor] -= 1
                if remaining_successors[predecessor] == 0:
                    output_cache.pop(predecessor, None)

            if remaining_successors[instance_name] == 0:
                output_cache.pop(instance_name, None)
        
        return simulation_result
    
    def _determine_block_mode_order_nx_method(self, instantiated_circuit):
        """
        Voltage signals at electrical ports are assumed to be constant
        for SParameterSimulations, but they are not known a priori, unless
        the voltage source is not dependent on an input signal.

        Since steady-state connections are assumemd to be uni-directional, this function is
        able to find the order in which electrical component voltages must
        be calculated to find the proper steady state.
        """
        try:
            return list(nx.topological_sort(instantiated_circuit.graph))
        except nx.NetworkXUnfeasible:
            raise ValueError("Failed to determine steady state order – circular dependencies detected")



    def _add_directionality_setting_to_s_parameter_components(
        self,
        flat_circuit,
        settings
    ):
        netlist = flat_circuit.netlist
        models = flat_circuit.models
        for instance_name, instance_data in netlist['instances'].items():
            component_class = models[instance_data['component']]
            if issubclass(component_class, SParameterSax):
                if not "port_directionality" in settings[instance_name].keys():
                    raise ValueError(f"Port directionality settings must be provided for S-parameter components. Missing for {instance_name}")
                if "sax_settings" not in settings[instance_name]:
                    meta_settings = {
                        k: v
                        for k, v in settings[instance_name].items()
                        if k in _S_PARAMETER_META_SETTINGS
                    }
                    sax_only = {
                        k: v
                        for k, v in settings[instance_name].items()
                        if k not in _S_PARAMETER_META_SETTINGS
                    }
                    settings[instance_name] = {"sax_settings": sax_only} | meta_settings

                # TODO: DOUBLE CHECK DEFAULT DICTIONARY CONSTRUCTION FOR EDGE CASES
                default_directionalities = {p.name:"output" if f"{instance_name},{p.name}" in netlist['connections'].keys() else "input" for p in component_class.ports}
                default_directionalities = {k:d if not (f"{instance_name},{k}" in netlist['connections'].keys() or not f"{instance_name},{k}" in netlist['connections'].values()) else "output" for k,d in default_directionalities.items()}
                settings[instance_name]["port_directionality"] = default_directionalities | settings[instance_name].get("port_directionality", {})

    def _count_remaining_successors(self, instantiated_circuit):
        return {
            instance_name: len(set(instantiated_circuit.graph.successors(instance_name)))
            for instance_name in instantiated_circuit.graph.nodes
        }

    def _group_external_ports_by_instance(self):
        if not isinstance(self.ports, dict):
            return {}

        external_port_sources = {}
        for circuit_port, port_ref in self.ports.items():
            instance_name, instance_port = port_ref.split(",", 1)
            external_port_sources.setdefault(instance_name, {})[circuit_port] = instance_port
        return external_port_sources

    def _should_store_component(self, instance_name, store_all_components, tracked_components):
        return store_all_components or instance_name in tracked_components
    
    
    # def _determine_block_mode_order(self):
    #     """
    #     Determine the order of components in block mode simulation.
    #     """
    #     graph = self.circuit.graph.copy()
    #     try:
    #         return list(nx.topological_sort(graph))
    #     except nx.NetworkXUnfeasible:
    #         raise ValueError("Failed to determine block order – circular dependencies detected")
        
