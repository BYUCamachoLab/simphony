from simphony.simulation.simulation import Simulation, SimulationResult, SimulationParameters, SimulationMode
from simphony.circuit.circuit import Circuit
import networkx as nx
from copy import deepcopy
from flax import struct
from dataclasses import field
import jax
from simphony.libraries.ideal.s_parameters import SParameterPlaceholder

@struct.dataclass
class BlockModeSimulationParameters(SimulationParameters):
    simulation_mode: SimulationMode = field(default_factory=lambda:SimulationMode.BLOCK_MODE)    
    directed: bool = True
    dt = 1e-14
    num_time_steps = 1000
    # Moved spectral range to vector fitting parameters in optical_s_parameters
    # spectral_range = (1.5e-6, 1.6e-6)
    # Optical Baseband Wavelengths will be used to control the frequency channels the simulator uses
    optical_baseband_wavelengths: jax.Array = field(default_factory=lambda:jax.numpy.array([1.54e-6, 1.55e-6, 1.56e-6]))
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
        # self.flat_circuit = circuit.flatten()
        self.settings = settings

        if ports is None:
            ports = self.circuit.netlist['top_level']['ports']

        
        self.ports = ports

    def run(
        self,
    )->BlockModeSimulationResult:
        # _add_directionality_settings_to_s_parameter_components(self.flat_circuit, self.settings)
        instantiated_circuit = self.circuit.instantiate(self.settings, self.simulation_parameters, directed=True)
        # instantiated_circuit.display()
        simulation_result = BlockModeSimulationResult(instantiated_circuit)
        output_cache = {}
        remaining_successors = self._count_remaining_successors(instantiated_circuit)
        external_port_sources = self._group_external_ports_by_instance()
        tracked_components = set(self.simulation_parameters.tracked_components)


        self.block_mode_order = self._determine_block_mode_order_nx_method(instantiated_circuit)
        
        # self._instantiate_components(self.settings)
        print(len(self.block_mode_order))
        for instance_name in self.block_mode_order:
            print("Hi")
            simulation_result._collect_component_inputs(instance_name)   
            inputs = simulation_result.component_inputs[instance_name]
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


    # def _determine_block_mode_order(self):
    #     """
    #     Determine the order of components in block mode simulation.
    #     """
    #     graph = self.circuit.graph.copy()
    #     try:
    #         return list(nx.topological_sort(graph))
    #     except nx.NetworkXUnfeasible:
    #         raise ValueError("Failed to determine block order – circular dependencies detected")
        
