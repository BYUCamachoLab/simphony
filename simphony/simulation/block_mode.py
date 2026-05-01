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
    dt: float = 1e-14
    num_time_steps: int = 1000
    optical_baseband_wavelengths: jax.Array = field(default_factory=lambda:jax.numpy.array([1.55e-6]))
    use_speed_up: bool = True

class BlockModeSimulationResult(SimulationResult):
    def __init__(self, circuit):
        self.circuit = deepcopy(circuit)
        self.component_inputs = {}
        self.component_outputs = {}
    
    def _collect_component_inputs(self, component)->dict:
        inputs = {}
        input_components = [u for u, v in self.circuit.graph.in_edges(component)]
        for input_component in input_components:
            input_edges = self.circuit.graph.get_edge_data(input_component, component)
            for edge_number, edge in input_edges.items():
                inputs[edge['dst_port']] = self.component_outputs[input_component][edge['src_port']]
                pass
        self.component_inputs[component] = inputs

        

class BlockModeSimulation(Simulation):
    def __init__(
        self, 
        circuit: Circuit,
        settings,
        tracked_ports: dict = None,
        simulation_parameters = None,
        # ports = None,
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
        self.tracked_ports = tracked_ports
        # if ports is None:
        #     ports = self.circuit.netlist['top_level']['ports']

        
        # self.ports = ports

    def run(
        self,
    )->BlockModeSimulationResult:
        # _add_directionality_settings_to_s_parameter_components(self.flat_circuit, self.settings)
        self._instantiated_circuit = self.circuit.instantiate(self.settings, self.simulation_parameters, tracked_ports=self.tracked_ports, directed=True)
        # instantiated_circuit.display()
        simulation_result = BlockModeSimulationResult(self._instantiated_circuit)


        self.block_mode_order = self._determine_block_mode_order_nx_method(self._instantiated_circuit)
        # self._instantiate_components(self.settings)
        print(len(self.block_mode_order))
        for instance_name in self.block_mode_order:
            
            simulation_result._collect_component_inputs(instance_name)   
            inputs = simulation_result.component_inputs[instance_name]
            component = self._instantiated_circuit.instantiated_flat_netlist['instances'][instance_name]['model']
            outputs = component.block_mode_response(inputs, self.simulation_parameters)
            simulation_result.component_outputs[instance_name] = outputs
        
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
        
