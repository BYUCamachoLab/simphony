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
    """Global settings for a Block mode simulation.

    Block mode evaluates a full time block at once. These parameters define the
    time grid, optical carrier wavelengths, and optical modes shared by every
    component in the circuit.

    Attributes
    ----------
    dt:
        Time step, in seconds, between adjacent samples in the block.
    num_time_steps:
        Number of samples processed by each component during one simulation run.
    optical_baseband_wavelengths:
        Carrier wavelengths, in meters, tracked by the optical envelope arrays.
    mode_identifiers:
        Inherited from `SimulationParameters`; labels for the optical modes
        represented on the last axis of a `BlockModeOpticalSignal`.
    use_speed_up:
        Enables implementation-specific acceleration paths where available.
    """
    simulation_mode: SimulationMode = field(default_factory=lambda:SimulationMode.BLOCK_MODE)    
    directed: bool = True
    dt: float = 1e-14
    num_time_steps: int = 1000
    optical_baseband_wavelengths: jax.Array = field(default_factory=lambda:jax.numpy.array([1.55e-6]))
    use_speed_up: bool = True

class BlockModeSimulationResult(SimulationResult):
    """Signals collected from a completed Block mode simulation.

    Attributes
    ----------
    input_signals:
        Signals observed on tracked ports when the tracked port corresponds to a
        component input.
    output_signals:
        Signals observed on tracked ports when the tracked port corresponds to a
        component output.
    """
    def __init__(self, input_signals, output_signals):
        self.input_signals = input_signals
        self.output_signals = output_signals

        

class BlockModeSimulation(Simulation):
    """Run a directed circuit by propagating full-block signals component by component.

    The simulator instantiates the circuit with the provided settings, determines
    a topological execution order, calls each component's block-mode response, and
    returns the signals available at tracked or top-level ports.

    Parameters
    ----------
    circuit:
        Circuit to simulate. For Block mode, the instantiated graph must be
        directed and acyclic.
    settings:
        Per-instance settings used when the circuit is instantiated. SAX
        S-parameter components typically need `sax_settings`,
        `port_directionality`, and `vector_fitting_parameters`.
    tracked_ports:
        Optional mapping of names to internal `"instance,port"` designators to
        expose in the returned result. Top-level circuit ports are tracked by
        default.
    simulation_parameters:
        Shared `BlockModeSimulationParameters`. If omitted, defaults are used.
    """
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
        self.component_inputs = {}
        self.component_outputs = {}
        # if ports is None:
        #     ports = self.circuit.netlist['top_level']['ports']

        
        # self.ports = ports

    def run(
        self,
    )->BlockModeSimulationResult:
        """Run the Block mode simulation and collect tracked port signals.

        The circuit is instantiated in directed mode, components are evaluated in
        topological order, and each component receives a full time block of input
        signals at once.
        """
        # _add_directionality_settings_to_s_parameter_components(self.flat_circuit, self.settings)
        self._instantiated_circuit = self.circuit.instantiate(self.settings, self.simulation_parameters, tracked_ports=self.tracked_ports, directed=True)
        # instantiated_circuit.display()
        # simulation_result = BlockModeSimulationResult(self._instantiated_circuit)


        self.block_mode_order = self._determine_block_mode_order_nx_method(self._instantiated_circuit)
        # self._instantiate_components(self.settings)
        # print(len(self.block_mode_order))
        for instance_name in self.block_mode_order:
            self._collect_component_inputs(instance_name)   
            inputs = self.component_inputs[instance_name]
            component = self._instantiated_circuit.instantiated_flat_netlist['instances'][instance_name]['model']
            outputs = component._block_mode_response(inputs, self.simulation_parameters)
            self.component_outputs[instance_name] = outputs
        
        input_signals = {}
        output_signals = {}
        for tracked_port_name, tracked_port_designator in self._instantiated_circuit.port_lookup_table.items():
            instance_name, port_name = tracked_port_designator.split(",")
            if port_name in self.component_inputs[instance_name]:
                input_signals[tracked_port_name] = self.component_inputs[instance_name][port_name]
            if port_name in self.component_outputs[instance_name]:
                output_signals[tracked_port_name] = self.component_outputs[instance_name][port_name]
        simulation_result = BlockModeSimulationResult(input_signals, output_signals)

        return simulation_result
    
    def _collect_component_inputs(self, component)->dict:
        inputs = {}
        input_components = [u for u, v in self._instantiated_circuit.graph.in_edges(component)]
        for input_component in input_components:
            input_edges = self._instantiated_circuit.graph.get_edge_data(input_component, component)
            for edge_number, edge in input_edges.items():
                inputs[edge['dst_port']] = self.component_outputs[input_component][edge['src_port']]
                pass
        self.component_inputs[component] = inputs
    
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
        
