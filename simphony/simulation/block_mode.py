from simphony.simulation.simulation import Simulation, SimulationResult, SimulationParameters, SimulationMode
from simphony.circuit.circuit import Circuit
import networkx as nx
from copy import deepcopy
from flax import struct
from dataclasses import field

from simphony.libraries.ideal.s_parameters import SParameterSax

@struct.dataclass
class BlockModeSimulationParameters(SimulationParameters):
    simulation_mode: SimulationMode = field(default_factory=lambda:SimulationMode.BLOCK_MODE)    
    directed: bool = True
    dt = 1e-14
    num_time_steps = 1000
    spectral_range = (1.5e-6, 1.6e-6)
    center_wavelength = 1.55e-6
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
    
    def _collect_component_inputs(self, component)->dict:
        inputs = {}
        # TODO: Determine if STEADYSTATESIMULATION needs this change too
        # input_components = nx.ancestors(self.circuit.graph, component)
        input_components = immediate_ancestors = [u for u, v in self.circuit.graph.in_edges(component)]
        for input_component in input_components:
            print(input_component)
            input_edges = self.circuit.graph.get_edge_data(input_component, component)
            for edge_number, edge in input_edges.items():
                inputs[edge['dst_port']] = self.component_outputs[input_component][edge['src_port']]
                pass
        
        self.component_inputs[component] = inputs

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


        self.block_mode_order = self._determine_block_mode_order_nx_method(instantiated_circuit)
        # self._instantiate_components(self.settings)
        for instance_name in self.block_mode_order:
            simulation_result._collect_component_inputs(instance_name)   
            inputs = simulation_result.component_inputs[instance_name]
            component = instantiated_circuit.instantiated_flat_netlist['instances'][instance_name]['model']
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
                if not "sax_settings" in settings[instance_name].keys():
                    settings[instance_name] = {"sax_settings": settings[instance_name]}
            
                settings[instance_name]
                # TODO: DOUBLE CHECK DEFAULT DICTIONARY CONSTRUCTION FOR EDGE CASES
                default_directionalities = {p.name:"output" if f"{instance_name},{p.name}" in netlist['connections'].keys() else "input" for p in component_class.ports}
                default_directionalities = {k:d if not (f"{instance_name},{k}" in netlist['connections'].keys() or not f"{instance_name},{k}" in netlist['connections'].values()) else "output" for k,d in default_directionalities.items()}
                settings[instance_name]["port_directionality"] = default_directionalities | settings[instance_name].get("port_directionality", {})
    # def _determine_block_mode_order(self):
    #     """
    #     Determine the order of components in block mode simulation.
    #     """
    #     graph = self.circuit.graph.copy()
    #     try:
    #         return list(nx.topological_sort(graph))
    #     except nx.NetworkXUnfeasible:
    #         raise ValueError("Failed to determine block order – circular dependencies detected")
        