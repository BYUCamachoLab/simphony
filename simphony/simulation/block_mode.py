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
        input_components = nx.ancestors(self.circuit.graph, component)
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
        
        # self.block_mode_order = self._determine_block_mode_order()


        # (self.all_components,
        #  self.electrical_components,
        #  self.optical_components,
        #  self.logic_components) = identify_component_types(self.circuit.graph)

    def run(
        self,
    )->BlockModeSimulationResult:
        self._add_directionality_setting_to_s_parameter_components(self.flat_circuit, self.settings)
        instantiated_circuit = self.flat_circuit.instantiate(self.settings, self.simulation_parameters)
        instantiated_circuit.display()
        simulation_result = BlockModeSimulationResult(instantiated_circuit)

        return simulation_result


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
        