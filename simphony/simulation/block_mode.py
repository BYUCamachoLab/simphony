from .simulation import Simulation, SimulationResult
from simphony.circuit import Circuit
import networkx as nx
from copy import deepcopy


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
    def __init__(self, circuit: Circuit,ports = None):
        self.circuit = circuit
        if ports is None:
            self.ports = self.circuit.netlist['ports']
        self.block_mode_order = self._determine_block_mode_order()

        # (self.all_components,
        #  self.electrical_components,
        #  self.optical_components,
        #  self.logic_components) = identify_component_types(self.circuit.graph)

    def run(self, t, input_signals: dict, **kwargs )->BlockModeSimulationResult:
        self.dt = kwargs["dt"]
        simulation_result = BlockModeSimulationResult(self.circuit)
        self._instantiate_components(kwargs.get("settings", {}))

        # tried something here but haven't tested it yet
        for component in self.block_mode_order:
            simulation_result._collect_component_inputs(component)   
            inputs = simulation_result.component_inputs[component]
            outputs = self.components[component].response(inputs)
            simulation_result.component_outputs[component] = outputs

        return simulation_result
        
    def _determine_block_mode_order(self):
        """
        Determine the order of components in block mode simulation.
        """
        graph = self.circuit.graph.copy()
        try:
            return list(nx.topological_sort(graph))
        except nx.NetworkXUnfeasible:
            raise ValueError("Failed to determine block order – circular dependencies detected")
        