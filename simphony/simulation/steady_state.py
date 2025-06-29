from .simulation import Simulation, SimulationResult
from simphony.circuit import Circuit
import networkx as nx
from copy import deepcopy

class SteadyStateSimulationResult(SimulationResult):
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
    # def add_outputs(self, component, outputs):
    #     self.component_outputs[component]=outputs

class SteadyStateSimulation(Simulation):
    def __init__(self, circuit: Circuit):
        self.circuit = circuit
        self.steady_state_order = self._determine_steady_state_order()
    
    def _determine_steady_state_order(self):
        """
        Voltage signals at electrical ports are assumed to be constant
        for SParameterSimulations, but they are not known a priori, unless
        the voltage source is not dependent on an input signal.

        Since steady-state connections are assumemd to be uni-directional, this function is
        able to find the order in which electrical component voltages must
        be calculated to find the proper steady state.
        """
        steady_state_order = []
        graph = self.circuit.graph.copy()
        # graph.remove_nodes_from(self.s_parameter_graph.nodes)
        while graph.number_of_nodes() > 0:
            root_nodes = [n for n in graph.nodes if graph.in_degree(n)==0]
            if len(root_nodes) == 0:
                break
            steady_state_order += root_nodes
            graph.remove_nodes_from(root_nodes)

        if graph.number_of_nodes() > 0:
            raise ValueError(
                "Failed to determine steady state order. " \
                "Hint: Steady state cannot be determined for circular connections."
            )
        return steady_state_order

    def run(
        self, 
        settings:dict = None
    ) -> SteadyStateSimulationResult:
        simulation_result = SteadyStateSimulationResult(self.circuit)

        self._instantiate_components(settings)
        for component in self.steady_state_order:
            simulation_result._collect_component_inputs(component)   
            inputs = simulation_result.component_inputs[component]
            outputs = self.components[component].steady_state(inputs)
            simulation_result.component_outputs[component] = outputs
        
        return simulation_result

