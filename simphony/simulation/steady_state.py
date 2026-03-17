from .simulation import Simulation, SimulationResult
from simphony.circuit.circuit import Circuit
import networkx as nx
from copy import deepcopy
from .simulation import Simulation, SimulationResult, SimulationParameters
from flax import struct

@struct.dataclass
class SteadyStateSimulationParameters(SimulationParameters):
    pass

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
    def __init__(
            self,         
            circuit: Circuit, 
            settings,
            simulation_parameters,
            ports=None,          
        ):
        self.circuit = circuit
        self.flat_circuit = circuit.flatten()
        self.settings = settings
        self.simulation_parameters = simulation_parameters
        
        if ports is None:
            ports = self.flat_circuit.netlist['ports']
        
        self.ports = ports

    def run(
        self, 
        # settings:dict = None
    ) -> SteadyStateSimulationResult:
        
        instantiated_circuit = self.circuit.instantiate(self.settings, self.simulation_parameters)
        simulation_result = SteadyStateSimulationResult(instantiated_circuit)
        
        self.steady_state_order = self._determine_steady_state_order_nx_method(instantiated_circuit)
        # self._instantiate_components(self.settings)
        for instance_name in self.steady_state_order:
            simulation_result._collect_component_inputs(instance_name)   
            inputs = simulation_result.component_inputs[instance_name]
            component = instantiated_circuit.instantiated_flat_netlist['instances'][instance_name]['model']
            outputs = component.steady_state(inputs, self.simulation_parameters)
            simulation_result.component_outputs[instance_name] = outputs
        
        return simulation_result
    
    # def _determine_steady_state_order(self, instantiated_circuit):
    #     """
    #     Voltage signals at electrical ports are assumed to be constant
    #     for SParameterSimulations, but they are not known a priori, unless
    #     the voltage source is not dependent on an input signal.

    #     Since steady-state connections are assumemd to be uni-directional, this function is
    #     able to find the order in which electrical component voltages must
    #     be calculated to find the proper steady state.
    #     """
    #     steady_state_order = []
    #     graph = instantiated_circuit.graph.copy()
    #     # graph.remove_nodes_from(self.s_parameter_graph.nodes)
    #     while graph.number_of_nodes() > 0:
    #         root_nodes = [n for n in graph.nodes if graph.in_degree(n)==0]
    #         if len(root_nodes) == 0:
    #             break
    #         steady_state_order += root_nodes
    #         graph.remove_nodes_from(root_nodes)

    #     if graph.number_of_nodes() > 0:
    #         raise ValueError(
    #             "Failed to determine steady state order. " \
    #             "Hint: Steady state cannot be determined for circular connections."
    #         )
    #     return steady_state_order
    
    #Matthew's Suggestions
    #Found this method while looking into the determine_steady_state algorithm.
    #Thought this might look cleaner then the custom implementation above and after a bit of testing
    #it appears to be exactly the same as the custom implementation. Though this depends
    #if we need a custom implementation depending on the circuit structure.
    def _determine_steady_state_order_nx_method(self, instantiated_circuit):
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



