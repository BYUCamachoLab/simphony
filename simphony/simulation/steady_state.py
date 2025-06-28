from .simulation import Simulation, SimulationResult
from simphony.circuit import Circuit

class SteadyStateSimulationResult(SimulationResult):
    def __init__(self):
        ...

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

    def run(self)->SteadyStateSimulationResult:
        ...