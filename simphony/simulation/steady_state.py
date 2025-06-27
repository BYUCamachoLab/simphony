from .simulation import Simulation, SimulationResult

class SteadyStateSimulationResult(SimulationResult):
    def __init__(self):
        ...

class SteadyStateSimulation(Simulation):
    def run(self)->SteadyStateSimulationResult:
        ...