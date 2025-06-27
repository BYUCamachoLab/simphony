from .simulation import Simulation, SimulationResult

class SampleModeSimulationResult(SimulationResult):
    def __init__(self):
        ...

class SampleModeSimulation(Simulation):
    def run(self)->SampleModeSimulationResult:
        ...
