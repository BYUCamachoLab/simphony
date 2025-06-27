from .simulation import Simulation, SimulationResult

class BlockModeSimulationResult(SimulationResult):
    def __init__(self):
        ...

class BlockModeSimulation(Simulation):
    def run(self)->BlockModeSimulationResult:
        ...