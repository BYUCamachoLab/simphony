from simphony.component.component import Component
from simphony.component.placeholder import Placeholder
from simphony.component.port import Port
from simphony.simulation.simulation import SimulationParameters

class ExternalPortPlaceholder(Placeholder):
    ports = [
        Port(
            name = "_0",
            type = "any",
            directionality = "bidirectional",
        )
    ]
    
    def __init__(
        self,
        simulation_parameters: SimulationParameters,
    ):
        pass