from simphony.component.component import Component
from simphony.component.port import Port
from simphony.simulation.simulation import SimulationParameters

class ExternalPortPlaceholder(Component):
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