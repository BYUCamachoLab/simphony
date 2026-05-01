from simphony.component.component import Component
from simphony.component.placeholder import Placeholder
from simphony.component.port import Port
from simphony.simulation.simulation import SimulationParameters

class DirectedPortLabel(Placeholder):
    ports = [
        Port(
            name = "in",
            type = "any",
            directionality = "input",
        ),
        Port(
            name = "out",
            type = "any",
            directionality = "output",
        )
    ]

class BidirectionalPortLabel(Placeholder):
    ports = [
        Port(
            name = "in",
            type = "any",
            directionality = "bidirectional",
        ),
        Port(
            name = "out",
            type = "any",
            directionality = "bidirectional",
        ),
    ]