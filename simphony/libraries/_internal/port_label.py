from simphony.component.placeholder import Placeholder
from simphony.component.port import Port


class PortLabel(Placeholder):
    def __init__(self, simulation_parameters, *, name=None, designator=None):
        self.name = name
        self.designator = designator


class DirectedPortLabel(PortLabel):
    ports = [
        Port(
            name="in",
            type="any",
            directionality="input",
        ),
        Port(
            name="out",
            type="any",
            directionality="output",
        ),
    ]


class BidirectionalPortLabel(PortLabel):
    ports = [
        Port(
            name="port1",
            type="any",
            directionality="bidirectional",
        ),
        Port(
            name="port2",
            type="any",
            directionality="bidirectional",
        ),
    ]
