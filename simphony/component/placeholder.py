from simphony.component.component import Component
from simphony.simulation.simulation import SimulationParameters


class Placeholder(Component):
    def __init__(
        self,
        simulation_mode: SimulationParameters,
        **kwargs,
    ):
        ...