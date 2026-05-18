from simphony.component.component import Component
from simphony.simulation.simulation import SimulationParameters


class Placeholder(Component):
    """Base class for temporary components replaced during instantiation.

    Placeholders are not meant to perform simulation work directly. They let a
    circuit carry model metadata, such as a raw SAX callable and its settings,
    until the active simulator knows how that model should be expanded. For
    example, S-parameter placeholders can later become vector-fitted Block mode
    components or frequency-domain S-parameter elements.
    """
    def __init__(
        self,
        simulation_mode: SimulationParameters,
        **kwargs,
    ):
        ...
