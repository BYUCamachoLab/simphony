from jax.typing import ArrayLike

from simphony.circuit.circuit import SteadyStateComponent, BlockModeComponent
from simphony.time_domain import SSFM


class Waveguide(SteadyStateComponent, BlockModeComponent):
    pass


class Fiber(SteadyStateComponent, BlockModeComponent):
    pass


class GRINFiber(SteadyStateComponent, BlockModeComponent):
    pass
