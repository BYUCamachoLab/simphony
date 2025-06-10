from simphony.time_domain import BlockModeSystem
from .component_types import OpticalComponent, ElectricalComponent
from jax.typing import ArrayLike

class Waveguide(BlockModeSystem, OpticalComponent):
    def __init__(self):
        optical_ports = ["o0","o1"]
        electrical_ports = None
        super().__init__(optical_ports, electrical_ports)

    def run(self, input_signal: ArrayLike, **kwargs):
        pass

class Fiber(OpticalComponent):
    pass

class GRINFiber(OpticalComponent):
    pass