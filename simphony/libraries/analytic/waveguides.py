from simphony.time_domain import BlockModeSystem, SSFM
from .component_types import OpticalComponent, ElectricalComponent
from jax.typing import ArrayLike

class Waveguide(BlockModeSystem, OpticalComponent):
    def __init__(self):
        optical_ports = ["o0","o1"]
        electrical_ports = None
        logic_ports = None
        super().__init__(
            optical_ports=optical_ports, 
            electrical_ports=electrical_ports, 
            logic_ports=logic_ports
        )

    def run(self, input_signal: ArrayLike, **kwargs):
        dt = kwargs.get('dt', None)
        carrier_freq = kwargs.get('carrier_freq', None)
        SSFM(input_signal, dt, carrier_freq)

class Fiber(OpticalComponent):
    pass

class GRINFiber(OpticalComponent):
    pass