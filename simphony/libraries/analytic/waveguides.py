from jax.typing import ArrayLike

from simphony.circuit import SpectralSystem
from simphony.time_domain import SSFM, BlockModeSystem


class Waveguide(SpectralSystem, BlockModeSystem):
    optical_ports = ["o0", "o1"]

    def __init__(self):
        pass
        # optical_ports = ["o0","o1"]
        # electrical_ports = None
        # logic_ports = None
        # super().__init__(
        #     optical_ports=optical_ports,
        #     electrical_ports=electrical_ports,
        #     logic_ports=logic_ports
        # )

    def run(self, input_signal: ArrayLike, **kwargs):
        dt = kwargs.get("dt", None)
        carrier_freq = kwargs.get("carrier_freq", None)
        SSFM(input_signal, dt, carrier_freq)


class Fiber(SpectralSystem, BlockModeSystem):
    pass


class GRINFiber(SpectralSystem, BlockModeSystem):
    pass
