from jax.typing import ArrayLike

from simphony.circuit import SteadyStateComponent, BlockModeComponent
from simphony.time_domain import SSFM


class Waveguide(SteadyStateComponent, BlockModeComponent):
    optical_ports = ["o0", "o1"]

    def __init__(self, length=0.0):
        self.length = length
        # super().__init__(**settings)
        # optical_ports = ["o0","o1"]
        # electrical_ports = None
        # logic_ports = None
        # super().__init__(
        #     optical_ports=optical_ports,
        #     electrical_ports=electrical_ports,
        #     logic_ports=logic_ports
        # )

    def run(self, input_signal: ArrayLike, **simulation_parameters):
        dt = simulation_parameters.get("dt", None)
        # carrier_freq = simulation_parameters.get("carrier_freq", None)
        # SSFM(input_signal, dt, carrier_freq)


class Fiber(SteadyStateComponent, BlockModeComponent):
    pass


class GRINFiber(SteadyStateComponent, BlockModeComponent):
    pass
