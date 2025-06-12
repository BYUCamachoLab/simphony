from jax.typing import ArrayLike

from simphony.circuit import SpectralSystem
from simphony.time_domain import BlockModeSystem, SampleModeSystem


class CWLaser(SpectralSystem, SampleModeSystem, BlockModeSystem):
    def __init__(self):
        pass


class VoltageSource(SpectralSystem, SampleModeSystem, BlockModeSystem):
    electrical_ports = ["e0"]

    def __init__(self):
        pass
        # optical_ports = None
        # electrical_ports = ['e0']
        # logic_ports = None
        # super().__init__(
        #     optical_ports=optical_ports,
        #     electrical_ports=electrical_ports,
        #     logic_ports=logic_ports
        # )

    def run(self, input_signal: ArrayLike, **kwargs):
        pass


class PRNG(SpectralSystem, SampleModeSystem, BlockModeSystem):
    logic_ports = ["l0"]

    def __init__(self):
        pass
        # optical_ports = None
        # electrical_ports = None
        # logic_ports = ['l0']
        # super().__init__(
        #     optical_ports=optical_ports,
        #     electrical_ports=electrical_ports,
        #     logic_ports=logic_ports
        # )

    def run(self, input_signal: ArrayLike, **kwargs):
        pass
