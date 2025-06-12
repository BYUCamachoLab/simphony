# class OpticalAmplitudeModulator():
#     pass
from simphony.circuit import SpectralSystem
from simphony.time_domain import BlockModeSystem, SampleModeSystem
from jax.typing import ArrayLike


class MachZehnderModulator(SpectralSystem, SampleModeSystem, BlockModeSystem):
    optical_ports = ["o0", "o1"]
    electrical_ports = ["e0", "e1"]
    def __init__(self):
        pass
        # optical_ports = ["o0", "o1"]
        # electrical_ports = ["e0", "e1"]
        # logic_ports = None
        # super().__init__(
        #     optical_ports=optical_ports, 
        #     electrical_ports=electrical_ports, 
        #     logic_ports=logic_ports
        # )
    def run(self, input_signal: ArrayLike, **kwargs):
        pass

class PhaseModulator(SpectralSystem, SampleModeSystem, BlockModeSystem):
    optical_ports = ["o0", "o1"]
    electrical_ports = ["e0"]
    def __init__(self):
        pass
        # optical_ports = ["o0", "o1"]
        # electrical_ports = ["e0"]
        # logic_ports = None
        # super().__init__(
        #     optical_ports=optical_ports, 
        #     electrical_ports=electrical_ports, 
        #     logic_ports=logic_ports
        # )
    
    def run(self, input_signal: ArrayLike, **kwargs):
        pass



# class AmplitudeModulator():
#     pass