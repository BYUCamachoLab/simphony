# class OpticalAmplitudeModulator():
#     pass
from simphony.circuit import OpticalComponent, ElectricalComponent
from simphony.time_domain import BlockModeSystem, SampleModeSystem
from jax.typing import ArrayLike
class MachZehnderModulator(BlockModeSystem, OpticalComponent, ElectricalComponent):
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

class PhaseModulator(BlockModeSystem, OpticalComponent, ElectricalComponent):
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