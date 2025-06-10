# class OpticalAmplitudeModulator():
#     pass
from .component_types import OpticalComponent, ElectricalComponent
from simphony.time_domain import BlockModeSystem, SampleModeSystem
from jax.typing import ArrayLike
class MachZehnderModulator(BlockModeSystem, OpticalComponent, ElectricalComponent):
    def __init__(self):
        optical_ports = ["o0", "o1"]
        electrical_ports = ["e0", "e1"]
        super().__init__(optical_ports, electrical_ports)
    def run(self, input_signal: ArrayLike, **kwargs):
        pass

class PhaseModulator(BlockModeSystem, OpticalComponent, ElectricalComponent):
    def __init__(self):
        optical_ports = ["o0", "o1"]
        electrical_ports = ["e0"]
        super().__init__(optical_ports, electrical_ports)
    
    def run(self, input_signal: ArrayLike, **kwargs):
        pass



# class AmplitudeModulator():
#     pass