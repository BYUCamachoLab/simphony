from simphony.time_domain import BlockModeSystem, SampleModeSystem
from .component_types import OpticalComponent, ElectricalComponent
from jax.typing import ArrayLike

class CWLaser(BlockModeSystem, SampleModeSystem, OpticalComponent):
    def __init__(self):
        pass

class VoltageSource(BlockModeSystem, ElectricalComponent):
    def __init__(self):
        optical_ports = None
        electrical_ports = ['e0']
        logic_ports = None
        super().__init__(optical_ports, electrical_ports, logic_ports)
    
    def run(self, input_signal: ArrayLike, **kwargs):
        pass