from simphony.time_domain import BlockModeSystem, SampleModeSystem
from simphony.circuit import OpticalComponent, ElectricalComponent, LogicComponent
from jax.typing import ArrayLike

class CWLaser(BlockModeSystem, SampleModeSystem, OpticalComponent):
    def __init__(self):
        pass

class VoltageSource(BlockModeSystem, ElectricalComponent):
    electrical_ports = ['e0']
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

class PRNG(BlockModeSystem, LogicComponent):
    logic_ports = ['l0']
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