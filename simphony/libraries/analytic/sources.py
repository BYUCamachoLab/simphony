from simphony.time_domain import BlockModeSystem, SampleModeSystem
from .component_types import OpticalComponent, ElectricalComponent


class CWLaser(BlockModeSystem, SampleModeSystem, OpticalComponent):
    def __init__(self):
        pass

class VoltageSource(ElectricalComponent):
    def __init__(self):
        pass