from typing import Type

from simphony.circuit.components import SteadyStateComponent
from simphony.circuit.components import BlockModeComponent, SampleModeComponent

def star_coupler(num_in: int, num_out: int) -> Type:
    """
    Component Factory
    """
    class_name = f"StarCoupler{num_in}x{num_out}"
    in_ports = [f"o{i}" for i in range(num_in)]
    out_ports = [f"o{i}" for i in range(num_in, num_in + num_out)]

    attr = {"optical_ports": in_ports + out_ports}

    return type(class_name, (SteadyStateComponent, SampleModeComponent, BlockModeComponent), attr)
