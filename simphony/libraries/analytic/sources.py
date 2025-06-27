import jax
from jax.typing import ArrayLike

from simphony.circuit import SteadyStateComponent
from simphony.time_domain import BlockModeComponent, SampleModeComponent
from simphony.signals import electrical_signal


class CWLaser(SteadyStateComponent, SampleModeComponent, BlockModeComponent):
    def __init__(self):
        pass


class VoltageSource(
    SteadyStateComponent, 
    # SampleModeComponent, 
    BlockModeComponent,
):
    electrical_ports = ["e0"]

    def __init__(
        self, 
        **settings,
    ):
        super().__init__(**settings)
        # optical_ports = None
        # electrical_ports = ['e0']
        # logic_ports = None
        # super().__init__(
        #     optical_ports=optical_ports,
        #     electrical_ports=electrical_ports,
        #     logic_ports=logic_ports
        # )

    def steady_state(
        self, 
        inputs: dict, 
        **settings,
    ):
        outputs = {
            "e0": electrical_signal(voltage=settings['steady_state_voltage'], wl=settings['wl'])
        }
        return outputs

    def run(self, input_signal: ArrayLike, **kwargs):
        pass


class PRNG(
    SteadyStateComponent, 
    # SampleModeComponent, 
    BlockModeComponent
):
    logic_ports = ["l0"]

    def __init__(self, **settings):
        super().__init__(**settings)
        # optical_ports = None
        # electrical_ports = None
        # logic_ports = ['l0']
        # super().__init__(
        #     optical_ports=optical_ports,
        #     electrical_ports=electrical_ports,
        #     logic_ports=logic_ports
        # )
    
    @jax.jit
    def steady_state(self, inputs: dict, default_output: int=0):
        outputs = {
            "l0": default_output
        }
        return outputs

    def run(self, inputs: dict, **kwargs):
        pass
