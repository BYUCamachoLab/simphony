import jax
from jax.typing import ArrayLike

from simphony.circuit import SteadyStateComponent
from simphony.circuit import BlockModeComponent, SampleModeComponent
from simphony.signals import electrical_signal, optical_signal, logic_signal
import jax.numpy as jnp
from typing import Union

class OpticalSource(SteadyStateComponent, SampleModeComponent, BlockModeComponent):
    optical_ports = ["o0"]

    def __init__(self, 
                 steady_state_signal = None, 
                 block_mode_signal= None, 
                 sample_mode_signal= None,
                 wl = 1550e-9, 
                 polarization = None,
                 **kwargs):
        
        self.steady_state_signal =  steady_state_signal
        self.block_mode_signal = block_mode_signal
        self.sample_mode_signal = sample_mode_signal
        self.wl = wl
        self.polarization = polarization
        
    def steady_state(
        self, 
        inputs: dict,
    ):
        if self.steady_state_signal is None:
            raise ValueError("Steady state signal must be provided for OpticalSource.")
        outputs = {
            "o0": optical_signal(field = self.steady_state_signal, 
                                 wl = self.wl, 
                                 polarization = self.polarization)  
        }
        return outputs
    
    def response (
        self, 
        inputs: dict,
    ):
        if self.block_mode_signal is None:
            raise ValueError("Block mode signal must be provided for OpticalSource.")
        outputs = {
            "o0": optical_signal(field = self.block_mode_signal, 
                                 wl = self.wl, 
                                 polarization = self.polarization),  
        }
        return outputs
    
    def step (
        self, 
        inputs: dict,
    ):
        if self.sample_mode_signal is None:
            raise ValueError("Sample mode signal must be provided for OpticalSource.")
        outputs = {
            "o0": optical_signal(field = self.sample_mode_signal, 
                                 wl = self.wl, 
                                 polarization = self.polarization),  
        }
        return outputs
    
        


class VoltageSource(
    SteadyStateComponent, 
    # SampleModeComponent, 
    BlockModeComponent,
):
    electrical_ports = ["e0"]

    def __init__(
        self, 
        steady_state_voltage=1.0,
        steady_state_wl=0,
    ):
        self.steady_state_voltage=steady_state_voltage
        self.steady_state_wl = steady_state_wl
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
    ):
        outputs = {
            "e0": electrical_signal(voltage=[self.steady_state_voltage], wl=[self.steady_state_wl])
        }
        return outputs

    def response(self, input_signal: ArrayLike, **kwargs):
        pass


class PRNG(
    SteadyStateComponent, 
    # SampleModeComponent, 
    BlockModeComponent
):
    logic_ports = ["l0"]

    def __init__(self, **settings):
        pass
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
