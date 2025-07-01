import jax
from jax.typing import ArrayLike

from simphony.circuit import SteadyStateComponent
from simphony.circuit import BlockModeComponent, SampleModeComponent
from simphony.signals import steady_state_electrical_signal, steady_state_optical_signal, steady_state_logic_signal
import jax.numpy as jnp
from typing import Union

class CWLaser(SteadyStateComponent, SampleModeComponent, BlockModeComponent):
    optical_ports = ["o0"]
    def __init__(self, steady_state_value = None, 
                 block_mode_value= None, 
                 sample_mode_value= None,
                 wl = 1550e-9,
                 t = None, 
                 polarization = None,
                 noise = False,
                 **kwargs):

        tau = kwargs.get("ramp_tau", 2e-12)         
        ramp = 1.0 - jnp.exp(-t / tau)              
        phase = kwargs.get("phase", 0.0)            
        self.sample_mode_signal = (
            sample_mode_value * ramp * jnp.exp(1j * phase)
        )
        self.block_mode_signal = (
        block_mode_value * ramp * jnp.exp(1j * phase)
        )
        self.wl = wl
        self.polarization = polarization
        if noise is True:
            mean = self.block_mode_signal.mean() if self.block_mode_signal is not None else 0.0
            raise NotImplementedError("Noise generation not implemented yet.")
        # extract or set defaults
        # tau       = kwargs.get("ramp_tau",      2e-9)   
        # tau_damp  = kwargs.get("damp_tau",      30e-9)   
        # f_rel     = kwargs.get("relax_freq",  2e9)      
        # A         = kwargs.get("relax_amp",    0.2)      
        # phase     = kwargs.get("phase",         0.0)     

        # # time-vector t already defined
        # exp_term  = 1.0 - jnp.exp(-t / tau)
        # osc_term  = A * jnp.exp(-t / tau_damp) * jnp.sin(2 * jnp.pi * f_rel * t)

        # ramp      = exp_term * (1 + osc_term)         # expo * (1 + damped sine)
        # complex_carrier = jnp.exp(1j * phase)

        # # block mode
        # self.block_mode_signal = block_mode_value * ramp * complex_carrier

        # # sample mode
        # self.sample_mode_signal = sample_mode_value * ramp * complex_carrier
            


    def steady_state(
        self, 
        inputs: dict,
    ):
        if self.steady_state_signal is None:
            raise ValueError("Steady state signal must be provided for CWLaser.")
        outputs = {
            "o0": steady_state_optical_signal(field = self.steady_state_signal, 
                                 wl = self.wl, 
                                 polarization = self.polarization)  
        }
        return outputs
    def response (
        self,
        inputs: dict,
    ):
        if self.block_mode_signal is None:
            raise ValueError("Block mode signal must be provided for CWLaser.")
        outputs = {
            "o0": steady_state_optical_signal(field = self.block_mode_signal, 
                                 wl = self.wl, 
                                 polarization = self.polarization),  
        }
        return outputs
    def step (
        self,
        inputs: dict,
    ):
        if self.sample_mode_signal is None:
            raise ValueError("Sample mode signal must be provided for CWLaser.")
        outputs = {
            "o0": steady_state_optical_signal(field = self.sample_mode_signal, 
                                 wl = self.wl, 
                                 polarization = self.polarization),  
        }
        return outputs
    


        
        
        
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
            "o0": steady_state_optical_signal(field = self.steady_state_signal, 
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
            "o0": steady_state_optical_signal(field = self.block_mode_signal, 
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
            "o0": steady_state_optical_signal(field = self.sample_mode_signal, 
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
            "e0": steady_state_electrical_signal(voltage=[self.steady_state_voltage], wl=[self.steady_state_wl])
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
