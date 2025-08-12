import jax.numpy as jnp
from simphony.circuit import SampleModeComponent
from simphony.signals import SampleModeLogicSignal, SampleModeElectricalSignal, SampleModeOpticalSignal

class ElectricalTermination(SampleModeComponent):
    delay_compensation = 0
    electrical_ports = ["out"]
    
    def initial_state(self):
        return jnp.array([0])
    
    def sample_mode_step(self, inputs: dict, state, simulation_parameters):
        outputs = {
            'out': SampleModeOpticalSignal(
                amplitude = ...,
                wavelength = ...,
            )
        }
        return outputs, state
    
class LogicTermination(SampleModeComponent):
    delay_compensation = 0
    logic_ports = ["out"]
    
    def initial_state(self):
        return jnp.array([0])
    
    def step(self, inputs: dict, state):
        outputs = {
            'out': SampleModeLogicSignal(
                value = 0,
            )
        }
        return outputs, state
    
class OpticalTermination(SampleModeComponent):
    delay_compensation = 0
    optical_ports = ["out"]

    def sample_mode_initial_state(self, simulation_parameters):
        return jnp.array([0])
    
    def sample_mode_step(self, inputs: dict, state, simulation_parameters):
        outputs = {
            'out': SampleModeOpticalSignal(
                amplitude = jnp.array([[0.0+0.0j]], dtype=complex),
                wavelength = jnp.array([1.55e-6]),
            )
        }
        return outputs, state

def _termination(termination_type='optical'):
    if termination_type == 'optical':
        return OpticalTermination
    if termination_type == 'electrical':
        return ElectricalTermination
    if termination_type == 'logic':
        return LogicTermination