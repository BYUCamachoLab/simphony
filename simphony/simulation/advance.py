import jax.numpy as jnp
from simphony.circuit import SampleModeComponent

class ElectricalAdvance(SampleModeComponent):
    delay_compensation = 2
    electrical_ports = ["in", "out"]
    
    def initial_state(self):
        return jnp.array([0])
    
    def sample_mode_step(self, inputs: dict, state):
        # outputs = {
        #     'in': inputs['out'],
        #     'out': inputs['in'],
        # }

        return inputs, state
    
class LogicAdvance(SampleModeComponent):
    delay_compensation = 2
    logic_ports = ["in", "out"]
    
    def initial_state(self):
        return jnp.array([0])
    
    def step(self, inputs: dict, state):
        # outputs = {
        #     'in': inputs['out'],
        #     'out': inputs['in'],
        # }

        return inputs, state
    
class OpticalAdvance(SampleModeComponent):
    delay_compensation = 2
    optical_ports = ["in", "out"]

    def sample_mode_initial_state(self, simulation_parameters):
        return jnp.array([0])
    
    def sample_mode_step(self, inputs: dict, state, simulation_parameters):
        outputs = {
            'in': inputs['out'],
            'out': inputs['in'],
        }
        return outputs, state

def _advance(advance_type='optical'):
    if advance_type == 'optical':
        return OpticalAdvance
    if advance_type == 'electrical':
        return ElectricalAdvance
    if advance_type == 'logic':
        return LogicAdvance