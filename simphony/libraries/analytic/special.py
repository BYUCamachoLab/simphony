from simphony.circuit import SampleModeComponent
import jax.numpy as jnp

class ElectricalAdvance(SampleModeComponent):
    delay_compensation = 1
    electrical_ports = ["in", "out"]
    
    def initial_state(self):
        return jnp.array([0])
    
    def step(self, inputs: dict, state):
        # outputs = {
        #     'in': inputs['out'],
        #     'out': inputs['in'],
        # }

        return inputs, state
    
class LogicAdvance(SampleModeComponent):
    delay_compensation = 1
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
    delay_compensation = 1
    optical_ports = ["in", "out"]

    def initial_state(self):
        return jnp.array([0])
    
    def step(self, inputs: dict, state):
        # outputs = {
        #     'in': inputs['out'],
        #     'out': inputs['in'],
        # }
        return inputs, state

def advance(delay_compensation=1, advance_type='optical'):
    if advance_type == 'optical':
        return OpticalAdvance
    if advance_type == 'electrical':
        return ElectricalAdvance
    if advance_type == 'logic':
        return LogicAdvance