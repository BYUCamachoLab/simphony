import jax.numpy as jnp
from simphony.component.component import Component, SampleModeComponent
from simphony.signal.sample_mode import SampleModeLogicSignal, SampleModeElectricalSignal, SampleModeOpticalSignal
from simphony.component.port import Port

class Terminator(Component):
    ports = [
        Port(
            name = "out",
            type = "any",
            directionality = "output",
        )
    ]
    def __init__(
        self, 
        simulation_parameters, 
        **kwargs,
    ):
        pass

class ElectricalTerminator(Terminator):
    ports = [
        Port(
            name = "out",
            type = "electrical",
            directionality = "output",
        )
    ]
    # delay_compensation = 0
    # electrical_ports = ["out"]
    
    # def initial_state(self):
    #     return jnp.array([0])
    
    # def sample_mode_step(self, inputs: dict, state, simulation_parameters):
    #     outputs = {
    #         'out': SampleModeOpticalSignal(
    #             amplitude = ...,
    #             wavelength = ...,
    #         )
    #     }
    #     return outputs, state
    
class LogicTerminator(Terminator):
    ports = [
        Port(
            name = "out",
            type = "logic",
            directionality = "output",
        )
    ]
    # delay_compensation = 0
    # logic_ports = ["out"]
    
    # def initial_state(self):
    #     return jnp.array([0])
    
    # def step(self, inputs: dict, state):
    #     outputs = {
    #         'out': SampleModeLogicSignal(
    #             value = 0,
    #         )
    #     }
    #     return outputs, state
    
class OpticalTerminator(Terminator):
    ports = [
        Port(
            name = "out",
            type = "optical",
            directionality = "output",
        )
    ]