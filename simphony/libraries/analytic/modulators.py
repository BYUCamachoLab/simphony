# class OpticalAmplitudeModulator():
#     pass
from simphony.circuit import SteadyStateComponent
from simphony.circuit import BlockModeComponent, SampleModeComponent
from jax.typing import ArrayLike
import jax
import jax.numpy as jnp
from typing import Callable
import sax

from simphony.signals import steady_state_optical_signal, steady_state_electrical_signal, complete_steady_state_inputs

class MachZehnderModulator(
    SteadyStateComponent, 
    # SampleModeComponent, 
    BlockModeComponent
):
    optical_ports = ["o0", "o1"]
    electrical_ports = ["e0", "e1"]

    
    # def __init__(self, **settings):
    #     super().__init__(**settings)

class OpticalModulator(
    SteadyStateComponent, 
    SampleModeComponent, 
    # BlockModeComponent
):
    optical_ports = ["o0", "o1"]
    electrical_ports = ["e0"]
    
    def __init__(
        self,
        *,
        # n_eff: Callable[[float, complex], float]=None, # Function of wavelength and voltage
        length: float = 1.0,
        operating_wl = 1.55e-6,
        absorption_coefficients: jnp.ndarray = jnp.asarray([0.0, 0.0, 0.0, 0.0]),
        phase_coefficients: jnp.ndarray = jnp.asarray([0.0, 0.0, jnp.pi, 0.0]),
        effective_index = 0.0,
    ):
        self.length = length
        self.absorption_coefficients = absorption_coefficients
        self.phase_coefficients = phase_coefficients
        self.operating_wl = operating_wl
        self.effective_index = effective_index

    def s_parameters(
        self,
        inputs: dict,
        wl: ArrayLike=1.55e-6,
    )->sax.SDict:    
        total_real_voltage = 0
        for v in inputs["e0"].voltage:
            total_real_voltage += jnp.real(v)

        phase_op = jnp.polyval(self.phase_coefficients, total_real_voltage)
        absorption_dB = jnp.polyval(self.absorption_coefficients, total_real_voltage)
        fraction_of_power_remaining = 10**(-absorption_dB*self.length/10)
        delta_n = self.operating_wl/(2*jnp.pi*self.length) * phase_op
        phase_shift = 2*jnp.pi/wl*(self.effective_index+delta_n)*self.length

        return {
            ("o0", "o1"): jnp.sqrt(fraction_of_power_remaining)*jnp.exp(1j*phase_shift),
            ("o1", "o0"): jnp.sqrt(fraction_of_power_remaining)*jnp.exp(1j*phase_shift),
            ("o0", "o0"): 0,
            ("o1", "o1"): 0,
        }

    def initial_state(self):
        return jnp.array([0])
    def step(self, inputs: dict,  state: jax.Array):
        # TODO: Complete this to acount for delay and phase shift
        return inputs, state
        

    # @staticmethod
    # @jax.jit
    def steady_state(
        self,
        inputs: dict,
        # settings
    ) -> dict:
        # TODO: Change complete_steady_state_inputs to be a method on the SteadyStateComponent
        # Base Class and have it give default values to ports with unspecified inputs
        # For now, I'll just use this work around.
        complete_steady_state_inputs(inputs)
        # self.s_parameters(inputs, jnp.linspace(1.5e-6, 1.6e-6, 1000))
        optical_wls = []
        if 'o0' in inputs:
            # Assuming they all have the same wl
            optical_wls = inputs["o0"].wl
        # if not 'o0' in inputs:    
            # inputs['o0'] = optical_signal(field=0)
        # if 'o1' not in inputs:
        #     inputs['o1'] = optical_signal(field=0)
        # We only consider DC voltage and assum
        total_real_voltage = 0
        for v in inputs["e0"].voltage:
            total_real_voltage += jnp.real(v)       

        o0_field_out = []
        o1_field_out = []
        for i, optical_wl in enumerate(optical_wls):
            o0_in = inputs["o0"].field[i]
            o1_in = inputs["o1"].field[i]
            
            phase_op = jnp.polyval(self.phase_coefficients, total_real_voltage)
            absorption_dB = jnp.polyval(self.absorption_coefficients, total_real_voltage)
            fraction_of_power_remaining = 10**(-absorption_dB*self.length/10)
            delta_n = self.operating_wl/(2*jnp.pi*self.length) * phase_op
            phase_shift = 2*jnp.pi/optical_wl * (self.effective_index+delta_n)* self.length

            o0_field_out.append(o1_in*jnp.sqrt(fraction_of_power_remaining)*jnp.exp(1j*phase_shift))
            o1_field_out.append(o0_in*jnp.sqrt(fraction_of_power_remaining)*jnp.exp(1j*phase_shift))

        outputs = {
            "o0": steady_state_optical_signal(field=o0_field_out, wl=optical_wls),
            "o1": steady_state_optical_signal(field=o1_field_out, wl=optical_wls),
            # "e0": electrical_signal(),
        }
        return outputs



# class AmplitudeModulator():
#     pass