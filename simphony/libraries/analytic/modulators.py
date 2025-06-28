# class OpticalAmplitudeModulator():
#     pass
from simphony.circuit import SteadyStateComponent
from simphony.circuit import BlockModeComponent, SampleModeComponent
from jax.typing import ArrayLike
import jax
import jax.numpy as jnp
from typing import Callable

from simphony.signals import optical_signal, electrical_signal, complete_steady_state_inputs

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
    # SampleModeComponent, 
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
        phase_coefficients: jnp.ndarray = jnp.asarray([0.0, 0.0, 0.0, jnp.pi]),
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
        wl: ArrayLike,
    ):
        wls = wl
        num_opt_ports = len(self.optical_ports)
        S = jnp.zeros((len(wls), num_opt_ports, num_opt_ports), dtype=complex)
        # self.n_eff()

    # @staticmethod
    # @jax.jit
    def steady_state(
        self,
        inputs: dict,
        # settings
    ) -> dict:
        complete_steady_state_inputs(inputs)
        # self.s_parameters(inputs, jnp.linspace(1.5e-6, 1.6e-6, 1000))
        
        # We only consider DC voltage and assum
        total_real_voltage = 0
        for v in inputs["e0"].voltage:
            total_real_voltage += jnp.real(v)
        
        # Assuming they all have the same wl
        optical_wls = inputs["o0"].wl

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
            "o0": optical_signal(field=o0_field_out, wl=optical_wls),
            "o1": optical_signal(field=o1_field_out, wl=optical_wls),
            # "e0": electrical_signal(),
        }
        return outputs



# class AmplitudeModulator():
#     pass