# class OpticalAmplitudeModulator():
#     pass
from simphony.circuit import SpectralSystem
from simphony.time_domain import BlockModeSystem, SampleModeSystem
from jax.typing import ArrayLike
import jax
import jax.numpy as jnp

from simphony.signals import optical_signal, electrical_signal

class MachZehnderModulator(
    SpectralSystem, 
    # SampleModeSystem, 
    BlockModeSystem
):
    optical_ports = ["o0", "o1"]
    electrical_ports = ["e0", "e1"]
    
    def __init__(self, **settings):
        super().__init__(**settings)

class PhaseModulator(
    SpectralSystem, 
    # SampleModeSystem, 
    # BlockModeSystem
):
    optical_ports = ["o0", "o1"]
    electrical_ports = ["e0"]
    
    def __init__(self, **settings):
        super().__init__(**settings)

    @staticmethod
    @jax.jit
    def steady_state(
        self, 
        inputs: dict,
        *,
        half_wave_voltage: float = 1.0
    ) -> dict:
        phase_shift = jnp.pi * inputs["e0"].voltage / half_wave_voltage
        ouputs = {
            "o0": optical_signal(
                jnp.exp(1j*phase_shift)*inputs["o1"].field,
                inputs["o1"].wl,
                inputs["o1"].polarization,
            ),
            "o1": optical_signal(
                jnp.exp(1j*phase_shift)*inputs["o0"].field,
                inputs["o0"].wl,
                inputs["o0"].polarization,
            ),
            "e0": electrical_signal(
                inputs["e0"].field,
                inputs["e0"].wl,
            ),
        }
        return ouputs



# class AmplitudeModulator():
#     pass