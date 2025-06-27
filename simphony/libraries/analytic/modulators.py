# class OpticalAmplitudeModulator():
#     pass
from simphony.circuit import SteadyStateComponent
from simphony.time_domain import BlockModeComponent, SampleModeComponent
from jax.typing import ArrayLike
import jax
import jax.numpy as jnp
from typing import Callable

from simphony.signals import optical_signal, electrical_signal

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
        length: float = 1.0e-3,
        absorption_coefficients: jnp.ndarray = jnp.asarray([[-0.005, 0.1, -0.5, -0.01], 
                                                            [-0.002, 0.05, -0.25, -0.005]]),
        phase_coefficients: jnp.ndarray = jnp.asarray([[-0.1, -0.5, -0.8, 0.0], 
                                                       [-0.05, -0.2, -0.3, 0.0]]),
        wl: jnp.ndarray = jnp.asarray([1.55e-6, 1.56e-6]),
    ):
        self.length = length
        self.absorption_coefficients = absorption_coefficients
        self.phase_coefficients = phase_coefficients
        self.wl = wl
        pass

    @staticmethod
    def n_eff(voltage, wls):
        pass

    def s_parameters(
        self,
        inputs: dict,
        wl: ArrayLike,
    ):
        wls = wl
        num_opt_ports = len(self.optical_ports)
        S = jnp.zeros((len(wls), num_opt_ports, num_opt_ports), dtype=complex)
        self.n_eff()

    # @staticmethod
    # @jax.jit
    def steady_state(
        self,
        inputs: dict,
        # settings
    ) -> dict:
        self.s_parameters(inputs, jnp.linspace(1.5e-6, 1.6e-6, 1000))
        # ouputs = {
        #     "o0": optical_signal(
        #         jnp.exp(1j*phase_shift)*inputs["o1"].field,
        #         inputs["o1"].wl,
        #         inputs["o1"].polarization,
        #     ),
        #     "o1": optical_signal(
        #         jnp.exp(1j*phase_shift)*inputs["o0"].field,
        #         inputs["o0"].wl,
        #         inputs["o0"].polarization,
        #     ),
        #     "e0": electrical_signal(
        #         inputs["e0"].voltage,
        #         inputs["e0"].wl,
        #     ),
        # }
        return None



# class AmplitudeModulator():
#     pass