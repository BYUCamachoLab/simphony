# class OpticalAmplitudeModulator():
#     pass
from simphony.component.component import SteadyStateComponent, BlockModeComponent, SampleModeComponent, SParameterComponent
from simphony.component.pcell import PCell

from simphony.signal.block_mode import BlockModeOpticalSignal

from jax.typing import ArrayLike
import jax
import jax.numpy as jnp
from typing import Callable
import sax

from simphony.signal.steady_state import SteadyStateOpticalSignal
from simphony.component.port import Port
from simphony.simulation.simulation import SimulationParameters

class DirectedOpticalModulator(
    BlockModeComponent
):
    """
    Phase coefficients can be set for each mode in mode identifiers

    It is assumed that all wavelengths are close together, thus, phase shifts across wavelength 
    are approximately equal
    """
    ports = [
        Port(
            name = "o0",
            type = "optical",
            directionality = "input",
        ),
        Port(
            name = "o1",
            type = "optical",
            directionality = "output",
        ),
        Port(
            name = "e0",
            type = "electrical",
            directionality = "input",
        )
    ]
    
    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        *,
        # n_eff: Callable[[float, complex], float]=None, # Function of wavelength and voltage
        length: float = 1.0,
        operating_wl = 1.55e-6,
        absorption_coefficients: jnp.ndarray = jnp.asarray([0.0, 0.0, 0.0, 0.0]),
        phase_coefficients: jnp.ndarray = jnp.asarray([0.0, 0.0, jnp.pi, 0.0]),
        effective_index = 0.0,
    ):
        self.length = length
        self.absorption_coefficients = jnp.atleast_2d(absorption_coefficients)
        self.phase_coefficients = jnp.atleast_2d(phase_coefficients)
        self.operating_wl = operating_wl
        self.effective_index = effective_index
    
    def block_mode_response(self, input_signals, simulation_parameters):
        outputs = {}
        input_amplitude = input_signals["o0"].amplitude
        wavelengths = input_signals["o0"].wavelength
        N = input_amplitude.shape[0]
        L = input_amplitude.shape[1]
        M = len(simulation_parameters.mode_identifiers) # Currently, ignores all but the first mode
        
        
        voltage = input_signals["e0"].voltage

        output_amplitude = jnp.zeros((N, L, M), dtype=complex)
        
        for m, mode in enumerate(simulation_parameters.mode_identifiers):
            phase_op = jnp.polyval(self.phase_coefficients[m], voltage)
            absorption_dB = jnp.polyval(self.absorption_coefficients[m], voltage)
            fraction_of_power_remaining = 10**(-absorption_dB*self.length/10)
            fraction_of_power_remaining = jnp.repeat(fraction_of_power_remaining[:, None], L, axis=1)
            phase_shift = jnp.repeat(phase_op[:, None], L, axis=1)

            output_amplitude = output_amplitude.at[:, :, m].set(
                jnp.sqrt(fraction_of_power_remaining)
                * jnp.exp(1j * phase_shift)
                * input_amplitude[:, :, m]
            )

        outputs["o1"] = BlockModeOpticalSignal(amplitude=output_amplitude, wavelength=wavelengths)

        return outputs
    



class OpticalModulator(
    SParameterComponent,
    SteadyStateComponent, 
    SampleModeComponent, 
    # BlockModeComponent
):
    """
    Single Mode Optical Modulator
    TODO: Define a nice way to make this multimodal
    """
    ports = [
        Port(
            name = "o0",
            type = "optical",
            directionality = "bidirectional",
            # directionality = "input",
            # directionality = "unknown",
        ),
        Port(
            name = "o1",
            type = "optical",
            directionality = "bidirectional",
            # directionality = "output",
            # directionality = "unknown",
        ),
        Port(
            name = "e0",
            type = "electrical",
            directionality = "input",
        )
    ]
    
    def __init__(
        self,
        simulation_parameters: SimulationParameters,
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
        voltage = inputs["e0"].voltage

        phase_op = jnp.polyval(self.phase_coefficients, voltage)
        absorption_dB = jnp.polyval(self.absorption_coefficients, voltage)
        fraction_of_power_remaining = 10**(-absorption_dB*self.length/10)
        phase_shift = phase_op
        # delta_n = self.operating_wl/(2*jnp.pi*self.length) * phase_op
        # phase_shift = 2*jnp.pi/wl*(self.effective_index+delta_n)*self.length

        # TODO: Make multimodal
        return {
            ("o0", "o1"): jnp.sqrt(fraction_of_power_remaining)*jnp.exp(1j*phase_shift),
            ("o1", "o0"): jnp.sqrt(fraction_of_power_remaining)*jnp.exp(1j*phase_shift),
            ("o0", "o0"): 0,
            ("o1", "o1"): 0,
        }

    def s_parameter_get_bias_ports(
        self,
    ):
        return ["e0"]

    def sample_mode_initial_state(self, simulation_parameters):
        return jnp.array([0])

    def sample_mode_step(self, inputs: dict, state: jax.Array, simulation_state, simulation_parameters):
        from simphony.signal.sample_mode import SampleModeOpticalSignal
        baseband_wls = simulation_parameters.optical_baseband_wavelengths
        n_modes      = len(simulation_parameters.mode_identifiers)
        zero_amp     = jnp.zeros((baseband_wls.shape[0], n_modes), dtype=complex)

        voltage       = inputs["e0"].voltage if "e0" in inputs else 0.0
        phase_op      = jnp.polyval(self.phase_coefficients, voltage)
        absorption_dB = jnp.polyval(self.absorption_coefficients, voltage)
        transfer      = jnp.sqrt(10 ** (-absorption_dB * self.length / 10)) * jnp.exp(1j * phase_op)

        o0_in = inputs["o0"].amplitude if "o0" in inputs else zero_amp
        o1_in = inputs["o1"].amplitude if "o1" in inputs else zero_amp

        outputs = {
            "o1": SampleModeOpticalSignal(amplitude=transfer * o0_in, wavelength=baseband_wls),
            "o0": SampleModeOpticalSignal(amplitude=transfer * o1_in, wavelength=baseband_wls),
        }
        return outputs, state
        
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
        # complete_steady_state_inputs(inputs)
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
        voltage = inputs["e0"].voltage

        o0_field_out = []
        o1_field_out = []
        for i, optical_wl in enumerate(optical_wls):
            o0_in = inputs["o0"].field[i]
            o1_in = inputs["o1"].field[i]
            
            phase_op = jnp.polyval(self.phase_coefficients, voltage)
            absorption_dB = jnp.polyval(self.absorption_coefficients, voltage)
            fraction_of_power_remaining = 10**(-absorption_dB*self.length/10)
            delta_n = self.operating_wl/(2*jnp.pi*self.length) * phase_op
            phase_shift = 2*jnp.pi/optical_wl * (self.effective_index+delta_n)* self.length

            o0_field_out.append(o1_in*jnp.sqrt(fraction_of_power_remaining)*jnp.exp(1j*phase_shift))
            o1_field_out.append(o0_in*jnp.sqrt(fraction_of_power_remaining)*jnp.exp(1j*phase_shift))

        outputs = {
            "o0": SteadyStateOpticalSignal(field=o0_field_out, wl=optical_wls),
            "o1": SteadyStateOpticalSignal(field=o1_field_out, wl=optical_wls),
            # "e0": electrical_signal(),
        }
        return outputs



# class MachZehnderModulator(PCell):
#     ports = [
#         Port(
#             name = "o0",
#             type = "optical",
#             directionality = "bidirectional",
#         ),
#         Port(
#             name = "o1",
#             type = "optical",
#             directionality = "bidirectional",
#         ),
#         Port(
#             name = "e0",
#             type = "electrical",
#             directionality = "unidirectional",
#         ),
#         Port(
#             name = "e1",
#             type = "electrical",
#             directionality = "unidirectional",
#         )
#     ]
#     def __init__(
#         self, 
#         NOT_IMPLEMENTED,         
#     ):
#         # IMPLEMENT ME
#         pass