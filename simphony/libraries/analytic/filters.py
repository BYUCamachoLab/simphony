from simphony.circuit import BlockModeComponent, SampleModeComponent
from simphony.signals.block_mode import BlockModeOpticalSignal
from simphony.signals.sample_mode import SampleModeOpticalSignal
import jax.numpy as jnp
from scipy.signal import lfilter
from scipy.constants import speed_of_light as SPEED_OF_LIGHT
import jax
from dataclasses import replace

class OpticalDiscreteFilter( 
    SampleModeComponent,
    BlockModeComponent,
):
    optical_ports = ["in", "out"]
    
    def __init__(
        self,
        *,
        b: jnp.ndarray = jnp.asarray([0.0, 1.0]),
        a: jnp.ndarray = jnp.asarray([1.0]),
        center_wl = 1.55e-6,
        delay_compensation = 0,
    ):
        # We need at least 1 filter for each mode
        b, a = jnp.atleast_2d(b), jnp.atleast_2d(a)
        self.filter_coefficients = b/a[:, 0], a/a[:, 0]
        self.center_wl = center_wl
        self.delay_compensation = 0

    def sample_mode_initial_state(self, simulation_parameters):
        b, a = self.filter_coefficients
        M, L = simulation_parameters.num_optical_modes, simulation_parameters.optical_baseband_wavelengths.shape[0]
        # weight_x = jnp.zeros((M, L), dtype=complex)
        x_hist = jnp.zeros((L, M, b.shape[1]), dtype=complex)
        # weight_y = jnp.zeros((M, L), dtype=complex)
        y_hist = jnp.zeros((L, M, a.shape[1]), dtype=complex)
        state = (x_hist, y_hist)
        return state
    
    def sample_mode_step(self, inputs: dict,  state: jax.Array, simulation_parameters):
        delay_compensation = self.delay_compensation
        
        x_hist, y_hist = state

        ###
        # TODO: Modulated the inputs by delta_omega
        ###

        input_signal = inputs["in"]
        input_amplitude = input_signal.amplitude
        input_wavelength = input_signal.wavelength

        output_amplitude = jnp.zeros_like(input_amplitude)
        output_wavelength = input_wavelength
        
        x_hist = jnp.roll(x_hist, shift=1, axis=2)
        x_hist = x_hist.at[:, :, 0].set(input_amplitude)
        y_hist = jnp.roll(y_hist, shift=1, axis=2)

        b, a = self.filter_coefficients
        for mode_idx in range(simulation_parameters.num_optical_modes):
            b_single_mode, a_single_mode = b[mode_idx], a[mode_idx]
            for wl_idx, wl in enumerate(input_wavelength):
                f_center = SPEED_OF_LIGHT / self.center_wl
                f = SPEED_OF_LIGHT/wl
                delta_omega = 2*jnp.pi*(f - f_center) / f_center
                
                weight_x = b_single_mode@x_hist[wl_idx, mode_idx]
                weight_y = a_single_mode[1:]@y_hist[wl_idx, mode_idx, 1:]
                y_single_mode = weight_x - weight_y
                output_amplitude = output_amplitude.at[wl_idx, mode_idx].set(y_single_mode)

        y_hist = y_hist.at[:, :, 0].set(output_amplitude) 

        outputs = {
            "in": SampleModeOpticalSignal(
                amplitude=jnp.zeros_like(output_amplitude),
                wavelength=output_wavelength
            ),
            "out": SampleModeOpticalSignal(
                amplitude=output_amplitude,
                wavelength=output_wavelength
            ),
        }

        state = (x_hist, y_hist)
        return outputs, state
    
    def block_mode_response(self, inputs: dict, simulation_parameters):
        input_signal = inputs['in']
        input_amplitude = input_signal.amplitude
        input_wavelength = input_signal.wavelength
        
        output_amplitude = jnp.zeros_like(input_signal.amplitude)
        output_wavelength = input_wavelength

        n = jnp.arange(0, output_amplitude.shape[0], 1)

        b, a = self.filter_coefficients
        for mode_idx in range(simulation_parameters.num_optical_modes):
            b_single_mode, a_single_mode = b[mode_idx], a[mode_idx]
            for wl_idx, wl in enumerate(input_wavelength):
                f_center = SPEED_OF_LIGHT / self.center_wl
                f = SPEED_OF_LIGHT/wl
                delta_omega = 2*jnp.pi*(f - f_center) / f_center
                x_single_mode = jnp.exp(-delta_omega*n*1j)*input_amplitude[:, wl_idx, mode_idx]
                y_single_mode = jnp.exp(delta_omega*n*1j)*lfilter(b_single_mode, a_single_mode, x_single_mode)
                output_amplitude = output_amplitude.at[:, wl_idx, mode_idx].set(y_single_mode)

        outputs = {
            'in': BlockModeOpticalSignal(
                amplitude=jnp.zeros_like(output_amplitude),
                wavelength=output_wavelength,
            ),
            'out': BlockModeOpticalSignal(
                amplitude=output_amplitude,
                wavelength=output_wavelength
            )
        }

        return outputs
    
class OpticalStateSpace( 
    SampleModeComponent,
):
    optical_ports = ["in", "out"]
    
    def __init__(
        self,
        *,
        A,
        B,
        C,
        D,
        center_wl = 1.55e-6,
        delay_compensation = 0,
    ):
        self.state_space_matrices = A, B, C, D
        self.center_wl = center_wl
        self.delay_compensation = delay_compensation

    def sample_mode_initial_state(self, simulation_parameters):
        state = ...
        return state
    
    def sample_mode_step(self, inputs: dict,  state: jax.Array, simulation_parameters):
        
        return inputs, state