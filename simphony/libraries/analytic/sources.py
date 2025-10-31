import jax
from jax.typing import ArrayLike

from simphony.circuit import SteadyStateComponent
from simphony.circuit import BlockModeComponent, SampleModeComponent
from simphony.signals import    BlockModeOpticalSignal, SteadyStateOpticalSignal, SampleModeOpticalSignal
import jax.numpy as jnp
import numpy as np # Used to avoid caching issues when generating random numbers
from typing import Union
from simphony.simulation import SimulationParameters, SampleModeSimulationParameters, BlockModeSimulationParameters

from scipy.ndimage import gaussian_filter1d
from scipy.signal import iirdesign, freqz
from scipy.signal import butter, lfilter, cheby1
from typing import Callable

def gaussian_kernel1d(sigma, truncate=4.0):
    radius = int(truncate * sigma + 0.5)
    x = jnp.arange(-radius, radius + 1)
    kernel = jnp.exp(-(x**2) / (2 * sigma**2))
    kernel /= jnp.sum(kernel)
    return kernel

def gaussian_filter1d_jax(x, sigma, truncate=4.0):
    kernel = gaussian_kernel1d(sigma, truncate)
    return jnp.convolve(x, kernel, mode='same')

import jax.numpy as jnp

def cubic_interp_1d(x: jnp.ndarray, new_len: int) -> jnp.ndarray:
    def catmull_rom(p0, p1, p2, p3, t):
        t2 = t * t
        t3 = t2 * t
        return 0.5 * (
            (2 * p1) +
            (-p0 + p2) * t +
            (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
            (-p0 + 3*p1 - 3*p2 + p3) * t3
        )

    old_len = x.shape[0]
    idxs_f = jnp.linspace(0, old_len - 1, new_len)
    idxs = jnp.floor(idxs_f).astype(int)
    t = idxs_f - idxs

    # Ensure indices stay within bounds
    idxs_m1 = jnp.clip(idxs - 1, 0, old_len - 1)
    idxs_p1 = jnp.clip(idxs + 1, 0, old_len - 1)
    idxs_p2 = jnp.clip(idxs + 2, 0, old_len - 1)

    p0 = x[idxs_m1]
    p1 = x[idxs]
    p2 = x[idxs_p1]
    p3 = x[idxs_p2]

    return catmull_rom(p0, p1, p2, p3, t)


class OpticalCombSource(SampleModeComponent, BlockModeComponent):
    optical_ports = ["o0"]
    def __init__(
        self,
        wavelength=jnp.array([1.53e-6, 1.54e-6, 1.55e-6, 1.56e-6, 1.57e-6]),
        linewidth=0.0,
    ):
        self.wavelength = wavelength
        self.linewidth = linewidth

    def block_mode_response (
        self,
        inputs: dict={},
        simulation_parameters: SimulationParameters = SimulationParameters(),
    ):
        N = simulation_parameters.num_time_steps
        num_wls = self.wavelength.shape[0]
        sampling_period = simulation_parameters.sampling_period
        t = jnp.arange(N) * sampling_period
        linewidth = self.linewidth
        
        key = simulation_parameters.prng_key
        delta_phi_std = jnp.sqrt(2*jnp.pi*self.linewidth*simulation_parameters.sampling_period)
        dphi = jax.random.normal(key, (simulation_parameters.num_time_steps, num_wls))*delta_phi_std
        phi = jnp.cumsum(dphi, axis=0)

        # Compute complex envelope
        A_t = jnp.exp(1j*phi)[:, :, None]

        outputs = {
            "o0": BlockModeOpticalSignal(
                amplitude=A_t,
                wavelength=self.wavelength
            ),
        }
        
        return outputs
    
    def sample_mode_initial_state(self, simulation_parameters):
        time_step = 0
        output_signal = self.block_mode_response(simulation_parameters=simulation_parameters)['o0']
        return time_step, output_signal

    def sample_mode_step(self, inputs, state, simulation_parameters):
        time_step, full_output_signal = state

        outputs = {
            "o0": SampleModeOpticalSignal(
                amplitude=full_output_signal.amplitude[time_step],
                wavelength=full_output_signal.wavelength
            ),
        }

        return outputs, (time_step+1, full_output_signal)


class CWLaser(SampleModeComponent, BlockModeComponent):
    """
    The CW Laser is meant to be used in time-domain simulations.
    """
    # delay_compensation = 0
    optical_ports = ["o0"]
    def __init__(
        self,
        wavelength=1.55e-6,
        linewidth=0,
        lineshape='lorentzian',
    ):
        self.wavelength = wavelength
        self.linewidth = linewidth
        self.lineshape = lineshape
        
        if self.lineshape.lower() == "lorentzian":
            self.phase_noise = self.lorentzian_phase_noise
            self.sample_mode_step = self.sample_mode_step_lorentzian
            self.sample_mode_initial_state = self.sample_mode_initial_state_lorentzian
        elif self.lineshape.lower() == "gaussian":
            self.gaussian_window_sigma = 170
            self.gaussian_window_period = 1e-14
            # gaussian_noise = jax.random.normal(key, shape=(N,))
            self.phase_noise = self.gaussian_phase_noise
            self.sample_mode_step = self.sample_mode_step_gaussian
            self.sample_mode_initial_state = self.sample_mode_initial_state_gaussian
        else:
            raise ValueError(f"Unrecognized name for lineshape parameter: {self.lineshape}")

    def lorentzian_phase_noise(self, simulation_parameters):
        key = simulation_parameters.prng_key
        delta_phi_std = jnp.sqrt(2*jnp.pi*self.linewidth*simulation_parameters.sampling_period)
        dphi = jax.random.normal(key, (simulation_parameters.num_time_steps,))*delta_phi_std
        phi = jnp.cumsum(dphi)
        
        return phi
    
    def sample_mode_step_lorentzian(self, inputs, state, simulation_parameters):
        phi_prev = state
        key = simulation_parameters.prng_key
        delta_phi_std = jnp.sqrt(2*jnp.pi*self.linewidth*simulation_parameters.sampling_period)
        dphi = jax.random.normal(key)*delta_phi_std
        phi = dphi + phi_prev
        
        A_t = jnp.exp(1j*phi)

        outputs = {
            "o0": SampleModeOpticalSignal(
                amplitude=A_t.reshape((1, 1)),
                wavelength=jnp.array([self.wavelength])
            ),
        }

        return outputs, phi
    
    def gaussian_phase_noise(self, simulation_parameters):
        dt_prime = self.gaussian_window_period
        dt = simulation_parameters.sampling_period
        sigma = self.gaussian_window_sigma
        N = simulation_parameters.num_time_steps
        M = int(dt/dt_prime*N)

        gaussian_noise = jax.random.normal(simulation_parameters.prng_key, shape=(M,))
        # b, a = butter(N=2, Wn=0.00187)  # 4th order low-pass
        # gaussian_noise = lfilter(b, a, gaussian_noise)
        gaussian_noise = gaussian_filter1d_jax(gaussian_noise, sigma=sigma)
        gaussian_noise /= jnp.std(gaussian_noise)
        f_instantaneous = (self.linewidth/2.355)*gaussian_noise
        phi = 2*jnp.pi*np.cumsum(f_instantaneous) * dt_prime
        phi = cubic_interp_1d(phi, N)
        return phi
    
    def sample_mode_step_gaussian(self, inputs, state, simulation_parameters):
        N = simulation_parameters.num_time_steps
        dt = simulation_parameters.sampling_period
        sigma = self.gaussian_window_sigma
        dt_prime = self.gaussian_window_period
        gaussian_filter1d
        iirdesign(sigma, -sigma, )
        return ...

    def block_mode_response (
        self,
        inputs: dict={},
        simulation_parameters: SimulationParameters = SimulationParameters(),
    ):
        N = simulation_parameters.num_time_steps
        sampling_period = simulation_parameters.sampling_period
        t = jnp.arange(N) * sampling_period
        linewidth = self.linewidth
        
        phi = self.phase_noise(simulation_parameters)

        # Compute complex envelope
        A_t = jnp.exp(1j*phi)

        outputs = {
            "o0": BlockModeOpticalSignal(
                amplitude=A_t.reshape((N, 1, 1)),
                wavelength=self.wavelength
            ),
        }

        return outputs
    
    def sample_mode_initial_state_gaussian(self, simulation_parameters):
        truncate = 4.0
        radius = int(truncate * self.gaussian_window_sigma + 0.5)
        x = np.arange(-radius, radius + 1)
        g = np.exp(-0.5 * (x / self.gaussian_window_sigma) ** 2)
        g /= g.sum()
        std_dev = np.sqrt(np.sum(g ** 2))
        return std_dev
    
    def sample_mode_initial_state_lorentzian(self, simulation_parameters):
        phi_prev = 0
        return phi_prev

        
class OpticalSource(SampleModeComponent, BlockModeComponent):
    optical_ports = ["o0"]

    def __init__(
        self, 
        wavelength = 1.55e-6,
        envelope_fn:Callable[[float], complex] = None 
    ):    
        self.wavelength = wavelength
        self.envelope_fn = envelope_fn
    
    def block_mode_response (
        self, 
        inputs: dict,
        simulation_parameters: BlockModeSimulationParameters,
    ):
        N = simulation_parameters.num_time_steps
        return ...
    
    def sample_mode_initial_state(self, simulation_parameters: SampleModeSimulationParameters):
        N = simulation_parameters.num_time_steps
        dt = simulation_parameters.sampling_period
        t = jnp.arange(0, N, 1)*dt
        self.envelope = self.envelope_fn(t)
        time_step = 0

        return jnp.array(time_step, dtype=int)

    def sample_mode_step (
        self, 
        inputs: dict,
        state,
        simulation_parameters: SampleModeSimulationParameters,
    ):
        current_time_step = state
        outputs = {
            "o0": SampleModeOpticalSignal(
                amplitude=jnp.array([[self.envelope[current_time_step]]], dtype=complex),
                wavelength=jnp.array([self.wavelength]),
            )
        }
        return outputs, state+1
    
class VoltageSource(
    SteadyStateComponent, 
    SampleModeComponent, 
    BlockModeComponent,
):
    electrical_ports = ["e0"]

    def __init__(
        self, 
        steady_state_voltage=1.0,
        steady_state_wl=0,
    ):
        self.steady_state_voltage=steady_state_voltage
        self.steady_state_wl = steady_state_wl
        # optical_ports = None
        # electrical_ports = ['e0']
        # logic_ports = None
        # super().__init__(
        #     optical_ports=optical_ports,
        #     electrical_ports=electrical_ports,
        #     logic_ports=logic_ports
        # )

    def steady_state(
        self, 
        inputs: dict,
    ):
        outputs = {
            "e0": SteadyStateOpticalSignal(voltage=[self.steady_state_voltage], wl=[self.steady_state_wl])
        }
        return outputs

    def block_mode_response(self, input_signal: ArrayLike, simulation_parameters):
        pass
    
    def sample_mode_step(self, inputs: dict, state: jax.Array, simulation_parameters):
        # TODO: Complete this to use the signal defined in settings
        return inputs, state
    
    def sample_mode_initial_state(self, simulation_parameters):
        return jnp.array([0])


class PRNG(
    SteadyStateComponent, 
    # SampleModeComponent, 
    BlockModeComponent
):
    logic_ports = ["l0"]

    def __init__(self, **settings):
        pass
        # optical_ports = None
        # electrical_ports = None
        # logic_ports = ['l0']
        # super().__init__(
        #     optical_ports=optical_ports,
        #     electrical_ports=electrical_ports,
        #     logic_ports=logic_ports
        # )
    
    @jax.jit
    def steady_state(self, inputs: dict, default_output: int=0):
        outputs = {
            "l0": default_output
        }
        return outputs

    def block_mode_response(self, inputs: dict, **kwargs):
        pass




# def gaussian_phase_noise(self, simulation_parameters):
#         N = simulation_parameters.num_time_steps
#         dt = simulation_parameters.sampling_period
#         tau = 1e-14
#         sigma = 100
#         M = int(N*dt/tau)
#         _gaussian_noise = jax.random.normal(simulation_parameters.prng_key, shape=(M,))
#         indices = jnp.round(jnp.linspace(0, M - 1, N)).astype(int)
#         gaussian_noise = _gaussian_noise[indices]
#         pass
        
#         # sigma =  250*( 1e-15 / simulation_parameters.sampling_period)
#         # # sigma = jnp.minimum(N, sigma)
        

#         # gaussian_noise = jax.random.normal(simulation_parameters.prng_key, shape=(N,))
#         f_instantaneous = gaussian_filter1d_jax(gaussian_noise, sigma=sigma)

#         # std_dev = jnp.std(f_instantaneous)        
#         ## We need to determine the scale factor 1/jnp.std(f_instantaneos) a priori ##
#         sigma_g = sigma
#         truncate = 4.0
#         radius = int(truncate * sigma_g + 0.5)
#         x = np.arange(-radius, radius + 1)
#         g = np.exp(-0.5 * (x / sigma_g) ** 2)
#         g /= g.sum()  # normalize like scipy
#         std_dev = np.sqrt(np.sum(g ** 2)) # sqrt(N/M) is the result of upsampling
#         ####
#         f_instantaneous *= (self.linewidth/2.355)/std_dev
#         f_instantaneous *= 2
#         phi = jnp.pi*np.cumsum(f_instantaneous) * dt
#         return phi

# def gaussian_phase_noise(self, simulation_parameters):
#         N = simulation_parameters.num_time_steps
#         dt = simulation_parameters.sampling_period
#         # tau = 1e-14
#         # sigma = 150
#         # M = int(N*dt/tau)
#         # _gaussian_noise = jax.random.normal(simulation_parameters.prng_key, shape=(M,))
#         # indices = jnp.round(jnp.linspace(0, M - 1, N)).astype(int)
#         # gaussian_noise = _gaussian_noise[indices]
#         pass
        
#         sigma =  500*( 1e-15 / simulation_parameters.sampling_period)
#         # sigma = jnp.minimum(N, sigma)
        

#         gaussian_noise = jax.random.normal(simulation_parameters.prng_key, shape=(N,))
#         f_instantaneous = gaussian_filter1d_jax(gaussian_noise, sigma=sigma)

#         # std_dev = jnp.std(f_instantaneous)        
#         ## We need to determine the scale factor 1/jnp.std(f_instantaneos) a priori ##
#         sigma_g = sigma
#         truncate = 4.0
#         radius = int(truncate * sigma_g + 0.5)
#         x = np.arange(-radius, radius + 1)
#         g = np.exp(-0.5 * (x / sigma_g) ** 2)
#         g /= g.sum()  # normalize like scipy
#         std_dev = np.sqrt(np.sum(g ** 2)) # sqrt(N/M) is the result of upsampling
#         ####
#         f_instantaneous *= (self.linewidth/2.355)/std_dev
#         f_instantaneous *= 2
#         phi = jnp.pi*np.cumsum(f_instantaneous) * dt
#         return phi