import jax
from jax.typing import ArrayLike

from simphony.component.component import SteadyStateComponent
from simphony.component.component import BlockModeComponent, SampleModeComponent
from simphony.signal.block_mode import BlockModeOpticalSignal, BlockModeElectricalSignal
from simphony.signal.steady_state import SteadyStateOpticalSignal, SteadyStateElectricalSignal
from simphony.signal.sample_mode import SampleModeOpticalSignal
import jax.numpy as jnp
import numpy as np # Used to avoid caching issues when generating random numbers
from typing import Union
from jaxtyping import Array, Float
from simphony.simulation.sample_mode import SampleModeSimulationParameters
from simphony.simulation.block_mode import BlockModeSimulationParameters

# from scipy.ndimage import gaussian_filter1d
# from scipy.signal import iirdesign
# from scipy.signal import freqz
# from scipy.signal import butter, lfilter, cheby1
from typing import Callable

from simphony.component.port import Port
from simphony.simulation.simulation import SimulationParameters

# def gaussian_kernel1d(sigma, truncate=4.0):
#     radius = int(truncate * sigma + 0.5)
#     x = jnp.arange(-radius, radius + 1)
#     kernel = jnp.exp(-(x**2) / (2 * sigma**2))
#     kernel /= jnp.sum(kernel)
#     return kernel

# def gaussian_filter1d_jax(x, sigma, truncate=4.0):
#     kernel = gaussian_kernel1d(sigma, truncate)
#     return jnp.convolve(x, kernel, mode='same')

import jax.numpy as jnp

# def cubic_interp_1d(x: jnp.ndarray, new_len: int) -> jnp.ndarray:
#     def catmull_rom(p0, p1, p2, p3, t):
#         t2 = t * t
#         t3 = t2 * t
#         return 0.5 * (
#             (2 * p1) +
#             (-p0 + p2) * t +
#             (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
#             (-p0 + 3*p1 - 3*p2 + p3) * t3
#         )

#     old_len = x.shape[0]
#     idxs_f = jnp.linspace(0, old_len - 1, new_len)
#     idxs = jnp.floor(idxs_f).astype(int)
#     t = idxs_f - idxs

#     # Ensure indices stay within bounds
#     idxs_m1 = jnp.clip(idxs - 1, 0, old_len - 1)
#     idxs_p1 = jnp.clip(idxs + 1, 0, old_len - 1)
#     idxs_p2 = jnp.clip(idxs + 2, 0, old_len - 1)

#     p0 = x[idxs_m1]
#     p1 = x[idxs]
#     p2 = x[idxs_p1]
#     p3 = x[idxs_p2]

#     return catmull_rom(p0, p1, p2, p3, t)


class OpticalCombSource(SampleModeComponent, BlockModeComponent):
    ports = [
        Port(
            name="o0",
            type="optical",
            directionality="output",
        ),
    ]

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        *,
        wavelength=jnp.array([1.53e-6, 1.54e-6, 1.55e-6, 1.56e-6, 1.57e-6]),
        linewidth=0.0,
    ):
        self.wavelength = jnp.asarray(wavelength)
        self.linewidth = linewidth

    def _generate_phase_noise(self, simulation_parameters):
        N = simulation_parameters.num_time_steps
        num_wls = self.wavelength.shape[0]
        dt = simulation_parameters.dt
        delta_phi_std = float(jnp.sqrt(2 * jnp.pi * self.linewidth * dt))
        rng = np.random.default_rng(simulation_parameters.seed)
        dphi = rng.standard_normal((N, num_wls)) * delta_phi_std
        return jnp.array(jnp.cumsum(dphi, axis=0))

    def block_mode_response(
        self,
        inputs: dict = {},
        simulation_parameters: BlockModeSimulationParameters = BlockModeSimulationParameters(),
    ):
        num_modes = len(simulation_parameters.mode_identifiers)
        phi = self._generate_phase_noise(simulation_parameters)
        # shape: (N, L, M) — one unit amplitude per wavelength, placed in mode 0
        A_t = jnp.zeros((*phi.shape, num_modes), dtype=complex)
        A_t = A_t.at[:, :, 0].set(jnp.exp(1j * phi))

        outputs = {
            "o0": BlockModeOpticalSignal(
                amplitude=A_t,
                wavelength=self.wavelength,
            ),
        }
        return outputs

    def sample_mode_initial_state(self, simulation_parameters):
        # State is just the accumulated phase per wavelength — O(L), not O(N×L).
        # block_mode_response is left unchanged for block-mode callers.
        L = self.wavelength.shape[0]
        return jnp.zeros((L,))

    def sample_mode_step(self, inputs, state, simulation_state, simulation_parameters):
        phi = state   # accumulated phase, shape (L,)
        L   = self.wavelength.shape[0]
        M   = len(simulation_parameters.mode_identifiers)

        if self.linewidth == 0.0:
            # CW: constant unit amplitude, phase never changes.
            new_phi = phi
        else:
            # Noisy laser: grow a random-walk phase one step at a time using the
            # per-step PRNG key already provided by the simulator.
            dt             = simulation_parameters.dt
            delta_phi_std  = jnp.sqrt(2 * jnp.pi * self.linewidth * dt)
            dphi           = delta_phi_std * jax.random.normal(
                                simulation_state.prng_key, shape=(L,))
            new_phi = phi + dphi

        amplitude = jnp.zeros((L, M), dtype=complex).at[:, 0].set(jnp.exp(1j * new_phi))

        outputs = {
            "o0": SampleModeOpticalSignal(
                amplitude=amplitude,
                wavelength=self.wavelength,
            ),
        }
        return outputs, new_phi


class CWLaser(SampleModeComponent, BlockModeComponent):
    """Continuous-wave optical source for time-domain simulations.

    The laser emits a complex optical envelope on output port `o0`. In Block
    mode, the returned `BlockModeOpticalSignal` spans the full simulation time
    block and uses `wavelength` as its carrier channel set.

    Parameters
    ----------
    wavelength:
        Optical carrier wavelength or wavelength array, in meters.
    linewidth:
        Lorentzian linewidth used to generate phase noise. A value of zero
        produces a deterministic constant-envelope source.
    lineshape:
        Phase-noise model name. Currently only `"lorentzian"` is implemented.
    mode_idx:
        Index of the optical mode that receives the source amplitude.
    """
    # delay_compensation = 0
    # optical_ports = ["o0"]
    ports = [
        Port(
            name = "o0",
            type = "optical",
            directionality = "output",
        ),
    ]
    def __init__(
        self,
        simulation_parameters,
        wavelength=1.55e-6,
        linewidth=0,
        lineshape='lorentzian',
        mode_idx = 0,
    ):
        self.wavelength = wavelength
        self.linewidth = linewidth
        self.lineshape = lineshape
        self.mode_idx = mode_idx
        
        if self.lineshape.lower() == "lorentzian":
            self.phase_noise = self.lorentzian_phase_noise
            self.sample_mode_step = self.sample_mode_step_lorentzian
            self.sample_mode_initial_state = self.sample_mode_initial_state_lorentzian
        elif self.lineshape.lower() == "gaussian":
            raise ValueError(f"Gaussian noise not yet implemented")
            # self.gaussian_window_sigma = 170
            # self.gaussian_window_period = 1e-14
            # # gaussian_noise = jax.random.normal(key, shape=(N,))
            # self.phase_noise = self.gaussian_phase_noise
            # self.sample_mode_step = self.sample_mode_step_gaussian
            # self.sample_mode_initial_state = self.sample_mode_initial_state_gaussian
        else:
            raise ValueError(f"Unrecognized name for lineshape parameter: {self.lineshape}")

    def lorentzian_phase_noise(self, simulation_parameters):
        key = simulation_parameters.prng_key
        delta_phi_std = jnp.sqrt(2*jnp.pi*self.linewidth*simulation_parameters.dt)
        dphi = jax.random.normal(key, (simulation_parameters.num_time_steps,))*delta_phi_std
        phi = jnp.cumsum(dphi)
        
        return phi
    
    def sample_mode_step_lorentzian(self, inputs, state, simulation_state, simulation_parameters):
        phi_prev = state
        key = simulation_state.prng_key
        delta_phi_std = jnp.sqrt(2*jnp.pi*self.linewidth*simulation_parameters.dt)
        dphi = jax.random.normal(key)*delta_phi_std
        phi = dphi + phi_prev
        
        A_t = jnp.exp(1j*phi)
        amplitude = jnp.zeros((1, len(simulation_parameters.mode_identifiers)), dtype=complex)
        amplitude = amplitude.at[0, self.mode_idx].set(A_t)
        outputs = {
            "o0": SampleModeOpticalSignal(
                amplitude=amplitude,
                wavelength=jnp.array([self.wavelength])
            ),
        }

        # amplitude = jnp.zeros((3, len(simulation_parameters.mode_identifiers)), dtype=complex)
        # amplitude = amplitude.at[0, 0].set(0.1 + 0j)
        # amplitude = amplitude.at[1, 0].set(0.2 + 0j)
        # amplitude = amplitude.at[2, 0].set(0.3 + 0j)
        # amplitude = amplitude.at[0, 1].set(1.1 + 0j)
        # amplitude = amplitude.at[1, 1].set(1.2 + 0j)
        # amplitude = amplitude.at[2, 1].set(1.3 + 0j)
        # wl = jnp.array([1.51e-6, 1.549e-6, 1.59e-6])
        # # TODO: REMOVE THIS HARDCODED TESTING CODE
        # outputs = {
        #     "o0": SampleModeOpticalSignal(
        #         amplitude=amplitude,
        #         wavelength=wl
        #     ),
        # }

        return outputs, phi
    
    # def gaussian_phase_noise(self, simulation_parameters):
    #     dt_prime = self.gaussian_window_period
    #     dt = simulation_parameters.sampling_period
    #     sigma = self.gaussian_window_sigma
    #     N = simulation_parameters.num_time_steps
    #     M = int(dt/dt_prime*N)

    #     gaussian_noise = jax.random.normal(simulation_parameters.prng_key, shape=(M,))
    #     # b, a = butter(N=2, Wn=0.00187)  # 4th order low-pass
    #     # gaussian_noise = lfilter(b, a, gaussian_noise)
    #     gaussian_noise = gaussian_filter1d_jax(gaussian_noise, sigma=sigma)
    #     gaussian_noise /= jnp.std(gaussian_noise)
    #     f_instantaneous = (self.linewidth/2.355)*gaussian_noise
    #     phi = 2*jnp.pi*np.cumsum(f_instantaneous) * dt_prime
    #     phi = cubic_interp_1d(phi, N)
    #     return phi
    
    # def sample_mode_step_gaussian(self, inputs, state, simulation_parameters):
    #     N = simulation_parameters.num_time_steps
    #     dt = simulation_parameters.sampling_period
    #     sigma = self.gaussian_window_sigma
    #     dt_prime = self.gaussian_window_period
    #     gaussian_filter1d
    #     iirdesign(sigma, -sigma, )
    #     return ...

    def block_mode_response (
        self,
        inputs: dict={},
        simulation_parameters: BlockModeSimulationParameters = BlockModeSimulationParameters(),
    ):
        N = simulation_parameters.num_time_steps
        sampling_period = simulation_parameters.dt
        t = jnp.arange(N) * sampling_period
        linewidth = self.linewidth
        
        phi = self.phase_noise(simulation_parameters)


        # Compute complex envelope
        A_t = jnp.exp(1j*phi)
        amplitude = jnp.zeros((A_t.shape[0], 1, len(simulation_parameters.mode_identifiers)), dtype=complex)
        amplitude = amplitude.at[:, 0, self.mode_idx].set(A_t)       

        outputs = {
            "o0": BlockModeOpticalSignal(
                amplitude=amplitude,
                wavelength=jnp.array([self.wavelength])
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
    """Optical source driven by a user-provided envelope.

    The source emits a `BlockModeOpticalSignal` on output port `o0`. Users can
    either provide a concrete `envelope` or an `envelope_fn` that creates one
    from the simulation time vector. Exactly one of those options must be
    supplied.

    If the envelope length does not match `simulation_parameters.num_time_steps`,
    it is truncated or padded with zeros so the emitted block has the simulation
    length.
    """
    optical_ports = ["o0"]

    def __init__(
        self, 
        simulation_parameters,
        # wavelength = 1.55e-6,
        envelope: BlockModeOpticalSignal = None,
        envelope_fn: Callable[[Float[Array, "n"]], BlockModeOpticalSignal] = None 
    ):    
        if envelope is not None and envelope_fn is not None:
            raise ValueError("Specify either evelope or envelope_fn, NOT both")
        if envelope is None and envelope_fn is None:
            raise ValueError("Parameter `envelope` or `envelope_fn` must be specified")
        
        # self.wavelength = wavelength
        self.envelope = envelope
        self.envelope_fn = envelope_fn
    
    def _calculate_envelope(self, simulation_parameters):
        N = simulation_parameters.num_time_steps
        dt = simulation_parameters.dt
        t = jnp.arange(0, N, 1)*dt

        if self.envelope_fn:
            self.envelope = self.envelope_fn(t)
        
        # Make envelope match the number of time steps, by truncating or appending zeros
        amplitude = self.envelope.amplitude
        T, L, M = amplitude.shape
        if amplitude.shape[0] < N:
            amplitude = jnp.concatenate([amplitude, jnp.zeros((N-T, L, M), dtype=complex)], axis=0)
        elif amplitude.shape[0] > N:
            amplitude = amplitude[:N, :, :]

        # TODO: RETURN, DON'T MUTATE
        self.envelope = BlockModeOpticalSignal(
            amplitude=amplitude,
            wavelength=self.envelope.wavelength
        )
    
    def block_mode_response (
        self, 
        inputs: dict,
        simulation_parameters: BlockModeSimulationParameters,
    ):
        self._calculate_envelope(simulation_parameters)

        outputs = {
            "o0": BlockModeOpticalSignal(
                amplitude=self.envelope.amplitude,
                wavelength=self.envelope.wavelength,
            )
        }
        return outputs
    
    def sample_mode_initial_state(self, simulation_parameters: SampleModeSimulationParameters):
        self._calculate_envelope(simulation_parameters)

        time_step = 0
        return jnp.array(time_step, dtype=int)

    def sample_mode_step (
        self, 
        inputs: dict,
        state,
        simulation_state,
        simulation_parameters: SampleModeSimulationParameters,
    ):
        current_time_step = state
        outputs = {
            "o0": SampleModeOpticalSignal(
                amplitude=self.envelope.amplitude[current_time_step],
                wavelength=self.envelope.wavelength,
            )
        }
        return outputs, state+1
    
class VoltageSource(
    SteadyStateComponent, 
    SampleModeComponent, 
    BlockModeComponent,
):
    """Electrical source for steady-state, sample-mode, and Block mode runs.

    In Block mode, the source emits a `BlockModeElectricalSignal` on `e0`.
    Users can supply a concrete electrical `envelope`, an `envelope_fn` that
    receives the simulation time vector, or neither. If neither is supplied, the
    source emits a constant voltage equal to `steady_state_voltage`.

    Parameters
    ----------
    envelope:
        Electrical signal to emit in time-domain simulations.
    envelope_fn:
        Callable that receives the time vector and returns a
        `BlockModeElectricalSignal`.
    steady_state_voltage:
        Constant voltage used for steady-state simulations and as the Block mode
        default when no envelope is supplied.
    """
    ports = [
        Port(
            name="e0",
            type="electrical",
            directionality="bidirectional",
        )
    ]

    def __init__(
        self, 
        simulation_parameters: SimulationParameters,
        *,
        envelope: BlockModeOpticalSignal = None,
        envelope_fn: Callable[[Float[Array, "n"]], BlockModeElectricalSignal] = None,
        steady_state_voltage=1.0,
    ):
        self.steady_state_voltage=steady_state_voltage
        
        
        if envelope is not None and envelope_fn is not None:
            raise ValueError("Specify either evelope or envelope_fn, NOT both")
        # if envelope is None and envelope_fn is None:
        #     raise ValueError("Parameter `envelope` or `envelope_fn` must be specified")
        
        # self.wavelength = wavelength
        self.envelope = envelope
        self.envelope_fn = envelope_fn
        
        # optical_ports = None
        # electrical_ports = ['e0']
        # logic_ports = None
        # super().__init__(
        #     optical_ports=optical_ports,
        #     electrical_ports=electrical_ports,
        #     logic_ports=logic_ports
        # )
    
    def _calculate_envelope(self, simulation_parameters):
        N = simulation_parameters.num_time_steps
        dt = simulation_parameters.dt
        t = jnp.arange(0, N, 1)*dt

        if self.envelope:
            pass
        elif self.envelope_fn:
            self.envelope = self.envelope_fn(t)
        else:
            self.envelope = BlockModeElectricalSignal(voltage=np.ones((len(t),), dtype=complex)*self.steady_state_voltage) 

        # Make envelope match the number of time steps, by truncating or appending zeros
        voltage = self.envelope.voltage
        # T = voltage.shape
        # if voltage.shape[0] < N:
        #     voltage = jnp.concatenate([voltage, jnp.zeros((N-T,), dtype=complex)], axis=0)
        # elif voltage.shape[0] > N:
        #     voltage = voltage[:N, :]

        return BlockModeElectricalSignal(voltage=voltage)
    
    def steady_state(
        self, 
        inputs: dict,
        simulation_parameters: SimulationParameters,
    ):
        outputs = {
            "e0": SteadyStateElectricalSignal(voltage=self.steady_state_voltage)
        }
        return outputs

    def block_mode_response(self, input_signal: ArrayLike, simulation_parameters):
        envelope = self._calculate_envelope(simulation_parameters)
        outputs = {
            "e0": envelope
        }
        return outputs
    
    def sample_mode_step(self, inputs: dict, state: jax.Array, simulation_state, simulation_parameters):
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
