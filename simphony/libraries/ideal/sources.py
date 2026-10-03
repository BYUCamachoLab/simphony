# from scipy.ndimage import gaussian_filter1d
# from scipy.signal import iirdesign
# from scipy.signal import freqz
# from scipy.signal import butter, lfilter, cheby1
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np  # Used to avoid caching issues when generating random numbers
from jax.typing import ArrayLike

from simphony.component.component import (
    BlockModeComponent,
    SampleModeComponent,
    SteadyStateComponent,
)
from simphony.component.port import Port
from simphony.signal.block_mode import BlockModeElectricalSignal, BlockModeOpticalSignal
from simphony.signal.sample_mode import (
    SampleModeElectricalSignal,
    SampleModeOpticalSignal,
)
from simphony.signal.steady_state import SteadyStateElectricalSignal
from simphony.simulation.block_mode import BlockModeSimulationParameters
from simphony.simulation.sample_mode import SampleModeSimulationParameters
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
        phi = state  # accumulated phase, shape (L,)
        L = self.wavelength.shape[0]
        M = len(simulation_parameters.mode_identifiers)

        if self.linewidth == 0.0:
            # CW: constant unit amplitude, phase never changes.
            new_phi = phi
        else:
            # Noisy laser: grow a random-walk phase one step at a time using the
            # per-step PRNG key already provided by the simulator.
            dt = simulation_parameters.dt
            delta_phi_std = jnp.sqrt(2 * jnp.pi * self.linewidth * dt)
            dphi = delta_phi_std * jax.random.normal(
                simulation_state.prng_key, shape=(L,)
            )
            new_phi = phi + dphi

        amplitude = jnp.zeros((L, M), dtype=complex).at[:, 0].set(jnp.exp(1j * new_phi))

        outputs = {
            "o0": SampleModeOpticalSignal(
                amplitude=amplitude,
                wavelength=self.wavelength,
            ),
        }
        return outputs, new_phi


FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))
# Separates the key stream used outside the per-tick keys (initial states,
# whole-run noise sequences) from the simulator's per-tick keys.
_DERIVED_KEY_SALT = 0x5EED


def _ou_step_coefficients(dt, sigma, tau):
    """Exact one-step coefficients of an Ornstein-Uhlenbeck frequency process.

    For a frequency nu with variance sigma**2 and correlation time tau, the
    next value and the integral of nu over a step dt, given the current value,
    are jointly Gaussian. With two standard normals xi0, xi1:

        nu_next  = a * nu + l11 * xi0
        integral = b * nu + l21 * xi0 + l22 * xi1

    This holds for any dt, so the update has no discretization error.
    """
    x = dt / tau
    a = jnp.exp(-x)
    one_minus_a = -jnp.expm1(-x)
    one_minus_a2 = -jnp.expm1(-2 * x)
    l11 = sigma * jnp.sqrt(one_minus_a2)
    l21 = sigma * tau * one_minus_a**2 / jnp.sqrt(one_minus_a2)
    # Conditional variance of the integral: 2 sigma^2 tau^2 (x - 2 tanh(x/2)),
    # with a series for small x where the difference cancels.
    g = jnp.where(
        x < 1e-2,
        x**3 / 12 - x**5 / 120 + 17 * x**7 / 20160,
        x - 2 * jnp.tanh(x / 2),
    )
    l22 = sigma * tau * jnp.sqrt(2 * g)
    return a, tau * one_minus_a, l11, l21, l22


class CWLaser(SampleModeComponent, BlockModeComponent):
    """Continuous-wave optical source for time-domain simulations.

    The laser emits a unit-power complex optical envelope on output port `o0`
    with phase noise set by `lineshape`. In Block mode, the returned
    `BlockModeOpticalSignal` spans the full simulation time block and uses
    `wavelength` as its carrier channel set.

    Line shapes (all FWHM values in Hz):

    - ``"lorentzian"``: white frequency noise, i.e. a phase random walk with
      step variance 2*pi*linewidth*dt. Exact at any time step. ``linewidth=0``
      (the default) gives a noiseless laser.
    - ``"gaussian"``: band-limited white frequency noise (Di Domenico, Schilt
      and Thomann, Appl. Opt. 49, 4801 (2010)) in its slow limit, where the
      line is a Gaussian of FWHM ``linewidth``. The frequency has rms
      sigma = linewidth / 2.355 and a flat spectrum up to ``noise_bandwidth``
      (f_c), zero above. The line is Gaussian only for f_c << sigma, so
      ``noise_bandwidth > 0.1 * sigma`` is rejected. The frequency sequence
      for the whole run is generated at the start (FFT shaping), so sample
      mode keeps ``num_time_steps`` floats in its state and cannot run past
      them (the output becomes NaN).
    - ``"voigt"``: white frequency noise (Lorentzian part) plus slow
      Ornstein-Uhlenbeck frequency noise (Gaussian part) with correlation time
      ``correlation_time``, updated exactly at any time step. ``linewidth`` is
      ``(lorentzian, gaussian)`` or ``{"lorentzian": ..., "gaussian": ...}``.
      The Gaussian part is Gaussian only for sigma * correlation_time >> 1, so
      sigma * correlation_time < 2.5 is rejected. With a zero Lorentzian part
      the line has the same Gaussian limit as ``"gaussian"`` but different far
      wings (OU frequency noise rolls off as 1/f^2 instead of a sharp cutoff).

    A Gaussian line is an ensemble property: a single record much shorter than
    the noise correlation time shows a narrow line at a wandering frequency.

    Parameters
    ----------
    wavelength:
        Optical carrier wavelength or wavelength array, in meters.
    linewidth:
        Line width (FWHM, Hz); its form depends on `lineshape` (see above).
    lineshape:
        ``"lorentzian"``, ``"gaussian"`` or ``"voigt"``.
    mode_idx:
        Index of the optical mode that receives the source amplitude.
    noise_bandwidth:
        ``"gaussian"`` only: cutoff f_c of the frequency noise, in Hz.
    correlation_time:
        ``"voigt"`` only: correlation time of the slow frequency noise, in s.
    seed:
        Component seed folded into the random keys, so that several lasers (or
        repeated runs) get independent noise. ``seed=0`` keeps the keys the
        simulator provides.

    Passing an argument that the chosen line shape does not use raises a
    ``ValueError``. The phase grows without bound, so run in float64
    (``jax_enable_x64``) for long simulations.
    """

    LINESHAPE_ARGUMENTS = {
        "lorentzian": ("linewidth",),
        "gaussian": ("linewidth", "noise_bandwidth"),
        "voigt": ("linewidth", "correlation_time"),
    }
    # Largest noise bandwidth (in units of sigma) for which the slow-noise
    # line shapes are accepted as Gaussian.
    MAX_BANDWIDTH_PER_SIGMA = 0.1

    ports = [
        Port(
            name="o0",
            type="optical",
            directionality="output",
        ),
    ]

    def __init__(
        self,
        simulation_parameters,
        wavelength=1.55e-6,
        linewidth=None,
        lineshape="lorentzian",
        mode_idx=0,
        noise_bandwidth=None,
        correlation_time=None,
        seed=0,
    ):
        lineshape = str(lineshape).lower()
        if lineshape not in self.LINESHAPE_ARGUMENTS:
            raise ValueError(
                f"Unrecognized lineshape '{lineshape}'; valid: "
                + ", ".join(self.LINESHAPE_ARGUMENTS)
            )
        valid = self.LINESHAPE_ARGUMENTS[lineshape]
        given = {
            "linewidth": linewidth,
            "noise_bandwidth": noise_bandwidth,
            "correlation_time": correlation_time,
        }
        for name, value in given.items():
            if value is not None and name not in valid:
                raise ValueError(
                    f"'{name}' is not a valid argument for the '{lineshape}' "
                    f"lineshape; valid: {', '.join(valid)}"
                )
        if lineshape != "lorentzian":
            for name in valid:
                if given[name] is None:
                    raise ValueError(
                        f"The '{lineshape}' lineshape requires '{name}'"
                    )

        self.wavelength = wavelength
        self.lineshape = lineshape
        self.mode_idx = mode_idx
        self.noise_bandwidth = noise_bandwidth
        self.correlation_time = correlation_time
        self.seed = int(seed)

        if lineshape == "lorentzian":
            self.linewidth = self._scalar_linewidth(0 if linewidth is None else linewidth, lineshape)
        elif lineshape == "gaussian":
            self.linewidth = self._scalar_linewidth(linewidth, lineshape)
            self.sigma = self.linewidth / FWHM_PER_SIGMA
            if not noise_bandwidth > 0:
                raise ValueError("noise_bandwidth must be positive")
            if noise_bandwidth > self.MAX_BANDWIDTH_PER_SIGMA * self.sigma:
                raise ValueError(
                    f"noise_bandwidth {noise_bandwidth:g} Hz is too large for a Gaussian "
                    f"line of FWHM {self.linewidth:g} Hz: it must be <= "
                    f"{self.MAX_BANDWIDTH_PER_SIGMA:g} * sigma = "
                    f"{self.MAX_BANDWIDTH_PER_SIGMA * self.sigma:g} Hz "
                    "(sigma = FWHM / 2.355); otherwise the line tends to a Lorentzian"
                )
        else:
            self.linewidth = self._voigt_linewidths(linewidth)
            self.sigma = self.linewidth[1] / FWHM_PER_SIGMA
            if not correlation_time > 0:
                raise ValueError("correlation_time must be positive")
            if self.linewidth[1] == 0:
                raise ValueError(
                    "The Gaussian part of a 'voigt' linewidth is zero; use the "
                    "'lorentzian' lineshape instead"
                )
            min_product = 1 / (4 * self.MAX_BANDWIDTH_PER_SIGMA)
            if self.sigma * correlation_time < min_product:
                raise ValueError(
                    f"correlation_time {correlation_time:g} s is too short for a Gaussian "
                    f"part of FWHM {self.linewidth[1]:g} Hz: sigma * correlation_time = "
                    f"{self.sigma * correlation_time:.3g} must be >= {min_product:g} "
                    "(sigma = FWHM / 2.355); otherwise the slow noise gives a Lorentzian"
                )

        self.phase_noise = getattr(self, f"{lineshape}_phase_noise")
        self.sample_mode_step = getattr(self, f"sample_mode_step_{lineshape}")
        self.sample_mode_initial_state = getattr(
            self, f"sample_mode_initial_state_{lineshape}"
        )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _scalar_linewidth(linewidth, lineshape):
        if isinstance(linewidth, (dict, tuple, list)) or np.ndim(linewidth) != 0:
            raise ValueError(
                f"The '{lineshape}' lineshape takes a single linewidth (FWHM in Hz)"
            )
        if not linewidth >= 0:
            raise ValueError("linewidth must be non-negative")
        return float(linewidth)

    @staticmethod
    def _voigt_linewidths(linewidth):
        message = (
            "voigt linewidth must be (lorentzian, gaussian) or "
            "{'lorentzian': ..., 'gaussian': ...} (FWHM in Hz)"
        )
        if isinstance(linewidth, dict):
            if set(linewidth) != {"lorentzian", "gaussian"}:
                raise ValueError(message)
            values = (linewidth["lorentzian"], linewidth["gaussian"])
        elif isinstance(linewidth, (tuple, list)) and len(linewidth) == 2:
            values = tuple(linewidth)
        else:
            raise ValueError(message)
        if any(np.ndim(v) != 0 or not v >= 0 for v in values):
            raise ValueError(message + "; both values must be non-negative numbers")
        return float(values[0]), float(values[1])

    def _step_key(self, key):
        return key if self.seed == 0 else jax.random.fold_in(key, self.seed)

    def _derived_key(self, simulation_parameters):
        # Key for draws made outside the per-tick stream (initial states and
        # whole-run sequences); the same in sample and block mode.
        key = jax.random.fold_in(
            jax.random.PRNGKey(simulation_parameters.seed), _DERIVED_KEY_SALT
        )
        return jax.random.fold_in(key, self.seed)

    def _outputs(self, phi, simulation_parameters):
        amplitude = jnp.zeros(
            (1, len(simulation_parameters.mode_identifiers)), dtype=complex
        )
        amplitude = amplitude.at[0, self.mode_idx].set(jnp.exp(1j * phi))
        return {
            "o0": SampleModeOpticalSignal(
                amplitude=amplitude, wavelength=jnp.array([self.wavelength])
            ),
        }

    # --------------------------------------------------------------- lorentzian

    def lorentzian_phase_noise(self, simulation_parameters):
        key = self._step_key(jax.random.PRNGKey(simulation_parameters.seed))
        delta_phi_std = jnp.sqrt(2 * jnp.pi * self.linewidth * simulation_parameters.dt)
        dphi = (
            jax.random.normal(key, (simulation_parameters.num_time_steps,))
            * delta_phi_std
        )
        return jnp.cumsum(dphi)

    def sample_mode_initial_state_lorentzian(self, simulation_parameters):
        # A float, like the phase the step returns: multirate scheduling gates
        # components with `lax.cond`, which needs both branches to agree on dtype.
        return jnp.asarray(0.0)

    def sample_mode_step_lorentzian(
        self, inputs, state, simulation_state, simulation_parameters
    ):
        key = self._step_key(simulation_state.prng_key)
        delta_phi_std = jnp.sqrt(2 * jnp.pi * self.linewidth * simulation_parameters.dt)
        phi = state + jax.random.normal(key) * delta_phi_std
        return self._outputs(phi, simulation_parameters), phi

    # ----------------------------------------------------------------- gaussian

    def _gaussian_frequency(self, simulation_parameters):
        # Band-limited white frequency noise for the whole run: flat one-sided
        # PSD h0 = sigma^2 / f_c up to f_c, zero above. Generated 4x longer and
        # cut from the middle so the FFT's circular wrap does not join the ends.
        n = int(simulation_parameters.num_time_steps)
        dt = simulation_parameters.dt
        fs = 1 / dt
        fc = self.noise_bandwidth
        if fc >= fs / 2:
            raise ValueError(
                f"noise_bandwidth {fc:g} Hz must be below the Nyquist frequency {fs / 2:g} Hz"
            )
        h0 = self.sigma**2 / fc
        m = 4 * n
        white = jax.random.normal(self._derived_key(simulation_parameters), (m,))
        spectrum = jnp.fft.rfft(white * jnp.sqrt(h0 * fs / 2))
        spectrum = jnp.where(jnp.fft.rfftfreq(m, dt) <= fc, spectrum, 0)
        start = m // 2 - n // 2
        return jnp.fft.irfft(spectrum, m)[start : start + n]

    def gaussian_phase_noise(self, simulation_parameters):
        frequency = self._gaussian_frequency(simulation_parameters)
        return 2 * jnp.pi * jnp.cumsum(frequency) * simulation_parameters.dt

    def sample_mode_initial_state_gaussian(self, simulation_parameters):
        return {
            "frequency": self._gaussian_frequency(simulation_parameters),
            "index": jnp.asarray(0, dtype=jnp.int32),
            "phi": jnp.asarray(0.0),
        }

    def sample_mode_step_gaussian(
        self, inputs, state, simulation_state, simulation_parameters
    ):
        frequency, index = state["frequency"], state["index"]
        n = frequency.shape[0]
        nu = jnp.where(index < n, frequency[jnp.minimum(index, n - 1)], jnp.nan)
        phi = state["phi"] + 2 * jnp.pi * nu * simulation_parameters.dt
        new_state = {"frequency": frequency, "index": index + 1, "phi": phi}
        return self._outputs(phi, simulation_parameters), new_state

    # -------------------------------------------------------------------- voigt

    def _voigt_update(self, state, xi, dt):
        # state = (nu, phi); xi = three standard normals.
        nu, phi = state[0], state[1]
        a, b, l11, l21, l22 = _ou_step_coefficients(dt, self.sigma, self.correlation_time)
        integral = b * nu + l21 * xi[0] + l22 * xi[1]
        white = jnp.sqrt(2 * jnp.pi * self.linewidth[0] * dt) * xi[2]
        phi = phi + 2 * jnp.pi * integral + white
        nu = a * nu + l11 * xi[0]
        return jnp.stack([nu, phi])

    def sample_mode_initial_state_voigt(self, simulation_parameters):
        # Stationary start: nu ~ N(0, sigma^2).
        key = self._derived_key(simulation_parameters)
        return jnp.stack([self.sigma * jax.random.normal(key), jnp.asarray(0.0)])

    def sample_mode_step_voigt(
        self, inputs, state, simulation_state, simulation_parameters
    ):
        xi = jax.random.normal(self._step_key(simulation_state.prng_key), (3,))
        state = self._voigt_update(state, xi, simulation_parameters.dt)
        return self._outputs(state[1], simulation_parameters), state

    def voigt_phase_noise(self, simulation_parameters, keys=None):
        """Block-mode phase: the sample-mode update scanned over all samples.

        With `keys` (one per sample, as the sample-mode simulator would pass
        them) the result equals that sample-mode run.
        """
        n = simulation_parameters.num_time_steps
        dt = simulation_parameters.dt
        if keys is None:
            keys = jax.random.split(
                jax.random.fold_in(self._derived_key(simulation_parameters), 1), n
            )
        xi = jax.vmap(lambda k: jax.random.normal(self._step_key(k), (3,)))(keys)
        state0 = self.sample_mode_initial_state_voigt(simulation_parameters)

        def step(state, x):
            state = self._voigt_update(state, x, dt)
            return state, state[1]

        _, phi = jax.lax.scan(step, state0, xi)
        return phi

    # --------------------------------------------------------------- block mode

    def block_mode_response(
        self,
        inputs: dict = {},
        simulation_parameters: BlockModeSimulationParameters = BlockModeSimulationParameters(),
    ):
        phi = self.phase_noise(simulation_parameters)

        # Compute complex envelope
        A_t = jnp.exp(1j * phi)
        amplitude = jnp.zeros(
            (A_t.shape[0], 1, len(simulation_parameters.mode_identifiers)),
            dtype=complex,
        )
        amplitude = amplitude.at[:, 0, self.mode_idx].set(A_t)

        outputs = {
            "o0": BlockModeOpticalSignal(
                amplitude=amplitude, wavelength=jnp.array([self.wavelength])
            ),
        }

        return outputs


class OpticalSource(SampleModeComponent, BlockModeComponent):
    """Optical source driven by a user-provided envelope.

    The source emits a `BlockModeOpticalSignal` on output port `o0`.
    Users can either provide a concrete `envelope` or an `envelope_fn`
    that creates one from the simulation time vector. Exactly one of
    those options must be supplied.

    If the envelope length does not match
    `simulation_parameters.num_time_steps`, it is truncated or padded
    with zeros so the emitted block has the simulation length.
    """

    optical_ports = ["o0"]

    def __init__(
        self,
        simulation_parameters,
        # wavelength = 1.55e-6,
        envelope: BlockModeOpticalSignal = None,
        envelope_fn: Callable[[ArrayLike], BlockModeOpticalSignal] = None,
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
        t = jnp.arange(0, N, 1) * dt

        if self.envelope_fn:
            self.envelope = self.envelope_fn(t)

        # Make envelope match the number of time steps, by truncating or appending zeros
        amplitude = self.envelope.amplitude
        T, L, M = amplitude.shape
        if amplitude.shape[0] < N:
            amplitude = jnp.concatenate(
                [amplitude, jnp.zeros((N - T, L, M), dtype=complex)], axis=0
            )
        elif amplitude.shape[0] > N:
            amplitude = amplitude[:N, :, :]

        # TODO: RETURN, DON'T MUTATE
        self.envelope = BlockModeOpticalSignal(
            amplitude=amplitude, wavelength=self.envelope.wavelength
        )

    def block_mode_response(
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

    def sample_mode_initial_state(
        self, simulation_parameters: SampleModeSimulationParameters
    ):
        self._calculate_envelope(simulation_parameters)

        time_step = 0
        return jnp.array(time_step, dtype=int)

    def sample_mode_step(
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
        return outputs, state + 1


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
        envelope_fn: Callable[[ArrayLike], BlockModeElectricalSignal] = None,
        steady_state_voltage=1.0,
    ):
        self.steady_state_voltage = steady_state_voltage

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
        t = jnp.arange(0, N, 1) * dt

        if self.envelope:
            pass
        elif self.envelope_fn:
            self.envelope = self.envelope_fn(t)
        else:
            self.envelope = BlockModeElectricalSignal(
                voltage=np.ones((len(t),), dtype=complex) * self.steady_state_voltage
            )

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
        outputs = {"e0": SteadyStateElectricalSignal(voltage=self.steady_state_voltage)}
        return outputs

    def block_mode_response(self, input_signal: ArrayLike, simulation_parameters):
        envelope = self._calculate_envelope(simulation_parameters)
        outputs = {"e0": envelope}
        return outputs

    def sample_mode_step(
        self, inputs: dict, state: jax.Array, simulation_state, simulation_parameters
    ):
        """Emit the voltage of the current sample on `e0`.

        The state is the sample counter n; the sample time is
        `time_offset + n * dt` of the source's rate region. `envelope_fn` is
        evaluated at that time, a concrete `envelope` is indexed by n (its last
        value is held once it runs out), and with neither the source emits
        `steady_state_voltage`.
        """
        n = state
        if self.envelope_fn is not None:
            t = simulation_parameters.time_offset + n * simulation_parameters.dt
            voltage = jnp.asarray(self.envelope_fn(jnp.atleast_1d(t)).voltage)[0]
        elif self.envelope is not None:
            samples = jnp.asarray(self.envelope.voltage)
            voltage = samples[jnp.minimum(n, samples.shape[0] - 1)]
        else:
            voltage = self.steady_state_voltage
        return {"e0": SampleModeElectricalSignal(voltage=voltage)}, n + 1

    def sample_mode_initial_state(self, simulation_parameters):
        return jnp.asarray(0)


class PRNG(
    SteadyStateComponent,
    # SampleModeComponent,
    BlockModeComponent,
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
    def steady_state(self, inputs: dict, default_output: int = 0):
        outputs = {"l0": default_output}
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
