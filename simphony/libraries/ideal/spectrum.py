"""Spectrum-analysis components: sample buffers, FFTs and a spectrum analyzer.

Optical inputs carry one envelope per tracked carrier. These components first
combine all carriers into a single complex baseband field per optical mode,
around a reference frequency (by default the mean tracked carrier):

    E(t) = sum_l A_l(t) exp(-i 2 pi (f_l - f_ref) t)

(Simphony's physicist convention, e^{-i w t}). They then keep the last N
samples of E. The frequency axis of an N-point spectrum of samples taken every
`dt` is `spectrum_frequencies(N, dt)`, relative to f_ref. Note that `dt` is the
period of the *buffered* signal, not the (possibly decimated) rate at which the
spectra are produced.

`spectrum_analyzer` is a PCell. In sample mode it is SampleBuffer -> Decimator
-> FFT: the buffer ingests every sample, but the FFT only runs on the decimated
frames. In block mode it is StridedBuffer -> FFT, which only builds the
decimated windows.
"""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import speed_of_light

from simphony.component.component import (
    BlockModeComponent,
    RateChanger,
    SampleModeComponent,
)
from simphony.component.pcell import PCell
from simphony.component.port import Port
from simphony.libraries.ideal.multirate import Decimator
from simphony.signal.block_mode import BlockModeVectorSignal
from simphony.signal.sample_mode import SampleModeVectorSignal
from simphony.simulation.simulation import SimulationMode

WINDOWS = ("rect", "hann")


def spectrum_frequencies(n_fft: int, dt: float) -> np.ndarray:
    """Frequency (Hz, relative to the reference carrier) of each FFT bin, in
    numpy's FFT order. Under Simphony's e^{-i w t} convention a field
    component at offset +f appears at the bin whose `fftfreq` is -f, so the
    axis is negated."""
    return -np.fft.fftfreq(n_fft, dt)


def window_function(name: str, n: int) -> jnp.ndarray:
    if name == "rect":
        return jnp.ones(n)
    if name == "hann":
        return 0.5 - 0.5 * jnp.cos(2 * jnp.pi * jnp.arange(n) / n)  # periodic Hann
    raise ValueError(f"window must be one of {WINDOWS}, got {name!r}")


def _reference_frequency(center_wavelength, wavelengths):
    if center_wavelength is not None:
        return speed_of_light / center_wavelength
    return jnp.mean(speed_of_light / wavelengths)


def _baseband_field(amplitude, wavelengths, t, f_ref):
    """Combine carriers: amplitude (..., L, M) at times t (...) -> (..., M)."""
    df = speed_of_light / wavelengths - f_ref  # (L,)
    rotation = jnp.exp(-1j * 2 * jnp.pi * df * jnp.asarray(t)[..., None])  # (..., L)
    return jnp.sum(amplitude * rotation[..., None], axis=-2)


class SampleBuffer(SampleModeComponent, BlockModeComponent):
    """Keep the last `n_samples` samples of the combined baseband field.

    Output port `out` (type `"vector"`) carries the window as an
    `(n_samples, M)` array. In sample mode it is a circular buffer (one O(1)
    write per sample); `signal.ordered()` returns it oldest sample first.
    Before `n_samples` samples have arrived, the window is zero-padded.

    In block mode the output is `(T, n_samples, M)`, i.e. one window per time
    step, which is large; prefer `StridedBuffer` (via `spectrum_analyzer`).
    """

    ports = [
        Port(name="in", type="optical", directionality="input"),
        Port(name="out", type="vector", directionality="output"),
    ]

    def __init__(
        self,
        simulation_parameters,
        *,
        n_samples: int = 1024,
        center_wavelength: float = None,
    ):
        self.n_samples = int(n_samples)
        self.center_wavelength = center_wavelength

    def _num_modes(self, simulation_parameters):
        return len(simulation_parameters.mode_identifiers)

    def sample_mode_output_template(self, port, simulation_parameters):
        if port.name == "out":
            M = self._num_modes(simulation_parameters)
            return SampleModeVectorSignal(
                jnp.zeros((self.n_samples, M), dtype=complex), jnp.asarray(0)
            )
        return super().sample_mode_output_template(port, simulation_parameters)

    def sample_mode_initial_state(self, simulation_parameters):
        M = self._num_modes(simulation_parameters)
        return (jnp.asarray(0), jnp.zeros((self.n_samples, M), dtype=complex))

    def sample_mode_step(self, inputs, state, simulation_state, simulation_parameters):
        # Circular buffer: write one entry per step (O(1)); the window starts at
        # the oldest entry, `start`. Consumers order it only when they use it.
        n, buffer = state
        signal = inputs["in"]
        t = simulation_parameters.time_offset + n * simulation_parameters.dt
        f_ref = _reference_frequency(self.center_wavelength, signal.wavelength)
        sample = _baseband_field(signal.amplitude, signal.wavelength, t, f_ref)  # (M,)
        buffer = buffer.at[n % self.n_samples].set(sample)
        start = (n + 1) % self.n_samples
        return {"out": SampleModeVectorSignal(buffer, start)}, (n + 1, buffer)

    def block_mode_response(self, inputs, simulation_parameters):
        signal = inputs["in"]
        T = signal.amplitude.shape[0]
        t = simulation_parameters.time_offset + jnp.arange(T) * simulation_parameters.dt
        f_ref = _reference_frequency(self.center_wavelength, signal.wavelength)
        E = _baseband_field(signal.amplitude, signal.wavelength, t, f_ref)  # (T, M)
        return {
            "out": BlockModeVectorSignal(_windows(E, self.n_samples, jnp.arange(T)))
        }


def _windows(E, n, ends):
    """Windows of `n` samples of E (T, M) ending at each index in `ends`."""
    padded = jnp.concatenate([jnp.zeros((n - 1,) + E.shape[1:], E.dtype), E], axis=0)
    return jax.vmap(lambda end: jax.lax.dynamic_slice_in_dim(padded, end, n, axis=0))(
        ends
    )


class FFT(SampleModeComponent, BlockModeComponent):
    """FFT of a vector signal along its first per-sample axis.

    Input and output are `"vector"` ports. The input window is multiplied by
    `window` ("rect" or "hann") before an unnormalized `jnp.fft.fft`. Bin
    frequencies are `spectrum_frequencies(N, dt_of_the_buffered_signal)`.
    """

    ports = [
        Port(name="in", type="vector", directionality="input"),
        Port(name="out", type="vector", directionality="output"),
    ]

    def __init__(self, simulation_parameters, *, window: str = "hann"):
        window_function(window, 2)  # validate
        self.window = window

    def _transform(self, x, axis):
        w = window_function(self.window, x.shape[axis])
        shape = [1] * x.ndim
        shape[axis] = -1
        return jnp.fft.fft(x * w.reshape(shape), axis=axis)

    def sample_mode_output_template(self, port, simulation_parameters):
        return None  # same shape as the input window

    def sample_mode_step(self, inputs, state, simulation_state, simulation_parameters):
        spectrum = self._transform(inputs["in"].ordered(), 0)
        return {
            "out": SampleModeVectorSignal(spectrum, jnp.zeros_like(inputs["in"].start))
        }, state

    def block_mode_response(self, inputs, simulation_parameters):
        return {"out": BlockModeVectorSignal(self._transform(inputs["in"].value, 1))}


class BufferedFFT(SampleBuffer):
    """`SampleBuffer` and `FFT` in one component: outputs the spectrum of the
    last `n_samples` samples on every sample."""

    def __init__(
        self,
        simulation_parameters,
        *,
        n_samples: int = 1024,
        window: str = "hann",
        center_wavelength: float = None,
    ):
        super().__init__(
            simulation_parameters,
            n_samples=n_samples,
            center_wavelength=center_wavelength,
        )
        self._fft = FFT(simulation_parameters, window=window)

    def sample_mode_step(self, inputs, state, simulation_state, simulation_parameters):
        outputs, state = super().sample_mode_step(
            inputs, state, simulation_state, simulation_parameters
        )
        window = outputs["out"]
        spectrum = self._fft._transform(window.ordered(), 0)
        return {
            "out": SampleModeVectorSignal(spectrum, jnp.zeros_like(window.start))
        }, state

    def block_mode_response(self, inputs, simulation_parameters):
        out = super().block_mode_response(inputs, simulation_parameters)["out"]
        return {"out": BlockModeVectorSignal(self._fft._transform(out.value, 1))}


class StridedBuffer(RateChanger, BlockModeComponent):
    """Block-mode equivalent of SampleBuffer followed by Decimator.

    Emits only the windows ending at samples `offset, offset + factor, ...`
    (output rate = input rate / `factor`), so it never builds the full
    `(T, n_samples, M)` array.
    """

    ports = [
        Port(name="in", type="optical", directionality="input"),
        Port(name="out", type="vector", directionality="output"),
    ]

    def __init__(
        self,
        simulation_parameters,
        *,
        n_samples: int = 1024,
        factor: int = 1,
        offset: int = 0,
        center_wavelength: float = None,
    ):
        self.n_samples, self.factor, self.offset = (
            int(n_samples),
            int(factor),
            int(offset),
        )
        self.center_wavelength = center_wavelength
        self.rate_ratio = Fraction(1, self.factor)

    def output_phase(self, input_phase, input_period):
        return input_phase + self.offset * input_period

    def block_mode_response(self, inputs, simulation_parameters):
        signal = inputs["in"]
        params = self.input_parameters  # the buffer runs at the input rate
        T = signal.amplitude.shape[0]
        t = params.time_offset + jnp.arange(T) * params.dt
        f_ref = _reference_frequency(self.center_wavelength, signal.wavelength)
        E = _baseband_field(signal.amplitude, signal.wavelength, t, f_ref)
        ends = jnp.arange(self.offset, T, self.factor)
        return {"out": BlockModeVectorSignal(_windows(E, self.n_samples, ends))}


class SpectrumAnalyzer(PCell):
    """Sliding-window spectrum of an optical signal, produced every
    `decimation` samples.

    Port `in` takes the optical signal. Port `out` emits an `(n_fft, M)`
    complex spectrum (unnormalized FFT of the windowed baseband field) at
    1/`decimation` of the input sample rate. The frame m covers the
    `n_fft` samples ending at input sample m*decimation + offset (one sample
    earlier in sample mode). Use `spectrum_frequencies(n_fft, dt_in)` for the
    frequency axis.

    Settings: `n_fft`, `decimation`, `offset`, `window` ("rect"/"hann"),
    `center_wavelength` (reference carrier; default: mean tracked carrier).
    """

    ports = [
        Port(name="in", type="optical", directionality="input"),
        Port(name="out", type="vector", directionality="output"),
    ]

    def __init__(
        self,
        simulation_parameters,
        *,
        n_fft: int = 1024,
        decimation: int = 1,
        offset: int = 0,
        window: str = "hann",
        center_wavelength: float = None,
    ):
        if simulation_parameters.simulation_mode == SimulationMode.BLOCK_MODE:
            self.netlist = {
                "instances": {"buffer": "strided_buffer", "fft": "fft"},
                "connections": {"buffer,out": "fft,in"},
                "ports": {"in": "buffer,in", "out": "fft,out"},
            }
            self.models = {"strided_buffer": StridedBuffer, "fft": FFT}
            self.settings = {
                "buffer": {
                    "n_samples": n_fft,
                    "factor": decimation,
                    "offset": offset,
                    "center_wavelength": center_wavelength,
                },
                "fft": {"window": window},
            }
        else:
            self.netlist = {
                "instances": {
                    "buffer": "buffer",
                    "decimator": "decimator",
                    "fft": "fft",
                },
                "connections": {
                    "buffer,out": "decimator,in",
                    "decimator,out": "fft,in",
                },
                "ports": {"in": "buffer,in", "out": "fft,out"},
            }
            self.models = {"buffer": SampleBuffer, "decimator": Decimator, "fft": FFT}
            self.settings = {
                "buffer": {"n_samples": n_fft, "center_wavelength": center_wavelength},
                "decimator": {"factor": decimation, "offset": offset},
                "fft": {"window": window},
            }
