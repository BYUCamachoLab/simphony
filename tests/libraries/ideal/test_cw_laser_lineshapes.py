"""CWLaser line shapes: argument validation, sample/block consistency, and
spectra against exact theory (Lorentzian, band-limited Gaussian, OU Voigt)."""

import warnings
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import sici

jax.config.update("jax_enable_x64", True)

from simphony.circuit.circuit import Circuit
from simphony.libraries.ideal.sources import FWHM_PER_SIGMA, CWLaser
from simphony.simulation.block_mode import BlockModeSimulation, BlockModeSimulationParameters
from simphony.simulation.sample_mode import SampleModeSimulation, SampleModeSimulationParameters

warnings.filterwarnings("ignore", message="Could not validate netlist")

FS = 50e9
NETLIST = {"instances": {"laser": "laser"}, "connections": {}, "ports": {"out": "laser,o0"}}


def sample_params(n, fs=FS):
    return SampleModeSimulationParameters(dt=1 / fs, num_time_steps=n, mode_identifiers=("te",),
                                          optical_baseband_wavelengths=jnp.array([1.55e-6]))


def block_params(n, fs=FS):
    return BlockModeSimulationParameters(dt=1 / fs, num_time_steps=n, mode_identifiers=("te",),
                                         optical_baseband_wavelengths=jnp.array([1.55e-6]))


class _State:
    # Stand-in for the simulator state: the laser only reads the key.
    def __init__(self, key):
        self.prng_key = key


# ----------------------------------------------------------------- theory helpers

def structure_lorentzian(tau, fwhm):
    return 2 * np.pi * fwhm * np.abs(tau)


def structure_bandlimited(tau, sigma, fc):
    tau = np.abs(tau)
    h0 = sigma**2 / fc
    return 4 * h0 * (np.pi * tau * sici(2 * np.pi * fc * tau)[0] - np.sin(np.pi * fc * tau) ** 2 / fc)


def structure_voigt(tau, lorentzian, sigma, tau_c):
    t = np.abs(tau)
    return (2 * np.pi) ** 2 * 2 * sigma**2 * tau_c**2 * (t / tau_c + np.expm1(-t / tau_c)) + 2 * np.pi * lorentzian * t


def expected_periodogram(structure, n, dt):
    # Exact E|FFT|^2/N^2 of an N-sample record of a stationary phase-noise field.
    k = np.arange(n)
    r = np.exp(-structure(k * dt) / 2) * (1 - k / n)
    full = np.concatenate([r, [0.0], r[:0:-1]])
    return np.real(np.fft.fft(full))[::2] / n


def whittle_fit(power, model, lo, hi):
    def cost(lp):
        m = np.maximum(model(np.exp(lp)), 1e-300)
        return np.sum(power / m + np.log(m))
    return np.exp(minimize_scalar(cost, bounds=(np.log(lo), np.log(hi)), method="bounded",
                                  options={"xatol": 1e-5}).x)


def mean_periodogram(make_laser, params, seeds):
    acc = 0
    for s in seeds:
        phi = np.asarray(make_laser(s).phase_noise(params))
        acc = acc + np.abs(np.fft.fft(np.exp(1j * phi))) ** 2 / phi.size**2
    return acc / len(seeds)


# --------------------------------------------------------------------- validation

class TestArguments:
    @pytest.mark.parametrize(
        "kwargs, message",
        [
            (dict(lineshape="lorentzian", linewidth=1e6, correlation_time=1e-6),
             "'correlation_time' is not a valid argument for the 'lorentzian' lineshape"),
            (dict(lineshape="lorentzian", noise_bandwidth=1e6),
             "'noise_bandwidth' is not a valid argument for the 'lorentzian' lineshape"),
            (dict(lineshape="gaussian", linewidth=1e9, noise_bandwidth=1e6, correlation_time=1e-6),
             "'correlation_time' is not a valid argument for the 'gaussian' lineshape"),
            (dict(lineshape="voigt", linewidth=(1e6, 1e9), correlation_time=1e-6, noise_bandwidth=1e6),
             "'noise_bandwidth' is not a valid argument for the 'voigt' lineshape"),
            (dict(lineshape="gaussian", linewidth=1e9), "requires 'noise_bandwidth'"),
            (dict(lineshape="voigt", linewidth=(1e6, 1e9)), "requires 'correlation_time'"),
            (dict(lineshape="gaussian", linewidth=1e9, noise_bandwidth=1e9), "too large for a Gaussian"),
            (dict(lineshape="voigt", linewidth=(1e6, 1e9), correlation_time=1e-10), "too short"),
            (dict(lineshape="voigt", linewidth=(1e6, 0.0), correlation_time=1e-6), "use the 'lorentzian'"),
            (dict(lineshape="voigt", linewidth=1e9, correlation_time=1e-6), "voigt linewidth must be"),
            (dict(lineshape="voigt", linewidth=(1e6, 1e9, 1.0), correlation_time=1e-6), "voigt linewidth must be"),
            (dict(lineshape="voigt", linewidth={"lorentzian": 1e6, "gauss": 1e9}, correlation_time=1e-6),
             "voigt linewidth must be"),
            (dict(lineshape="voigt", linewidth=(-1e6, 1e9), correlation_time=1e-6), "non-negative"),
            (dict(lineshape="lorentzian", linewidth=(1e6, 1e9)), "takes a single linewidth"),
            (dict(lineshape="square"), "Unrecognized lineshape"),
        ],
    )
    def test_invalid(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            CWLaser(sample_params(16), **kwargs)

    def test_defaults_and_case(self):
        assert CWLaser(sample_params(16)).linewidth == 0.0
        assert CWLaser(sample_params(16), lineshape="Lorentzian", linewidth=1e6).lineshape == "lorentzian"

    def test_voigt_tuple_equals_dict(self):
        p = block_params(256)
        a = CWLaser(p, lineshape="voigt", linewidth=(5e7, 1e9), correlation_time=1e-8)
        b = CWLaser(p, lineshape="voigt", linewidth={"gaussian": 1e9, "lorentzian": 5e7}, correlation_time=1e-8)
        assert a.linewidth == b.linewidth == (5e7, 1e9)
        np.testing.assert_array_equal(a.phase_noise(p), b.phase_noise(p))

    def test_seed_is_a_dataclass_field(self):
        assert replace(sample_params(16), seed=7).seed == 7
        assert replace(block_params(16), seed=7).seed == 7


# ----------------------------------------------------------------- consistency

class TestConsistency:
    def test_lorentzian_unchanged(self):
        # Same numbers as the original implementation: PRNGKey(0) in block mode,
        # the simulator's key unchanged in sample mode.
        p = block_params(1000)
        laser = CWLaser(p, linewidth=1e9)
        std = np.sqrt(2 * np.pi * 1e9 * p.dt)
        expected = np.cumsum(np.asarray(jax.random.normal(jax.random.PRNGKey(0), (1000,))) * std)
        np.testing.assert_allclose(laser.phase_noise(p), expected, rtol=1e-12)
        key = jax.random.PRNGKey(5)
        _, phi = laser.sample_mode_step({}, jnp.asarray(0.0), _State(key), sample_params(1000))
        np.testing.assert_allclose(phi, jax.random.normal(key) * std, rtol=1e-12)

    def test_component_seed_changes_noise(self):
        p = block_params(256)
        a = CWLaser(p, linewidth=1e9).phase_noise(p)
        b = CWLaser(p, linewidth=1e9, seed=1).phase_noise(p)
        assert not np.allclose(a, b)

    def test_gaussian_sample_equals_block(self):
        n = 4096
        settings = {"laser": dict(lineshape="gaussian", linewidth=1e9, noise_bandwidth=2e7, seed=3)}
        fields = []
        for sim, params in ((SampleModeSimulation, sample_params(n)), (BlockModeSimulation, block_params(n))):
            result = sim(Circuit(NETLIST, {"laser": CWLaser}), settings, simulation_parameters=params).run()
            fields.append(np.asarray(result.output_signals["out"].amplitude).reshape(n, -1)[:, 0])
        np.testing.assert_allclose(fields[0], fields[1], atol=1e-9)

    def test_gaussian_overrun_is_nan(self):
        p = sample_params(8)
        laser = CWLaser(p, lineshape="gaussian", linewidth=1e9, noise_bandwidth=2e7)
        state = laser.sample_mode_initial_state(p)
        state = state | {"index": jnp.asarray(8, dtype=jnp.int32)}
        out, _ = laser.sample_mode_step({}, state, _State(jax.random.PRNGKey(0)), p)
        assert np.isnan(np.asarray(out["o0"].amplitude)).all()

    @pytest.mark.parametrize("seed", [0, 4])
    def test_voigt_sample_steps_equal_block_scan(self, seed):
        n = 512
        p = sample_params(n)
        laser = CWLaser(p, lineshape="voigt", linewidth=(5e7, 1e9), correlation_time=1e-8, seed=seed)
        keys = jax.random.split(jax.random.PRNGKey(11), n)
        state = laser.sample_mode_initial_state(p)
        phases = []
        for k in keys:
            _, state = laser.sample_mode_step({}, state, _State(k), p)
            phases.append(state[1])
        np.testing.assert_allclose(np.array(phases), laser.voigt_phase_noise(p, keys=keys), rtol=1e-10, atol=1e-10)


# -------------------------------------------------------------- spectra vs theory

class TestSpectra:
    def test_lorentzian_width(self):
        n, fwhm = 2**14, 1e9
        p = block_params(n)
        power = mean_periodogram(lambda s: CWLaser(p, linewidth=fwhm, seed=s), p, range(1, 9))
        model = lambda w: expected_periodogram(lambda t: structure_lorentzian(t, w), n, p.dt)
        assert whittle_fit(power, model, 0.3 * fwhm, 3 * fwhm) == pytest.approx(fwhm, rel=0.05)

    @pytest.mark.parametrize("fs", [12.5e9, 50e9])
    def test_gaussian_spectrum_independent_of_sample_rate(self, fs):
        # Linear statistics of the averaged spectrum (unbiased, unlike a Whittle
        # fit, whose bins are far from independent when the frequency wanders
        # slowly): second moment and power near the carrier vs exact theory.
        fwhm, fc = 1e9, 2e7
        n = int(2 ** np.ceil(np.log2(30 / fc * fs)))
        p = block_params(n, fs)
        power = mean_periodogram(
            lambda s: CWLaser(p, lineshape="gaussian", linewidth=fwhm, noise_bandwidth=fc, seed=s), p, range(1, 49))
        expected = expected_periodogram(lambda t: structure_bandlimited(t, fwhm / FWHM_PER_SIGMA, fc), n, p.dt)
        f = np.fft.fftfreq(n, p.dt)
        band, core = np.abs(f) < 3e9, np.abs(f) < 0.6e9
        moment = np.sum((f**2 * power)[band]) / np.sum((f**2 * expected)[band])
        assert np.sqrt(moment) == pytest.approx(1, abs=0.04)
        assert power[core].sum() == pytest.approx(expected[core].sum(), abs=0.015)

    def test_voigt_gaussian_part(self):
        lorentzian, gaussian, tau_c = 5e7, 1e9, 2e-8
        n = 2**15
        p = block_params(n)
        power = mean_periodogram(
            lambda s: CWLaser(p, lineshape="voigt", linewidth=(lorentzian, gaussian), correlation_time=tau_c, seed=s),
            p, range(1, 25))
        model = lambda w: expected_periodogram(
            lambda t: structure_voigt(t, lorentzian, w / FWHM_PER_SIGMA, tau_c), n, p.dt)
        assert whittle_fit(power, model, 0.3 * gaussian, 3 * gaussian) == pytest.approx(gaussian, rel=0.1)
