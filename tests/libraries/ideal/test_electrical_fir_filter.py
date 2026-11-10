"""Tests for ElectricalFIRFilter and envelope playback with OpticalSource."""

import jax.numpy as jnp
import numpy as np
import pytest

from simphony.circuit.circuit import Circuit
from simphony.libraries.ideal.electrical_circuits import ElectricalFIRFilter
from simphony.libraries.ideal.multirate import Interpolator
from simphony.libraries.ideal.sources import OpticalSource, VoltageSource
from simphony.signal.block_mode import (
    BlockModeElectricalSignal,
    BlockModeOpticalSignal,
)
from simphony.simulation.block_mode import (
    BlockModeSimulation,
    BlockModeSimulationParameters,
)
from simphony.simulation.sample_mode import (
    SampleModeSimulation,
    SampleModeSimulationParameters,
)

DT = 1e-12
TAPS = np.array([0.5, 0.3, 0.15, 0.05])


def _fir_netlist():
    return {
        "instances": {"src": "source", "fir": "fir"},
        "connections": {"src,e0": "fir,e0"},
        "ports": {"out": "fir,e1"},
    }


MODELS = {"source": VoltageSource, "fir": ElectricalFIRFilter}


def _impulse(n):
    v = np.zeros(n)
    v[0] = 1.0
    return BlockModeElectricalSignal(voltage=jnp.asarray(v))


def test_block_mode_impulse_response_equals_taps():
    n = 16
    params = BlockModeSimulationParameters(dt=DT, num_time_steps=n, mode_identifiers=("te",))
    settings = {"src": {"envelope": _impulse(n)}, "fir": {"taps": TAPS}}
    result = BlockModeSimulation(
        Circuit(_fir_netlist(), MODELS), settings, simulation_parameters=params
    ).run()
    out = np.asarray(result.output_signals["out"].voltage)
    expected = np.zeros(n)
    expected[: len(TAPS)] = TAPS
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_sample_mode_impulse_response_equals_taps():
    n = 16
    params = SampleModeSimulationParameters(dt=DT, num_time_steps=n, mode_identifiers=("te",))
    settings = {"src": {"envelope": _impulse(n)}, "fir": {"taps": TAPS}}
    result = SampleModeSimulation(
        Circuit(_fir_netlist(), MODELS), settings, simulation_parameters=params
    ).run()
    out = np.real(np.asarray(result.output_signals["out"].voltage)).ravel()
    # Sample mode adds one sample of latency per edge (source -> filter).
    nz = np.flatnonzero(np.abs(out) > 1e-12)
    np.testing.assert_allclose(out[nz[0] : nz[0] + len(TAPS)], TAPS, atol=1e-6)


def test_group_delay_compensation_aligns_symmetric_filter():
    n = 32
    taps = np.array([0.25, 0.5, 0.25])
    params = BlockModeSimulationParameters(dt=DT, num_time_steps=n, mode_identifiers=("te",))
    ramp = BlockModeElectricalSignal(voltage=jnp.arange(n, dtype=float))
    settings = {
        "src": {"envelope": ramp},
        "fir": {"taps": taps, "delay_compensation": "group"},
    }
    result = BlockModeSimulation(
        Circuit(_fir_netlist(), MODELS), settings, simulation_parameters=params
    ).run()
    out = np.asarray(result.output_signals["out"].voltage)
    # A symmetric smoother leaves the interior of a ramp unchanged once aligned.
    np.testing.assert_allclose(out[1:-1], np.arange(n)[1:-1], atol=1e-5)


def test_sample_mode_rejects_delay_compensation():
    params = SampleModeSimulationParameters(dt=DT, num_time_steps=8, mode_identifiers=("te",))
    fir = ElectricalFIRFilter(params, taps=TAPS, delay_compensation=2)
    with pytest.raises(ValueError, match="causal"):
        fir.sample_mode_initial_state(params)


@pytest.mark.parametrize("bad", [[], [[1.0, 2.0]], -1, 1.5])
def test_invalid_settings(bad):
    params = BlockModeSimulationParameters(dt=DT, num_time_steps=8, mode_identifiers=("te",))
    if isinstance(bad, list):
        with pytest.raises(ValueError):
            ElectricalFIRFilter(params, taps=bad)
    else:
        with pytest.raises(ValueError):
            ElectricalFIRFilter(params, taps=TAPS, delay_compensation=bad)


def test_dac_chain_hold_then_filter():
    """VoltageSource (symbol rate) -> Interpolator(hold) -> FIR = staircase * taps."""
    factor, n_sym = 8, 12
    symbols = np.random.default_rng(0).uniform(-1, 1, n_sym)
    taps = np.hanning(7) / np.hanning(7).sum()
    netlist = {
        "instances": {"src": "source", "dac": "interp", "fir": "fir"},
        "connections": {"src,e0": "dac,in", "dac,out": "fir,e0"},
        "ports": {"out": "fir,e1"},
    }
    models = {**MODELS, "interp": Interpolator}
    settings = {
        "src": {"envelope": BlockModeElectricalSignal(voltage=jnp.asarray(symbols))},
        "dac": {"factor": factor, "mode": "hold"},
        "fir": {"taps": taps},
    }
    params = BlockModeSimulationParameters(
        dt=DT, num_time_steps=n_sym, mode_identifiers=("te",)
    )
    result = BlockModeSimulation(
        Circuit(netlist, models), settings, simulation_parameters=params
    ).run()
    out = np.asarray(result.output_signals["out"].voltage)
    expected = np.convolve(np.repeat(symbols, factor), taps)[: n_sym * factor]
    assert out.shape == (n_sym * factor,)
    np.testing.assert_allclose(out, expected, atol=1e-5)
    assert result.sample_periods["out"] == pytest.approx(DT / factor)


def test_optical_source_replays_envelope_exactly():
    n, wavelengths = 20, jnp.array([1.54e-6, 1.55e-6])
    rng = np.random.default_rng(1)
    amplitude = rng.normal(size=(n, 2, 1)) + 1j * rng.normal(size=(n, 2, 1))
    envelope = BlockModeOpticalSignal(amplitude=jnp.asarray(amplitude), wavelength=wavelengths)
    netlist = {"instances": {"src": "source"}, "connections": {}, "ports": {"out": "src,o0"}}
    params = BlockModeSimulationParameters(
        dt=DT,
        num_time_steps=n + 5,
        mode_identifiers=("te",),
        optical_baseband_wavelengths=wavelengths,
    )
    source = OpticalSource(params, envelope=envelope)
    out = source.block_mode_response({}, params)["o0"]
    np.testing.assert_array_equal(np.asarray(out.amplitude[:n]), amplitude)
    np.testing.assert_array_equal(np.asarray(out.amplitude[n:]), 0)
    # Not mutated: a shorter run still sees the full original envelope.
    assert source.envelope.amplitude.shape[0] == n

    result = BlockModeSimulation(
        Circuit(netlist, {"source": OpticalSource}),
        {"src": {"envelope": envelope}},
        simulation_parameters=params.replace(num_time_steps=n),
    ).run()
    out = result.output_signals["out"].amplitude
    np.testing.assert_array_equal(np.asarray(out), amplitude)
