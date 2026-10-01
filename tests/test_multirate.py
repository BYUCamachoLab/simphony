"""Tests for multirate simulation: rate inference, decimators, interpolators,
and the spectrum analyzer."""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from simphony.circuit.circuit import Circuit
from simphony.circuit.rates import infer_sample_rates
from simphony.component.component import BlockModeComponent, SampleModeComponent
from simphony.component.port import Port
from simphony.libraries import old_ideal
from simphony.libraries.ideal.multirate import Decimator, Interpolator
from simphony.libraries.ideal.spectrum import (
    BufferedFFT,
    SpectrumAnalyzer,
    spectrum_frequencies,
)
from simphony.signal.block_mode import (
    BlockModeElectricalSignal,
    BlockModeOpticalSignal,
    BlockModeVectorSignal,
)
from simphony.signal.sample_mode import (
    SampleModeElectricalSignal,
    SampleModeOpticalSignal,
)
from simphony.signal.utils import decimate_block, upsample_block
from simphony.simulation.block_mode import (
    BlockModeSimulation,
    BlockModeSimulationParameters,
)
from simphony.simulation.sample_mode import (
    SampleModeSimulation,
    SampleModeSimulationParameters,
)

DT = 1e-12


class Ramp(SampleModeComponent, BlockModeComponent):
    """Electrical source emitting its own sample index n (0, 1, 2, ...)."""

    ports = [Port(name="e0", type="electrical", directionality="output")]

    def __init__(self, simulation_parameters, **kwargs):
        pass

    def sample_mode_initial_state(self, simulation_parameters):
        return jnp.asarray(0.0)

    def sample_mode_step(self, inputs, state, simulation_state, simulation_parameters):
        return {"e0": SampleModeElectricalSignal(state)}, state + 1.0

    def block_mode_response(self, inputs, simulation_parameters):
        n = jnp.arange(simulation_parameters.num_time_steps, dtype=float)
        return {"e0": BlockModeElectricalSignal(n)}


class Counter(SampleModeComponent, BlockModeComponent):
    """Counts how many times it is evaluated; passes its input through."""

    ports = [
        Port(name="e0", type="electrical", directionality="input"),
        Port(name="e1", type="electrical", directionality="output"),
    ]

    def __init__(self, simulation_parameters, **kwargs):
        pass

    def sample_mode_initial_state(self, simulation_parameters):
        return jnp.asarray(0.0)

    def sample_mode_step(self, inputs, state, simulation_state, simulation_parameters):
        return {"e1": SampleModeElectricalSignal(state + 1.0)}, state + 1.0

    def block_mode_response(self, inputs, simulation_parameters):
        return {"e1": inputs["e0"]}


class Laser(SampleModeComponent):
    """CW carrier at a given offset (Hz) from the tracked carrier."""

    ports = [Port(name="o0", type="optical", directionality="output")]

    def __init__(self, simulation_parameters, *, offset=0.0, wavelength=1.55e-6):
        self.offset, self.wavelength = offset, wavelength

    def sample_mode_initial_state(self, simulation_parameters):
        return jnp.asarray(0.0)

    def sample_mode_step(self, inputs, state, simulation_state, simulation_parameters):
        t = simulation_parameters.time_offset + state * simulation_parameters.dt
        a = jnp.exp(-2j * jnp.pi * self.offset * t)
        amplitude = jnp.zeros((1, 1), complex).at[0, 0].set(a)
        return {
            "o0": SampleModeOpticalSignal(amplitude, jnp.array([self.wavelength]))
        }, state + 1


def chain(*stages, out_port="e1"):
    """Netlist: ramp -> stage0 -> stage1 -> ... with `out` at the last stage."""
    instances = {"src": "ramp"}
    connections = {}
    prev = "src,e0"
    for i, model in enumerate(stages):
        name = f"s{i}"
        instances[name] = model
        in_port = "e0" if model == "counter" else "in"
        connections[prev] = f"{name},{in_port}"
        prev = f"{name},{'e1' if model == 'counter' else 'out'}"
    return {"instances": instances, "connections": connections, "ports": {"out": prev}}


MODELS = {"ramp": Ramp, "counter": Counter, "dec": Decimator, "interp": Interpolator}


def sample_params(n, **kw):
    return SampleModeSimulationParameters(
        dt=DT, num_time_steps=n, mode_identifiers=("te",), **kw
    )


# ------------------------------------------------------------------ inference
def test_rate_inference_periods_and_phases():
    netlist = chain("dec", "counter", "interp")
    settings = {"s0": {"factor": 8, "offset": 3}, "s2": {"factor": 8}}
    ic = Circuit(netlist, MODELS).instantiate(settings, sample_params(64))
    schedule = infer_sample_rates(ic, sample_params(64))
    assert schedule.instance_domain("src").period == 1
    assert schedule.instance_domain("s0").period == 8
    assert schedule.instance_domain("s1").period == 8
    assert schedule.instance_domain("s1").phase == 3
    assert schedule.instance_domain("s2").period == 1
    assert schedule.instance_domain("s1").dt == pytest.approx(8 * DT)
    assert schedule.instance_domain("s1").num_time_steps == 8
    assert schedule.hyperperiod == 8


def test_rate_inference_rejects_inconsistent_merge():
    class Adder(SampleModeComponent):
        ports = [
            Port(name="a", type="electrical", directionality="input"),
            Port(name="b", type="electrical", directionality="input"),
            Port(name="o", type="electrical", directionality="output"),
        ]

        def __init__(self, simulation_parameters, **kwargs):
            pass

    # src -> /4 -> add.a, and a second source straight into add.b
    netlist = {
        "instances": {"src": "ramp", "src2": "ramp", "dec": "dec", "add": "adder"},
        "connections": {"src,e0": "dec,in", "dec,out": "add,a", "src2,e0": "add,b"},
        "ports": {"out": "add,o"},
    }
    models = dict(MODELS, adder=Adder)
    ic = Circuit(netlist, models).instantiate({"dec": {"factor": 4}}, sample_params(16))
    with pytest.raises(ValueError, match="different net up/down-sampling"):
        infer_sample_rates(ic, sample_params(16))


def test_num_time_steps_must_cover_whole_periods():
    ic = Circuit(chain("dec"), MODELS).instantiate(
        {"s0": {"factor": 8}}, sample_params(20)
    )
    with pytest.raises(ValueError, match="num_time_steps=24"):
        infer_sample_rates(ic, sample_params(20))


# ---------------------------------------------------------------- block mode
def test_block_helpers_any_signal_type():
    x = jnp.arange(12.0)
    optical = BlockModeOpticalSignal(
        jnp.stack([x, -x], 1)[:, None, :], jnp.array([1.55e-6])
    )
    vec = BlockModeVectorSignal(jnp.stack([x, 2 * x], 1))
    for sig in (BlockModeElectricalSignal(x), optical, vec):
        d = decimate_block(sig, 4, 1)
        field = type(sig)._data_fields[0]
        np.testing.assert_array_equal(getattr(d, field), getattr(sig, field)[1::4])
    np.testing.assert_array_equal(
        decimate_block(optical, 4, 1).wavelength, optical.wavelength
    )
    up = upsample_block(
        BlockModeElectricalSignal(jnp.arange(3.0)), 3, 1, "zeros"
    ).voltage
    np.testing.assert_array_equal(up, [0, 0, 0, 0, 1, 0, 0, 2, 0])
    hold = upsample_block(
        BlockModeElectricalSignal(jnp.arange(3.0)), 3, 0, "hold"
    ).voltage
    np.testing.assert_array_equal(hold, [0, 0, 0, 1, 1, 1, 2, 2, 2])


@pytest.mark.parametrize("offset", [0, 3, 7])
def test_block_mode_decimate_then_interpolate(offset):
    params = BlockModeSimulationParameters(
        dt=DT, num_time_steps=64, mode_identifiers=("te",)
    )
    settings = {
        "s0": {"factor": 8, "offset": offset},
        "s1": {"factor": 8, "mode": "hold"},
    }
    netlist = chain("dec", "interp")
    netlist["ports"]["mid"] = "s0,out"
    result = BlockModeSimulation(
        Circuit(netlist, MODELS), settings, simulation_parameters=params
    ).run()
    mid = np.asarray(result.output_signals["mid"].voltage)
    out = np.asarray(result.output_signals["out"].voltage)
    np.testing.assert_array_equal(mid, np.arange(64)[offset::8])
    np.testing.assert_array_equal(out, np.repeat(np.arange(64)[offset::8], 8))
    assert result.sample_periods["mid"] == pytest.approx(8 * DT)


# --------------------------------------------------------------- sample mode
@pytest.mark.parametrize("offset", range(8))
def test_sample_mode_decimator_offset(offset):
    settings = {"s0": {"factor": 8, "offset": offset}}
    result = SampleModeSimulation(
        Circuit(chain("dec"), MODELS), settings, simulation_parameters=sample_params(64)
    ).run()
    out = np.asarray(result.output_signals["out"].voltage)
    # block-mode equation plus the usual one-sample edge latency
    expected = np.arange(64)[offset::8] - 1.0
    np.testing.assert_array_equal(out, np.maximum(expected, 0))
    assert result.sample_periods["out"] == pytest.approx(8 * DT)


def test_slow_components_fire_at_the_slow_rate():
    settings = {"s0": {"factor": 8}}
    result = SampleModeSimulation(
        Circuit(chain("dec", "counter"), MODELS),
        settings,
        simulation_parameters=sample_params(64),
    ).run()
    count = np.asarray(result.output_signals["out"].voltage)
    assert count.shape == (8,)  # recorded at the slow rate
    np.testing.assert_array_equal(count, np.arange(1, 9))  # evaluated 8 times, not 64


@pytest.mark.parametrize(
    "mode, offset", [("hold", 0), ("hold", 2), ("zeros", 0), ("zeros", 2)]
)
def test_sample_mode_interpolator(mode, offset):
    settings = {
        "s0": {"factor": 4},
        "s1": {"factor": 4, "offset": offset, "mode": mode},
    }
    result = SampleModeSimulation(
        Circuit(chain("dec", "interp"), MODELS),
        settings,
        simulation_parameters=sample_params(32),
    ).run()
    out = np.asarray(result.output_signals["out"].voltage)
    slow = np.maximum(np.arange(32)[::4] - 1.0, 0)  # decimator output (sample mode)
    # interpolator: block-mode equation applied to the previous slow sample
    previous = np.concatenate([[0.0], slow[:-1]])
    block = np.asarray(
        upsample_block(
            BlockModeElectricalSignal(jnp.asarray(previous)), 4, offset, mode
        ).voltage
    )
    np.testing.assert_array_equal(out, block)


@pytest.mark.parametrize(
    "spectral_range, warns",
    [((1.5499e-6, 1.5501e-6), False), ((1.549e-6, 1.551e-6), True)],
)
def test_s_parameter_element_fits_at_local_dt(spectral_range, warns):
    from simphony.libraries.ideal.s_parameters import SParameterElement
    from simphony.simulation.sample_mode import SampleModeSimulation as Sim

    netlist = {
        "instances": {"laser": "laser", "dec": "dec", "wg": "waveguide"},
        "connections": {"laser,o0": "dec,in", "dec,out": "wg,o0"},
        "ports": {"out": "wg,o1"},
    }
    models = {"laser": Laser, "dec": Decimator, "waveguide": old_ideal.waveguide}
    params = SampleModeSimulationParameters(
        dt=DT,
        num_time_steps=64,
        mode_identifiers=("te",),
        optical_baseband_wavelengths=jnp.array([1.55e-6]),
    )
    settings = {
        "dec": {"factor": 8},
        "wg": {
            "sax_settings": {"length": 10.0},
            "vector_fitting_parameters": {
                "model_order": 4,
                "num_frequency_samples": 50,
                "spectral_range": spectral_range,
            },
        },
    }
    sim = Sim(Circuit(netlist, models), settings, simulation_parameters=params)
    # the /8 region samples at 125 GHz: a 250 GHz fit band aliases, 25 GHz does not
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.run()
    assert any("will alias" in str(w.message) for w in caught) == warns
    wg = next(c for c in sim.components.values() if isinstance(c, SParameterElement))
    (key,) = wg._state_space_cache.keys()
    assert key[2] == pytest.approx(8 * DT)


# ------------------------------------------------------------ spectrum analyzer
@pytest.mark.parametrize("decimation", [1, 16])
def test_spectrum_analyzer_finds_tone(decimation):
    n_fft, n = 64, 256
    df = 1 / (n_fft * DT)
    netlist = {
        "instances": {"laser": "laser", "sa": "sa"},
        "connections": {"laser,o0": "sa,in"},
        "ports": {"spectrum": "sa,out"},
    }
    params = SampleModeSimulationParameters(
        dt=DT,
        num_time_steps=n,
        mode_identifiers=("te",),
        optical_baseband_wavelengths=jnp.array([1.55e-6]),
    )
    settings = {
        "laser": {"offset": 5 * df},
        "sa": {"n_fft": n_fft, "decimation": decimation, "window": "rect"},
    }
    result = SampleModeSimulation(
        Circuit(netlist, {"laser": Laser, "sa": SpectrumAnalyzer}),
        settings,
        simulation_parameters=params,
    ).run()
    spectra = np.asarray(result.output_signals["spectrum"].value)
    assert spectra.shape == (n // decimation, n_fft, 1)
    freqs = spectrum_frequencies(n_fft, DT)
    peak = freqs[np.argmax(np.abs(spectra[-1, :, 0]))]
    assert peak == pytest.approx(5 * df)


def test_buffer_decimate_fft_matches_buffered_fft_decimate():
    n_fft, n, M = 32, 128, 8
    params = SampleModeSimulationParameters(
        dt=DT,
        num_time_steps=n,
        mode_identifiers=("te",),
        optical_baseband_wavelengths=jnp.array([1.55e-6]),
    )
    laser = {"offset": 3.3 / (n_fft * DT)}
    a = (
        SampleModeSimulation(
            Circuit(
                {
                    "instances": {"laser": "laser", "sa": "sa"},
                    "connections": {"laser,o0": "sa,in"},
                    "ports": {"s": "sa,out"},
                },
                {"laser": Laser, "sa": SpectrumAnalyzer},
            ),
            {"laser": laser, "sa": {"n_fft": n_fft, "decimation": M, "offset": 5}},
            simulation_parameters=params,
        )
        .run()
        .output_signals["s"]
        .value
    )
    b = (
        SampleModeSimulation(
            Circuit(
                {
                    "instances": {"laser": "laser", "fft": "fft", "dec": "dec"},
                    "connections": {"laser,o0": "fft,in", "fft,out": "dec,in"},
                    "ports": {"s": "dec,out"},
                },
                {"laser": Laser, "fft": BufferedFFT, "dec": Decimator},
            ),
            {
                "laser": laser,
                "fft": {"n_samples": n_fft},
                "dec": {"factor": M, "offset": 5},
            },
            simulation_parameters=params,
        )
        .run()
        .output_signals["s"]
        .value
    )
    # In sample mode every edge adds one sample of latency at the producing
    # rate. In the analyzer the FFT sits after the decimator, so its edge costs
    # one *slow* sample: analyzer frame m equals BufferedFFT -> decimator frame m-1.
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape == (n // M, n_fft, 1)
    np.testing.assert_allclose(a[1:], b[:-1], atol=1e-9)


def test_cond_fallback_matches_static_schedule(monkeypatch):
    settings = {
        "s0": {"factor": 4, "offset": 1},
        "s2": {"factor": 4, "offset": 2, "mode": "zeros"},
    }
    netlist = chain("dec", "counter", "interp")
    static = SampleModeSimulation(
        Circuit(netlist, MODELS), settings, simulation_parameters=sample_params(32)
    ).run()
    monkeypatch.setattr(SampleModeSimulation, "max_schedule_runs", 0)
    dynamic = SampleModeSimulation(
        Circuit(netlist, MODELS), settings, simulation_parameters=sample_params(32)
    ).run()
    np.testing.assert_array_equal(
        np.asarray(static.output_signals["out"].voltage),
        np.asarray(dynamic.output_signals["out"].voltage),
    )


def test_block_mode_spectrum_analyzer():
    class BlockLaser(BlockModeComponent):
        ports = [Port(name="o0", type="optical", directionality="output")]

        def __init__(self, simulation_parameters, *, offset=0.0):
            self.offset = offset

        def block_mode_response(self, inputs, simulation_parameters):
            t = (
                jnp.arange(simulation_parameters.num_time_steps)
                * simulation_parameters.dt
            )
            a = jnp.exp(-2j * jnp.pi * self.offset * t)[:, None, None]
            return {
                "o0": BlockModeOpticalSignal(
                    a, simulation_parameters.optical_baseband_wavelengths
                )
            }

    n_fft, n, M = 64, 512, 32
    params = BlockModeSimulationParameters(
        dt=DT,
        num_time_steps=n,
        mode_identifiers=("te",),
        optical_baseband_wavelengths=jnp.array([1.55e-6]),
    )
    df = 1 / (n_fft * DT)
    netlist = {
        "instances": {"laser": "laser", "sa": "sa"},
        "connections": {"laser,o0": "sa,in"},
        "ports": {"spectrum": "sa,out"},
    }
    result = BlockModeSimulation(
        Circuit(netlist, {"laser": BlockLaser, "sa": SpectrumAnalyzer}),
        {
            "laser": {"offset": -7 * df},
            "sa": {"n_fft": n_fft, "decimation": M, "offset": 3, "window": "rect"},
        },
        simulation_parameters=params,
    ).run()
    spectra = np.asarray(result.output_signals["spectrum"].value)
    assert spectra.shape == (n // M, n_fft, 1)
    assert result.sample_periods["spectrum"] == pytest.approx(M * DT)
    freqs = spectrum_frequencies(n_fft, DT)
    assert freqs[np.argmax(np.abs(spectra[-1, :, 0]))] == pytest.approx(-7 * df)
    # the last window is full: |X| at the peak equals n_fft for a unit tone
    assert np.max(np.abs(spectra[-1])) == pytest.approx(n_fft, rel=1e-9)
