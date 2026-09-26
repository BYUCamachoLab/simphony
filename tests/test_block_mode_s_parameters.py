"""Tests for the SAX-interpreting `SParameterElement` in block mode: per-instance
directionality, fusing of adjacent elements, and the backward pass."""

import jax.numpy as jnp
import numpy as np
import pytest
import sax

import simphony.libraries.ideal.s_parameters as s_parameters
from simphony.circuit.circuit import Circuit
from simphony.component.component import BlockModeComponent
from simphony.component.port import Port
from simphony.libraries import old_ideal
from simphony.libraries.ideal.modulators import DirectedOpticalModulator
from simphony.libraries.ideal.s_parameters import (
    SParameterElement,
    merge_vector_fitting_parameters,
    optical_s_parameter,
)
from simphony.libraries.ideal.sources import VoltageSource
from simphony.signal.block_mode import BlockModeOpticalSignal
from simphony.simulation.block_mode import (
    BlockModeSimulation,
    BlockModeSimulationParameters,
)
from simphony.simulation.sample_mode import SampleModeSimulationParameters

WAVELENGTHS = jnp.array([1.549e-6, 1.55e-6, 1.551e-6])
VECTOR_FIT = {"model_order": 20, "num_frequency_samples": 400}


class Laser(BlockModeComponent):
    ports = [Port(name="o0", type="optical", directionality="output")]

    def __init__(self, simulation_parameters, *, amplitude=1.0):
        self.amplitude = amplitude

    def block_mode_response(self, inputs, simulation_parameters):
        T = simulation_parameters.num_time_steps
        wls = simulation_parameters.optical_baseband_wavelengths
        M = len(simulation_parameters.mode_identifiers)
        amplitude = jnp.zeros((T, wls.shape[0], M), dtype=complex)
        amplitude = amplitude.at[:, :, 0].set(self.amplitude)
        return {"o0": BlockModeOpticalSignal(amplitude=amplitude, wavelength=wls)}


def mirror(r=0.3, t=0.9):
    """Wavelength-independent, symmetric partial reflector."""
    return {
        ("o0", "o0"): r,
        ("o1", "o1"): r,
        ("o0", "o1"): t,
        ("o1", "o0"): t,
    }


def block_parameters(**kwargs):
    return BlockModeSimulationParameters(
        mode_identifiers=("te",),
        dt=1e-14,
        num_time_steps=400,
        optical_baseband_wavelengths=WAVELENGTHS,
        **kwargs,
    )


def waveguide_chain(n=3, lengths=(10.0, 25.0, 40.0)):
    instances = {"laser": "laser"}
    connections = {"laser,o0": "wg0,o0"}
    for i in range(n):
        instances[f"wg{i}"] = "waveguide"
        if i > 0:
            connections[f"wg{i - 1},o1"] = f"wg{i},o0"
    netlist = {
        "instances": instances,
        "connections": connections,
        "ports": {"out": f"wg{n - 1},o1"},
    }
    models = {"laser": Laser, "waveguide": old_ideal.waveguide}
    settings = {
        f"wg{i}": {
            "sax_settings": {"length": lengths[i]},
            "vector_fitting_parameters": dict(VECTOR_FIT),
        }
        for i in range(n)
    }
    return netlist, models, settings


def sax_transmission(lengths):
    wl_um = np.asarray(WAVELENGTHS) * 1e6
    total = np.ones_like(wl_um, dtype=complex)
    for length in lengths:
        total = total * np.asarray(
            old_ideal.waveguide(wl=wl_um, length=length)[("o0", "o1")]
        )
    return total


def s_parameter_elements(simulation):
    instances = simulation._instantiated_circuit.instantiated_flat_netlist["instances"]
    return {
        name: data["model"]
        for name, data in instances.items()
        if isinstance(data["model"], SParameterElement)
    }


def test_cw_steady_state_matches_sax():
    netlist, models, settings = waveguide_chain()
    result = BlockModeSimulation(
        Circuit(netlist, models), settings, simulation_parameters=block_parameters()
    ).run()
    steady = np.asarray(result.output_signals["out"].amplitude[-1, :, 0])
    np.testing.assert_allclose(steady, sax_transmission((10.0, 25.0, 40.0)), atol=2e-3)


def test_per_instance_port_directionality():
    netlist = {
        "instances": {"laser": "laser", "wg0": "waveguide", "wg1": "waveguide"},
        # wg1 is traversed from o1 to o0
        "connections": {"laser,o0": "wg0,o0", "wg0,o1": "wg1,o1"},
        "ports": {"out": "wg1,o0"},
    }
    models = {"laser": Laser, "waveguide": old_ideal.waveguide}
    settings = {
        "wg0": {
            "sax_settings": {"length": 10.0},
            "vector_fitting_parameters": VECTOR_FIT,
        },
        "wg1": {
            "sax_settings": {"length": 25.0},
            "vector_fitting_parameters": VECTOR_FIT,
        },
    }
    simulation = BlockModeSimulation(
        Circuit(netlist, models), settings, simulation_parameters=block_parameters()
    )
    result = simulation.run()
    elements = s_parameter_elements(simulation)

    assert set(elements["wg0"]._input_port_lookup_table) == {"o0"}
    assert set(elements["wg1"]._input_port_lookup_table) == {"o1"}
    # The class itself is untouched
    assert all(p.directionality == "bidirectional" for p in type(elements["wg0"]).ports)

    steady = np.asarray(result.output_signals["out"].amplitude[-1, :, 0])
    np.testing.assert_allclose(steady, sax_transmission((10.0, 25.0)), atol=2e-3)


@pytest.mark.parametrize("backward_pass, expected", [(False, 1), (True, 2)])
def test_fit_is_directional_without_backward_pass(backward_pass, expected):
    netlist, models, settings = waveguide_chain(n=1)
    params = block_parameters(backward_pass=backward_pass)
    simulation = BlockModeSimulation(
        Circuit(netlist, models), settings, simulation_parameters=params
    )
    simulation.run()
    state_space = s_parameter_elements(simulation)["wg0"]._state_space(params)
    assert len(state_space["inputs"]) == expected
    assert len(state_space["outputs"]) == expected


def test_fused_matches_unfused_with_fewer_fits(monkeypatch):
    calls = []
    original = s_parameters.vector_fitting_discrete

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(s_parameters, "vector_fitting_discrete", counting)

    netlist, models, settings = waveguide_chain()
    for settings_i in settings.values():
        settings_i["vector_fitting_parameters"] = {
            "model_order": 20,
            "num_frequency_samples": 400,
        }
    results = {}
    for fuse in (False, True):
        calls.clear()
        simulation = BlockModeSimulation(
            Circuit(netlist, models),
            settings,
            simulation_parameters=block_parameters(),
            fuse_s_parameters=fuse,
        )
        results[fuse] = simulation.run()
        results[fuse, "calls"] = len(calls)
        results[fuse, "elements"] = s_parameter_elements(simulation)

    assert results[False, "calls"] == 3
    assert results[True, "calls"] == 1
    (fused,) = results[True, "elements"].values()
    assert set(fused.members) == {"wg0", "wg1", "wg2"}

    unfused = np.asarray(results[False].output_signals["out"].amplitude[-1])
    fused_out = np.asarray(results[True].output_signals["out"].amplitude[-1])
    np.testing.assert_allclose(fused_out, unfused, atol=2e-3)


@pytest.mark.parametrize(
    "groups, expected_elements",
    [
        ((0, 0, 0), 1),
        ((0, -1, 0), 3),
        ((0, 0, 1), 2),
        ((-1, -1, -1), 3),
    ],
)
def test_s_parameter_groups(groups, expected_elements):
    netlist, models, settings = waveguide_chain()
    for i, group in enumerate(groups):
        settings[f"wg{i}"]["s_parameter_group"] = group
    simulation = BlockModeSimulation(
        Circuit(netlist, models),
        settings,
        simulation_parameters=block_parameters(),
        fuse_s_parameters=True,
    )
    simulation.run()
    assert len(s_parameter_elements(simulation)) == expected_elements


def test_fused_vector_fitting_parameters_are_merged_and_overridable():
    params = block_parameters()
    element_class = optical_s_parameter(old_ideal.waveguide)
    members = {
        "a": element_class(
            params,
            vector_fitting_parameters={
                "min_model_order": 4,
                "max_model_order": 10,
                "spectral_range": (1.52e-6, 1.56e-6),
            },
        ),
        "b": element_class(
            params,
            vector_fitting_parameters={
                "model_order": 20,
                "spectral_range": (1.54e-6, 1.58e-6),
            },
        ),
    }
    merged = merge_vector_fitting_parameters("g", members)
    assert merged["spectral_range"] == (1.52e-6, 1.58e-6)
    assert merged["model_order"] is None
    assert (merged["min_model_order"], merged["max_model_order"]) == (4, 20)

    members["b"].settings["vector_fitting_parameters"]["center_wavelength"] = 1.56e-6
    with pytest.raises(ValueError, match="center_wavelength"):
        merge_vector_fitting_parameters("g", members)

    netlist, models, settings = waveguide_chain()
    simulation = BlockModeSimulation(
        Circuit(netlist, models),
        settings,
        simulation_parameters=params,
        fuse_s_parameters=True,
        s_parameter_group_settings={
            0: {"vector_fitting_parameters": {"model_order": 6}}
        },
    )
    simulation.run()
    (fused,) = s_parameter_elements(simulation).values()
    assert fused.settings["vector_fitting_parameters"]["model_order"] == 6


def test_fusing_is_block_mode_only():
    netlist, models, settings = waveguide_chain()
    with pytest.raises(NotImplementedError):
        Circuit(netlist, models).instantiate(
            settings,
            SampleModeSimulationParameters(mode_identifiers=("te",)),
            fuse_s_parameters=True,
        )


def _mirror_circuit(with_modulator):
    instances = {"laser": "laser", "m1": "mirror", "m2": "mirror"}
    if with_modulator:
        instances.update(mod="modulator", vs="voltage_source")
        connections = {
            "laser,o0": "m1,o0",
            "m1,o1": "mod,o0",
            "mod,o1": "m2,o0",
            "vs,e0": "mod,e0",
        }
    else:
        connections = {"laser,o0": "m1,o0", "m1,o1": "m2,o0"}
    netlist = {
        "instances": instances,
        "connections": connections,
        "ports": {"in": "m1,o0", "out": "m2,o1"},
    }
    models = {
        "laser": Laser,
        "mirror": mirror,
        "modulator": DirectedOpticalModulator,
        "voltage_source": VoltageSource,
    }
    settings = {
        "m1": {"r": 0.3, "t": 0.9},
        "m2": {"r": 0.2, "t": 0.95},
        "mod": {"phase_coefficients": jnp.array([0.0, 0.0, jnp.pi / 2, 0.0])},
        "vs": {"steady_state_voltage": 1.0},
    }
    settings = {k: v for k, v in settings.items() if k in instances}
    return Circuit(netlist, models), settings


@pytest.mark.parametrize("with_modulator", [False, True])
def test_backward_pass_accumulates_reflections(with_modulator):
    circuit, settings = _mirror_circuit(with_modulator)
    result = BlockModeSimulation(
        circuit, settings, simulation_parameters=block_parameters(backward_pass=True)
    ).run()

    # First order in back-reflection: r1 + t1 * (m * r2 * m) * t1, where the
    # modulator transfer m is j (a pi/2 phase shift) in both directions.
    m = 1j if with_modulator else 1.0
    expected_reflection = 0.3 + 0.9 * m * 0.2 * m * 0.9
    reflection = np.asarray(result.backward_signals["in"].amplitude[:, :, 0])
    np.testing.assert_allclose(reflection, expected_reflection, atol=1e-12)

    transmission = np.asarray(result.output_signals["out"].amplitude[:, :, 0])
    np.testing.assert_allclose(transmission, 0.9 * m * 0.95, atol=1e-12)


def test_no_backward_signals_without_backward_pass():
    circuit, settings = _mirror_circuit(with_modulator=False)
    result = BlockModeSimulation(
        circuit, settings, simulation_parameters=block_parameters()
    ).run()
    assert result.backward_signals == {}


def test_fusing_that_creates_a_loop_raises():
    # cp1 and cp2 are directly connected (o3 -> o2) and also through a
    # modulator (o1 -> mod -> o0); fusing them would create a feedback loop.
    netlist = {
        "instances": {
            "laser": "laser",
            "cp1": "coupler",
            "cp2": "coupler",
            "mod": "modulator",
            "vs": "voltage_source",
        },
        "connections": {
            "laser,o0": "cp1,o0",
            "cp1,o1": "mod,o0",
            "mod,o1": "cp2,o0",
            "cp1,o3": "cp2,o2",
            "vs,e0": "mod,e0",
        },
        "ports": {"out": "cp2,o1"},
    }
    models = {
        "laser": Laser,
        "coupler": old_ideal.coupler,
        "modulator": DirectedOpticalModulator,
        "voltage_source": VoltageSource,
    }
    settings = {"cp1": {}, "cp2": {}, "mod": {}, "vs": {}}
    simulation = BlockModeSimulation(
        Circuit(netlist, models),
        settings,
        simulation_parameters=block_parameters(),
        fuse_s_parameters=True,
    )
    with pytest.raises(ValueError, match="s_parameter_group"):
        simulation.run()


def test_unknown_settings_are_rejected():
    element_class = optical_s_parameter(old_ideal.waveguide)
    with pytest.raises(TypeError, match="sax_settings"):
        element_class(block_parameters(), length=10.0, vector_fitting_parameters={})
