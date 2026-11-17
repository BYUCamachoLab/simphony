"""CWLaser Jones vector: output split over several modes, in sample and block mode."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from simphony.circuit.circuit import Circuit
from simphony.libraries.ideal.sources import CWLaser
from simphony.simulation.block_mode import BlockModeSimulation, BlockModeSimulationParameters
from simphony.simulation.sample_mode import SampleModeSimulation, SampleModeSimulationParameters

warnings.filterwarnings("ignore", message="Could not validate netlist")

NETLIST = {"instances": {"laser": "laser"}, "connections": {}, "ports": {"out": "laser,o0"}}
MODES = ("te", "tm")


def run(mode, settings, n=8, modes=MODES):
    common = dict(dt=1e-11, num_time_steps=n, mode_identifiers=modes, optical_baseband_wavelengths=jnp.array([1.55e-6]))
    circuit = Circuit(NETLIST, {"laser": CWLaser})
    if mode == "sample":
        sim = SampleModeSimulation(circuit, {"laser": settings}, simulation_parameters=SampleModeSimulationParameters(**common))
    else:
        sim = BlockModeSimulation(circuit, {"laser": settings}, simulation_parameters=BlockModeSimulationParameters(**common))
    return np.asarray(sim.run().output_signals["out"].amplitude)


@pytest.mark.parametrize("mode", ["sample", "block"])
def test_default_puts_all_power_in_mode_idx(mode):
    for idx in (0, 1):
        a = run(mode, {"wavelength": 1.55e-6, "mode_idx": idx})
        assert np.allclose(np.abs(a[..., idx]), 1)
        assert np.allclose(a[..., 1 - idx], 0)


@pytest.mark.parametrize("mode", ["sample", "block"])
def test_jones_split_and_phase(mode):
    a = run(mode, {"wavelength": 1.55e-6, "jones": [1.0, 1j]})
    te, tm = a[..., 0].ravel(), a[..., 1].ravel()
    assert np.allclose(np.abs(te) ** 2, 0.5) and np.allclose(np.abs(tm) ** 2, 0.5)
    assert np.allclose(tm / te, 1j)                       # relative phase is kept


@pytest.mark.parametrize("mode", ["sample", "block"])
def test_jones_is_normalised(mode):
    a = run(mode, {"wavelength": 1.55e-6, "jones": [3.0, 4.0]})
    assert np.allclose(np.sum(np.abs(a) ** 2, axis=-1), 1)
    assert np.allclose(np.abs(a[..., 1] / a[..., 0]), 4 / 3)


def test_jones_matches_sample_and_block():
    s = run("sample", {"wavelength": 1.55e-6, "jones": [0.6, 0.8j]})
    b = run("block", {"wavelength": 1.55e-6, "jones": [0.6, 0.8j]})
    assert np.allclose(s.reshape(-1, 2), b.reshape(-1, 2))


def test_jones_validation():
    with pytest.raises(ValueError, match="entries"):
        run("sample", {"wavelength": 1.55e-6, "jones": [1.0, 0.0, 0.0]})
    with pytest.raises(ValueError, match="non-zero"):
        run("sample", {"wavelength": 1.55e-6, "jones": [0.0, 0.0]})
