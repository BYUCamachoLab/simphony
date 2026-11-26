"""One-way ports connected to bidirectional ports, and the ideal OpticalIsolator.

A bidirectional port connected to a one-way port receives no wave from it; sample mode feeds such ports a zero
signal (as block mode already does) instead of failing. The isolator passes in -> out and reports the wave
travelling back into `out` on `isolated`.
"""

import jax.numpy as jnp
import numpy as np

from simphony.circuit.circuit import Circuit
from simphony.libraries.ideal.s_parameters import optical_s_parameter
from simphony.libraries.ideal.sources import OpticalIsolator, OpticalSource
from simphony.signal.block_mode import BlockModeOpticalSignal
from simphony.simulation.block_mode import BlockModeSimulation, BlockModeSimulationParameters
from simphony.simulation.sample_mode import SampleModeSimulation, SampleModeSimulationParameters

DT = 1e-12
N = 12
WL = jnp.array([1.55e-6])
S_TEE = {("o1", "o0"): 0.6, ("o2", "o0"): 0.5j, ("o0", "o0"): 0.1,     # a 3-port with a reflection at o0
         ("o0", "o1"): 0.6, ("o0", "o2"): 0.5j, ("o1", "o1"): 0.0, ("o2", "o2"): 0.0,
         ("o1", "o2"): 0.0, ("o2", "o1"): 0.0}


def tee():
    """Frequency-independent, so the element uses S directly (no fit): every port bidirectional by default."""
    return {k: jnp.asarray(v + 0j) for k, v in S_TEE.items()}


def impulse():
    a = np.zeros((N, 1, 1), complex)
    a[0] = 1.0
    return BlockModeOpticalSignal(amplitude=jnp.asarray(a), wavelength=WL)


def run_sample(netlist, models):
    params = SampleModeSimulationParameters(dt=DT, num_time_steps=N, mode_identifiers=("te",),
                                            optical_baseband_wavelengths=WL)
    settings = {"src": {"envelope": impulse()}}
    result = SampleModeSimulation(Circuit(netlist, models), settings, simulation_parameters=params).run()
    return {k: np.asarray(v.amplitude).reshape(N) for k, v in result.output_signals.items()}


def nonzero(x):
    """The single nonzero sample of an impulse response (value), checking that there is exactly one."""
    k = np.flatnonzero(np.abs(x) > 1e-12)
    assert k.size == 1, x
    return x[k[0]]


def test_bidirectional_element_feeding_one_way_ports():
    # tee o1, o2 (bidirectional) -> isolator inputs (one-way): nothing comes back into o1, o2
    netlist = {"instances": {"src": "source", "tee": "tee", "p1": "iso", "p2": "iso"},
               "connections": {"src,o0": "tee,o0", "tee,o1": "p1,in", "tee,o2": "p2,in"},
               "ports": {"t1": "p1,out", "t2": "p2,out"}}
    out = run_sample(netlist, {"source": OpticalSource, "tee": optical_s_parameter(tee), "iso": OpticalIsolator})
    np.testing.assert_allclose(nonzero(out["t1"]), 0.6)
    np.testing.assert_allclose(nonzero(out["t2"]), 0.5j)


def test_isolator_reports_the_back_reflection():
    # source -> isolator -> tee (bidirectional both ways): the tee's reflection at o0 returns to the isolator
    netlist = {"instances": {"src": "source", "iso": "iso", "tee": "tee", "p1": "iso"},
               "connections": {"src,o0": "iso,in", "iso,out": "tee,o0", "tee,o1": "p1,in"},
               "ports": {"isolated": "iso,isolated", "t1": "p1,out"}}
    out = run_sample(netlist, {"source": OpticalSource, "tee": optical_s_parameter(tee), "iso": OpticalIsolator})
    np.testing.assert_allclose(nonzero(out["isolated"]), 0.1)
    np.testing.assert_allclose(nonzero(out["t1"]), 0.6)


def test_isolator_block_mode():
    netlist = {"instances": {"src": "source", "iso": "iso"}, "connections": {"src,o0": "iso,in"},
               "ports": {"out": "iso,out", "isolated": "iso,isolated"}}
    params = BlockModeSimulationParameters(dt=DT, num_time_steps=N, mode_identifiers=("te",),
                                           optical_baseband_wavelengths=WL)
    result = BlockModeSimulation(Circuit(netlist, {"source": OpticalSource, "iso": OpticalIsolator}),
                                 {"src": {"envelope": impulse()}}, simulation_parameters=params).run()
    np.testing.assert_array_equal(np.asarray(result.output_signals["out"].amplitude), np.asarray(impulse().amplitude))
    np.testing.assert_array_equal(np.asarray(result.output_signals["isolated"].amplitude), 0)
