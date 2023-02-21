# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import jax.numpy as np
#import numpy as np
import pytest
import os

from simphony.libraries import siepic
from simphony.tools import wl2freq

mzi_s_parameters = np.load(os.path.join(os.path.dirname(__file__), "mzi_sparameters.npy"))

mzi_monte_carlo_s_parameters = np.load(os.path.join(os.path.dirname(__file__), "mzi_monte_carlo_sparameters.npy"))


@pytest.fixture(scope="module")
def freqs():
    return np.linspace(wl2freq(1600e-9), wl2freq(1500e-9))


@pytest.fixture
def mzi():
    gc_input = siepic.GratingCoupler()
    y_splitter = siepic.YBranch()
    wg_long = siepic.Waveguide(length=150e-6)
    wg_short = siepic.Waveguide(length=50e-6)
    y_recombiner = siepic.YBranch()
    gc_output = siepic.GratingCoupler()

    y_splitter.multiconnect(gc_input, wg_long, wg_short)
    y_recombiner.multiconnect(gc_output, wg_short, wg_long)

    return y_splitter.circuit


class TestCircuit:
    def test_s_parameters(self, freqs, mzi):
        assert np.allclose(mzi_s_parameters, mzi.s_parameters(freqs))

    def test_monte_carlo_s_parameters(self, freqs, mzi):
        assert not np.allclose(
            mzi_monte_carlo_s_parameters,
            mzi.to_subcircuit(permanent=False).monte_carlo_s_parameters(freqs),
        )
