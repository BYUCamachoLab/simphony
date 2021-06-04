# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import numpy as np
import pytest

from simphony.libraries import siepic
from simphony.simulators import MonteCarloSweepSimulator, SweepSimulator
from simphony.tools import wl2freq

sweep_freqs = np.array(
    [
        1.87370286e14,
        1.87496462e14,
        1.87622637e14,
        1.87748812e14,
        1.87874987e14,
        1.88001163e14,
        1.88127338e14,
        1.88253513e14,
        1.88379688e14,
        1.88505864e14,
        1.88632039e14,
        1.88758214e14,
        1.88884390e14,
        1.89010565e14,
        1.89136740e14,
        1.89262915e14,
        1.89389091e14,
        1.89515266e14,
        1.89641441e14,
        1.89767617e14,
        1.89893792e14,
        1.90019967e14,
        1.90146142e14,
        1.90272318e14,
        1.90398493e14,
        1.90524668e14,
        1.90650843e14,
        1.90777019e14,
        1.90903194e14,
        1.91029369e14,
        1.91155545e14,
        1.91281720e14,
        1.91407895e14,
        1.91534070e14,
        1.91660246e14,
        1.91786421e14,
        1.91912596e14,
        1.92038771e14,
        1.92164947e14,
        1.92291122e14,
        1.92417297e14,
        1.92543473e14,
        1.92669648e14,
        1.92795823e14,
        1.92921998e14,
        1.93048174e14,
        1.93174349e14,
        1.93300524e14,
        1.93426700e14,
        1.93552875e14,
        1.93679050e14,
        1.93805225e14,
        1.93931401e14,
        1.94057576e14,
        1.94183751e14,
        1.94309926e14,
        1.94436102e14,
        1.94562277e14,
        1.94688452e14,
        1.94814628e14,
        1.94940803e14,
        1.95066978e14,
        1.95193153e14,
        1.95319329e14,
        1.95445504e14,
        1.95571679e14,
        1.95697855e14,
        1.95824030e14,
        1.95950205e14,
        1.96076380e14,
        1.96202556e14,
        1.96328731e14,
        1.96454906e14,
        1.96581081e14,
        1.96707257e14,
        1.96833432e14,
        1.96959607e14,
        1.97085783e14,
        1.97211958e14,
        1.97338133e14,
        1.97464308e14,
        1.97590484e14,
        1.97716659e14,
        1.97842834e14,
        1.97969010e14,
        1.98095185e14,
        1.98221360e14,
        1.98347535e14,
        1.98473711e14,
        1.98599886e14,
        1.98726061e14,
        1.98852236e14,
        1.98978412e14,
        1.99104587e14,
        1.99230762e14,
        1.99356938e14,
        1.99483113e14,
        1.99609288e14,
        1.99735463e14,
        1.99861639e14,
    ]
)

sweep_p = np.array(
    [
        1.71199084e-03,
        3.78843973e-07,
        2.18324526e-03,
        7.85996226e-03,
        1.11580912e-02,
        8.38701406e-03,
        2.00621467e-03,
        5.44148399e-04,
        9.19632301e-03,
        2.09451840e-02,
        2.43724047e-02,
        1.31872950e-02,
        1.24921052e-03,
        4.93177315e-03,
        2.83897743e-02,
        5.02930655e-02,
        4.56398576e-02,
        1.90348908e-02,
        1.15736342e-06,
        2.39553431e-02,
        7.23175373e-02,
        1.00818788e-01,
        7.16354129e-02,
        1.58822732e-02,
        4.82851035e-03,
        6.46677674e-02,
        1.49986292e-01,
        1.60387606e-01,
        8.80805599e-02,
        6.51302677e-03,
        2.81797542e-02,
        1.44305873e-01,
        2.35684642e-01,
        2.01484517e-01,
        6.84566717e-02,
        4.78743424e-06,
        7.30113658e-02,
        2.31014534e-01,
        2.89874429e-01,
        1.90946877e-01,
        4.02326810e-02,
        9.79865508e-03,
        1.43789246e-01,
        2.85855210e-01,
        2.87463988e-01,
        1.36946712e-01,
        1.11149217e-02,
        3.89245364e-02,
        1.92270393e-01,
        2.96960558e-01,
        2.37300012e-01,
        8.48951856e-02,
        9.01931022e-06,
        7.91604173e-02,
        2.19371435e-01,
        2.66699112e-01,
        1.69303081e-01,
        3.39420091e-02,
        8.07051546e-03,
        1.06313868e-01,
        2.16222230e-01,
        2.11339212e-01,
        1.02614780e-01,
        7.73068795e-03,
        2.54759746e-02,
        1.23960073e-01,
        1.86280841e-01,
        1.47118654e-01,
        4.86542895e-02,
        2.61944277e-05,
        4.11046933e-02,
        1.17762742e-01,
        1.40077770e-01,
        8.60027456e-02,
        1.77404914e-02,
        3.25619681e-03,
        4.81210414e-02,
        9.23125414e-02,
        8.71136269e-02,
        4.03162783e-02,
        3.20815022e-03,
        8.40733331e-03,
        3.95952304e-02,
        5.70825805e-02,
        4.21573597e-02,
        1.36957182e-02,
        2.28107867e-05,
        9.03438777e-03,
        2.39272397e-02,
        2.63536601e-02,
        1.50679997e-02,
        2.88705305e-03,
        4.00496435e-04,
        5.86648576e-03,
        1.05883667e-02,
        9.17052210e-03,
        3.95107275e-03,
        3.17090176e-04,
        6.03833606e-04,
        2.79487239e-03,
    ]
)


@pytest.fixture
def freqs():
    return np.linspace(wl2freq(1600e-9), wl2freq(1500e-9), 100)


@pytest.fixture
def monte(freqs):
    return MonteCarloSweepSimulator(freqs[0], freqs[-1], len(freqs))


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


@pytest.fixture
def sweep(freqs):
    return SweepSimulator(freqs[0], freqs[-1], len(freqs))


@pytest.fixture
def sweep_wl(wls):
    return SweepSimulator(wls[0], wls[-1], len(wls))


@pytest.fixture
def wls():
    return np.linspace(1500e-9, 1600e-9, 100)


class TestSimulators:
    def test_sweep_simulator(self, mzi, sweep):
        _, gc_input, _, _, _, gc_output = mzi
        sweep.multiconnect(gc_input, gc_output)

        freqs, p = sweep.simulate()
        assert np.allclose(freqs, sweep_freqs)
        assert np.allclose(p, sweep_p)

    def test_monte_carlo_sweep_simulator(self, mzi, monte):
        _, gc_input, _, _, _, gc_output = mzi
        monte.multiconnect(gc_input, gc_output)

        results = monte.simulate()

        assert len(results) == 10
        assert np.allclose(results[0][0], sweep_freqs)
        assert np.allclose(results[0][1], sweep_p)
        assert np.allclose(results[1][0], sweep_freqs)
        assert not np.allclose(results[1][1], sweep_p)

    def test_modes(self, mzi, sweep):
        _, gc_input, _, _, _, gc_output = mzi
        sweep.multiconnect(gc_input, gc_output)

        wl, p = sweep.simulate(mode="wl")
        assert np.allclose(wl2freq(wl), sweep_freqs)

    def test_auto_mode_conversion(self, mzi, sweep_wl):
        _, gc_input, _, _, _, gc_output = mzi
        sweep_wl.multiconnect(gc_input, gc_output)

        wl, p = sweep_wl.simulate()
        assert np.allclose(wl2freq(wl), sweep_freqs)
