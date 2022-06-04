# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import numpy as np
import pytest

from simphony.libraries import siepic
from simphony.simulation import Detector, DifferentialDetector, Laser, Simulation
from simphony.tools import wl2freq


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

    return (gc_input, gc_output)


@pytest.fixture
def oh():
    x1 = siepic.GratingCoupler(name="x1")
    s = siepic.GratingCoupler(name="s")
    p1 = siepic.GratingCoupler(name="p1")
    p2 = siepic.GratingCoupler(name="p2")
    lo = siepic.GratingCoupler(name="lo")
    x2 = siepic.GratingCoupler(name="x2")

    xdc = siepic.BidirectionalCoupler()
    lodc = siepic.BidirectionalCoupler()
    pdc = siepic.BidirectionalCoupler()

    x1_xdc = siepic.Waveguide(length=514e-6)
    x2_xdc = siepic.Waveguide(length=514e-6)
    s_y = siepic.Waveguide(length=208e-6)
    lo_lodc = siepic.Waveguide(length=208e-6)
    p1_pdc = siepic.Waveguide(length=81e-6)
    p2_pdc = siepic.Waveguide(length=81e-6)
    y_xdc = siepic.Waveguide(length=12e-6)
    y_pdc = siepic.Waveguide(length=12e-6)
    pdc_lodc = siepic.Waveguide(length=12e-6)
    xdc_lodc = siepic.Waveguide(length=12e-6)

    y = siepic.YBranch()
    terminator = siepic.Terminator()

    xdc.multiconnect(y_xdc, xdc_lodc, x1_xdc, x2_xdc)
    lodc.multiconnect(lo_lodc, terminator, pdc_lodc, xdc_lodc)
    pdc.multiconnect(p1_pdc, p2_pdc, y_pdc, pdc_lodc)
    y.multiconnect(s_y, y_xdc, y_pdc)

    x1.connect(x1_xdc)
    s.connect(s_y)
    p1.connect(p1_pdc)
    p2.connect(p2_pdc)
    lo.connect(lo_lodc)
    x2.connect(x2_xdc)

    return (x1, s, p1, p2, lo, x2)


class TestSimulation:
    seed117 = [
        0.00017481,
        0.01219353,
        -0.01773873,
        0.02061959,
        -0.00290609,
        -0.0066712,
        0.00846216,
        0.00488167,
        -0.01002604,
        0.00672506,
        -0.01299871,
        0.0126199,
        0.0007396,
        0.00115915,
        -0.00602,
        0.00979,
        -0.00520642,
        -0.01741927,
        -0.0240019,
        0.03115938,
        -0.00537727,
        -0.00066326,
        -0.00495342,
        0.0002517,
        -0.01819794,
        -0.00936641,
        0.00736962,
        -0.01756158,
        0.01517604,
        0.00298318,
        0.00553522,
        -0.00281899,
        0.01784163,
        0.00610215,
        -0.00944377,
        -0.00967335,
        0.03266932,
        -0.00754913,
        -0.00785714,
        0.03044863,
        -0.00879942,
        0.02543895,
        -0.00322589,
        -0.00785712,
        0.00815186,
        -0.01540587,
        0.00631346,
        0.01470638,
        -0.0051735,
        0.00150219,
        0.01991704,
        -0.00193712,
        0.01432663,
        0.00699449,
        0.00281496,
        -0.0075551,
        0.00341335,
        0.01141054,
        -0.00696104,
        0.00628623,
        -0.00156238,
        0.00271096,
        -0.00631849,
        0.00724422,
        0.00808875,
        0.00742942,
        -0.02009245,
        0.0071186,
        -0.00098557,
        -0.01329963,
        -0.00692713,
        0.01484593,
        0.01073398,
        0.01623651,
        -0.00623136,
        -0.01092318,
        -0.00766223,
        -0.00344117,
        0.01897063,
        0.01066724,
        -0.00842774,
        -0.01002413,
        0.01600654,
        -0.00786538,
        0.01610357,
        0.01215284,
        0.0039726,
        0.0194278,
        -0.00150813,
        -0.00359058,
        -0.00125099,
        0.01863215,
        -0.01533298,
        -0.00367189,
        0.005698,
        -0.00949113,
        0.00948224,
        -0.00325547,
        0.01019897,
        0.00419238,
        -0.00354101,
    ]

    def test_context(self, mzi):
        gc_input, gc_output = mzi

        with Simulation() as sim1:
            assert sim1.circuit is None

            l1 = Laser().connect(gc_input)
            d1 = Detector().connect(gc_output)

            assert l1.circuit == gc_input.circuit == sim1.circuit
            assert d1.circuit == gc_output.circuit == sim1.circuit

        with Simulation() as _:
            assert sim1.circuit is None
            assert l1.circuit != gc_input.circuit
            assert d1.circuit != gc_output.circuit

    def test_sampling(self, mzi):
        gc_input, gc_output = mzi

        with Simulation() as sim1:
            Laser().connect(gc_input)
            Detector().connect(gc_output)

            assert len(sim1.sample(100)[0][0]) == 100
            assert len(sim1.sample(101)[0][0]) == 101

    def test_seed(self, mzi):
        gc_input, gc_output = mzi

        with Simulation(seed=117) as sim1:
            Laser(power=1e-3, wl=1550e-9).connect(gc_input)
            Detector().connect(gc_output)

            data = sim1.sample(101)

            assert np.allclose(data[0][0], self.seed117, rtol=0, atol=1e-8)

        with Simulation(seed=118) as sim2:
            Laser(power=1e-3, wl=1550e-9).connect(gc_input)
            Detector().connect(gc_output)

            data = sim2.sample(101)

            assert not np.allclose(data[0][0], self.seed117, rtol=0, atol=1e-8)

        with Simulation() as sim3:
            Laser(power=1e-3, wl=1550e-9).connect(gc_input)
            Detector().connect(gc_output)

            data = sim3.sample(101)

            assert not np.allclose(data[0][0], self.seed117, rtol=0, atol=1e-8)

        with Simulation(seed=117) as sim4:
            Laser(power=1e-3, wl=1550e-9).connect(gc_input)
            Detector().connect(gc_output)

            data = sim4.sample(101)

            assert np.allclose(data[0][0], self.seed117, rtol=0, atol=1e-8)

    def test_sampling_frequency(self, mzi):
        gc_input, gc_output = mzi

        data1 = None
        with Simulation(fs=10e9, seed=117) as sim:
            Laser(power=1e-3, wl=1550e-9).connect(gc_input)
            Detector().connect(gc_output)

            data1 = sim.sample(1001)

        data2 = None
        with Simulation(fs=10e9, seed=117) as sim:
            Laser(power=1e-3, wl=1550e-9).connect(gc_input)
            Detector().connect(gc_output)

            data2 = sim.sample(1001)

        assert np.allclose(data1[0][0], data2[0][0], rtol=0, atol=1e-11)

    def test_layout_aware(self, mzi):
        gc_input, gc_output = mzi

        data = None
        with Simulation(fs=10e9, seed=117) as sim:
            Laser(power=1e-3, wl=1550e-9).connect(gc_input)
            Detector().connect(gc_output)

            data = sim.layout_aware_simulation(runs=2, num_samples=101)

        assert len(data) == 2
        assert len(data[0][0][0]) == 101

class TestSingleDetector:
    result = 0.00017544
    results = [
        1.80576404e-04,
        1.08063217e-02,
        -1.84591717e-02,
        2.11631266e-02,
        -4.24527434e-03,
        -5.53885990e-03,
        8.67396297e-03,
        5.28644276e-03,
        -1.02520694e-02,
        8.05882087e-03,
        -1.25512983e-02,
        1.18939574e-02,
        -3.92095769e-06,
        3.61245566e-03,
        -6.60295137e-03,
        9.18355753e-03,
        -2.92043587e-03,
        -1.80968121e-02,
        -2.20941667e-02,
        3.09025569e-02,
        -5.98374595e-03,
        -6.09039074e-05,
        -6.12987780e-03,
    ]

    def test_single_sample(self, mzi):
        gc_input, gc_output = mzi

        with Simulation() as sim:
            Laser(power=1e-3, wl=1550e-9).connect(gc_input)
            Detector().connect(gc_output)

            data = sim.sample()

            assert np.allclose(data[0][0], [self.result], rtol=0, atol=1e-8)

    def test_conversion_gain(self, mzi):
        gc_input, gc_output = mzi

        with Simulation() as sim:
            Laser(power=1e-3, wl=1550e-9).connect(gc_input)
            Detector(conversion_gain=7).connect(gc_output)

            data = sim.sample()

            assert np.allclose(data[0][0], [self.result * 7], rtol=0, atol=1e-7)

    def test_noise(self, mzi):
        gc_input, gc_output = mzi

        with Simulation(seed=117) as sim:
            Laser(power=1e-3, wl=1550e-9).connect(gc_input)
            Detector(noise=1e-3).connect(gc_output)

            data = sim.sample(23)

            assert np.allclose(data[0][0], self.results, rtol=0, atol=1e-8)


class TestDifferentialDetector:
    cmrr_x = (
        [
            0.00185109,
            0.00438573,
            -0.00161808,
            0.00039819,
            0.00595933,
            -0.00064074,
            0.00427077,
            0.00205663,
            0.00079463,
            -0.00233682,
            0.0009754,
            0.00339042,
            -0.00215775,
            0.00184282,
            -0.00052728,
            0.0007634,
            -0.00196386,
            0.00213241,
            0.00238699,
            0.00218791,
            -0.00612339,
            0.00209437,
            -0.00035317,
        ],
    )
    cmrr_p = [
        -0.00470416,
        0.00939605,
        0.00633034,
        0.00376072,
        -0.00135223,
        -0.00634246,
        -0.00374243,
        0.00145856,
        -0.00100338,
        0.00687123,
        -0.00553187,
        0.00345907,
        -0.00358338,
        -0.00438295,
        -0.00392697,
        -0.00550157,
        0.00577888,
        -0.00182664,
        -0.00431197,
        0.00058432,
        0.00173179,
        0.00035304,
        0.00030706,
    ]
    result = [
        6.820933398426216e-05,
        -2.51779027237116e-06,
        7.072712425663332e-05,
        7.528059784829445e-05,
        -2.2353588319872576e-06,
        7.751595668028171e-05,
    ]
    x1results = [
        7.25832807e-05,
        6.45213384e-03,
        -1.16782238e-02,
        1.32502362e-02,
        -2.92429196e-03,
        -3.29487901e-03,
        5.40473883e-03,
        3.32631865e-03,
        -6.47341674e-03,
        5.21927531e-03,
        -7.78813016e-03,
        7.24665505e-03,
        -1.74786835e-04,
        2.64408006e-03,
        -4.26117438e-03,
        5.57803566e-03,
        -1.45885813e-03,
        -1.14445296e-02,
        -1.34812942e-02,
        1.91818955e-02,
        -3.87934796e-03,
        2.70018878e-05,
        -4.07081299e-03,
    ]
    xresults = [
        0.00185109,
        0.0043857,
        -0.00161804,
        0.00039815,
        0.00595934,
        -0.00064072,
        0.00427075,
        0.00205662,
        0.00079465,
        -0.00233683,
        0.00097543,
        0.00339039,
        -0.00215776,
        0.00184282,
        -0.00052726,
        0.00076338,
        -0.00196385,
        0.00213244,
        0.00238704,
        0.00218784,
        -0.00612338,
        0.00209437,
        -0.00035316,
    ]
    x2results = [
        7.51125467e-05,
        6.59171046e-03,
        -1.18798537e-02,
        1.34858866e-02,
        -2.95698094e-03,
        -3.37021160e-03,
        5.50204769e-03,
        3.38258002e-03,
        -6.58734386e-03,
        5.29622841e-03,
        -7.93593533e-03,
        7.39107533e-03,
        -1.65775053e-04,
        2.65791212e-03,
        -4.32937195e-03,
        5.69003709e-03,
        -1.51750747e-03,
        -1.16429994e-02,
        -1.37543005e-02,
        1.95380050e-02,
        -3.94001069e-03,
        1.98397377e-05,
        -4.12678903e-03,
    ]
    p1results = [
        -0.00012833,
        0.00777554,
        -0.01088095,
        0.01399814,
        -0.00186632,
        -0.00567741,
        0.00499696,
        0.00325294,
        -0.00684533,
        0.00448832,
        -0.0088845,
        0.00729312,
        0.00244817,
        -0.00056501,
        -0.00356172,
        0.00697164,
        -0.00434339,
        -0.01221006,
        -0.0154076,
        0.02053649,
        -0.00297821,
        -0.00016577,
        -0.00318841,
    ]
    presults = [
        -0.00470416,
        0.00939602,
        0.00633038,
        0.00376067,
        -0.00135222,
        -0.00634245,
        -0.00374245,
        0.00145855,
        -0.00100335,
        0.00687121,
        -0.00553184,
        0.00345904,
        -0.00358338,
        -0.00438295,
        -0.00392696,
        -0.00550159,
        0.0057789,
        -0.0018266,
        -0.00431191,
        0.00058425,
        0.0017318,
        0.00035304,
        0.00030707,
    ]
    p2results = [
        -0.00012596,
        0.00789399,
        -0.01105161,
        0.01419768,
        -0.00189359,
        -0.00574133,
        0.0050792,
        0.00330085,
        -0.00694156,
        0.00455373,
        -0.00900964,
        0.00741552,
        0.00245592,
        -0.00055348,
        -0.00361925,
        0.00706657,
        -0.00439297,
        -0.0123776,
        -0.01563877,
        0.02083776,
        -0.00302955,
        -0.00017171,
        -0.00323547,
    ]

    def test_single_sample(self, oh):
        x1, s, p1, p2, lo, x2 = oh

        with Simulation() as sim:
            Laser(power=1e-3, wl=1550e-9).connect(lo)
            DifferentialDetector().multiconnect(x1, x2)
            DifferentialDetector().multiconnect(p1, p2)

            x1, x, x2, p1, p, p2 = sim.sample()

            assert np.allclose(
                [
                    x1[0][0][0],
                    x[0][0][0],
                    x2[0][0][0],
                    p1[0][0][0],
                    p[0][0][0],
                    p2[0][0][0],
                ],
                self.result,
            )

    def test_conversion_gain(self, oh):
        x1, s, p1, p2, lo, x2 = oh

        with Simulation() as sim:
            Laser(power=1e-3, wl=1550e-9).connect(lo)
            DifferentialDetector(
                monitor_conversion_gain=7, rf_conversion_gain=7
            ).multiconnect(x1, x2)
            DifferentialDetector(
                monitor_conversion_gain=7, rf_conversion_gain=7
            ).multiconnect(p1, p2)

            x1, x, x2, p1, p, p2 = sim.sample()

            assert np.allclose(
                [
                    x1[0][0][0],
                    x[0][0][0],
                    x2[0][0][0],
                    p1[0][0][0],
                    p[0][0][0],
                    p2[0][0][0],
                ],
                np.array(self.result) * 7,
            )

    def test_noise(self, oh):
        x1, s, p1, p2, lo, x2 = oh

        with Simulation(seed=117) as sim:
            Laser(power=1e-3, wl=1550e-9).connect(lo)
            DifferentialDetector(monitor_noise=800e-6, rf_noise=4e-3).multiconnect(
                x1, x2
            )
            DifferentialDetector(monitor_noise=800e-6, rf_noise=4e-3).multiconnect(
                p1, p2
            )

            x1, x, x2, p1, p, p2 = sim.sample(23)

            assert np.allclose(x1[0][0], self.x1results, rtol=0, atol=1e-7)
            assert np.allclose(x[0][0], self.xresults, rtol=0, atol=1e-7)
            assert np.allclose(x2[0][0], self.x2results, rtol=0, atol=1e-7)
            assert np.allclose(p1[0][0], self.p1results, rtol=0, atol=1e-7)
            assert np.allclose(p[0][0], self.presults, rtol=0, atol=1e-7)
            assert np.allclose(p2[0][0], self.p2results, rtol=0, atol=1e-7)

    def test_cmrr(self, oh):
        x1, s, p1, p2, lo, x2 = oh

        with Simulation(seed=117) as sim:
            Laser(power=1e-3, rin=-145, wl=1550e-9).connect(lo)
            DifferentialDetector(
                monitor_noise=800e-6, rf_cmrr=60, rf_noise=4e-3
            ).multiconnect(x1, x2)
            DifferentialDetector(
                monitor_noise=800e-6, rf_cmrr=60, rf_noise=4e-3
            ).multiconnect(p1, p2)

            _, x, _, _, p, _ = sim.sample(23)

            assert np.allclose(x[0][0], self.cmrr_x, rtol=0, atol=1e-7)
            assert np.allclose(p[0][0], self.cmrr_p, rtol=0, atol=1e-7)


class TestLaser:
    freqs = [
        1.87370286e14,
        1.87487466e14,
        1.87604792e14,
        1.87722265e14,
        1.87839886e14,
        1.87957654e14,
        1.88075570e14,
        1.88193633e14,
        1.88311845e14,
        1.88430206e14,
        1.88548716e14,
        1.88667374e14,
        1.88786183e14,
        1.88905141e14,
        1.89024248e14,
        1.89143507e14,
        1.89262915e14,
        1.89382475e14,
        1.89502186e14,
        1.89622048e14,
        1.89742062e14,
        1.89862228e14,
        1.89982546e14,
        1.90103017e14,
        1.90223641e14,
        1.90344418e14,
        1.90465348e14,
        1.90586432e14,
        1.90707670e14,
        1.90829063e14,
        1.90950610e14,
        1.91072312e14,
        1.91194170e14,
        1.91316183e14,
        1.91438351e14,
        1.91560676e14,
        1.91683157e14,
        1.91805795e14,
        1.91928590e14,
        1.92051543e14,
        1.92174653e14,
        1.92297920e14,
        1.92421347e14,
        1.92544931e14,
        1.92668675e14,
        1.92792577e14,
        1.92916640e14,
        1.93040862e14,
        1.93165244e14,
        1.93289786e14,
        1.93414489e14,
        1.93539353e14,
        1.93664379e14,
        1.93789566e14,
        1.93914915e14,
        1.94040426e14,
        1.94166100e14,
        1.94291936e14,
        1.94417936e14,
        1.94544100e14,
        1.94670427e14,
        1.94796919e14,
        1.94923575e14,
        1.95050396e14,
        1.95177382e14,
        1.95304533e14,
        1.95431850e14,
        1.95559333e14,
        1.95686983e14,
        1.95814799e14,
        1.95942783e14,
        1.96070934e14,
        1.96199253e14,
        1.96327739e14,
        1.96456394e14,
        1.96585218e14,
        1.96714211e14,
        1.96843374e14,
        1.96972706e14,
        1.97102208e14,
        1.97231880e14,
        1.97361724e14,
        1.97491738e14,
        1.97621924e14,
        1.97752281e14,
        1.97882811e14,
        1.98013513e14,
        1.98144387e14,
        1.98275435e14,
        1.98406657e14,
        1.98538052e14,
        1.98669621e14,
        1.98801365e14,
        1.98933283e14,
        1.99065377e14,
        1.99197647e14,
        1.99330092e14,
        1.99462713e14,
        1.99595511e14,
        1.99728486e14,
        1.99861639e14,
    ]

    powers = [
        0.001,
        0.0011984,
        0.00139679,
        0.00159519,
        0.00179359,
        0.00199198,
        0.00219038,
        0.00238878,
        0.00258717,
        0.00278557,
        0.00298397,
        0.00318236,
        0.00338076,
        0.00357916,
        0.00377756,
        0.00397595,
        0.00417435,
        0.00437275,
        0.00457114,
        0.00476954,
        0.00496794,
        0.00516633,
        0.00536473,
        0.00556313,
        0.00576152,
        0.00595992,
        0.00615832,
        0.00635671,
        0.00655511,
        0.00675351,
        0.0069519,
        0.0071503,
        0.0073487,
        0.00754709,
        0.00774549,
        0.00794389,
        0.00814228,
        0.00834068,
        0.00853908,
        0.00873747,
        0.00893587,
        0.00913427,
        0.00933267,
        0.00953106,
        0.00972946,
        0.00992786,
        0.01012625,
        0.01032465,
        0.01052305,
        0.01072144,
        0.01091984,
        0.01111824,
        0.01131663,
        0.01151503,
        0.01171343,
        0.01191182,
        0.01211022,
        0.01230862,
        0.01250701,
        0.01270541,
        0.01290381,
        0.0131022,
        0.0133006,
        0.013499,
        0.01369739,
        0.01389579,
        0.01409419,
        0.01429259,
        0.01449098,
        0.01468938,
        0.01488778,
        0.01508617,
        0.01528457,
        0.01548297,
        0.01568136,
        0.01587976,
        0.01607816,
        0.01627655,
        0.01647495,
        0.01667335,
        0.01687174,
        0.01707014,
        0.01726854,
        0.01746693,
        0.01766533,
        0.01786373,
        0.01806212,
        0.01826052,
        0.01845892,
        0.01865731,
        0.01885571,
        0.01905411,
        0.01925251,
        0.0194509,
        0.0196493,
        0.0198477,
        0.02004609,
        0.02024449,
        0.02044289,
        0.02064128,
        0.02083968,
        0.02103808,
        0.02123647,
        0.02143487,
        0.02163327,
        0.02183166,
        0.02203006,
        0.02222846,
        0.02242685,
        0.02262525,
        0.02282365,
        0.02302204,
        0.02322044,
        0.02341884,
        0.02361723,
        0.02381563,
        0.02401403,
        0.02421242,
        0.02441082,
        0.02460922,
        0.02480762,
        0.02500601,
        0.02520441,
        0.02540281,
        0.0256012,
        0.0257996,
        0.025998,
        0.02619639,
        0.02639479,
        0.02659319,
        0.02679158,
        0.02698998,
        0.02718838,
        0.02738677,
        0.02758517,
        0.02778357,
        0.02798196,
        0.02818036,
        0.02837876,
        0.02857715,
        0.02877555,
        0.02897395,
        0.02917234,
        0.02937074,
        0.02956914,
        0.02976754,
        0.02996593,
        0.03016433,
        0.03036273,
        0.03056112,
        0.03075952,
        0.03095792,
        0.03115631,
        0.03135471,
        0.03155311,
        0.0317515,
        0.0319499,
        0.0321483,
        0.03234669,
        0.03254509,
        0.03274349,
        0.03294188,
        0.03314028,
        0.03333868,
        0.03353707,
        0.03373547,
        0.03393387,
        0.03413226,
        0.03433066,
        0.03452906,
        0.03472745,
        0.03492585,
        0.03512425,
        0.03532265,
        0.03552104,
        0.03571944,
        0.03591784,
        0.03611623,
        0.03631463,
        0.03651303,
        0.03671142,
        0.03690982,
        0.03710822,
        0.03730661,
        0.03750501,
        0.03770341,
        0.0379018,
        0.0381002,
        0.0382986,
        0.03849699,
        0.03869539,
        0.03889379,
        0.03909218,
        0.03929058,
        0.03948898,
        0.03968737,
        0.03988577,
        0.04008417,
        0.04028257,
        0.04048096,
        0.04067936,
        0.04087776,
        0.04107615,
        0.04127455,
        0.04147295,
        0.04167134,
        0.04186974,
        0.04206814,
        0.04226653,
        0.04246493,
        0.04266333,
        0.04286172,
        0.04306012,
        0.04325852,
        0.04345691,
        0.04365531,
        0.04385371,
        0.0440521,
        0.0442505,
        0.0444489,
        0.04464729,
        0.04484569,
        0.04504409,
        0.04524248,
        0.04544088,
        0.04563928,
        0.04583768,
        0.04603607,
        0.04623447,
        0.04643287,
        0.04663126,
        0.04682966,
        0.04702806,
        0.04722645,
        0.04742485,
        0.04762325,
        0.04782164,
        0.04802004,
        0.04821844,
        0.04841683,
        0.04861523,
        0.04881363,
        0.04901202,
        0.04921042,
        0.04940882,
        0.04960721,
        0.04980561,
        0.05000401,
        0.0502024,
        0.0504008,
        0.0505992,
        0.0507976,
        0.05099599,
        0.05119439,
        0.05139279,
        0.05159118,
        0.05178958,
        0.05198798,
        0.05218637,
        0.05238477,
        0.05258317,
        0.05278156,
        0.05297996,
        0.05317836,
        0.05337675,
        0.05357515,
        0.05377355,
        0.05397194,
        0.05417034,
        0.05436874,
        0.05456713,
        0.05476553,
        0.05496393,
        0.05516232,
        0.05536072,
        0.05555912,
        0.05575752,
        0.05595591,
        0.05615431,
        0.05635271,
        0.0565511,
        0.0567495,
        0.0569479,
        0.05714629,
        0.05734469,
        0.05754309,
        0.05774148,
        0.05793988,
        0.05813828,
        0.05833667,
        0.05853507,
        0.05873347,
        0.05893186,
        0.05913026,
        0.05932866,
        0.05952705,
        0.05972545,
        0.05992385,
        0.06012224,
        0.06032064,
        0.06051904,
        0.06071743,
        0.06091583,
        0.06111423,
        0.06131263,
        0.06151102,
        0.06170942,
        0.06190782,
        0.06210621,
        0.06230461,
        0.06250301,
        0.0627014,
        0.0628998,
        0.0630982,
        0.06329659,
        0.06349499,
        0.06369339,
        0.06389178,
        0.06409018,
        0.06428858,
        0.06448697,
        0.06468537,
        0.06488377,
        0.06508216,
        0.06528056,
        0.06547896,
        0.06567735,
        0.06587575,
        0.06607415,
        0.06627255,
        0.06647094,
        0.06666934,
        0.06686774,
        0.06706613,
        0.06726453,
        0.06746293,
        0.06766132,
        0.06785972,
        0.06805812,
        0.06825651,
        0.06845491,
        0.06865331,
        0.0688517,
        0.0690501,
        0.0692485,
        0.06944689,
        0.06964529,
        0.06984369,
        0.07004208,
        0.07024048,
        0.07043888,
        0.07063727,
        0.07083567,
        0.07103407,
        0.07123246,
        0.07143086,
        0.07162926,
        0.07182766,
        0.07202605,
        0.07222445,
        0.07242285,
        0.07262124,
        0.07281964,
        0.07301804,
        0.07321643,
        0.07341483,
        0.07361323,
        0.07381162,
        0.07401002,
        0.07420842,
        0.07440681,
        0.07460521,
        0.07480361,
        0.075002,
        0.0752004,
        0.0753988,
        0.07559719,
        0.07579559,
        0.07599399,
        0.07619238,
        0.07639078,
        0.07658918,
        0.07678758,
        0.07698597,
        0.07718437,
        0.07738277,
        0.07758116,
        0.07777956,
        0.07797796,
        0.07817635,
        0.07837475,
        0.07857315,
        0.07877154,
        0.07896994,
        0.07916834,
        0.07936673,
        0.07956513,
        0.07976353,
        0.07996192,
        0.08016032,
        0.08035872,
        0.08055711,
        0.08075551,
        0.08095391,
        0.0811523,
        0.0813507,
        0.0815491,
        0.08174749,
        0.08194589,
        0.08214429,
        0.08234269,
        0.08254108,
        0.08273948,
        0.08293788,
        0.08313627,
        0.08333467,
        0.08353307,
        0.08373146,
        0.08392986,
        0.08412826,
        0.08432665,
        0.08452505,
        0.08472345,
        0.08492184,
        0.08512024,
        0.08531864,
        0.08551703,
        0.08571543,
        0.08591383,
        0.08611222,
        0.08631062,
        0.08650902,
        0.08670741,
        0.08690581,
        0.08710421,
        0.08730261,
        0.087501,
        0.0876994,
        0.0878978,
        0.08809619,
        0.08829459,
        0.08849299,
        0.08869138,
        0.08888978,
        0.08908818,
        0.08928657,
        0.08948497,
        0.08968337,
        0.08988176,
        0.09008016,
        0.09027856,
        0.09047695,
        0.09067535,
        0.09087375,
        0.09107214,
        0.09127054,
        0.09146894,
        0.09166733,
        0.09186573,
        0.09206413,
        0.09226253,
        0.09246092,
        0.09265932,
        0.09285772,
        0.09305611,
        0.09325451,
        0.09345291,
        0.0936513,
        0.0938497,
        0.0940481,
        0.09424649,
        0.09444489,
        0.09464329,
        0.09484168,
        0.09504008,
        0.09523848,
        0.09543687,
        0.09563527,
        0.09583367,
        0.09603206,
        0.09623046,
        0.09642886,
        0.09662725,
        0.09682565,
        0.09702405,
        0.09722244,
        0.09742084,
        0.09761924,
        0.09781764,
        0.09801603,
        0.09821443,
        0.09841283,
        0.09861122,
        0.09880962,
        0.09900802,
        0.09920641,
        0.09940481,
        0.09960321,
        0.0998016,
        0.1,
    ]

    rin_results = [
        0.00017534,
        0.00019062,
        0.00015289,
        0.00020163,
        0.00017138,
        0.00016665,
        0.0001857,
        0.00018142,
        0.00016255,
        0.00018353,
        0.00015875,
        0.0001913,
        0.00017599,
        0.00017676,
        0.00016767,
        0.00018743,
        0.00016871,
        0.00015348,
        0.00014547,
        0.00021439,
        0.00016853,
        0.00017464,
        0.00016882,
    ]

    def test_wlsweep(self, mzi):
        gc_input, gc_output = mzi

        with Simulation(seed=117) as sim:
            l = Laser(power=1e-3).wlsweep(1500e-9, 1600e-9, 101).connect(gc_input)
            Detector().connect(gc_output)

            data = sim.sample()

            assert len(data) == 101
            assert np.allclose(l.freqs, self.freqs)

    def test_freqsweep(self, mzi):
        gc_input, gc_output = mzi

        with Simulation(seed=117) as sim:
            l = (
                Laser(power=1e-3)
                .freqsweep(1.87370286e14, 1.99861639e14, 101)
                .connect(gc_input)
            )
            Detector().connect(gc_output)

            data = sim.sample()

            assert len(data) == 101
            assert np.allclose(l.freqs, self.freqs, rtol=0, atol=1e12)

    def test_powersweep(self, mzi):
        gc_input, gc_output = mzi

        with Simulation(seed=117) as sim:
            l = Laser(wl=1550e-9).powersweep(1e-3, 100e-3).connect(gc_input)
            Detector().connect(gc_output)

            data = sim.sample()

            assert len(data[0]) == 500
            assert np.allclose(l.powers, self.powers, rtol=0, atol=1e-8)

    def test_freqs(self, oh, mzi):
        x1, s, p1, p2, lo, x2 = oh
        gc_input, gc_output = mzi

        with Simulation(seed=117) as sim:
            Laser(freq=1.94888531e14, power=1e-3).connect(gc_input)
            Detector().connect(gc_output)

            data = sim.sample()

        with Simulation(seed=117) as sim:
            l1 = Laser(power=1e-3).connect(lo)
            l2 = Laser(power=1e-3).connect(s)

            l1._freqs = wl2freq(
                np.array(
                    [
                        1500e-9,
                        1510e-9,
                        1520e-9,
                        1530e-9,
                        1540e-9,
                        1550e-9,
                        1560e-9,
                        1570e-9,
                        1580e-9,
                        1590e-9,
                        1600e-9,
                    ]
                )
            )
            l2._freqs = wl2freq(
                np.array([1500e-9, 1520e-9, 1540e-9, 1560e-9, 1580e-9, 1600e-9])
            )

            DifferentialDetector().multiconnect(x1, x2)
            DifferentialDetector().multiconnect(p1, p2)

            assert len(l1.freqs) == len(l2.freqs) == 0

            data = sim.sample()

            assert len(data) == len(l1.freqs) == 6
            assert np.equal(l1.freqs, l2.freqs).all()
            assert np.equal(l1.freqs, sim.freqs).all()

    def test_rin(self, mzi):
        gc_input, gc_output = mzi

        with Simulation(seed=117) as sim:
            Laser(power=1e-3, rin=-145).connect(gc_input)
            Detector().connect(gc_output)

            data = sim.sample(23)

            assert np.allclose(data[0][0], self.rin_results, rtol=0, atol=1e-7)
