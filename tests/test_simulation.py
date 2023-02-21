# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

try:
    import gdsfactory as gf
    _has_gf = True
except ImportError:
    _has_gf = False
import jax.numpy as np
# import numpy as np
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
def mzi_gf():
    if _has_gf:
        gc_input = siepic.GratingCoupler(name="gc_input")
        y_splitter = siepic.YBranch(name="y_splitter")
        wg_long = siepic.Waveguide(name="wg_long", length=150e-6)
        wg_short = siepic.Waveguide(name="wg_short", length=50e-6)
        y_recombiner = siepic.YBranch(name="y_recombiner")
        gc_output = siepic.GratingCoupler(name="gc_output")

        c = gf.Component("mzi")

        ysplit = c << y_splitter.component

        gcin = c << gc_input.component

        gcout = c << gc_output.component

        yrecomb = c << y_recombiner.component

        yrecomb.move(destination=(0, -55.5))
        gcout.move(destination=(-20.4, -55.5))
        gcin.move(destination=(-20.4, 0))

        gc_input["pin2"].connect(y_splitter, gcin, ysplit)
        gc_output["pin2"].connect(y_recombiner["pin1"], gcout, yrecomb)
        y_splitter["pin2"].connect(wg_long)
        y_recombiner["pin3"].connect(wg_long)
        y_splitter["pin3"].connect(wg_short)
        y_recombiner["pin2"].connect(wg_short)

        wg_long_ref = gf.routing.get_route_from_steps(
            ysplit.ports["pin2"],
            yrecomb.ports["pin3"],
            steps=[{"dx": 91.75 / 2}, {"dy": -61}],
        )
        wg_short_ref = gf.routing.get_route_from_steps(
            ysplit.ports["pin3"],
            yrecomb.ports["pin2"],
            steps=[{"dx": 47.25 / 2}, {"dy": -50}],
        )

        wg_long.path = wg_long_ref
        print(wg_long.path)
        wg_short.path = wg_short_ref

        c.add(wg_long_ref.references)
        c.add(wg_short_ref.references)

        c.add_port("o1", port=gcin.ports["pin2"])
        c.add_port("o2", port=gcout.ports["pin2"])

        return (c, gc_input, gc_output)

    return


@pytest.fixture
def mzi_unconnected():
    gc_input = siepic.GratingCoupler()
    y_splitter = siepic.YBranch()
    wg_long = siepic.Waveguide(length=150e-6)
    wg_short = siepic.Waveguide(length=50e-6)
    y_recombiner = siepic.YBranch()
    gc_output = siepic.GratingCoupler()

    return (gc_input, y_splitter, wg_long, wg_short, y_recombiner, gc_output)


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
        0.1754380152914969,
        0.17544695442565544,
        0.17544287567595707,
        0.17544503090527674,
        0.17543844500459105,
        0.17544186707373638,
        0.17544228589377417,
        0.1754404004359048,
        0.17543849716481452,
        0.17543692274591904,
        0.17543927226317219,
        0.17543999507279218,
        0.1754353380751127,
        0.17544193128079277,
        0.17543568794589912,
        0.17543832658642994,
        0.17543009244831398,
        0.17544583253229848,
        0.17544363014090936,
        0.17544256643083592,
        0.17544512241030039,
        0.1754456227379844,
        0.17543984487221026,
        0.17544758726885815,
        0.17543322936671696,
        0.17544361540240627,
        0.17544239803184677,
        0.1754448437948928,
        0.17544114541753658,
        0.17544239136728484,
        0.17543947616222758,
        0.17544552661963644,
        0.17544637220513903,
        0.17544162600891977,
        0.1754367026993376,
        0.1754471000129131,
        0.17543902094573183,
        0.17543615213308428,
        0.17544967047420865,
        0.17543146412086139,
        0.17544296807772886,
        0.17543449876934214,
        0.17544351274836711,
        0.17544819614655285,
        0.17543593118939826,
        0.1754422038724275,
        0.17544157410536837,
        0.1754397534953444,
        0.1754476727504654,
        0.17544280557328554,
        0.1754403282834214,
        0.17544255758794788,
        0.17543850331638874,
        0.175449876038959,
        0.17544753049531048,
        0.1754442724674327,
        0.17543741410327007,
        0.1754416430539068,
        0.17544524544178575,
        0.17543818035838332,
        0.17544267011085032,
        0.17543843013793015,
        0.17544483226104743,
        0.17545063486127643,
        0.17544132150635,
        0.17544385967116777,
        0.17544266639427417,
        0.1754417823614324,
        0.17544548022687032,
        0.17543783484496195,
        0.17544516534316243,
        0.17543704705898236,
        0.17543960111607987,
        0.1754430161365464,
        0.1754409539497875,
        0.1754371891859795,
        0.17543988729244112,
        0.17544361142951453,
        0.17544093933979865,
        0.1754443515407936,
        0.17544592826652938,
        0.17544733736115362,
        0.17544438165823237,
        0.1754446377171548,
        0.17543386349149767,
        0.17542962249331512,
        0.1754448082955394,
        0.17543616751201996,
        0.17543694978757066,
        0.1754513865067572,
        0.17544069673708843,
        0.17544706810162158,
        0.17543943322936553,
        0.17545438232342542,
        0.17544124909719455,
        0.1754407823461406,
        0.175441107867302,
        0.1754490941482381,
        0.17544355427149344,
        0.17544306970650564,
        0.17544362552722872,
    ]

    def test_context(self, mzi):
        gc_input, gc_output = mzi

        with Simulation() as sim1:
            assert sim1.circuit is None

            l1 = Laser().connect(gc_input)
            d1 = Detector().connect(gc_output)

            assert l1.circuit == gc_input.circuit == sim1.circuit
            assert d1.circuit == gc_output.circuit == sim1.circuit

        # with Simulation() as _:
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

    def test_layout_aware(self, mzi_gf):
        if not _has_gf:
            return
        c, gc_input, gc_output = mzi_gf

        with Simulation(fs=10e9, seed=117) as sim:
            Laser(power=1e-3, wl=1550e-9).connect(gc_input)
            Detector().connect(gc_output)

            data = sim.layout_aware_simulation(
                component_or_circuit=c, runs=2, num_samples=101
            )

        assert len(data) == 2
        assert len(data[0][0][0]) == 101


class TestSingleDetector:
    result = 0.17544204
    results = [
        0.1754446446318916,
        0.1740570477966146,
        0.17472088224910245,
        0.17600042687369985,
        0.17409707193348012,
        0.17656878808228374,
        0.17564879343100195,
        0.17584860763997665,
        0.17521402003433048,
        0.1767711660212849,
        0.17588534947132822,
        0.17472234709003542,
        0.17469337624014103,
        0.177897858739015,
        0.17485959291174613,
        0.17483231815278333,
        0.17772861062695466,
        0.1747695127456315,
        0.1773630911610061,
        0.1751848508997118,
        0.17483786830788275,
        0.17605227005695107,
        0.17426008721781772,
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
    cmrr_x = [
        -0.0006649530079477117,
        0.001862986930585692,
        -0.004127633806731973,
        -0.002114743819325535,
        0.0034426013129120617,
        -0.0031508670468650933,
        0.0017569831950194803,
        -0.00045637284640092724,
        -0.0017272818749597511,
        -0.004850761198680061,
        -0.0015435978785956952,
        0.0008714199831552534,
        -0.004669033696690775,
        -0.0006717397085453862,
        -0.0030448708403816773,
        -0.001751682348939884,
        -0.004483208665495104,
        -0.0003865227882897124,
        -0.0001270245108348886,
        -0.0003380476733973268,
        -0.008645386160323062,
        -0.00042084948912773923,
        -0.0028703445512053907,
    ]
    cmrr_p = [
        -0.0069360569388826985,
        0.007162650436662034,
        0.004092972479890926,
        0.0015257977103456232,
        -0.003595451576560762,
        -0.008574945982064151,
        -0.005975918317579784,
        -0.0007794603488169448,
        -0.003243277451558061,
        0.0046364044604873655,
        -0.007765984954843569,
        0.0012219494023251371,
        -0.0058133270530027845,
        -0.006623138756001309,
        -0.0061581015123797645,
        -0.007722807298276565,
        0.0035515242719080064,
        -0.004062776678878426,
        -0.006549156884781828,
        -0.0016477338915286788,
        -0.0004981163877181849,
        -0.001881633727400377,
        -0.0019333666109683496,
    ]
    result = [
        0.06820933398426215,
        -0.00251779027237109,
        0.07072712425663324,
        0.07528059784829455,
        -0.002235358831987083,
        0.07751595668028163,
    ]
    x1results = [
        0.06821699257144385,
        0.0670952382875827,
        0.0676372896391082,
        0.06864269173955594,
        0.06714010175082658,
        0.06911780934124957,
        0.0683777068291349,
        0.06853477945343182,
        0.06802755771604438,
        0.06927209160924779,
        0.06856504613437195,
        0.06762827665802736,
        0.06761848293287445,
        0.07017466869076468,
        0.06774859678775505,
        0.06772307342540554,
        0.07003315633076466,
        0.06766455420703217,
        0.06973587278660405,
        0.06799847701751302,
        0.06771901819968199,
        0.06869352418909015,
        0.06726606144321046,
    ]
    xresults = [
        -0.0006649530075707654,
        0.001862978025611869,
        -0.004127620532783833,
        -0.002114758967834034,
        0.0034426035961630276,
        -0.0031508619736904902,
        0.0017569770547843028,
        -0.00045637633363159943,
        -0.0017272743160151303,
        -0.004850766051893397,
        -0.0015435881169640177,
        0.000871410762092511,
        -0.0046690341148760235,
        -0.0006717404374501456,
        -0.00304486624987099,
        -0.001751689473161437,
        -0.004483204677955724,
        -0.00038650975098166373,
        -0.0001270065960240251,
        -0.0003380706316670535,
        -0.008645382045841199,
        -0.00042084886758710733,
        -0.002870340750907864,
    ]
    x2results = [
        0.07073557241476708,
        0.06962048630932399,
        0.07014952579962609,
        0.07115810916114472,
        0.06965928854947752,
        0.07163069372397597,
        0.07089428677003226,
        0.07105036822199548,
        0.07055188889683411,
        0.07178828771761253,
        0.07108649658419547,
        0.07014976030079044,
        0.07013233734189801,
        0.07269184520627828,
        0.07026865262442902,
        0.07024049933599255,
        0.07255522821774693,
        0.07018568554401064,
        0.07225263048236619,
        0.07052716312848073,
        0.07024368425679589,
        0.07121114361795099,
        0.06978575600305406,
    ]
    p1results = [
        0.07507507044765914,
        0.07510679224036242,
        0.07605473709785131,
        0.0758074532418194,
        0.07534979484947107,
        0.07401383966670384,
        0.07477621118497266,
        0.07537404353752364,
        0.0750368698817334,
        0.0754033528579343,
        0.0749498710706332,
        0.07434496003925792,
        0.07728657396969658,
        0.07399235689447041,
        0.07569985687109715,
        0.07588374315571056,
        0.07438786494439528,
        0.07452169107365568,
        0.07563111880182889,
        0.0754454167542434,
        0.07586818593121604,
        0.07558788502434018,
        0.07537426412710296,
    ]
    presults = [
        -0.006936056938517612,
        0.007162641811828369,
        0.004092985336256041,
        0.0015257830383916472,
        -0.0035954493651382275,
        -0.00857494106848574,
        -0.005975924264650281,
        -0.0007794637263433804,
        -0.0032432701304090806,
        0.004636399759950497,
        -0.0077659755003015125,
        0.0012219404713463438,
        -0.005813327458032411,
        -0.006623139461975567,
        -0.006158097066281225,
        -0.007722814198378362,
        0.003551528134004016,
        -0.004062764051708979,
        -0.006549139533550153,
        -0.0016477561275578447,
        -0.0004981124026731531,
        -0.0018816331254126921,
        -0.0019333629302237811,
    ]
    p2results = [
        0.07730934862206633,
        0.07734260484804065,
        0.07829441379130923,
        0.07804450462276571,
        0.07759550072649996,
        0.07624850679999039,
        0.07701193228801893,
        0.07761452762506892,
        0.0772790473182888,
        0.07764034092892877,
        0.0771860494407,
        0.07658431582555629,
        0.07951881968959741,
        0.07623457534148566,
        0.0779332797203656,
        0.07810707383472705,
        0.07661760312883541,
        0.07676017718114204,
        0.0778705881359709,
        0.07767956087555943,
        0.07810036385564213,
        0.0778247090533542,
        0.0776171483539101,
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
        0.175438900210304,
        0.1754444250920227,
        0.17544089750284503,
        0.17545771248659853,
        0.17543611656310257,
        0.17543613344239561,
        0.175436990691907,
        0.17544404948283715,
        0.17543988702881463,
        0.17543755870755723,
        0.1754376205860491,
        0.1754488221489465,
        0.1754369607099373,
        0.175444662035645,
        0.17544263609826882,
        0.17543897867673636,
        0.1754429191962544,
        0.17544672691250823,
        0.17545523227288715,
        0.17544229604828684,
        0.17544420564473528,
        0.17545003776460372,
        0.17543633153874252,
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
