# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

from simphony.libraries import siepic
from simphony.libraries import sipann


class TestSiepicHash:

    def test_hash(self):

        bdc1 = siepic.BidirectionalCoupler(thickness=2.2e-7, width=5e-7)
        bdc2 = siepic.BidirectionalCoupler(thickness=2.2e-7, width=5e-7)
        assert bdc1.__hash__() == bdc2.__hash__()

        bdc1 = siepic.BidirectionalCoupler(thickness=2.1e-7)
        bdc2 = siepic.BidirectionalCoupler(thickness=2.2e-7)
        assert bdc1.__hash__() != bdc2.__hash__()

        bdc1 = siepic.BidirectionalCoupler(width=5.2e-7)
        bdc2 = siepic.BidirectionalCoupler(width=5e-7)
        assert bdc1.__hash__() != bdc2.__hash__()

        hr1siepic = siepic.HalfRing(gap=1e-7, radius=1e-5, width=5e-7, thickness=2.2e-7)
        hr2siepic = siepic.HalfRing(gap=1e-7, radius=1e-5, width=5e-7, thickness=2.2e-7)
        assert hr1siepic.__hash__() == hr2siepic.__hash__()

        hr1siepic = siepic.HalfRing(gap=8e-8, radius=1e-5, width=5e-7, thickness=2.1e-7)
        hr2siepic = siepic.HalfRing(gap=1e-7, radius=1e-5, width=5e-7, thickness=2.2e-7)
        assert hr1siepic.__hash__() != hr2siepic.__hash__()

        hr1siepic = siepic.HalfRing(gap=1e-7, radius=5e-6, width=5e-7, thickness=2.2e-7)
        hr2siepic = siepic.HalfRing(gap=1e-7, radius=1e-5, width=5e-7, thickness=2.2e-7)
        assert hr1siepic.__hash__() != hr2siepic.__hash__()

        dc1 = siepic.DirectionalCoupler()
        dc2 = siepic.DirectionalCoupler()
        assert dc1.__hash__() == dc2.__hash__()

        dc1 = siepic.DirectionalCoupler(Lc=0)
        dc2 = siepic.DirectionalCoupler()
        assert dc1.__hash__() != dc2.__hash__()

        term1 = siepic.Terminator()
        term2 = siepic.Terminator()
        assert term1.__hash__() == term2.__hash__()

        wg1 = siepic.Waveguide(length=150e-6)
        wg2 = siepic.Waveguide(length=150e-6)
        assert wg1.__hash__() == wg2.__hash__()

        wg1 = siepic.Waveguide(length=150e-6)
        wg2 = siepic.Waveguide(length=50e-6)
        assert wg1.__hash__() != wg2.__hash__()

        wg1 = siepic.Waveguide(width=5.2e-7)
        wg2 = siepic.Waveguide()
        assert wg1.__hash__() != wg2.__hash__()

        wg1 = siepic.Waveguide(height=2e-7)
        wg2 = siepic.Waveguide()
        assert wg1.__hash__() != wg2.__hash__()


class TestSipannHash:

    def test_hash(self):

        hr1 = sipann.HalfRing(gap=2e-7, radius=1e-5, width=5e-7, thickness=2.2e-7)
        hr2 = sipann.HalfRing(gap=2e-7, radius=1e-5, width=5e-7, thickness=2.2e-7)
        assert hr1.__hash__() == hr2.__hash__()

        hr1 = sipann.HalfRing(gap=1.5e-7, radius=1e-5, width=5e-7, thickness=2.2e-7)
        hr2 = sipann.HalfRing(gap=2e-7, radius=1e-5, width=5e-7, thickness=2.2e-7)
        assert hr1.__hash__() != hr2.__hash__()

        hr1 = sipann.HalfRing(gap=2e-7, radius=8e-6, width=5e-7, thickness=2.2e-7)
        hr2 = sipann.HalfRing(gap=2e-7, radius=1e-5, width=5e-7, thickness=2.2e-7)
        assert hr1.__hash__() != hr2.__hash__()

        hr1 = sipann.HalfRing(gap=2e-7, radius=1e-5, width=5.2e-7, thickness=2.2e-7)
        hr2 = sipann.HalfRing(gap=2e-7, radius=1e-5, width=5e-7, thickness=2.2e-7)
        assert hr1.__hash__() != hr2.__hash__()

        hr1 = sipann.HalfRing(gap=2e-7, radius=1e-5, width=5e-7, thickness=2.1e-7)
        hr2 = sipann.HalfRing(gap=2e-7, radius=1e-5, width=5e-7, thickness=2.2e-7)
        assert hr1.__hash__() != hr2.__hash__()

        hra1 = sipann.HalfRacetrack(gap=1e-7, radius=1e-5, width=5e-7, thickness=2.2e-7, length=1e-7)
        hra2 = sipann.HalfRacetrack(gap=1e-7, radius=1e-5, width=5e-7, thickness=2.2e-7, length=1e-7)
        assert hra1.__hash__() == hra2.__hash__()

        hra1 = sipann.HalfRacetrack(gap=1.5e-7, radius=10e-6, width=5e-7, thickness=2.2e-7, length=2e-7)
        hra2 = sipann.HalfRacetrack(gap=2e-7, radius=1e-5, width=5e-7, thickness=2.2e-7, length=2e-7)
        assert hra1.__hash__() != hra2.__hash__()

        hra1 = sipann.HalfRacetrack(gap=2e-7, radius=5e-6, width=5e-7, thickness=2.2e-7, length=2e-7)
        hra2 = sipann.HalfRacetrack(gap=2e-7, radius=1e-5, width=5e-7, thickness=2.2e-7, length=2e-7)
        assert hra1.__hash__() != hra2.__hash__()

        hra1 = sipann.HalfRacetrack(gap=2e-7, radius=1e-5, width=5.2e-7, thickness=2.2e-7, length=2e-7)
        hra2 = sipann.HalfRacetrack(gap=2e-7, radius=1e-5, width=5e-7, thickness=2.2e-7, length=2e-7)
        assert hra1.__hash__() != hra2.__hash__()

        hra1 = sipann.HalfRacetrack(gap=2e-7, radius=1e-5, width=5e-7, thickness=2.1e-7, length=2e-7)
        hra2 = sipann.HalfRacetrack(gap=2e-7, radius=1e-5, width=5e-7, thickness=2.2e-7, length=2e-7)
        assert hra1.__hash__() != hra2.__hash__()

        hra1 = sipann.HalfRacetrack(gap=2e-7, radius=1e-5, width=5e-7, thickness=2.2e-7, length=2.2e-7)
        hra2 = sipann.HalfRacetrack(gap=2e-7, radius=1e-5, width=5e-7, thickness=2.2e-7, length=2e-7)
        assert hra1.__hash__() != hra2.__hash__()

        sc1 = sipann.StraightCoupler(width=5e-7, thickness=2e-7, gap=2e-7, length=1e-7)
        sc2 = sipann.StraightCoupler(width=5e-7, thickness=2e-7, gap=2e-7, length=1e-7)
        assert sc1.__hash__() == sc2.__hash__()

        sc1 = sipann.StraightCoupler(width=5.8e-7, thickness=2e-7, gap=2e-7, length=2e-7)
        sc2 = sipann.StraightCoupler(width=5e-7, thickness=2e-7, gap=2e-7, length=2e-7)
        assert sc1.__hash__() != sc2.__hash__()

        sc1 = sipann.StraightCoupler(width=5e-7, thickness=2.2e-7, gap=2e-7, length=2e-7)
        sc2 = sipann.StraightCoupler(width=5e-7, thickness=2e-7, gap=2e-7, length=2e-7)
        assert sc1.__hash__() != sc2.__hash__()

        sc1 = sipann.StraightCoupler(width=5e-7, thickness=2e-7, gap=2.2e-7, length=2e-7)
        sc2 = sipann.StraightCoupler(width=5e-7, thickness=2e-7, gap=2e-7, length=2e-7)
        assert sc1.__hash__() != sc2.__hash__()

        sc1 = sipann.StraightCoupler(width=5e-7, thickness=2e-7, gap=2e-7, length=2.2e-7)
        sc2 = sipann.StraightCoupler(width=5e-7, thickness=2e-7, gap=2e-7, length=2e-7)
        assert sc1.__hash__() != sc2.__hash__()

        wg1 = sipann.Waveguide(length=150e-6, width=5e-7, thickness=2e-7)
        wg2 = sipann.Waveguide(length=150e-6, width=5e-7, thickness=2e-7)
        assert wg1.__hash__() == wg2.__hash__()

        wg1 = sipann.Waveguide(length=150e-6, width=5e-7, thickness=2e-7)
        wg2 = sipann.Waveguide(length=50e-6, width=5e-7, thickness=2e-7)
        assert wg1.__hash__() != wg2.__hash__()

        wg1 = sipann.Waveguide(length=50e-6, width=5.2e-7, thickness=2e-7)
        wg2 = sipann.Waveguide(length=50e-6, width=5e-7, thickness=2e-7)
        assert wg1.__hash__() != wg2.__hash__()

        wg1 = sipann.Waveguide(length=50e-6, width=5e-7, thickness=2.3e-7)
        wg2 = sipann.Waveguide(length=50e-6, width=5e-7, thickness=2e-7)
        assert wg1.__hash__() != wg2.__hash__()

        stcoup1 = sipann.Standard(width=5e-7, thickness=2.3e-7, gap=2e-7, length=10e-6, horizontal=1e-6, vertical=1e-6)
        stcoup2 = sipann.Standard(width=5e-7, thickness=2.3e-7, gap=2e-7, length=10e-6, horizontal=1e-6, vertical=1e-6)
        assert stcoup1.__hash__() == stcoup2.__hash__()

        stcoup1 = sipann.Standard(width=5.8e-7, thickness=2.2e-7, gap=2e-7, length=10e-6, horizontal=2e-6, vertical=2e-6)
        stcoup2 = sipann.Standard(width=5e-7, thickness=2.2e-7, gap=2e-7, length=10e-6, horizontal=2e-6, vertical=2e-6)
        assert stcoup1.__hash__() != stcoup2.__hash__()

        stcoup1 = sipann.Standard(width=5e-7, thickness=2.3e-7, gap=2e-7, length=10e-6, horizontal=2e-6, vertical=2e-6)
        stcoup2 = sipann.Standard(width=5e-7, thickness=2.2e-7, gap=2e-7, length=10e-6, horizontal=2e-6, vertical=2e-6)
        assert stcoup1.__hash__() != stcoup2.__hash__()

        stcoup1 = sipann.Standard(width=5e-7, thickness=2.2e-7, gap=2.3e-7, length=10e-6, horizontal=2e-6, vertical=2e-6)
        stcoup2 = sipann.Standard(width=5e-7, thickness=2.2e-7, gap=2e-7, length=10e-6, horizontal=2e-6, vertical=2e-6)
        assert stcoup1.__hash__() != stcoup2.__hash__()

        stcoup1 = sipann.Standard(width=5e-7, thickness=2.2e-7, gap=2e-7, length=9e-6, horizontal=2e-6, vertical=2e-6)
        stcoup2 = sipann.Standard(width=5e-7, thickness=2.2e-7, gap=2e-7, length=10e-6, horizontal=2e-6, vertical=2e-6)
        assert stcoup1.__hash__() != stcoup2.__hash__()

        dhr1 = sipann.DoubleHalfRing(width=5e-7, thickness=2.3e-7, radius=1e-5, gap=2e-7)
        dhr2 = sipann.DoubleHalfRing(width=5e-7, thickness=2.3e-7, radius=1e-5, gap=2e-7)
        assert dhr1.__hash__() == dhr2.__hash__()

        dhr1 = sipann.DoubleHalfRing(width=5.8e-7, thickness=2.2e-7, radius=1e-5, gap=2e-7)
        dhr2 = sipann.DoubleHalfRing(width=5e-7, thickness=2.2e-7, radius=1e-5, gap=2e-7)
        assert dhr1.__hash__() != dhr2.__hash__()

        dhr1 = sipann.DoubleHalfRing(width=5e-7, thickness=2.3e-7, radius=1e-5, gap=2e-7)
        dhr2 = sipann.DoubleHalfRing(width=5e-7, thickness=2.2e-7, radius=1e-5, gap=2e-7)
        assert dhr1.__hash__() != dhr2.__hash__()

        dhr1 = sipann.DoubleHalfRing(width=5e-7, thickness=2.2e-7, radius=5e-6, gap=2e-7)
        dhr2 = sipann.DoubleHalfRing(width=5e-7, thickness=2.2e-7, radius=1e-5, gap=2e-7)
        assert dhr1.__hash__() != dhr2.__hash__()

        dhr1 = sipann.DoubleHalfRing(width=5e-7, thickness=2.2e-7, radius=1e-5, gap=1e-7)
        dhr2 = sipann.DoubleHalfRing(width=5e-7, thickness=2.2e-7, radius=1e-5, gap=2e-7)
        assert dhr1.__hash__() != dhr2.__hash__()

        ahr1 = sipann.AngledHalfRing(width=5e-7, thickness=2.3e-7, radius=1e-5, theta=45, gap=2e-7)
        ahr2 = sipann.AngledHalfRing(width=5e-7, thickness=2.3e-7, radius=1e-5, theta=45, gap=2e-7)
        assert ahr1.__hash__() == ahr2.__hash__()

        ahr1 = sipann.AngledHalfRing(width=5.8e-7, thickness=2.2e-7, radius=1e-5, theta=45, gap=2e-7)
        ahr2 = sipann.AngledHalfRing(width=5e-7, thickness=2.2e-7, radius=1e-5, theta=45, gap=2e-7)
        assert ahr1.__hash__() != ahr2.__hash__()

        ahr1 = sipann.AngledHalfRing(width=5e-7, thickness=2.3e-7, radius=1e-5, theta=45, gap=2e-7)
        ahr2 = sipann.AngledHalfRing(width=5e-7, thickness=2.2e-7, radius=1e-5, theta=45, gap=2e-7)
        assert ahr1.__hash__() != ahr2.__hash__()

        ahr1 = sipann.AngledHalfRing(width=5e-7, thickness=2.2e-7, radius=5e-6, theta=45, gap=2e-7)
        ahr2 = sipann.AngledHalfRing(width=5e-7, thickness=2.2e-7, radius=1e-5, theta=45, gap=2e-7)
        assert ahr1.__hash__() != ahr2.__hash__()

        ahr1 = sipann.AngledHalfRing(width=5e-7, thickness=2.2e-7, radius=1e-5, theta=50, gap=2e-7)
        ahr2 = sipann.AngledHalfRing(width=5e-7, thickness=2.2e-7, radius=1e-5, theta=45, gap=2e-7)
        assert ahr1.__hash__() != ahr2.__hash__()

        ahr1 = sipann.AngledHalfRing(width=5e-7, thickness=2.2e-7, radius=1e-5, theta=45, gap=1e-7)
        ahr2 = sipann.AngledHalfRing(width=5e-7, thickness=2.2e-7, radius=1e-5, theta=45, gap=2e-7)
        assert ahr1.__hash__() != ahr2.__hash__()
