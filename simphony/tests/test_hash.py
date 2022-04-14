# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import numpy as np

from simphony.libraries import siepic
from simphony.libraries import sipann


class TestSiepicHash:

    def test_hash():

        bdc1 = siepic.BidirectionalCoupler()
        bdc2 = siepic.BidirectionalCoupler()
        assert np.allclose(bdc1.__hash__(), bdc2.__hash__())

        bdc1 = siepic.BidirectionalCoupler(2.1e-7)
        bdc2 = siepic.BidirectionalCoupler(2.3e-7)
        assert not np.allclose(bdc1.__hash__(), bdc2.__hash__())

        hr1siepic = siepic.HalfRing(gap=1e-7, radius=1e-5, width=5e-7, thickness=2.2e-7)
        hr2siepic = siepic.HalfRing(gap=1e-7, radius=1e-5, width=5e-7, thickness=2.2e-7)
        assert np.allclose(hr1siepic.__hash__(), hr2siepic.__hash__())

        hr1siepic = siepic.HalfRing(gap=8e-8, radius=10e-6, width=5.2e-7, thickness=2.1e-7)
        hr2siepic = siepic.HalfRing(gap=1e-7, radius=1e-5, width=5e-7, thickness=2.2e-7)
        assert not np.allclose(hr1siepic.__hash__(), hr2siepic.__hash__())

        dc1 = siepic.DirectionalCoupler()
        dc2 = siepic.DirectionalCoupler()
        assert np.allclose(dc1.__hash__(), dc2.__hash__())

        dc1 = siepic.DirectionalCoupler(gap=2e-7, Lc=10e-6)
        dc2 = siepic.DirectionalCoupler()
        assert not np.allclose(dc1.__hash__(), dc2.__hash__())

        term1 = siepic.Terminator()
        term2 = siepic.Terminator()
        assert np.allclose(term1.__hash__(), term2.__hash__())

        term1 = siepic.Terminator(w1=5e-7)     # this autocorrects to default attribute values everytime
        term2 = siepic.Terminator()
        assert not np.allclose(term1.__hash__(), term2.__hash__())

        # wg1 and wg2 hash should be the same
        wg1 = siepic.Waveguide(length=150e-6)
        wg2 = siepic.Waveguide(length=150e-6)
        assert np.allclose(wg1.__hash__(), wg2.__hash__())

        wg1 = siepic.Waveguide(length=150e-6)
        wg2 = siepic.Waveguide(length=50e-6)
        assert not np.allclose(wg1.__hash__(), wg2.__hash__())


class TestSipannHash:

    def test_hash():
                
        hr1siepic = sipann.HalfRing(gap=2e-7, radius=1e-5, width=5e-7, thickness=2.2e-7)
        hr2siepic = sipann.HalfRing(gap=2e-7, radius=1e-5, width=5e-7, thickness=2.2e-7)
        assert np.allclose(hr1siepic.__hash__(), hr2siepic.__hash__())

        hr1siepic = sipann.HalfRing(gap=1.5e-7, radius=10e-6, width=5.2e-7, thickness=2.1e-7)
        hr2siepic = sipann.HalfRing(gap=2e-7, radius=1e-5, width=5e-7, thickness=2.2e-7)
        assert not np.allclose(hr1siepic.__hash__(), hr2siepic.__hash__())

        hra1siepic = sipann.HalfRacetrack(gap=1e-7, radius=1e-5, width=5e-7, thickness=2.2e-7, length=1e-7)
        hra2siepic = sipann.HalfRacetrack(gap=1e-7, radius=1e-5, width=5e-7, thickness=2.2e-7, length=1e-7)
        assert np.allclose(hra1siepic.__hash__(), hra2siepic.__hash__())

        hra1siepic = sipann.HalfRacetrack(gap=1.5e-7, radius=10e-6, width=5.2e-7, thickness=2.1e-7, length=1e-7)
        hra2siepic = sipann.HalfRacetrack(gap=2e-7, radius=1e-5, width=5e-7, thickness=2.2e-7, length=2e-7)
        assert not np.allclose(hra1siepic.__hash__(), hra2siepic.__hash__())

        sc1 = sipann.StraightCoupler(width=5e-7, thickness=2e-7, gap=2e-7, length=1e-7)
        sc2 = sipann.StraightCoupler(width=5e-7, thickness=2e-7, gap=2e-7, length=1e-7)
        assert np.allclose(sc1.__hash__(), sc2.__hash__())

        sc1 = sipann.StraightCoupler(width=5.8e-7, thickness=2e-7, gap=3e-7, length=2e-7)
        sc2 = sipann.StraightCoupler(width=5e-7, thickness=2.3e-7, gap=2e-7, length=1e-7)
        assert not np.allclose(sc1.__hash__(), sc2.__hash__())

        wg1 = sipann.Waveguide(length=150e-6, width=5e-7, thickness=2e-7)
        wg2 = sipann.Waveguide(length=150e-6, width=5e-7, thickness=2e-7)
        assert np.allclose(wg1.__hash__(), wg2.__hash__())
        
        wg1 = sipann.Waveguide(length=150e-6, width=5e-7, thickness=2e-7)
        wg2 = sipann.Waveguide(length=50e-6, width=6e-7, thickness=2.3e-7)
        assert not np.allclose(wg1.__hash__(), wg2.__hash__())

        stcoup1 = sipann.Standard(width=5e-7, thickness=2.3e-7, gap=2e-7, length=10e-6, horizontal=1e-6, vertical=1e-6)
        stcoup2 = sipann.Standard(width=5e-7, thickness=2.3e-7, gap=2e-7, length=10e-6, horizontal=1e-6, vertical=1e-6)
        assert np.allclose(stcoup1.__hash__(), stcoup2.__hash__())

        stcoup1 = sipann.Standard(width=5.8e-7, thickness=2.2e-7, gap=3e-7, length=9e-6, horizontal=2e-6, vertical=2e-6)
        stcoup2 = sipann.Standard(width=5e-7, thickness=2.3e-7, gap=2e-7, length=10e-6, horizontal=1e-6, vertical=1e-6)
        assert not np.allclose(stcoup1.__hash__(), stcoup2.__hash__())

        dhr1 = sipann.DoubleHalfRing(width=5e-7, thickness=2.3e-7, radius=1e-5,gap=2e-7)
        dhr2 = sipann.DoubleHalfRing(width=5e-7, thickness=2.3e-7, radius=1e-5,gap=2e-7)
        assert np.allclose(dhr1.__hash__(), dhr2.__hash__())

        dhr1 = sipann.DoubleHalfRing(width=5.8e-7, thickness=2.2e-7, radius=9e-6,gap=3e-7)
        dhr2 = sipann.DoubleHalfRing(width=5e-7, thickness=2.3e-7, radius=1e-5,gap=2e-7)
        assert not np.allclose(dhr1.__hash__(), dhr2.__hash__())

        ahr1 = sipann.AngledHalfRing(width=5e-7, thickness=2.3e-7, radius=1e-5,theta=45,gap=2e-7)
        ahr2 = sipann.AngledHalfRing(width=5e-7, thickness=2.3e-7, radius=1e-5,theta=45,gap=2e-7)
        assert np.allclose(ahr1.__hash__(), ahr2.__hash__())

        ahr1 = sipann.AngledHalfRing(width=5.8e-7, thickness=2.2e-7, radius=9e-6,theta=50,gap=3e-7)
        ahr2 = sipann.AngledHalfRing(width=5e-7, thickness=2.3e-7, radius=1e-5,theta=45,gap=2e-7)
        assert not np.allclose(ahr1.__hash__(), ahr2.__hash__())
