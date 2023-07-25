import pytest
import numpy as np

from simphony.libraries import siepic


class TestBidirectionalCoupler:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            siepic.BidirectionalCouplerTE(pol="tem")
        with pytest.raises(ValueError):
            siepic.BidirectionalCouplerTE(pol="te", thickness=200)

    def test_instantiable(self):
        bdc = siepic.BidirectionalCouplerTE(thickness=220, width=500)

    def test_s_params(self, std_wl_um):
        bdc = siepic.BidirectionalCouplerTE()
        s = bdc.s_params(std_wl_um)


class TestDirectionalCoupler:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            siepic.DirectionalCoupler(gap=300)

    def test_instantiable(self):
        siepic.DirectionalCoupler(gap=200, coupling_length=45)

    def test_s_params(self, std_wl_um):
        dc = siepic.DirectionalCoupler(gap=200, coupling_length=45)
        s = dc.s_params(std_wl_um)


class TestGratingCoupler:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            siepic.GratingCoupler(pol="tem")
        with pytest.raises(ValueError):
            siepic.GratingCoupler(pol="te", thickness=200)

    def test_instantiable(self):
        gc = siepic.GratingCoupler(pol="te", thickness=220, dwidth=0)

    def test_s_params(self, std_wl_um):
        gc = siepic.GratingCoupler(pol="te")
        s = gc.s_params(std_wl_um)


class TestHalfRing:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            siepic.HalfRing(pol="tem")
        with pytest.raises(ValueError):
            siepic.HalfRing(pol="te", gap=30, radius=10)

    def test_instantiable(self):
        siepic.HalfRing(
            pol="te", gap=50, radius=5, width=500, thickness=220, coupling_length=0
        )

    def test_s_params(self, std_wl_um):
        ring = siepic.HalfRing(
            pol="te", gap=50, radius=5, width=500, thickness=220, coupling_length=0
        )
        s = ring.s_params(std_wl_um)


class TestTaper:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            siepic.Taper(w1=0.3)
        with pytest.raises(ValueError):
            siepic.Taper(w2=0.5)

    def test_instantiable(self):
        siepic.Taper(w1=0.5, w2=1.0, length=10.0)

    def test_s_params(self, std_wl_um):
        taper = siepic.Taper(w1=0.5, w2=1.0, length=10.0)
        s = taper.s_params(std_wl_um)


class TestTerminator:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            siepic.Terminator(pol="tem")

    def test_instantiable(self):
        term = siepic.Terminator(pol="te")
        term = siepic.Terminator(pol="tm")

    def test_s_params(self, std_wl_um):
        term = siepic.Terminator(pol="te")
        s = term.s_params(std_wl_um)
        term = siepic.Terminator(pol="tm")
        s = term.s_params(std_wl_um)


class TestWaveguide:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            siepic.Waveguide(pol="tem")
        with pytest.raises(ValueError):
            siepic.Waveguide(height=200)

    def test_instantiable(self):
        siepic.Waveguide(pol="te", length=100, width=500, height=220, loss=2)

    def test_s_params(self, std_wl_um):
        wg = siepic.Waveguide(pol="te", length=100, width=500, height=220, loss=2)
        s = wg.s_params(std_wl_um)


class TestYBranch:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            siepic.YBranch(pol="tem")

    def test_instantiable(self):
        yb = siepic.YBranch(pol="te", thickness=220, width=500)

    def test_s_params(self, std_wl_um):
        yb = siepic.YBranch(pol="te")
        s = yb.s_params(std_wl_um)
