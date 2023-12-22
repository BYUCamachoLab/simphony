import pytest

from simphony.libraries import siepic


class TestBidirectional_coupler:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            siepic.bidirectional_coupler(pol="tem")
        with pytest.raises(ValueError):
            siepic.bidirectional_coupler(pol="te", thickness=200)

    def test_instantiable(self):
        siepic.bidirectional_coupler(thickness=220, width=500)

    def test_s_params(self, std_wl_um):
        bdc = siepic.bidirectional_coupler()
        bdc(std_wl_um)


class Testdirectional_coupler:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            siepic.directional_coupler(gap=300)

    def test_instantiable(self):
        siepic.directional_coupler(gap=200, coupling_length=45)

    def test_s_params(self, std_wl_um):
        dc = siepic.directional_coupler(gap=200, coupling_length=45)
        dc(std_wl_um)


class Testgrating_coupler:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            siepic.grating_coupler(pol="tem")
        with pytest.raises(ValueError):
            siepic.grating_coupler(pol="te", thickness=200)

    def test_instantiable(self):
        siepic.grating_coupler(pol="te", thickness=220, dwidth=0)

    def test_s_params(self, std_wl_um):
        gc = siepic.grating_coupler(pol="te")
        gc(std_wl_um)


class Testhalf_ring:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            siepic.half_ring(pol="tem")
        with pytest.raises(ValueError):
            siepic.half_ring(pol="te", gap=30, radius=10)

    def test_instantiable(self):
        siepic.half_ring(
            pol="te", gap=50, radius=5, width=500, thickness=220, coupling_length=0
        )

    def test_s_params(self, std_wl_um):
        ring = siepic.half_ring(
            pol="te", gap=50, radius=5, width=500, thickness=220, coupling_length=0
        )
        ring(std_wl_um)


class Testtaper:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            siepic.taper(w1=0.3)
        with pytest.raises(ValueError):
            siepic.taper(w2=0.5)

    def test_instantiable(self):
        siepic.taper(w1=0.5, w2=1.0, length=10.0)

    def test_s_params(self, std_wl_um):
        taper = siepic.taper(w1=0.5, w2=1.0, length=10.0)
        taper(std_wl_um)


class Testterminator:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            siepic.terminator(pol="tem")

    def test_instantiable(self):
        siepic.terminator(pol="te")
        siepic.terminator(pol="tm")

    def test_s_params(self, std_wl_um):
        term = siepic.terminator(pol="te")
        term(std_wl_um)
        term = siepic.terminator(pol="tm")
        term(std_wl_um)


class Testwaveguide:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            siepic.waveguide(pol="tem")
        with pytest.raises(ValueError):
            siepic.waveguide(height=200)

    def test_instantiable(self):
        siepic.waveguide(pol="te", length=100, width=500, height=220, loss=2)

    def test_s_params(self, std_wl_um):
        wg = siepic.waveguide(pol="te", length=100, width=500, height=220, loss=2)
        wg(std_wl_um)


class Testy_branch:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            siepic.y_branch(pol="tem")

    def test_instantiable(self):
        siepic.y_branch(pol="te", thickness=220, width=500)

    def test_s_params(self, std_wl_um):
        yb = siepic.y_branch(pol="te")
        yb(std_wl_um)
