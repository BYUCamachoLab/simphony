import pytest

from simphony.libraries import siepic


class TestBidirectionalCoupler:
    def test_invalid_parameters(self):
        with pytest.raises(TypeError):
            siepic.bidirectional_coupler(pol="tem")  # type: ignore
        with pytest.raises(TypeError):
            siepic.bidirectional_coupler(pol="te", thickness=200)  # type: ignore

    def test_instantiable(self):
        siepic.bidirectional_coupler(thickness=220, width=500)

    def test_s_params(self, std_wl_um):
        siepic.bidirectional_coupler(wl=std_wl_um)


class TestDirectionalCoupler:
    def test_invalid_parameters(self):
        with pytest.raises(FileNotFoundError):
            siepic.directional_coupler(gap=300)

    def test_instantiable(self):
        siepic.directional_coupler(gap=200, coupling_length=45)

    def test_s_params(self, std_wl_um):
        siepic.directional_coupler(wl=std_wl_um, gap=200, coupling_length=45)


class TestGratingCoupler:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            siepic.grating_coupler(pol="tem")  # type: ignore
        with pytest.raises(ValueError):
            siepic.grating_coupler(pol="te", thickness=200)

    def test_instantiable(self):
        siepic.grating_coupler(pol="te", thickness=220, dwidth=0)

    def test_s_params(self, std_wl_um):
        siepic.grating_coupler(wl=std_wl_um, pol="te")


class TestHalfRing:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            siepic.half_ring(pol="tem")  # type: ignore
        with pytest.raises(FileNotFoundError):
            siepic.half_ring(pol="te", gap=30, radius=10)

    def test_instantiable(self):
        siepic.half_ring(
            pol="te", gap=50, radius=5, width=500, thickness=220, coupling_length=0
        )

    def test_s_params(self, std_wl_um):
        siepic.half_ring(
            wl=std_wl_um,
            pol="te",
            gap=50,
            radius=5,
            width=500,
            thickness=220,
            coupling_length=0,
        )


class TestTaper:
    def test_invalid_parameters(self):
        with pytest.raises(FileNotFoundError):
            siepic.taper(w1=0.3)
        with pytest.raises(FileNotFoundError):
            siepic.taper(w2=0.5)

    def test_instantiable(self):
        siepic.taper(w1=0.5, w2=1.0, length=10.0)

    def test_s_params(self, std_wl_um):
        siepic.taper(wl=std_wl_um, w1=0.5, w2=1.0, length=10.0)


class TestTerminator:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            siepic.terminator(pol="tem")  # type: ignore

    def test_instantiable(self):
        siepic.terminator(pol="te")
        siepic.terminator(pol="tm")

    def test_s_params(self, std_wl_um):
        siepic.terminator(wl=std_wl_um, pol="te")
        siepic.terminator(wl=std_wl_um, pol="tm")


class TestWaveguide:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            siepic.waveguide(pol="tem")  # type: ignore
        with pytest.raises(FileNotFoundError):
            siepic.waveguide(height=200)

    def test_instantiable(self):
        siepic.waveguide(pol="te", length=100, width=500, height=220, loss=2)

    def test_s_params(self, std_wl_um):
        siepic.waveguide(
            wl=std_wl_um, pol="te", length=100, width=500, height=220, loss=2
        )


class TestYBranch:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            siepic.y_branch(pol="tem")  # type: ignore

    def test_instantiable(self):
        siepic.y_branch(pol="te", thickness=220, width=500)

    def test_s_params(self, std_wl_um):
        siepic.y_branch(wl=std_wl_um, pol="te")
