import pytest
import numpy as np

try:
    from simphony.libraries import sipann
except ImportError:
    SIPANN_AVAILABLE = False


class TestGapFuncSymmetric:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.GapFuncSymmetric(
                width=350,
                thickness=160,
                gap=(lambda x: x * 3),
                dgap=(lambda x: 3),
                zmin=0.0,
                zmax=1.0,
            )

    def test_instantiable(self):
        dev = sipann.GapFuncSymmetric(
            width=500,
            thickness=220,
            gap=(lambda x: x * 3),
            dgap=(lambda x: 3),
            zmin=0.0,
            zmax=1.0,
        )

    def test_s_params(self, std_wl_um):
        dev = sipann.GapFuncSymmetric(
            width=500,
            thickness=220,
            gap=(lambda x: x * 3),
            dgap=(lambda x: 3),
            zmin=0.0,
            zmax=1.0,
        )
        s = dev.s_params(std_wl_um)


# class TestGapFuncAntiSymmetric:
#     def test_invalid_parameters(self):
#         with pytest.raises(ValueError):
#             sipann.GapFuncAntiSymmetric(gap=300)

#     def test_instantiable(self):
#         siepic.DirectionalCoupler(gap=200, coupling_length=45)

#     def test_s_params(self, std_wl_um):
#         dc = siepic.DirectionalCoupler(gap=200, coupling_length=45)
#         s = dc.s_params(std_wl_um)


class TestHalfRing:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.HalfRing(width=350, thickness=160, radius=5000, gap=50)

    def test_instantiable(self):
        dev = sipann.HalfRing(width=500, thickness=220, radius=5000, gap=100)

    def test_s_params(self, std_wl_um):
        dev = sipann.HalfRing(width=500, thickness=220, radius=5000, gap=100)
        s = dev.s_params(std_wl_um)


class TestHalfRacetrack:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.HalfRacetrack(
                width=625, thickness=245, radius=5000, gap=100, length=1000
            )

    def test_instantiable(self):
        dev = sipann.HalfRacetrack(
            width=500, thickness=220, radius=5000, gap=100, length=1000
        )

    def test_s_params(self, std_wl_um):
        dev = sipann.HalfRacetrack(
            width=500, thickness=220, radius=5000, gap=100, length=1000
        )
        s = dev.s_params(std_wl_um)


class TestStraightCoupler:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.StraightCoupler(width=400, thickness=160, gap=50, length=1000)

    def test_instantiable(self):
        dev = sipann.StraightCoupler(width=500, thickness=220, gap=150, length=1000)

    def test_s_params(self, std_wl_um):
        dev = sipann.StraightCoupler(width=500, thickness=220, gap=180, length=1000)
        s = dev.s_params(std_wl_um)


class TestStandardCoupler:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.StandardCoupler(
                width=399,
                thickness=240.1,
                gap=180,
                length=2000,
                horizontal=2000,
                vertical=2000,
            )

    def test_instantiable(self):
        dev = sipann.StandardCoupler(
            width=500,
            thickness=220,
            gap=180,
            length=2000,
            horizontal=2000,
            vertical=2000,
        )

    def test_s_params(self, std_wl_um):
        dev = sipann.StandardCoupler(
            width=500,
            thickness=220,
            gap=180,
            length=2000,
            horizontal=2000,
            vertical=2000,
        )
        s = dev.s_params(std_wl_um)


class TestDoubleHalfRing:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.DoubleHalfRing(width=380, thickness=250, radius=5000, gap=100)

    def test_instantiable(self):
        dev = sipann.DoubleHalfRing(width=500, thickness=220, radius=5000, gap=100)

    def test_s_params(self, std_wl_um):
        dev = sipann.DoubleHalfRing(width=500, thickness=220, radius=5000, gap=100)
        s = dev.s_params(std_wl_um)


class TestAngledHalfRing:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.AngledHalfRing(
                width=375, thickness=175, radius=5000, gap=150, theta=0.5
            )

    def test_instantiable(self):
        dev = sipann.AngledHalfRing(
            width=500, thickness=220, radius=5000, gap=150, theta=0.5
        )

    def test_s_params(self, std_wl_um):
        dev = sipann.AngledHalfRing(
            width=500, thickness=220, radius=5000, gap=150, theta=0.5
        )
        s = dev.s_params(std_wl_um)


class TestWaveguide:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.Waveguide(width=350, thickness=250, length=10000)

    def test_instantiable(self):
        dev = sipann.Waveguide(width=500, thickness=220, length=10000)

    def test_s_params(self, std_wl_um):
        dev = sipann.Waveguide(width=500, thickness=220, length=10000)
        s = dev.s_params(std_wl_um)


class TestRacetrack:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.Racetrack(width=625, thickness=175, radius=5000, gap=80, length=5000)

    def test_instantiable(self):
        dev = sipann.Racetrack(
            width=500, thickness=220, radius=5000, gap=150, length=2000
        )

    def test_s_params(self, std_wl_um):
        dev = sipann.Racetrack(
            width=500, thickness=220, radius=5000, gap=150, length=2000
        )
        s = dev.s_params(std_wl_um)


# class TestPremadeCoupler:
#     def test_invalid_parameters(self):
#         with pytest.raises(ValueError):
#             sipann.PremadeCoupler(pol="tem")

#     def test_instantiable(self):
#         yb = siepic.YBranch(pol="te", thickness=220, width=500)

#     def test_s_params(self, std_wl_um):
#         yb = siepic.YBranch(pol="te")
#         s = yb.s_params(std_wl_um)
