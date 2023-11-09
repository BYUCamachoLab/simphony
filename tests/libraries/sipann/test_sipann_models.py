import pytest

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


class Testhalf_ring:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.half_ring(width=350, thickness=160, radius=5000, gap=50)

    def test_instantiable(self):
        dev = sipann.half_ring(width=500, thickness=220, radius=5000, gap=100)

    def test_s_params(self, std_wl_um):
        dev = sipann.half_ring(width=500, thickness=220, radius=5000, gap=100)
        s = dev.s_params(std_wl_um)


class TestHalfracetrack:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.Halfracetrack(
                width=625, thickness=245, radius=5000, gap=100, length=1000
            )

    def test_instantiable(self):
        dev = sipann.Halfracetrack(
            width=500, thickness=220, radius=5000, gap=100, length=1000
        )

    def test_s_params(self, std_wl_um):
        dev = sipann.Halfracetrack(
            width=500, thickness=220, radius=5000, gap=100, length=1000
        )
        s = dev.s_params(std_wl_um)


class Teststraight_coupler:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.straight_coupler(width=400, thickness=160, gap=50, length=1000)

    def test_instantiable(self):
        dev = sipann.straight_coupler(width=500, thickness=220, gap=150, length=1000)

    def test_s_params(self, std_wl_um):
        dev = sipann.straight_coupler(width=500, thickness=220, gap=180, length=1000)
        s = dev.s_params(std_wl_um)


class Teststandard_coupler:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.standard_coupler(
                width=399,
                thickness=240.1,
                gap=180,
                length=2000,
                horizontal=2000,
                vertical=2000,
            )

    def test_instantiable(self):
        dev = sipann.standard_coupler(
            width=500,
            thickness=220,
            gap=180,
            length=2000,
            horizontal=2000,
            vertical=2000,
        )

    def test_s_params(self, std_wl_um):
        dev = sipann.standard_coupler(
            width=500,
            thickness=220,
            gap=180,
            length=2000,
            horizontal=2000,
            vertical=2000,
        )
        s = dev.s_params(std_wl_um)


class Testdouble_half_ring:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.double_half_ring(width=380, thickness=250, radius=5000, gap=100)

    def test_instantiable(self):
        dev = sipann.double_half_ring(width=500, thickness=220, radius=5000, gap=100)

    def test_s_params(self, std_wl_um):
        dev = sipann.double_half_ring(width=500, thickness=220, radius=5000, gap=100)
        s = dev.s_params(std_wl_um)


class Testangled_half_ring:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.angled_half_ring(
                width=375, thickness=175, radius=5000, gap=150, theta=0.5
            )

    def test_instantiable(self):
        dev = sipann.angled_half_ring(
            width=500, thickness=220, radius=5000, gap=150, theta=0.5
        )

    def test_s_params(self, std_wl_um):
        dev = sipann.angled_half_ring(
            width=500, thickness=220, radius=5000, gap=150, theta=0.5
        )
        s = dev.s_params(std_wl_um)


class Testwaveguide:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.waveguide(width=350, thickness=250, length=10000)

    def test_instantiable(self):
        dev = sipann.waveguide(width=500, thickness=220, length=10000)

    def test_s_params(self, std_wl_um):
        dev = sipann.waveguide(width=500, thickness=220, length=10000)
        s = dev.s_params(std_wl_um)


class Testracetrack:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.racetrack(width=625, thickness=175, radius=5000, gap=80, length=5000)

    def test_instantiable(self):
        dev = sipann.racetrack(
            width=500, thickness=220, radius=5000, gap=150, length=2000
        )

    def test_s_params(self, std_wl_um):
        dev = sipann.racetrack(
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
