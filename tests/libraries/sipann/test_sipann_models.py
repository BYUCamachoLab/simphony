from sax import SDict

import pytest

try:
    from simphony.libraries import sipann
except ImportError:
    SIPANN_AVAILABLE = False


class TestGapFuncSymmetric:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.gap_func_symmetric(
                width=350,
                thickness=160,
                gap=(lambda x: x * 3),
                dgap=(lambda x: 3),
                zmin=0.0,
                zmax=1.0,
            )

    def test_instantiable(self):
        dev: SDict = sipann.gap_func_symmetric(
            width=500,
            thickness=220,
            gap=(lambda x: x * 3),
            dgap=(lambda x: 3),
            zmin=0.0,
            zmax=1.0,
        )
        # TODO: Test the actual values returned in the SDict.

    def test_s_params(self, std_wl_um):
        dev: SDict = sipann.gap_func_symmetric(
            width=500,
            thickness=220,
            gap=(lambda x: x * 3),
            dgap=(lambda x: 3),
            zmin=0.0,
            zmax=1.0,
        )
        # TODO: Test the actual values returned in the SDict.


class TestGapFuncAntiSymmetric:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.gap_func_antisymmetric(gap=lambda _: 50)

    # TODO: The below tests don't run. Why? Where is siepic?
    #       Why do we have any siepic tests in here anyway? They should be in test_siepic_models.py
    
    # def test_instantiable(self):
    #     siepic.DirectionalCoupler(gap=200, coupling_length=45)

    # def test_s_params(self, std_wl_um):
    #     dc = siepic.DirectionalCoupler(gap=200, coupling_length=45)
    #     s = dc.s_params(std_wl_um)


class Testhalf_ring:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.half_ring(width=350, thickness=160, radius=5000, gap=50)

    def test_instantiable(self):
        sipann.half_ring(width=500, thickness=220, radius=5000, gap=100)

    def test_s_params(self, std_wl_um):
        dev: SDict = sipann.half_ring(width=500, thickness=220, radius=5000, gap=100)
        # TODO: Test the actual values returned in the SDict.


class Teststraight_coupler:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.straight_coupler(width=400, thickness=160, gap=50, length=1000)

    def test_instantiable(self):
        sipann.straight_coupler(width=500, thickness=220, gap=150, length=1000)

    def test_s_params(self, std_wl_um):
        dev: SDict = sipann.straight_coupler(width=500, thickness=220, gap=180, length=1000)
        # TODO: Test the actual values returned in the SDict.


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
        sipann.standard_coupler(
            width=500,
            thickness=220,
            gap=180,
            length=2000,
            horizontal=2000,
            vertical=2000,
        )

    def test_s_params(self, std_wl_um):
        dev: SDict = sipann.standard_coupler(
            width=500,
            thickness=220,
            gap=180,
            length=2000,
            horizontal=2000,
            vertical=2000,
        )
        # TODO: Test the actual values returned in the SDict.


class Testdouble_half_ring:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.double_half_ring(width=380, thickness=250, radius=5000, gap=100)

    def test_instantiable(self):
        sipann.double_half_ring(width=500, thickness=220, radius=5000, gap=100)

    def test_s_params(self, std_wl_um):
        dev: SDict = sipann.double_half_ring(width=500, thickness=220, radius=5000, gap=100)
        # TODO: Test the actual values returned in the SDict.


class Testangled_half_ring:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.angled_half_ring(
                width=375, thickness=175, radius=5000, gap=150, theta=0.5
            )

    def test_instantiable(self):
        sipann.angled_half_ring(
            width=500, thickness=220, radius=5000, gap=150, theta=0.5
        )

    def test_s_params(self, std_wl_um):
        dev: SDict = sipann.angled_half_ring(
            width=500, thickness=220, radius=5000, gap=150, theta=0.5
        )
        # TODO: Test the actual values returned in the SDict.


class Testwaveguide:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.waveguide(width=350, thickness=250, length=10000)

    def test_instantiable(self):
        sipann.waveguide(width=500, thickness=220, length=10000)

    def test_s_params(self, std_wl_um):
        dev: SDict = sipann.waveguide(width=500, thickness=220, length=10000)
        # TODO: Test the actual values returned in the SDict.


class Testracetrack:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.racetrack(width=625, thickness=175, radius=5000, gap=80, length=5000)

    def test_instantiable(self):
        sipann.racetrack(width=500, thickness=220, radius=5000, gap=150, length=2000)

    def test_s_params(self, std_wl_um):
        dev: SDict = sipann.racetrack(
            width=500, thickness=220, radius=5000, gap=150, length=2000
        )
        # TODO: Test the actual values returned in the SDict.
