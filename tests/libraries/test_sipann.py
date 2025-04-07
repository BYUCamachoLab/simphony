from tests.utils import is_sdict

import pytest
from itertools import product

try:
    from simphony.libraries import sipann
except ImportError:
    sipann = None


@pytest.fixture(scope="module", autouse=True)
def skip_if_no_sipann():
    if sipann is None:
        pytest.skip("SIPANN not available.")


class TestGapFuncSymmetric:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.gap_func_symmetric(  # type: ignore[call-arg]
                width=350,
                thickness=160,
                gap=(lambda x: x * 3),
                dgap=(lambda x: 3),
                zmin=0.0,
                zmax=1.0,
            )

    def test_instantiable(self):
        # Test accross valid parameter ranges
        params = product(
            [400, 450, 500, 550, 600],  # width
            [180, 200, 220, 240],  # thickness
            [100, 150, 200],  # gap
            [80, 85, 90],  # sw_angle
        )
        for w, t, g, sw in params:
            result = sipann.gap_func_symmetric(width=w, thickness=t, gap=lambda _: g, sw_angle=sw)  # type: ignore[call-arg]
            assert is_sdict(
                result
            ), f"Failed for width={w}, thickness={t}, gap={g}, sw_angle={sw}"

        # Test for various wavelengths
        wavelengths = [1.55, 1.6, 1.65]  # in micrometers
        for wl in wavelengths:
            result = sipann.gap_func_symmetric(wl=wl)  # type: ignore[call-arg]
            assert is_sdict(result), f"Failed for wavelength={wl}"

        # Test for array of wavelengths
        wl_array = [1.55, 1.6, 1.65]  # in micrometers
        result = sipann.gap_func_symmetric(wl=wl_array)  # type: ignore[call-arg]
        assert is_sdict(result), f"Failed for wavelength array={wl_array}"

        # NOTE: this instantiation test currently does not test accross values of dgap, zmin, or zmax


class TestGapFuncAntiSymmetric:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.gap_func_antisymmetric(gap=lambda _: 50)  # type: ignore[call-arg]

    def test_instantiable(self):
        # Test accross valid parameter ranges
        params = product(
            [400, 450, 500, 550, 600],  # width
            [180, 200, 220, 240],  # thickness
            [100, 150, 200],  # gap
            [80, 85, 90],  # sw_angle
        )
        for w, t, g, sw in params:
            result = sipann.gap_func_antisymmetric(width=w, thickness=t, gap=lambda _: g, sw_angle=sw)  # type: ignore[call-arg]
            assert is_sdict(
                result
            ), f"Failed for width={w}, thickness={t}, gap={g}, sw_angle={sw}"

        # Test for various wavelengths
        wavelengths = [1.55, 1.6, 1.65]  # in micrometers
        for wl in wavelengths:
            result = sipann.gap_func_antisymmetric(wl=wl)  # type: ignore[call-arg]
            assert is_sdict(result), f"Failed for wavelength={wl}"

        # Test for array of wavelengths
        wl_array = [1.55, 1.6, 1.65]  # in micrometers
        result = sipann.gap_func_antisymmetric(wl=wl_array)  # type: ignore[call-arg]
        assert is_sdict(result), f"Failed for wavelength array={wl_array}"

        # NOTE: this does not currently test zmin, zmax, arc1-4


class TestHalfRing:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.half_ring(width=350, thickness=160, radius=5000, gap=50)  # type: ignore[call-arg]

    def test_instantiable(self):
        # Test accross valid parameter ranges
        params = product(
            [400, 450, 500, 550, 600],  # width
            [180, 200, 220, 240],  # thickness
            [10, 15, 20],  # radius
            [100, 150, 200],  # gap
            [80, 85, 90],  # sw_angle
        )
        for w, t, r, g, sw in params:
            result = sipann.half_ring(width=w, thickness=t, radius=r, gap=g, sw_angle=sw)  # type: ignore[call-arg]
            assert is_sdict(
                result
            ), f"Failed for width={w}, thickness={t}, radius={r}, gap={g}, sw_angle={sw}"

        # Test for various wavelengths
        wavelengths = [1.55, 1.6, 1.65]  # in micrometers
        for wl in wavelengths:
            result = sipann.half_ring(wl=wl)  # type: ignore[call-arg]
            assert is_sdict(result), f"Failed for wavelength={wl}"

        # Test for array of wavelengths
        wl_array = [1.55, 1.6, 1.65]  # in micrometers
        result = sipann.half_ring(wl=wl_array)  # type: ignore[call-arg]
        assert is_sdict(result), f"Failed for wavelength array={wl_array}"


class TestStraightCoupler:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.straight_coupler(width=400, thickness=160, gap=50, length=1000)  # type: ignore[call-arg]

    def test_instantiable(self):
        # Test accross valid parameter ranges
        params = product(
            [400, 450, 500, 550, 600],  # width
            [180, 200, 220, 240],  # thickness
            [100, 150, 200],  # gap
            [1000, 1500, 2000],  # length
            [80, 85, 90],  # sw_angle
        )
        for w, t, g, l, sw in params:
            result = sipann.straight_coupler(width=w, thickness=t, gap=g, length=l, sw_angle=sw)  # type: ignore[call-arg]
            assert is_sdict(
                result
            ), f"Failed for width={w}, thickness={t}, gap={g}, length={l}, sw_angle={sw}"

        # Test for various wavelengths
        wavelengths = [1.55, 1.6, 1.65]  # in micrometers
        for wl in wavelengths:
            result = sipann.straight_coupler(wl=wl)  # type: ignore[call-arg]
            assert is_sdict(result), f"Failed for wavelength={wl}"

        # Test for array of wavelengths
        wl_array = [1.55, 1.6, 1.65]  # in micrometers
        result = sipann.straight_coupler(wl=wl_array)  # type: ignore[call-arg]
        assert is_sdict(result), f"Failed for wavelength array={wl_array}"


class TestStandardCoupler:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.standard_coupler(  # type: ignore[call-arg]
                width=399,
                thickness=240.1,
                gap=180,
                length=2000,
                horizontal=2000,
                vertical=2000,
            )

    def test_instantiable(self):
        # Test accross valid parameter ranges
        params = product(
            [400, 450, 500, 550, 600],  # width
            [180, 200, 220, 240],  # thickness
            [100, 150, 200],  # gap
            [1000, 1500, 2000],  # length
            [10e3, 20e3],  # horizontal
            [10e3, 20e3],  # vertical
            [80, 85, 90],  # sw_angle
        )
        for w, t, g, l, h, v, sw in params:
            result = sipann.standard_coupler(width=w, thickness=t, gap=g, length=l, horizontal=h, vertical=v, sw_angle=sw)  # type: ignore[call-arg]
            assert is_sdict(
                result
            ), f"Failed for width={w}, thickness={t}, gap={g}, length={l}, horizontal={h}, vertical={v}, sw_angle={sw}"

        # Test for various wavelengths
        wavelengths = [1.55, 1.6, 1.65]  # in micrometers
        for wl in wavelengths:
            result = sipann.standard_coupler(wl=wl)  # type: ignore[call-arg]
            assert is_sdict(result), f"Failed for wavelength={wl}"

        # Test for array of wavelengths
        wl_array = [1.55, 1.6, 1.65]  # in micrometers
        result = sipann.standard_coupler(wl=wl_array)  # type: ignore[call-arg]
        assert is_sdict(result), f"Failed for wavelength array={wl_array}"


class TestDoubleHalfRing:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.double_half_ring(width=380, thickness=250, radius=5000, gap=100)  # type: ignore[call-arg]

    def test_instantiable(self):
        # Test accross valid parameter ranges
        params = product(
            [400, 450, 500, 550, 600],  # width
            [180, 200, 220, 240],  # thickness
            [10e3, 20e3],  # radius
            [100, 150, 200],  # gap
            [80, 85, 90],  # sw_angle
        )
        for w, t, r, g, sw in params:
            result = sipann.double_half_ring(width=w, thickness=t, radius=r, gap=g, sw_angle=sw)  # type: ignore[call-arg]
            assert is_sdict(
                result
            ), f"Failed for width={w}, thickness={t}, radius={r}, gap={g}, sw_angle={sw}"

        # Test for various wavelengths
        wavelengths = [1.55, 1.6, 1.65]  # in micrometers
        for wl in wavelengths:
            result = sipann.double_half_ring(wl=wl)  # type: ignore[call-arg]
            assert is_sdict(result), f"Failed for wavelength={wl}"

        # Test for array of wavelengths
        wl_array = [1.55, 1.6, 1.65]  # in micrometers
        result = sipann.double_half_ring(wl=wl_array)  # type: ignore[call-arg]
        assert is_sdict(result), f"Failed for wavelength array={wl_array}"


class TestAngledHalfRing:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.angled_half_ring(  # type: ignore[call-arg]
                width=375, thickness=175, radius=5000, gap=150, theta=0.5
            )

    def test_instantiable(self):
        # Test accross valid parameter ranges
        params = product(
            [400, 450, 500, 550, 600],  # width
            [180, 200, 220, 240],  # thickness
            [10e3, 20e3],  # radius
            [100, 150, 200],  # gap
            [0, 0.5, 1.0],  # theta
            [80, 85, 90],  # sw_angle
        )
        for w, t, r, g, th, sw in params:
            result = sipann.angled_half_ring(width=w, thickness=t, radius=r, gap=g, theta=th, sw_angle=sw)  # type: ignore[call-arg]
            assert is_sdict(
                result
            ), f"Failed for width={w}, thickness={t}, radius={r}, gap={g}, theta={th}, sw_angle={sw}"

        # Test for various wavelengths
        wavelengths = [1.55, 1.6, 1.65]  # in micrometers
        for wl in wavelengths:
            result = sipann.angled_half_ring(wl=wl)  # type: ignore[call-arg]
            assert is_sdict(result), f"Failed for wavelength={wl}"

        # Test for array of wavelengths
        wl_array = [1.55, 1.6, 1.65]  # in micrometers
        result = sipann.angled_half_ring(wl=wl_array)  # type: ignore[call-arg]
        assert is_sdict(result), f"Failed for wavelength array={wl_array}"


class TestWaveguide:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.waveguide(width=350, thickness=250, length=10000)  # type: ignore[call-arg]

    def test_instantiable(self):
        # Test accross valid parameter ranges
        params = product(
            [400, 450, 500, 550, 600],  # width
            [180, 200, 220, 240],  # thickness
            [1000, 1500, 2000],  # length
            [80, 85, 90],  # sw_angle
        )
        for w, t, l, sw in params:
            result = sipann.waveguide(width=w, thickness=t, length=l, sw_angle=sw)  # type: ignore[call-arg]
            assert is_sdict(
                result
            ), f"Failed for width={w}, thickness={t}, length={l}, sw_angle={sw}"

        # Test for various wavelengths
        wavelengths = [1.55, 1.6, 1.65]  # in micrometers
        for wl in wavelengths:
            result = sipann.waveguide(wl=wl)  # type: ignore[call-arg]
            assert is_sdict(result), f"Failed for wavelength={wl}"

        # Test for array of wavelengths
        wl_array = [1.55, 1.6, 1.65]
        # in micrometers
        result = sipann.waveguide(wl=wl_array)  # type: ignore[call-arg]
        assert is_sdict(result), f"Failed for wavelength array={wl_array}"


class TestRacetrack:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            sipann.racetrack(width=625, thickness=175, radius=5000, gap=80, length=5000)  # type: ignore[call-arg]

    def test_instantiable(self):
        # Test accross valid parameter ranges
        params = product(
            [400, 450, 500, 550, 600],  # width
            [180, 200, 220, 240],  # thickness
            [10e3, 20e3],  # radius
            [100, 150, 200],  # gap
            [10e3, 20e3],  # length
            [80, 85, 90],  # sw_angle
        )
        for w, t, r, g, l, sw in params:
            result = sipann.racetrack(width=w, thickness=t, radius=r, gap=g, length=l, sw_angle=sw)  # type: ignore[call-arg]
            assert is_sdict(
                result
            ), f"Failed for width={w}, thickness={t}, radius={r}, gap={g}, length={l}, sw_angle={sw}"

        # Test for various wavelengths
        wavelengths = [1.55, 1.6, 1.65]
        for wl in wavelengths:
            result = sipann.racetrack(wl=wl)  # type: ignore[call-arg]
            assert is_sdict(result), f"Failed for wavelength={wl}"

        # Test for array of wavelengths
        wl_array = [1.55, 1.6, 1.65]
        result = sipann.racetrack(wl=wl_array)  # type: ignore[call-arg]
        assert is_sdict(result), f"Failed for wavelength array={wl_array}"


class TestPremadeCoupler:
    def test_instantiable(self):
        # Test accross valid parameter ranges
        for s in [10, 20, 30, 40, 50, 100]:
            result = sipann.premade_coupler(split=s)  # type: ignore[call-arg]
            assert is_sdict(result), f"Failed for split={s}"

        # Test for various wavelengths
        wavelengths = [1.55, 1.6, 1.65]
        for wl in wavelengths:
            result = sipann.premade_coupler(wl=wl)  # type: ignore[call-arg]
            assert is_sdict(result), f"Failed for wavelength={wl}"

        # Test for array of wavelengths
        wl_array = [1.55, 1.6, 1.65]
        result = sipann.premade_coupler(wl=wl_array)  # type: ignore[call-arg]
        assert is_sdict(result), f"Failed for wavelength array={wl_array}"
