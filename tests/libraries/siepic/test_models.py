import pytest
import re

from simphony.libraries import siepic
import jax.numpy as jnp
from tests.utils import is_sdict


class TestBidirectionalCoupler:
    def test_invalid_parameters(self):
        with pytest.raises(TypeError):
            siepic.bidirectional_coupler(pol="tem")  # type: ignore
        with pytest.raises(TypeError):
            siepic.bidirectional_coupler(pol="te", thickness=200)  # type: ignore

    def test_instantiable(self):
        # Valid thicknesses and widths
        for t in [210, 220, 230]:
            for w in [480, 500, 520]:
                result = siepic.bidirectional_coupler(thickness=t, width=w)
                assert is_sdict(result), "Result is not a valid SDict object"

        # Various wavelengths
        for wl in [0.5, 0.6, 0.7]:
            result = siepic.bidirectional_coupler(wl=wl)
            assert is_sdict(result), "Result is not a valid SDict object"

        # Array of wavelengths
        result = siepic.bidirectional_coupler(wl=jnp.array([1.50, 1.55, 1.60]))
        assert is_sdict(result), "Result is not a valid SDict object"

    def test_s_params(self, std_wl_um):
        siepic.bidirectional_coupler(wl=std_wl_um)


class TestDirectionalCoupler:
    def test_invalid_parameters(self):
        with pytest.raises(FileNotFoundError):
            siepic.directional_coupler(gap=300)

    def test_instantiable(self):
        # Valid gaps and coupling lengths
        gap = 200
        for length in [
            0,
            2.5,
            5,
            7.5,
            10,
            12.5,
            15,
            17.5,
            20,
            22.5,
            25,
            27.5,
            30,
            32.5,
            35,
            37.5,
            40,
            42.5,
            45,
            47.5,
        ]:
            result = siepic.directional_coupler(gap=gap, coupling_length=length)
            assert is_sdict(result), "Result is not a valid SDict object"

        # Various wavelengths
        for wl in [0.5, 0.6, 0.7]:
            result = siepic.directional_coupler(wl=wl)
            assert is_sdict(result), "Result is not a valid SDict object"

        # Array of wavelengths
        result = siepic.bidirectional_coupler(wl=jnp.array([1.50, 1.55, 1.60]))

    def test_s_params(self, std_wl_um):
        siepic.directional_coupler(wl=std_wl_um, gap=200, coupling_length=45)


class TestHalfRing:
    @classmethod
    def setup_class(cls):
        # Allowed parameter combinations (taken from docstring)
        cls.allowed_combinations_str = """
            te                     0      480          210         3     70
            te                     0      480          210         3     80
            te                     0      480          210         3    100
            te                     0      480          210         3    120
            te                     0      480          210         5     70
            te                     0      480          210         5     80
            te                     0      480          210         5    120
            te                     0      480          210        10    120
            te                     0      480          210        10    170
            te                     0      480          230         3     70
            te                     0      480          230         3     80
            te                     0      480          230         3    100
            te                     0      480          230         3    120
            te                     0      480          230         5     70
            te                     0      480          230         5     80
            te                     0      480          230         5    120
            te                     0      480          230        10    120
            te                     0      480          230        10    170
            te                     0      500          220         3     50
            te                     0      500          220         3     60
            te                     0      500          220         3     80
            te                     0      500          220         3    100
            te                     0      500          220         5     50
            te                     0      500          220         5     60
            te                     0      500          220         5    100
            te                     0      500          220        10    100
            te                     0      500          220        10    150
            te                     0      500          220        18    200
            te                     0      520          210         3     30
            te                     0      520          210         3     40
            te                     0      520          210         3     60
            te                     0      520          210         3     80
            te                     0      520          210         5     30
            te                     0      520          210         5     40
            te                     0      520          210         5     80
            te                     0      520          210        10     80
            te                     0      520          210        10    130
            te                     0      520          230         3     30
            te                     0      520          230         3     40
            te                     0      520          230         3     60
            te                     0      520          230         3     80
            te                     0      520          230         5     30
            te                     0      520          230         5     40
            te                     0      520          230         5     80
            te                     0      520          230        10     80
            te                     0      520          230        10    130
            te                     4      500          220        10    200
            tm                     0      480          210         5    320
            tm                     0      480          230         5    320
            tm                     0      500          220         5    300
            tm                     0      520          210         5    280
            tm                     0      520          230         5    280
        """

        # Parse the allowed combinations into a list of dictionaries
        pattern = re.compile(r"(\w+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)")
        cls.allowed_param_combinations = [
            {
                "pol": match[0],
                "coupling_length": int(match[1]),
                "width": int(match[2]),
                "thickness": int(match[3]),
                "radius": int(match[4]),
                "gap": int(match[5]),
            }
            for match in pattern.findall(cls.allowed_combinations_str)
        ]

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            siepic.half_ring(pol="tem")  # type: ignore
        with pytest.raises(FileNotFoundError):
            siepic.half_ring(pol="te", gap=30, radius=10)

    def test_instantiable(self):
        # Iterate over the allowed combinations and test each one
        for params in self.allowed_param_combinations:
            result = siepic.half_ring(
                pol=params["pol"],
                gap=params["gap"],
                radius=params["radius"],
                width=params["width"],
                thickness=params["thickness"],
                coupling_length=params["coupling_length"],
            )
            assert is_sdict(
                result
            ), f"Result is not a valid SDict object for params: {params}"

    def test_functionality(self, std_wl_um):
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
    @classmethod
    def setup_class(cls):
        cls.allowed_combinations_str = """
            0.4     1         1
            0.4     1         2
            0.4     1         3
            0.4     1         4
            0.4     1         5
            0.4     1         6
            0.4     1         7
            0.4     1         8
            0.4     1         9
            0.4     1        10
            0.4     1        11
            0.4     1        12
            0.4     1        13
            0.4     1        14
            0.4     1        15
            0.4     1        16
            0.4     1        17
            0.4     1        18
            0.4     1        19
            0.4     1        20
            0.4     2         1
            0.4     2         2
            0.4     2         3
            0.4     2         4
            0.4     2         5
            0.4     2         6
            0.4     2         7
            0.4     2         8
            0.4     2         9
            0.4     2        10
            0.4     2        11
            0.4     2        12
            0.4     2        13
            0.4     2        14
            0.4     2        15
            0.4     2        16
            0.4     2        17
            0.4     2        18
            0.4     2        19
            0.4     2        20
            0.4     3         1
            0.4     3         2
            0.4     3         3
            0.4     3         4
            0.4     3         5
            0.4     3         6
            0.4     3         7
            0.4     3         8
            0.4     3         9
            0.4     3        10
            0.4     3        11
            0.4     3        12
            0.4     3        13
            0.4     3        14
            0.4     3        15
            0.4     3        16
            0.4     3        17
            0.4     3        18
            0.4     3        19
            0.4     3        20
            0.5     1         1
            0.5     1         2
            0.5     1         3
            0.5     1         4
            0.5     1         5
            0.5     1         6
            0.5     1         7
            0.5     1         8
            0.5     1         9
            0.5     1        10
            0.5     1        11
            0.5     1        12
            0.5     1        13
            0.5     1        14
            0.5     1        15
            0.5     1        16
            0.5     1        17
            0.5     1        18
            0.5     1        19
            0.5     1        20
            0.5     2         1
            0.5     2         2
            0.5     2         3
            0.5     2         4
            0.5     2         5
            0.5     2         6
            0.5     2         7
            0.5     2         8
            0.5     2         9
            0.5     2        10
            0.5     2        11
            0.5     2        12
            0.5     2        13
            0.5     2        14
            0.5     2        15
            0.5     2        16
            0.5     2        17
            0.5     2        18
            0.5     2        19
            0.5     2        20
            0.5     3         1
            0.5     3         2
            0.5     3         3
            0.5     3         4
            0.5     3         5
            0.5     3         6
            0.5     3         7
            0.5     3         8
            0.5     3         9
            0.5     3        10
            0.5     3        11
            0.5     3        12
            0.5     3        13
            0.5     3        14
            0.5     3        15
            0.5     3        16
            0.5     3        17
            0.5     3        18
            0.5     3        19
            0.5     3        20
            0.6     1         1
            0.6     1         2
            0.6     1         3
            0.6     1         4
            0.6     1         5
            0.6     1         6
            0.6     1         7
            0.6     1         8
            0.6     1         9
            0.6     1        10
            0.6     1        11
            0.6     1        12
            0.6     1        13
            0.6     1        14
            0.6     1        15
            0.6     1        16
            0.6     1        17
            0.6     1        18
            0.6     1        19
            0.6     1        20
            0.6     2         1
            0.6     2         2
            0.6     2         3
            0.6     2         4
            0.6     2         5
            0.6     2         6
            0.6     2         7
            0.6     2         8
            0.6     2         9
            0.6     2        10
            0.6     2        11
            0.6     2        12
            0.6     2        13
            0.6     2        14
            0.6     2        15
            0.6     2        16
            0.6     2        17
            0.6     2        18
            0.6     2        19
            0.6     2        20
            0.6     3         1
            0.6     3         2
            0.6     3         3
            0.6     3         4
            0.6     3         5
            0.6     3         6
            0.6     3         7
            0.6     3         8
            0.6     3         9
            0.6     3        10
            0.6     3        11
            0.6     3        12
            0.6     3        13
            0.6     3        14
            0.6     3        15
            0.6     3        16
            0.6     3        17
            0.6     3        18
            0.6     3        19
            0.6     3        20
        """

        # Parse the allowed combinations into a list of dictionaries
        pattern = re.compile(r"(\d+\.\d+)\s+(\d+)\s+(\d+)")
        cls.allowed_param_combinations = [
            {"w1": float(match[0]), "w2": int(match[1]), "length": int(match[2])}
            for match in pattern.findall(cls.allowed_combinations_str)
        ]

    def test_invalid_parameters(self):
        with pytest.raises(FileNotFoundError):
            siepic.taper(w1=0.3)
        with pytest.raises(FileNotFoundError):
            siepic.taper(w2=0.5)

    def test_instantiable(self):
        # Iterate over the allowed combinations and test each one
        for params in self.allowed_param_combinations:
            result = siepic.taper(
                w1=params["w1"],
                w2=params["w2"],
                length=params["length"],
            )
            assert is_sdict(
                result
            ), f"Result is not a valid SDict object for params: {params}"

    def test_s_params(self, std_wl_um):
        siepic.taper(wl=std_wl_um, w1=0.5, w2=1.0, length=10.0)


class TestTerminator:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            siepic.terminator(pol="tem")  # type: ignore

    def test_instantiable(self):
        for pol in ["te", "tm"]:
            result = siepic.terminator(pol=pol)  # type: ignore
            assert is_sdict(result), "Result is not a valid SDict object"

        # Test for different wavelengths
        for wl in [1.4, 1.5, 1.6]:
            result = siepic.terminator(wl=wl)
            assert is_sdict(result), "Result is not a valid SDict object"

        # Test for array of wavelengths
        result = siepic.terminator(wl=jnp.array([1.50, 1.55, 1.60]))
        assert is_sdict(result), "Result is not a valid SDict object"

    def test_s_params(self, std_wl_um):
        siepic.terminator(wl=std_wl_um, pol="te")
        siepic.terminator(wl=std_wl_um, pol="tm")


class TestGratingCoupler:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            siepic.grating_coupler(pol="tem")  # type: ignore
        with pytest.raises(ValueError):
            siepic.grating_coupler(pol="te", thickness=200)

    def test_instantiable(self):
        # Test accross all combinations of parameters
        for pol in ["te", "tm"]:
            for thickness in [210, 220, 230]:
                for dwidth in [-20, 0, 20]:
                    result = siepic.grating_coupler(pol=pol, thickness=thickness, dwidth=dwidth)  # type: ignore
                    assert is_sdict(result), "Result is not a valid SDict object"

        # Test for different wavelengths
        for wl in [1.4, 1.5, 1.6]:
            result = siepic.grating_coupler(wl=wl)
            assert is_sdict(result), "Result is not a valid SDict object"

        # Test for array of wavelengths
        result = siepic.grating_coupler(wl=jnp.array([1.50, 1.55, 1.60]))
        assert is_sdict(result), "Result is not a valid SDict object"

    def test_s_params(self, std_wl_um):
        siepic.grating_coupler(wl=std_wl_um, pol="te")


class TestWaveguide:
    @classmethod
    def setup_class(cls):
        cls.allowed_combinations_str = """
            210      400
            210      420
            210      440
            210      460
            210      480
            210      500
            210      520
            210      540
            210      560
            210      580
            210      600
            210      640
            210      680
            210      720
            210      760
            210      800
            210      840
            210      880
            210      920
            210      960
            210     1000
            210     1040
            210     1080
            210     1120
            210     1160
            210     1200
            210     1240
            210     1280
            210     1320
            210     1360
            210     1400
            210     1500
            210     1600
            210     1700
            210     1800
            210     1900
            210     2000
            210     2100
            210     2200
            210     2300
            210     2400
            210     2500
            210     2600
            210     2700
            210     2800
            210     2900
            210     3000
            210     3100
            210     3200
            210     3300
            210     3400
            210     3500
            220      400
            220      420
            220      440
            220      460
            220      480
            220      500
            220      520
            220      540
            220      560
            220      580
            220      600
            220      640
            220      680
            220      720
            220      760
            220      800
            220      840
            220      880
            220      920
            220      960
            220     1000
            220     1040
            220     1080
            220     1120
            220     1160
            220     1200
            220     1240
            220     1280
            220     1320
            220     1360
            220     1400
            220     1500
            220     1600
            220     1700
            220     1800
            220     1900
            220     2000
            220     2100
            220     2200
            220     2300
            220     2400
            220     2500
            220     2600
            220     2700
            220     2800
            220     2900
            220     3000
            220     3100
            220     3200
            220     3300
            220     3400
            220     3500
            230      400
            230      420
            230      440
            230      460
            230      480
            230      500
            230      520
            230      540
            230      560
            230      580
            230      600
            230      640
            230      680
            230      720
            230      760
            230      800
            230      840
            230      880
            230      920
            230      960
            230     1000
            230     1040
            230     1080
            230     1120
            230     1160
            230     1200
            230     1240
            230     1280
            230     1320
            230     1360
            230     1400
            230     1500
            230     1600
            230     1700
            230     1800
            230     1900
            230     2000
            230     2100
            230     2200
            230     2300
            230     2400
            230     2500
            230     2600
            230     2700
            230     2800
            230     2900
            230     3000
            230     3100
            230     3200
            230     3300
            230     3400
            230     3500
        """

        # Parse the allowed combinations into a list of dictionaries
        pattern = re.compile(r"(\d+)\s+(\d+)")
        cls.allowed_param_combinations = [
            {"height": int(match[0]), "width": int(match[1])}
            for match in pattern.findall(cls.allowed_combinations_str)
        ]

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            siepic.waveguide(pol="tem")  # type: ignore
        with pytest.raises(FileNotFoundError):
            siepic.waveguide(height=200)

    def test_instantiable(self):
        # Test across all combinations of parameters
        for pol in ["te", "tm"]:
            for width_height_combo in self.allowed_param_combinations:
                height = width_height_combo["height"]
                width = width_height_combo["width"]
                result = siepic.waveguide(pol=pol, length=100, width=width, height=height, loss=2)  # type: ignore
                assert is_sdict(result), "Result is not a valid SDict object"

        # Test for different wavelengths
        for wl in [1.4, 1.5, 1.6]:
            result = siepic.waveguide(wl=wl, length=100, width=500, height=220, loss=2)
            assert is_sdict(result), "Result is not a valid SDict object"

        # Test for array of wavelengths
        result = siepic.waveguide(
            wl=jnp.array([1.50, 1.55, 1.60]), length=100, width=500, height=220, loss=2
        )
        assert is_sdict(result), "Result is not a valid SDict object"

        # Test for several loss values
        for loss in [0.1, 0.2, 0.3]:
            result = siepic.waveguide(
                wl=1.55, length=100, width=500, height=220, loss=loss
            )
            assert is_sdict(result), "Result is not a valid SDict object"

    def test_s_params(self, std_wl_um):
        siepic.waveguide(
            wl=std_wl_um, pol="te", length=100, width=500, height=220, loss=2
        )


class TestYBranch:
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            siepic.y_branch(pol="tem")  # type: ignore

    def test_instantiable(self):
        for pol in ["te", "tm"]:
            for thickness in [210, 220, 230]:
                for width in [480, 500, 520]:
                    result = siepic.y_branch(pol=pol, thickness=thickness, width=width)  # type: ignore
                    assert is_sdict(result), "Result is not a valid SDict object"

        # Test for different wavelengths
        for wl in [1.4, 1.5, 1.6]:
            result = siepic.y_branch(wl=wl)
            assert is_sdict(result), "Result is not a valid SDict object"

        # Test for array of wavelengths
        result = siepic.y_branch(wl=jnp.array([1.50, 1.55, 1.60]))
        assert is_sdict(result), "Result is not a valid SDict object"

    def test_s_params(self, std_wl_um: float):
        siepic.y_branch(wl=std_wl_um, pol="te")
