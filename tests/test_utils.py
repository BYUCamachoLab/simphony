# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import pytest

from simphony import utils
import jax.numpy as jnp


class TestRect:
    def test_funcionality(self):
        r = 2
        phi = jnp.pi / 4
        res = utils.rect(r, phi)
        assert res == jnp.sqrt(2) + jnp.sqrt(2) * 1j

    def test_boundary(self):
        # 1 + 0j
        r = 1
        phi = 0
        res = utils.rect(r, phi)
        assert res == pytest.approx(1, abs=1e-15)

        # 0 + 1j
        r = 1
        phi = jnp.pi / 2
        res = utils.rect(r, phi)
        assert res == pytest.approx(1j, abs=1e-15)

        # -1 + 0j
        r = 1
        phi = jnp.pi
        res = utils.rect(r, phi)
        assert res == pytest.approx(-1, abs=1e-15)

        # 0 - 1j
        r = 1
        phi = 3 * jnp.pi / 2
        res = utils.rect(r, phi)
        assert res == pytest.approx(-1j, abs=1e-15)

        # 0 + 0j
        r = 0
        phi = 0
        res = utils.rect(r, phi)
        assert res == pytest.approx(0, abs=1e-15)

    def test_exceptions(self):
        # Negative radius
        with pytest.raises(ValueError):
            utils.rect(-1, 0)


class TestPolar:
    def test_funcionality(self):
        z = 1 + 1j
        res = utils.polar(z)
        assert res == (jnp.sqrt(2), jnp.pi / 4)

    def test_boundary(self):
        # 1 + 0j
        z = 1
        res = utils.polar(z)
        assert res == (1, 0)

        # 0 + 1j
        z = 1j
        res = utils.polar(z)
        assert res == (1, jnp.pi / 2)

        # -1 + 0j
        z = -1
        res = utils.polar(z)
        assert res == (1, jnp.pi)

        # 0 - 1j
        z = -1j
        res = utils.polar(z)
        assert res == (1, -jnp.pi / 2)

        # 0 + 0j
        z = 0
        res = utils.polar(z)
        assert res == (0, 0)

    def test_exceptions(self):
        pass


class TestAddPolar:
    def test_functionality(self):
        z1 = (1, jnp.pi / 4)  # r, phi
        z2 = (1, jnp.pi / 4)
        res = utils.add_polar(z1, z2)
        assert jnp.allclose(res[0], 2)  # r
        assert jnp.allclose(res[1], jnp.pi / 4)  # phi

    def test_exceptions(self):
        pass

    def test_boundary(self):
        # Test adding two zero vectors
        z1 = (0, 0)
        z2 = (0, 0)
        res = utils.add_polar(z1, z2)
        assert jnp.allclose(res[0], 0)  # r
        assert jnp.allclose(res[1], 0)  # phi

        # Test adding a zero vector to a non-zero vector
        z1 = (0, 0)
        z2 = (1, jnp.pi / 4)
        res = utils.add_polar(z1, z2)
        assert jnp.allclose(res[0], 1)
        assert jnp.allclose(res[1], jnp.pi / 4)


class TestMulPolar:
    def test_functionality(self):
        z1 = (1, jnp.pi / 4)  # r, phi
        z2 = (2, jnp.pi / 4)
        res = utils.mul_polar(z1, z2)
        assert jnp.allclose(res[0], 2)  # r
        assert jnp.allclose(res[1], jnp.pi / 2)  # phi

    def test_exceptions(self):
        pass

    def test_boundary(self):
        # Test multiplying two zero vectors
        z1 = (0, 0)
        z2 = (0, 0)
        res = utils.mul_polar(z1, z2)
        assert jnp.allclose(res[0], 0)  # r
        assert jnp.allclose(res[1], 0)  # phi

        # test multiplying a zero vector with a non-zero vector
        z1 = (0, 0)
        z2 = (1, jnp.pi / 4)
        res = utils.mul_polar(z1, z2)
        assert jnp.allclose(res[0], 0)


class TestMatMulPolar:
    # TODO: learn how this works and write tests
    pass


class TestMatAddPolar:
    # TODO: learn how this works and write tests
    pass


class TestString2Float:
    def test_no_suffix(self):
        assert utils.str2float("2.53") == 2.53

    def test_femto(self):
        assert utils.str2float("17.83f") == 17.83e-15

    def test_pico(self):
        assert utils.str2float("-15.37p") == -15.37e-12

    def test_nano(self):
        assert utils.str2float("158.784n") == 158.784e-9

    def test_micro(self):
        assert utils.str2float("15.26u") == 15.26e-06

    def test_milli(self):
        assert utils.str2float("-15.781m") == -15.781e-3

    def test_centi(self):
        assert utils.str2float("14.5c") == 14.5e-2

    def test_kilo(self):
        assert utils.str2float("-0.257k") == -0.257e3

    def test_Mega(self):
        assert utils.str2float("15.26M") == 15.26e6

    def test_Giga(self):
        assert utils.str2float("-8.73G") == -8.73e9

    def test_Tera(self):
        assert utils.str2float("183.4T") == 183.4e12

    def test_e(self):
        assert utils.str2float("15.2e-6") == 15.2e-6

    def test_E(self):
        assert utils.str2float("0.4E6") == 0.4e6

    def test_unrecognized(self):
        with pytest.raises(ValueError):
            utils.str2float("17.3o")

    def test_malformed(self):
        with pytest.raises(ValueError):
            utils.str2float("17.3.5e7")


class TestFreq2Wl:
    def test_funcionality(self):
        # Test converting a few random frequencies to wavelengths
        assert utils.freq2wl(299792458) == 1
        assert utils.freq2wl(149896229) == 2
        assert utils.freq2wl(99930819.33333333) == 3
        assert utils.freq2wl(4) == 74948114.5
        assert utils.freq2wl(5) == 59958491.6
        assert utils.freq2wl(6) == 49965409.666666664

    def test_exceptions(self):
        # Test with negative frequency
        with pytest.raises(ValueError):
            utils.freq2wl(-1)

        # Test with non-numeric frequency
        with pytest.raises(TypeError):
            utils.freq2wl("not_a_number")  # type: ignore

        # Test with zero frequency
        with pytest.raises(Exception):
            utils.freq2wl(0)

    def test_boundary(self):
        pass


class TestWl2Freq:
    def test_funcionality(self):
        # Test converting a few random wavelengths to frequencies
        assert utils.wl2freq(1) == 299792458
        assert utils.wl2freq(2) == 149896229
        assert utils.wl2freq(3) == 99930819.33333333

    def test_exceptions(self):
        # Test with negative wavelength
        with pytest.raises(ValueError):
            utils.wl2freq(-1)

        # Test with non-numeric wavelength
        with pytest.raises(TypeError):
            utils.wl2freq("not_a_number")  # type: ignore

        # Test with zero wavelength
        with pytest.raises(Exception):
            utils.wl2freq(0)

    def test_boundary(self):
        pass


class TestWlum2Freq:
    def test_funcionality(self):
        # Test converting a few random wavelengths in microns to frequencies
        assert utils.wlum2freq(1) == utils.wl2freq(1e-6)
        assert utils.wlum2freq(2) == utils.wl2freq(2e-6)
        assert utils.wlum2freq(3) == utils.wl2freq(3e-6)

    def test_exceptions(self):
        # Test with negative wavelength in microns
        with pytest.raises(ValueError):
            utils.wlum2freq(-1)

        # Test with non-numeric wavelength in microns
        with pytest.raises(TypeError):
            utils.wlum2freq("not_a_number")  # type: ignore

        # Test with zero wavelength in microns
        with pytest.raises(Exception):
            utils.wlum2freq(0)

    def test_boundary(self):
        pass


class TestXxppToXpxp:
    def test_funcionality(self):
        # [1, 2, 3, 4] -> [1, 3, 2, 4]
        xxpp = jnp.array([1, 2, 3, 4])
        xpxp = utils.xxpp_to_xpxp(xxpp)
        assert jnp.array_equal(xpxp, jnp.array([1, 3, 2, 4]))

        # TODO: more advanced functionality tests

    # TODO: exception and boundary-value tests


class TestXpxpToXxpp:
    def test_functionality(self):
        xpxp = jnp.array([1, 3, 2, 4])
        xxpp = utils.xpxp_to_xxpp(xpxp)
        assert jnp.array_equal(xxpp, jnp.array([1, 2, 3, 4]))

        # TODO: more advanced functionality tests

    # TODO: exception and boundary-value tests


class TestDictToMatrix:
    def test_functionality(self):
        # Simple case
        d = {("in0", "out0"): jnp.array(3.0)}
        matrix = utils.dict_to_matrix(d)
        # expected_matrix = jnp.array([[0, 3], [0, 0]])
        expected_matrix = jnp.array([[[0 + 0j, 3.0 + 0j], [0 + 0j, 0 + 0j]]])
        assert jnp.array_equal(matrix, expected_matrix)

        # Slightly more complex
        d = {
            ("a", "b"): jnp.array(1),
            ("b", "a"): jnp.array(2),
            ("c", "c"): jnp.array(3),
        }
        matrix = utils.dict_to_matrix(d)
        expected_matrix = jnp.array(
            [
                [
                    [0 + 0j, 1 + 0j, 0 + 0j],
                    [2 + 0j, 0 + 0j, 0 + 0j],
                    [0 + 0j, 0 + 0j, 3 + 0j],
                ]
            ]
        )
        assert jnp.array_equal(matrix, expected_matrix)

        # TODO: more advanced functionality tests

    # TODO: exception and boundary-value tests


class TestValidateModel:
    # TODO: learn how this works and write tests
    pass


class TestResample:
    # TODO: learn how this works and write tests
    pass
