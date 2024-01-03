# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import pytest

from simphony.utils import str2float, wl2freq, freq2wl, wlum2freq


def test_wl2freq():
    # Test converting a few random wavelengths to frequencies
    assert wl2freq(1) == 299792458
    assert wl2freq(2) == 149896229
    assert wl2freq(3) == 99930819.33333333


def test_freq2wl():
    # Test converting a few random frequencies to wavelengths
    assert freq2wl(4) == 74948114.5
    assert freq2wl(5) == 59958491.6
    assert freq2wl(6) == 49965409.666666664


def test_wlum2freq():
    # Test converting a few random wavelengths in microns to frequencies
    assert wlum2freq(1) == wl2freq(1e-6)
    assert wlum2freq(2) == wl2freq(2e-6)
    assert wlum2freq(3) == wl2freq(3e-6)


class TestString2Float:
    def test_no_suffix(self):
        assert str2float("2.53") == 2.53

    def test_femto(self):
        assert str2float("17.83f") == 17.83e-15

    def test_pico(self):
        assert str2float("-15.37p") == -15.37e-12

    def test_nano(self):
        assert str2float("158.784n") == 158.784e-9

    def test_micro(self):
        assert str2float("15.26u") == 15.26e-06

    def test_milli(self):
        assert str2float("-15.781m") == -15.781e-3

    def test_centi(self):
        assert str2float("14.5c") == 14.5e-2

    def test_kilo(self):
        assert str2float("-0.257k") == -0.257e3

    def test_Mega(self):
        assert str2float("15.26M") == 15.26e6

    def test_Giga(self):
        assert str2float("-8.73G") == -8.73e9

    def test_Tera(self):
        assert str2float("183.4T") == 183.4e12

    def test_e(self):
        assert str2float("15.2e-6") == 15.2e-6

    def test_E(self):
        assert str2float("0.4E6") == 0.4e6

    def test_unrecognized(self):
        with pytest.raises(ValueError):
            str2float("17.3o")

    def test_malformed(self):
        with pytest.raises(ValueError):
            str2float("17.3.5e7")
