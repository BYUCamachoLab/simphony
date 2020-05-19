# -*- coding: utf-8 -*-
# Copyright © 2019-2020 Simphony Project Contributors and others (see AUTHORS.txt).
# The resources, libraries, and some source files under other terms (see NOTICE.txt).
#
# This file is part of Simphony.
#
# Simphony is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Simphony is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Simphony. If not, see <https://www.gnu.org/licenses/>.

import pytest

from simphony.tools import freq2wl, str2float, wl2freq


def test_wl2freq():
    pass


def test_freq2wl():
    pass


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
