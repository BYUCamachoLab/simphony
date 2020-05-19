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

import numpy as np
import pytest

from simphony.library import ebeam, siepic
from simphony.tools import wl2freq

f = np.linspace(wl2freq(1600e-9), wl2freq(1500e-9))


def is_equal(c1, c2):
    return np.allclose(c1.s_parameters(f), c2.s_parameters(f))


class TestReimplementedComponents:
    def test_ebeam_y_1550(self):
        assert is_equal(siepic.ebeam_y_1550(), ebeam.ebeam_y_1550())

    def test_ebeam_bdc_te1550(self):
        assert is_equal(siepic.ebeam_bdc_te1550(), ebeam.ebeam_bdc_te1550())

    def test_ebeam_dc_halfring_straight(self):
        d1 = siepic.ebeam_dc_halfring_straight(
            gap=30e-9, radius=3e-6, width=520e-9, thickness=210e-9
        )
        d2 = ebeam.ebeam_dc_halfring_te1550()
        s1 = d1.s_parameters(f)
        s2 = np.transpose(d2.s_parameters(f), (0, 2, 1))
        assert np.allclose(s1, s2)

    def test_ebeam_dc_te1550(self):
        assert is_equal(siepic.ebeam_dc_te1550(), ebeam.ebeam_dc_te1550())

    def test_ebeam_terminator_te1550(self):
        assert is_equal(
            siepic.ebeam_terminator_te1550(), ebeam.ebeam_terminator_te1550()
        )

    def test_ebeam_ebeam_wg_integral_1550(self):
        assert is_equal(
            siepic.ebeam_wg_integral_1550(100e-6), ebeam.ebeam_wg_integral_1550(100e-6)
        )

    def test_ebeam_gc_te1550(self):
        assert is_equal(siepic.ebeam_gc_te1550(), ebeam.ebeam_gc_te1550())
