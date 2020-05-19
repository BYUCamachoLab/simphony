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

from simphony.library.ebeam import ebeam_wg_integral_1550
from simphony.persist import export_model, import_model
from simphony.tools import wl2freq


def test_ebeam_wg_integral_1550(tmpdir):
    """Tests whether the results of `s_parameters()` for the same frequency on
    an exported and reloaded model are (close) to the same."""
    p = tmpdir.mkdir("persist").join("export")
    wg1 = ebeam_wg_integral_1550(100e-9)
    export_model(wg1, p, wl=np.linspace(1520e-9, 1580e-9, 51))

    wg = import_model(p + ".mdl")
    wg2 = wg()
    freq = np.linspace(wl2freq(1540e-9), wl2freq(1560e-9), 50)
    assert np.allclose(wg1.s_parameters(freq), wg2.s_parameters(freq))


def test_load_wrong_ext(tmpdir):
    """Tests whether the results of `s_parameters()` for the same frequency on
    an exported and reloaded model are (close) to the same."""
    p = tmpdir.mkdir("persist").join("export")
    with pytest.raises(ValueError):
        import_model(p)
