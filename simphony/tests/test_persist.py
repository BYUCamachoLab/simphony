# -*- coding: utf-8 -*-
#
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import numpy as np
import pytest

from simphony.library.ebeam import ebeam_wg_integral_1550
from simphony.persist import export_model, import_model
from simphony.tools import wl2freq


def test_ebeam_wg_integral_1550(tmpdir):
    """
    Tests whether the results of `s_parameters()` for the same frequency on
    an exported and reloaded model are (close) to the same.
    """
    p = tmpdir.mkdir("persist").join("export")
    wg1 = ebeam_wg_integral_1550(100e-9)
    export_model(wg1, p, wl=np.linspace(1520e-9, 1580e-9, 51))

    wg = import_model(p + ".mdl")
    wg2 = wg()
    freq = np.linspace(wl2freq(1540e-9), wl2freq(1560e-9), 50)
    assert np.allclose(wg1.s_parameters(freq), wg2.s_parameters(freq))


def test_load_wrong_ext(tmpdir):
    """
    Tests whether the results of `s_parameters()` for the same frequency on
    an exported and reloaded model are (close) to the same.
    """
    p = tmpdir.mkdir("persist").join("export")
    with pytest.raises(ValueError):
        import_model(p)
