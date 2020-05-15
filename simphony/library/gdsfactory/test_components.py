# this code has been automatically generated from write_tests.py
import numpy as np
import simphony.library.gdsfactory as cl


def test_mmi1x2(data_regression):
    c = cl.mmi1x2()

    wav = np.linspace(1520, 1570, 3) * 1e-9
    f = 3e8 / wav
    s = c.s_parameters(freq=f)
    _, rows, cols = np.shape(s)
    sdict = {
        f"S{i+1}{j+1}": np.abs(s[:, i, j]).tolist()
        for i in range(rows)
        for j in range(cols)
    }
    data_regression.check(sdict)


def test_mmi1x2(data_regression):
    c = cl.mmi1x2()

    wav = np.linspace(1520, 1570, 3) * 1e-9
    f = 3e8 / wav
    s = c.s_parameters(freq=f)
    _, rows, cols = np.shape(s)
    sdict = {
        f"S{i+1}{j+1}": np.abs(s[:, i, j]).tolist()
        for i in range(rows)
        for j in range(cols)
    }
    data_regression.check(sdict)


def test_coupler_ring(data_regression):
    c = cl.coupler_ring()

    wav = np.linspace(1520, 1570, 3) * 1e-9
    f = 3e8 / wav
    s = c.s_parameters(freq=f)
    _, rows, cols = np.shape(s)
    sdict = {
        f"S{i+1}{j+1}": np.abs(s[:, i, j]).tolist()
        for i in range(rows)
        for j in range(cols)
    }
    data_regression.check(sdict)
