import pytest

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp
    from simphony.utils import jax

    JAX_AVAILABLE = False

from simphony.circuit import Circuit
from simphony.libraries.ideal import Coupler, Waveguide

class TestCircuit:
    def test_circuit(self):
        ckt = Circuit()
        assert ckt is not None

    def test_connect_o2o(self):
        # Test model.o() to model.o()
        ckt = Circuit()
        wg0 = Waveguide(length=1.0)
        wg1 = Waveguide(length=2.0)
        ckt.connect(wg0.o(0), wg1.o(0))
        assert set([wg0, wg1]) <= set(ckt.components)
        assert len(ckt.components) == 2
        assert set(wg0._oports + wg1._oports) <= set(ckt._internal_oports)
        assert len(ckt._internal_oports) == 4
        assert (wg0.o(0), wg1.o(0)) in ckt._onodes
        assert len(ckt._onodes) == 1
        
    def test_connect_o2e(self):
        pass
 
    def test_connect_e2o(self):
        pass

    def test_connect_e2e(self):
        pass

    def test_connect_o2m(self):
        ckt = Circuit()
        wg0 = Waveguide(length=1.0)
        wg1 = Waveguide(length=2.0)
        ckt.connect(wg0.o(0), wg1)
        assert set([wg0, wg1]) <= set(ckt.components)
        assert len(ckt.components) == 2
        assert set(wg0._oports + wg1._oports) <= set(ckt._internal_oports)
        assert len(ckt._internal_oports) == 4
        assert (wg0.o(0), wg1.o(0)) in ckt._onodes
        assert len(ckt._onodes) == 1

    def test_connect_e2m(self):
        pass

    def test_connect_m2o(self):
        ckt = Circuit()
        wg0 = Waveguide(length=1.0)
        wg1 = Waveguide(length=2.0)
        ckt.connect(wg0, wg1.o(0))
        assert set([wg0, wg1]) <= set(ckt.components)
        assert len(ckt.components) == 2
        assert set(wg0._oports + wg1._oports) <= set(ckt._internal_oports)
        assert len(ckt._internal_oports) == 4
        assert (wg0.o(0), wg1.o(0)) in ckt._onodes

    def test_connect_m2e(self):
        pass

    def test_connect_m2m(self):
        pass
