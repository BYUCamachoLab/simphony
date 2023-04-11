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
from simphony.models import Model, OPort, EPort
from simphony.libraries.ideal import Coupler, Waveguide

@pytest.fixture
def wg0():
    return Waveguide(length=1.0)

@pytest.fixture
def wg1():
    return Waveguide(length=2.0)

@pytest.fixture
def model_with_eport():
    class ModelWithEPorts(Model):
        onames = ["o0", "o1"]
        enames = ["e0"]

        def s_params(self, wl):
            return jnp.eye((2, 2))
    return ModelWithEPorts()

@pytest.fixture
def model_with_two_eports():
    class ModelWithEPorts(Model):
        onames = ["o0", "o1"]
        enames = ["e0", "e1"]

        def s_params(self, wl):
            return jnp.eye((2, 2))
    return ModelWithEPorts()

@pytest.fixture
def model_with_three_eports():
    class ModelWithEPorts(Model):
        onames = ["o0", "o1"]
        enames = ["e0", "e1", "e2"]

        def s_params(self, wl):
            return jnp.eye((2, 2))
    return ModelWithEPorts()

@pytest.fixture
def coupler():
    return Coupler()

@pytest.fixture
def ckt():
    return Circuit()

class TestCircuit:
    def test_circuit(self):
        ckt = Circuit()
        assert ckt is not None

    def test_connect_o2o(self, ckt, wg0, wg1):
        # Test model.o() to model.o()
        ckt.connect(wg0.o(0), wg1.o(0))
        assert set([wg0, wg1]) <= set(ckt.components)
        assert len(ckt.components) == 2
        assert set(wg0._oports + wg1._oports) <= set(ckt._internal_oports)
        assert len(ckt._internal_oports) == 4
        assert (wg0.o(0), wg1.o(0)) in ckt._onodes
        assert len(ckt._onodes) == 1
        
    def test_connect_o2e(self, ckt, wg0, model_with_eport):
        with pytest.raises(ValueError):
            ckt.connect(wg0.o(0), model_with_eport.e(0))
 
    def test_connect_e2o(self, ckt, wg0, model_with_eport):
        with pytest.raises(ValueError):
            ckt.connect(model_with_eport.e(0), wg0.o(0))

    def test_connect_e2e(self, ckt, model_with_eport, model_with_two_eports):
        ckt.connect(model_with_eport.e(0), model_with_two_eports.e(0))
        assert set([model_with_eport, model_with_two_eports]) <= set(ckt.components)
        assert len(ckt.components) == 2
        assert set(model_with_eport._eports + model_with_two_eports._eports) <= set(ckt._internal_eports)
        assert len(ckt._internal_eports) == 3
        assert len(ckt._enodes) == 1

    def test_connect_o2m(self, ckt, wg0, wg1):
        ckt.connect(wg0.o(0), wg1)
        assert set([wg0, wg1]) <= set(ckt.components)
        assert len(ckt.components) == 2
        assert set(wg0._oports + wg1._oports) <= set(ckt._internal_oports)
        assert len(ckt._internal_oports) == 4
        assert (wg0.o(0), wg1.o(0)) in ckt._onodes
        assert len(ckt._onodes) == 1

    def test_connect_e2m(self, ckt, model_with_eport, model_with_two_eports):
        ckt.connect(model_with_eport.e(0), model_with_two_eports)
        assert set([model_with_eport, model_with_two_eports]) <= set(ckt.components)
        assert len(ckt.components) == 2
        assert set(model_with_eport._eports + model_with_two_eports._eports) <= set(ckt._internal_eports)
        assert len(ckt._internal_eports) == 3
        assert len(ckt._enodes) == 1

    def test_connect_m2o(self, ckt, wg0, wg1):
        ckt.connect(wg0, wg1.o(0))
        assert set([wg0, wg1]) <= set(ckt.components)
        assert len(ckt.components) == 2
        assert set(wg0._oports + wg1._oports) <= set(ckt._internal_oports)
        assert len(ckt._internal_oports) == 4
        assert (wg0.o(0), wg1.o(0)) in ckt._onodes

    def test_connect_m2e(self, ckt, model_with_eport, model_with_two_eports):
        # set all oports to connected
        ckt.connect(model_with_eport, [model_with_two_eports.o(0), model_with_two_eports.o(1)])
        ckt.connect(model_with_eport, model_with_two_eports.e(0)) # test m2e
        assert set([model_with_eport, model_with_two_eports]) <= set(ckt.components)
        assert len(ckt.components) == 2
        assert set(model_with_eport._eports + model_with_two_eports._eports) <= set(ckt._internal_eports)
        assert len(ckt._internal_eports) == 3
        assert len(ckt._enodes) == 1

    def test_connect_m2m_oports_and_eports(self, ckt, model_with_two_eports, model_with_three_eports):
        ckt.connect(model_with_two_eports, model_with_three_eports)
        assert set([model_with_three_eports, model_with_two_eports]) <= set(ckt.components)
        assert len(ckt.components) == 2
        assert set(model_with_three_eports._oports + model_with_two_eports._oports) <= set(ckt._internal_oports)
        assert len(ckt._internal_oports) == 4
        assert set(model_with_three_eports._eports + model_with_two_eports._eports) <= set(ckt._internal_eports)
        assert len(ckt._internal_eports) == 5
        assert len(ckt._enodes) == 2

    def test_connect_m2m(self, ckt, wg0, wg1, coupler):
        ckt.connect(wg0.o(0), coupler)
        ckt.connect(coupler, wg1)
        assert set([wg0, wg1, coupler]) <= set(ckt.components)
        assert len(ckt.components) == 3
        assert set(wg0._oports + wg1._oports + coupler._oports) <= set(ckt._internal_oports)
        assert len(ckt._internal_oports) == 8
        assert (wg0.o(0), coupler.o(0)) in ckt._onodes
        assert (coupler.o(1), wg1.o(0)) in ckt._onodes
        assert (coupler.o(2), wg1.o(1)) in ckt._onodes # Failed here
        assert len(ckt._onodes) == 3
