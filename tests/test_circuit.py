import typing
from copy import deepcopy

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
from simphony.models import Model, OPort, EPort, clear_name_register
from simphony.libraries.ideal import Coupler, Waveguide


@pytest.fixture
def wg0():
    return Waveguide(length=1.0)


@pytest.fixture
def wg1():
    return Waveguide(length=100.0)


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
    return Coupler(coupling=0.45)


@pytest.fixture
def coupler2():
    return Coupler(coupling=0.58)


@pytest.fixture
def ckt():
    return Circuit()


@pytest.fixture
def detached_s_params(data_dir):
    import numpy as np

    return np.load(data_dir / "detached_s_params.npy")


@pytest.fixture
def exposed_s_params(data_dir):
    import numpy as np

    return np.load(data_dir / "exposed_s_params.npy")


class TestCircuit:
    def test_circuit(self):
        ckt = Circuit()
        assert ckt is not None

    def test_connect_o2o(self, ckt, wg0, wg1):
        # Test model.o() to model.o()
        ckt.connect(wg0.o(0), wg1.o(0))
        assert {wg0, wg1} <= set(ckt.components)
        assert len(ckt.components) == 2
        assert {wg0.o(1), wg1.o(1)} <= set(ckt._oports)
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
        assert {model_with_eport, model_with_two_eports} <= set(ckt.components)
        assert len(ckt.components) == 2
        assert model_with_eport.e(0)._connections <= set(model_with_two_eports.e(0))
        assert model_with_two_eports.e(0)._connections <= set(model_with_eport.e(0))
        assert len(ckt._enodes) == 1

    def test_connect_o2m(self, ckt, wg0, wg1):
        ckt.connect(wg0.o(0), wg1)
        assert {wg0, wg1} <= set(ckt.components)
        assert len(ckt.components) == 2
        assert wg0.o(0)._connections <= set(wg1.o(0))
        assert wg1.o(0)._connections <= set(wg0.o(0))
        assert (wg0.o(0), wg1.o(0)) in ckt._onodes
        assert len(ckt._onodes) == 1

    def test_connect_e2m(self, ckt, model_with_eport, model_with_two_eports):
        ckt.connect(model_with_eport.e(0), model_with_two_eports)
        assert {model_with_eport, model_with_two_eports} <= set(ckt.components)
        assert len(ckt.components) == 2
        assert model_with_eport.e(0)._connections <= set(model_with_two_eports.e(0))
        assert model_with_two_eports.e(0)._connections <= set(model_with_eport.e(0))
        assert len(ckt._enodes) == 1

    def test_connect_m2o(self, ckt, wg0, wg1):
        ckt.connect(wg0, wg1.o(0))
        assert {wg0, wg1} <= set(ckt.components)
        assert len(ckt.components) == 2
        assert wg0.o(0)._connections <= set(wg1.o(0))
        assert wg1.o(0)._connections <= set(wg0.o(0))
        assert (wg0.o(0), wg1.o(0)) in ckt._onodes

    def test_connect_m2e(self, ckt, model_with_eport, model_with_two_eports):
        # set all oports to connected
        ckt.connect(
            model_with_eport, [model_with_two_eports.o(0), model_with_two_eports.o(1)]
        )
        ckt.connect(model_with_eport, model_with_two_eports.e(0))  # test m2e
        assert {model_with_eport, model_with_two_eports} <= set(ckt.components)
        assert len(ckt.components) == 2
        assert model_with_eport.e(0)._connections <= set(model_with_two_eports.e(0))
        assert model_with_two_eports.e(0)._connections <= set(model_with_eport.e(0))
        assert model_with_two_eports.e(1)._connections <= set(model_with_eport.e(0))
        assert len(ckt._enodes) == 1

    def test_connect_m2m_oports_and_eports(
        self, ckt, model_with_two_eports, model_with_three_eports
    ):
        ckt.connect(model_with_two_eports, model_with_three_eports)
        assert {model_with_three_eports, model_with_two_eports} <= set(ckt.components)
        assert len(ckt.components) == 2
        assert model_with_two_eports.o(0)._connections <= set(
            model_with_three_eports.o(0)
        )
        assert model_with_three_eports.o(0)._connections <= set(
            model_with_two_eports.o(0)
        )
        assert model_with_two_eports.o(1)._connections <= set(
            model_with_three_eports.o(1)
        )
        assert model_with_three_eports.o(1)._connections <= set(
            model_with_two_eports.o(1)
        )
        assert model_with_two_eports.e(0)._connections <= set(
            model_with_three_eports.e(0)
        )
        assert model_with_three_eports.e(0)._connections <= set(
            model_with_two_eports.e(0)
        )
        assert model_with_two_eports.e(1)._connections <= set(
            model_with_three_eports.e(1)
        )
        assert model_with_three_eports.e(1)._connections <= set(
            model_with_two_eports.e(1)
        )
        assert len(ckt._onodes) == 2
        assert len(ckt._enodes) == 2

    def test_connect_m2m(self, ckt, wg0, wg1, coupler):
        ckt.connect(wg0.o(0), coupler)
        ckt.connect(coupler, wg1)
        assert {wg0, wg1, coupler} <= set(ckt.components)
        assert len(ckt.components) == 3
        assert (wg0.o(0), coupler.o(0)) in ckt._onodes
        assert (coupler.o(1), wg1.o(0)) in ckt._onodes
        assert (coupler.o(2), wg1.o(1)) in ckt._onodes
        assert len(ckt._onodes) == 3

    def test_subnetwork_growth(self, ckt, coupler, coupler2, wg0, wg1):
        connect_arr = [
            ckt.connect(a, b)
            for (a, b) in zip(
                [coupler.o(2), coupler.o(3), coupler2.o(0), coupler2.o(1)],
                [wg0, wg1, wg0, wg1],
            )
        ]
        ckt.expose([coupler.o(0), coupler.o(1), coupler2.o(2), coupler2.o(3)])
        wls = jnp.array([1.55])
        field_in = jnp.array([1.0, 0, 0, 0]).T

        c1_sparams = coupler.s_params(wls)
        c2_sparams = coupler2.s_params(wls)
        wg0_sparams = wg0.s_params(wls)
        wg1_sparams = wg1.s_params(wls)

        # Ignores reflections since these are ideal models
        # Fout_1 = (t_1 * wg0 * t_2 * + k_1 * wg1 * k_2) * Fin
        field_out_man_1 = (
            c1_sparams[:, 2, 0] * wg0_sparams[:, 1, 0] * c2_sparams[:, 2, 0]
            + c1_sparams[:, 3, 0] * wg1_sparams[:, 1, 0] * c2_sparams[:, 2, 1]
        ) * field_in[0]
        # Fout_2 = (t_1 * wg0 * k_2 * + k_1 * wg1 * t_2) * Fin
        field_out_man_2 = (
            c1_sparams[:, 2, 0] * wg0_sparams[:, 1, 0] * c2_sparams[:, 3, 0]
            + c1_sparams[:, 3, 0] * wg1_sparams[:, 1, 0] * c2_sparams[:, 3, 1]
        ) * field_in[0]

        ckt.s_params(wls)
        field_out = [ckt.s_params(wls)[wl_ind] @ field_in for wl_ind in range(len(wls))]
        field_out = jnp.stack(field_out, axis=0)

        assert jnp.allclose(field_out[:, 2], field_out_man_1)
        assert jnp.allclose(field_out[:, 3], field_out_man_2)

    def test_add_to_circuit(self, ckt, wg0, wg1):
        ckt.add(wg0)
        assert {wg0} <= set(ckt.components)
        assert len(ckt.components) == 1
        assert {wg0.o(0)} <= set(ckt._oports)
        ckt.add(wg1)
        assert {wg0, wg1} <= set(ckt.components)
        assert len(ckt.components) == 2
        assert {wg0.o(0), wg1.o(0)} <= set(ckt._oports)
        assert ckt._onodes == []
        assert ckt._enodes == []

    def test_auto_connect(self, ckt, coupler, coupler2):
        coupler.rename_oports(["in1", "in2", "con1", "con2"])
        coupler2.rename_oports(["con1", "con2", "out1", "out2"])
        ckt.autoconnect(coupler, coupler2)
        assert len(ckt.components) == 2
        assert (coupler.o("con1"), coupler2.o("con1")) in ckt._onodes
        assert (coupler.o("con2"), coupler2.o("con2")) in ckt._onodes
        assert len(ckt._onodes) == 2

    def test_detached_s_params(self, ckt, coupler, wg0, wg1, detached_s_params):
        import numpy as np

        ckt.add(coupler)
        ckt.add(wg0)
        ckt.add(wg1)
        s = ckt.s_params([1.55])
        np.testing.assert_allclose(s, detached_s_params)  # , atol=1e-3)

    def test_expose(self, ckt, coupler, wg0, wg1, exposed_s_params):
        ckt.connect(coupler.o(2), wg0)
        ckt.connect(coupler.o(3), wg1)
        ckt.expose([wg0.o(1), coupler.o(0), wg1.o(1), coupler.o(1)])
        assert len(ckt.exposed_ports) == 4
        s = ckt.s_params([1.55])
        np.testing.assert_allclose(s, exposed_s_params)  # , atol=1e-3)

    def test_circuit_copy(self, ckt):
        # copied from test_circuit_deepcopy(), same behavior
        wg1 = Waveguide(length=1.0)
        wg2 = Waveguide(length=2.0)

        ckt1 = Circuit()
        ckt1.connect(wg1, wg2)

        ckt2 = deepcopy(ckt1)
        assert ckt1 is not ckt2
        for comp1, comp2 in zip(ckt1.components, ckt2.components):
            assert comp1 is not comp2
            assert comp1 == comp2
            for port1, port2 in zip(comp1._oports, comp2._oports):
                assert port1 is not port2
                assert port1 != port2
                assert port1.instance is not port2.instance
                assert port1.instance is comp1
                assert port2.instance is comp2
            for port1, port2 in zip(comp1._eports, comp2._eports):
                assert port1 is not port2
                assert port1 != port2
                assert port1.instance is not port2.instance
                assert port1.instance is comp1
                assert port2.instance is comp2
        for onode1, onode2 in zip(ckt1._onodes, ckt2._onodes):
            assert onode1 is not onode2
            for port1, port2 in zip(onode1, onode2):
                assert port1 is not port2
                assert port1 != port2
                assert port1.instance is not port2.instance
        for enode1, enode2 in zip(ckt1._enodes, ckt2._enodes):
            assert enode1 is not enode2
            for port1, port2 in zip(enode1, enode2):
                assert port1 is not port2
                assert port1 != port2
                assert port1.instance is not port2.instance

    def test_circuit_deepcopy(self):
        wg1 = Waveguide(length=1.0)
        wg2 = Waveguide(length=2.0)

        ckt1 = Circuit()
        ckt1.connect(wg1, wg2)

        ckt2 = deepcopy(ckt1)
        assert ckt1 is not ckt2
        for comp1, comp2 in zip(ckt1.components, ckt2.components):
            assert comp1 is not comp2
            assert comp1 == comp2
            for port1, port2 in zip(comp1._oports, comp2._oports):
                assert port1 is not port2
                assert port1 != port2
                assert port1.instance is not port2.instance
                assert port1.instance is comp1
                assert port2.instance is comp2
            for port1, port2 in zip(comp1._eports, comp2._eports):
                assert port1 is not port2
                assert port1 != port2
                assert port1.instance is not port2.instance
                assert port1.instance is comp1
                assert port2.instance is comp2
        for onode1, onode2 in zip(ckt1._onodes, ckt2._onodes):
            assert onode1 is not onode2
            for port1, port2 in zip(onode1, onode2):
                assert port1 is not port2
                assert port1 != port2
                assert port1.instance is not port2.instance
        for enode1, enode2 in zip(ckt1._enodes, ckt2._enodes):
            assert enode1 is not enode2
            for port1, port2 in zip(enode1, enode2):
                assert port1 is not port2
                assert port1 != port2
                assert port1.instance is not port2.instance

    def test_circuit_equality(self):
        c1 = Circuit()
        c2 = c1
        assert c1 == c2

    def test_circuit_inequality(self):
        c1 = Circuit()
        c2 = Circuit()
        assert c1 != c2

    def test_circuit_hash(self):
        c1 = Circuit()
        assert isinstance(c1, typing.Hashable)


@pytest.fixture
def mzi_s_params(data_dir):
    import numpy as np

    return np.load(data_dir / "mzi_s_params.npy")


class TestMZIExample:
    def test_mzi_s_params(self, ckt, mzi_s_params):
        from simphony.libraries import siepic
        import numpy as np

        gc_input = siepic.GratingCoupler()
        gc_input.rename_oports(["o0", "input"])
        y_splitter = siepic.YBranch()
        wg_long = siepic.Waveguide(length=150)
        wg_short = siepic.Waveguide(length=50)
        y_recombiner = siepic.YBranch()
        gc_output = siepic.GratingCoupler()
        gc_output.rename_oports(["o0", "output"])

        ckt.connect(gc_input.o(0), y_splitter.o(0))
        ckt.connect(y_splitter, [wg_short, wg_long])
        ckt.connect(gc_output.o(), y_recombiner)
        ckt.connect(y_recombiner, [wg_short, wg_long])

        wl = np.linspace(1.5, 1.6, 1000)

        ckt1 = deepcopy(ckt)
        ckt1.expose([ckt1.o("input"), ckt1.o("output")])
        res_s = ckt1.s_params(wl)
        np.testing.assert_allclose(
            res_s[:, [1, 0], :][:, :, [1, 0]], mzi_s_params
        )  # , atol=1e-3)

        ckt2 = deepcopy(ckt)
        ckt2.expose([ckt2.o("output"), ckt2.o("input")])
        res_s = ckt2.s_params(wl)
        np.testing.assert_allclose(res_s, mzi_s_params)  # , atol=1e-3)


class TestCircuitNaming:
    def test_circuit_naming(self):
        clear_name_register()
        ckt = Circuit(name="MyCircuit")
        assert ckt.name == "MyCircuit"

    def test_circuit_autonaming(self):
        clear_name_register()
        ckt = Circuit()
        assert ckt.name == "Circuit0"

    def test_circuit_duplicate_naming(self):
        clear_name_register()
        ckt = Circuit(name="MyCircuit")
        with pytest.raises(ValueError):
            Circuit(name="MyCircuit")
