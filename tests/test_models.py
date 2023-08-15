from copy import deepcopy

import pytest
import numpy as np

from simphony.models import (
    Model,
    Port,
    OPort,
    EPort,
    _NAME_REGISTER,
    clear_name_register,
)
from simphony.exceptions import ModelValidationError
from simphony.libraries.siepic import YBranch
from simphony.libraries.ideal import Waveguide


class TestPort:
    def test_port(self):
        port = Port("test")
        assert port.name == "test"

    def test_port_deepcopy(self):
        port = Port("test")
        port.instance = object()

        port_copy = deepcopy(port)
        assert port_copy.name == "test"
        assert port_copy is not port
        assert port_copy != port
        assert port_copy.instance is not port.instance

    def test_port_equality(self):
        port1 = Port("test")
        port2 = Port("test")

        l = [port1]
        with pytest.raises(ValueError):
            l.index(port2)


class TestModelDeclaration:
    def test_missing_sparams(self):
        """Should fail because s-params function is missing."""
        with pytest.raises(ModelValidationError):

            class BadModel(Model):
                pass

    def test_missing_onames(self):
        """Should fail because ocount or onames is missing."""
        with pytest.raises(ModelValidationError):

            class BadModel(Model):
                def s_params(self, wl):
                    pass

            BadModel()

    def test_missing_ocount(self):
        with pytest.raises(ModelValidationError):

            class BadModel(Model):
                def s_params(self, wl):
                    pass

            BadModel()

    def test_ocount_and_onames_mismatch(self):
        """Should fail because both ocount and onames are defined."""
        with pytest.raises(ModelValidationError):

            class BadModel(Model):
                ocount = 3
                onames = ["o0", "o1"]

                def s_params(self, wl):
                    pass

            BadModel()

    def test_ocount_and_onames_length_match(self):
        """Should fail because both ocount and onames are defined."""
        with pytest.raises(ModelValidationError):

            class GoodModel(Model):
                ocount = 3
                onames = ["o0", "o1", "o2"]

                def s_params(self, wl):
                    pass

            GoodModel()

    def test_good_model_onames(self):
        class GoodModel(Model):
            onames = ["o0", "o1", "o2"]

            def s_params(self, wl):
                pass

        GoodModel()

    def test_good_model_ocount(self):
        class GoodModel(Model):
            ocount = 3

            def s_params(self, wl):
                pass

        GoodModel()


class TestModelContextAccessibility:
    def test_model(self):
        pass


@pytest.fixture
def test_model():
    class TestModel(Model):
        onames = ["o0", "o1"]

        def s_params(self, wl):
            pass

    return TestModel()


@pytest.fixture
def test_model_with_eports():
    class TestModel(Model):
        onames = ["o0", "o1"]
        enames = ["e0", "e1"]

        def s_params(self, wl):
            pass

    return TestModel()


@pytest.fixture
def test_model_three_ports():
    class TestModel(Model):
        onames = ["o0", "o1", "o2"]
        enames = ["e0", "e1", "e2"]

        def s_params(self, wl):
            pass

    return TestModel()


@pytest.fixture
def oport():
    return OPort(name="test_port", instance=None)


@pytest.fixture
def eport():
    return EPort(name="test_port", instance=None)


class TestModelPorts:
    def test_model_str(self, test_model, test_model_with_eports):
        modelstr = test_model.__str__()
        assert "o: [o0, o1]" in modelstr
        assert "TestModel" in modelstr
        assert "e: [None]" in modelstr
        modelstr = test_model_with_eports.__str__()
        assert "o: [o0, o1]" in modelstr
        assert "TestModel" in modelstr
        assert "e: [e0, e1]" in modelstr

    def test_oport_by_name_and_index(self, test_model):
        assert test_model.o("o0") == test_model.o(0)
        assert test_model.o("o1") == test_model.o(1)

    def test_eport_by_name_and_index(self, test_model_with_eports):
        assert test_model_with_eports.e("e0") == test_model_with_eports.e(0)
        assert test_model_with_eports.e("e1") == test_model_with_eports.e(1)

    def test_next_unconnected_oport(self, test_model, oport):
        assert test_model.o(0).connected is False
        assert test_model.o(1).connected is False
        assert test_model.next_unconnected_oport() == test_model.o(0)
        test_model.o(0).connect_to(oport)
        assert test_model.next_unconnected_oport() == test_model.o(1)

    def test_next_unconnected_eport(self, test_model_with_eports, eport):
        assert test_model_with_eports.e(0).connected is False
        assert test_model_with_eports.e(1).connected is False
        assert (
            test_model_with_eports.next_unconnected_eport()
            == test_model_with_eports.e(0)
        )
        test_model_with_eports.e(0).connect_to(eport)
        assert (
            test_model_with_eports.next_unconnected_eport()
            == test_model_with_eports.e(1)
        )

    def test_next_unconnected_oport_all_taken(
        self, test_model, test_model_three_ports, oport
    ):
        test_model.o(0).connect_to(test_model_three_ports.o(0))
        test_model.o(1).connect_to(oport)
        assert test_model.next_unconnected_oport() is None

    def test_next_unconnected_eport_all_taken(
        self, test_model_with_eports, test_model_three_ports, eport
    ):
        test_model_with_eports.e(0).connect_to(test_model_three_ports.e(0))
        test_model_with_eports.e(1).connect_to(eport)
        assert test_model_with_eports.next_unconnected_eport() is None

    def test_duplicate_oport_name(self):
        pass

    def test_duplicate_eport_name(self):
        pass


class TestModelEquality:
    def test_parameter_equality(self):
        comp1 = Waveguide(length=10)
        comp2 = Waveguide(length=10)
        assert comp1 == comp2
        assert comp1 is not comp2

    def test_copy_equality(self):
        comp1 = Waveguide(length=10)
        comp2 = deepcopy(comp1)
        assert comp1 == comp2
        assert comp1 is not comp2


class TestModelHashability:
    def test_hashability(self):
        comp1 = Waveguide(length=10)
        assert hash(comp1)

    def test_copy_hashability(self):
        comp1 = Waveguide(length=10)
        comp2 = deepcopy(comp1)
        assert hash(comp2)


class TestModelCopying:
    def test_shallow_copy(self):
        pass

    def test_deep_copy(self):
        comp1 = Waveguide(length=10)
        comp2 = deepcopy(comp1)

        for port1, port2 in zip(comp1._oports, comp2._oports):
            assert port1 is not port2
            assert port1.instance is not port2.instance
            assert port1.instance is comp1
            assert port2.instance is comp2
        for port1, port2 in zip(comp1._eports, comp2._eports):
            assert port1 is not port2
            assert port1.instance is not port2.instance
            assert port1.instance is comp1
            assert port2.instance is comp2


class TestModelCaching:
    def test_same_model_two_instances_attributes_identical(self, std_wl_um):
        yb1 = YBranch(pol="te")
        yb2 = YBranch(pol="te")

        s1 = yb1._s(tuple(std_wl_um))
        s2 = yb2._s(tuple(std_wl_um))

        np.testing.assert_array_equal(s1, s2)

    def test_same_model_two_instances_attributes_different(self, std_wl_um):
        yb1 = YBranch(pol="te")
        yb2 = YBranch(pol="tm")
        assert yb1 is not yb2
        assert yb1 != yb2

        s1 = yb1._s(tuple(std_wl_um))
        s2 = yb2._s(tuple(std_wl_um))

        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(s1, s2)

    def test_different_wl_array(self):
        yb1 = YBranch()
        yb2 = YBranch()
        assert yb1 is not yb2

        arr1 = np.linspace(1.5, 1.6, 1000)
        arr2 = np.linspace(1.52, 1.58, 1000)

        s1 = yb1._s(tuple(arr1))
        s2 = yb2._s(tuple(arr2))

        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(s1, s2)

    def test_s_params_are_ordered_by_wavelength(self, std_wl_um):
        yb = YBranch()

        s1 = yb._s(tuple(std_wl_um))
        s2 = yb._s(tuple(std_wl_um[::-1]))
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(s1, s2)


@pytest.fixture
def dummy_model():
    class MyModel(Model):
        ocount = 2

        def s_params(self, wl):
            return wl

    return MyModel


class TestModelNaming:
    def test_model_name(self, dummy_model):
        clear_name_register()
        assert dummy_model(name="Waveguide").name == "Waveguide"

    def test_model_name_unique(self, dummy_model):
        clear_name_register()
        wg1 = dummy_model(name="Waveguide")
        with pytest.raises(ValueError):
            wg2 = dummy_model(name="Waveguide")

    def test_auto_naming(self, dummy_model):
        clear_name_register()
        m1 = dummy_model()
        m2 = dummy_model()
        assert m1.name == "MyModel0"
        assert m2.name == "MyModel1"

    def test_auto_naming_with_name(self, dummy_model):
        clear_name_register()
        m1 = dummy_model(name="You can")
        m2 = dummy_model(name="name models")
        assert m1.name == "You can"
        assert m2.name == "name models"

    def test_auto_naming_with_name_and_auto(self, dummy_model):
        clear_name_register()
        m1 = dummy_model(name="You can")
        m2 = dummy_model(name="name models")
        m3 = dummy_model()
        with pytest.raises(ValueError):
            m4 = dummy_model(name="MyModel0")
