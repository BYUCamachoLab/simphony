# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import pytest

from simphony.libraries import siepic
from simphony.models import Model, Subcircuit


@pytest.fixture
def wg1():
    return siepic.Waveguide(150e-6)


@pytest.fixture
def wg2():
    return siepic.Waveguide(100e-6)


@pytest.fixture
def wg3():
    return siepic.Waveguide(50e-6)


@pytest.fixture
def y1():
    return siepic.YBranch()


class TestConnections:
    def test_connect(self, wg1, wg2):
        wg1.connect(wg2["pin2"])
        wg1.connect(wg2)

        assert wg1["pin1"]._connection == wg2["pin2"]
        assert wg1["pin2"]._connection == wg2["pin1"]

    def test_multiconnect(self, y1, wg1, wg2, wg3):
        y1.multiconnect(wg2, wg3, wg1["pin2"])

        assert y1["pin1"]._connection == wg2["pin1"]
        assert y1["pin2"]._connection == wg3["pin1"]
        assert y1["pin3"]._connection == wg1["pin2"]

    def test_interface(self, wg1, wg2):
        wg1.rename_pins("a", "b")
        wg2.rename_pins("b", "a")

        wg1.interface(wg2)

        assert wg1.pins[0]._connection == wg2.pins[1]
        assert wg1.pins[1]._connection == wg2.pins[0]


class TestCircuitContainment:
    def test_separate_components(self, wg1, wg2):
        assert wg1.circuit != wg2.circuit

    def test_connected_components(self, wg1, wg2):
        wg1.connect(wg2)
        assert wg1.circuit == wg2.circuit

    def test_detached_component(self, wg1, wg2, wg3):
        wg1.connect(wg2)
        wg1.connect(wg3)

        assert wg1.circuit == wg2.circuit
        assert wg2.circuit == wg3.circuit

        wg3.disconnect()

        assert wg1.circuit == wg2.circuit
        assert wg2.circuit != wg3.circuit


class TestModelExtension:
    def test_branch(self, y1, wg1, wg2, wg3):
        class YBranch(Model):
            pin_count = 3

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

                self.y1 = y1

            def s_parameters(self, freqs):
                return self.y1.s_parameters(freqs)

        brancher = YBranch()
        brancher.multiconnect(wg1, wg2, wg3)

        assert brancher.circuit == wg1.circuit
        assert wg1.circuit == wg2.circuit
        assert wg2.circuit == wg3.circuit


class TestSubcircuitExtension:
    def test_subcircuit_factory(self, y1, wg1, wg2, wg3):
        def create_brancher(y1, wg1, wg2):
            y1.multiconnect(wg1, wg2)
            wg1.connect(wg2)

            return y1.circuit.to_subcircuit()

        brancher = create_brancher(y1, wg1, wg2)
        brancher.connect(wg3)

        assert [pin.name for pin in brancher.pins] == ["pin3"]
        assert brancher["pin3"]._connection == wg3["pin1"]

    def test_subcircuit_class(self, y1, wg1, wg2, wg3):
        class Brancher(Subcircuit):
            def __init__(self, **kwargs):
                y1.multiconnect(wg1, wg2)
                wg1.connect(wg2)

                super().__init__(y1.circuit, **kwargs)

        brancher = Brancher()
        brancher.connect(wg3)

        assert [pin.name for pin in brancher.pins] == ["pin3"]
        assert brancher["pin3"]._connection == wg3["pin1"]


class TestModelComponent:
    def testcomponent(self, y1):
        y1 = y1
        y2 = siepic.YBranch(name="y2")

        assert y1.component != y2.component
