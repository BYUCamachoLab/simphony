# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import numpy as np
import pytest

from simphony.library import siepic
from simphony.netlist import Subcircuit
from simphony.simulation import MonteCarloSweepSimulation, SweepSimulation


@pytest.fixture(scope="class")
def result_norm():
    # Declare the models used in the circuit
    gc = siepic.ebeam_gc_te1550()
    y = siepic.ebeam_y_1550()
    wg150 = siepic.ebeam_wg_integral_1550(length=150e-6)
    wg50 = siepic.ebeam_wg_integral_1550(length=50e-6)

    # Create the circuit, add all individual instances
    circuit = Subcircuit("MZI")
    e = circuit.add(
        [
            (gc, "input"),
            (gc, "output"),
            (y, "splitter"),
            (y, "recombiner"),
            (wg150, "wg_long"),
            (wg50, "wg_short"),
        ]
    )

    # You can set pin names individually:
    circuit.elements["input"].pins["n2"] = "input"
    circuit.elements["output"].pins["n2"] = "output"

    # Or you can rename all the pins simultaneously:
    circuit.elements["splitter"].pins = ("in1", "out1", "out2")
    circuit.elements["recombiner"].pins = ("out1", "in2", "in1")

    # Circuits can be connected using the elements' string names:
    circuit.connect_many(
        [
            ("input", "n1", "splitter", "in1"),
            ("splitter", "out1", "wg_long", "n1"),
            ("splitter", "out2", "wg_short", "n1"),
            ("recombiner", "in1", "wg_long", "n2"),
            ("recombiner", "in2", "wg_short", "n2"),
            ("output", "n1", "recombiner", "out1"),
        ]
    )
    sim = SweepSimulation(circuit)
    return sim.simulate()


@pytest.fixture(scope="class")
def result_monte():
    # Declare the models used in the circuit
    gc = siepic.ebeam_gc_te1550()
    y = siepic.ebeam_y_1550()
    wg150 = siepic.ebeam_wg_integral_1550(length=150e-6)
    wg50 = siepic.ebeam_wg_integral_1550(length=50e-6)

    # Create the circuit, add all individual instances
    circuit = Subcircuit("MZI")
    e = circuit.add(
        [
            (gc, "input"),
            (gc, "output"),
            (y, "splitter"),
            (y, "recombiner"),
            (wg150, "wg_long"),
            (wg50, "wg_short"),
        ]
    )

    # You can set pin names individually:
    circuit.elements["input"].pins["n2"] = "input"
    circuit.elements["output"].pins["n2"] = "output"

    # Or you can rename all the pins simultaneously:
    circuit.elements["splitter"].pins = ("in1", "out1", "out2")
    circuit.elements["recombiner"].pins = ("out1", "in2", "in1")

    # Circuits can be connected using the elements' string names:
    circuit.connect_many(
        [
            ("input", "n1", "splitter", "in1"),
            ("splitter", "out1", "wg_long", "n1"),
            ("splitter", "out2", "wg_short", "n1"),
            ("recombiner", "in1", "wg_long", "n2"),
            ("recombiner", "in2", "wg_short", "n2"),
            ("output", "n1", "recombiner", "out1"),
        ]
    )
    sim = MonteCarloSweepSimulation(circuit)
    return sim.simulate()


class TestNormalPinNames:
    def test_singlein(self, result_norm):
        result_norm.data("input", "output")
        result_norm.data("input", "output", 1)

    def test_multiin(self, result_norm):
        result_norm.data(["input", "output"], "output")
        result_norm.data(["input", "output"], "output", [1, 0])

    def test_results(self, result_norm):
        one = result_norm.data(["input", "output"], "output", [1, 0])
        two = result_norm.data("input", "output")
        assert np.allclose(one, two)

        one = result_norm.data(["input", "output"], "input", [0, 1])
        two = result_norm.data("output", "input")
        assert np.allclose(one, two)

    def test_badinput(self, result_norm):
        with pytest.raises(ValueError) as e:
            result_norm.data(["input", "output"], "output", [0])
        with pytest.raises(ValueError) as e:
            result_norm.data(["input", "output"], "output", 0)


class TestNormalPinNumbers:
    def test_singlein(self, result_norm):
        result_norm.data(0, 1)
        result_norm.data(0, 1, 1)

    def test_multiin(self, result_norm):
        result_norm.data([0, 1], 1)
        result_norm.data([0, 1], 1, [1, 0])

    def test_results(self, result_norm):
        one = result_norm.data([0, 1], 1, [1, 0])
        two = result_norm.data(0, 1)
        assert np.allclose(one, two)

        one = result_norm.data([0, 1], 0, [0, 1])
        two = result_norm.data(1, 0)
        assert np.allclose(one, two)


class TestMontePinNames:
    def test_singlein(self, result_monte):
        result_monte.data("input", "output", 2)
        result_monte.data("input", "output", 2, 1)

    def test_multiin(self, result_monte):
        result_monte.data(["input", "output"], "output", 2)
        result_monte.data(["input", "output"], "output", 2, [1, 0])

    def test_results(self, result_monte):
        one = result_monte.data(["input", "output"], "output", 2, [1, 0])
        two = result_monte.data("input", "output", 2)
        assert np.allclose(one, two)

        one = result_monte.data(["input", "output"], "input", 2, [0, 1])
        two = result_monte.data("output", "input", 2)
        assert np.allclose(one, two)

    def test_badinput(self, result_monte):
        with pytest.raises(ValueError) as e:
            result_monte.data(["input", "output"], "output", 2, [0])
        with pytest.raises(ValueError) as e:
            result_monte.data(["input", "output"], "output", 2, 0)


class TestMontePinNumbers:
    def test_singlein(self, result_monte):
        result_monte.data(0, 1, 2)
        result_monte.data(0, 1, 2, 1)

    def test_multiin(self, result_monte):
        result_monte.data([0, 1], 1, 2)
        result_monte.data([0, 1], 1, 2, [1, 0])

    def test_results(self, result_monte):
        one = result_monte.data([0, 1], 1, 2, [1, 0])
        two = result_monte.data(0, 1, 2)
        assert np.allclose(one, two)

        one = result_monte.data([0, 1], 0, 2, [0, 1])
        two = result_monte.data(1, 0, 2)
        assert np.allclose(one, two)
