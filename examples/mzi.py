#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np

from simphony.library import ebeam
from simphony.netlist import Subcircuit
from simphony.simulation import SweepSimulation, MonteCarloSweepSimulation

# Declare the models used in the circuit
gc = ebeam.ebeam_gc_te1550()
y = ebeam.ebeam_y_1550()
wg150 = ebeam.ebeam_wg_integral_1550(length=150e-6)
wg50 = ebeam.ebeam_wg_integral_1550(length=50e-6)

# Create the circuit, add all individual instances
circuit = Subcircuit('MZI')
e = circuit.add([
    (gc, 'input'),
    (gc, 'output'),
    (y, 'splitter'),
    (y, 'recombiner'),
    (wg150, 'wg_long'),
    (wg50, 'wg_short'),
])

# You can set pin names individually:
circuit.elements['input'].pins['n2'] = 'input'
circuit.elements['output'].pins['n2'] = 'output'

# Or you can rename all the pins simultaneously:
circuit.elements['splitter'].pins = ('in1', 'out1', 'out2')
circuit.elements['recombiner'].pins = ('out1', 'in2', 'in1')

# Circuits can be connected using the elements' string names:
circuit.connect_many([
    ('input', 'n1', 'splitter', 'in1'),
    ('splitter', 'out1', 'wg_long', 'n1'),
    ('splitter', 'out2', 'wg_short', 'n1'),
    ('recombiner', 'in1', 'wg_long', 'n2'),
    ('recombiner', 'in2', 'wg_short', 'n2'),
    ('output', 'n1', 'recombiner', 'out1'),
])

# or by using the actual object reference.
# circuit.connect(e[0], e[0].pin[0], e[2], e[2].pin[0])

# At this point, your circuit is defined. This file can serve as a description
# for a subcircuit that is used in a larger circuit, and can simply be imported
# using the Python import system (``from mzi import circuit``).

# Run a simulation on the netlist.
simulation = SweepSimulation(circuit, 1500e-9, 1600e-9)
result = simulation.simulate()

f, s = result.data('input', 'output')
plt.plot(f, s)
plt.title("MZI")
plt.tight_layout()
plt.show()


# We can run a monte carlo simulation on the circuit, too
simulation = MonteCarloSweepSimulation(circuit, 1500e-9, 1600e-9)
runs = 10
result = simulation.simulate(runs=runs)
for i in range(1, runs + 1):
    f, s = result.data('input', 'output', i)
    plt.plot(f, s)

# The data located at the 0 position is the ideal values.
f, s = result.data('input', 'output', 0)
plt.plot(f, s, 'k')
plt.title("MZI Monte Carlo")
plt.tight_layout()
plt.show()
