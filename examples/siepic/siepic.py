#!/usr/bin/env python3
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import os

import matplotlib.pyplot as plt
import numpy as np

from simphony.netlist import Subcircuit
from simphony.simulation import SweepSimulation, MonteCarloSweepSimulation
from simphony.plugins.siepic import load

# The siepic plugin only needs the path to the main spice file.
filename = os.path.join(os.path.dirname(__file__), "MZI4", "MZI4_main.spi")

# Loading the main file includes any other files linked to it internally.
built = load(filename)

# The file ``MZI4_main.spi`` declares one single-sweep simulation.
analyses = built["analyses"]
sim = analyses[0]

# We can see what the external pins of the circuit are.
sim.circuit.pins
# ['ebeam_gc_te1550_laser1', 'ebeam_gc_te1550_detector2']

# The simulation object is returned pre-simulated. We run the simulatio and
# take the data from the ports, using the pin names we looked up earlier.
res = sim.simulate()
f, s = res.data("ebeam_gc_te1550_laser1", "ebeam_gc_te1550_detector2")
plt.plot(f, s)
plt.title("MZI")
plt.tight_layout()
plt.show()
