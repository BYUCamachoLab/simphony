#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright © 2019-2020 Simphony Project Contributors and others (see AUTHORS.txt).
# The resources, libraries, and some source files under other terms (see NOTICE.txt).
#
# This file is part of Simphony.
#
# Simphony is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Simphony is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Simphony. If not, see <https://www.gnu.org/licenses/>.

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
