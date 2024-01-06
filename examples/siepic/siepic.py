#!/usr/bin/env python3
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)
#
# This file is deprecated and will not run with the latest version of simphony.
# It is retained for legacy purposes only.

import os

import matplotlib.pyplot as plt

from simphony.formatters import CircuitSiEPICFormatter
from simphony.layout import Circuit

# The siepic plugin only needs the path to the main spice file.
filename = os.path.join(os.path.dirname(__file__), "MZI4", "MZI4_main.spi")

# Loading the main file includes any other files linked to it internally.
circuit = Circuit.from_file(filename, formatter=CircuitSiEPICFormatter())

# We can see the connections of the circuit by printing it.
print(circuit)

# We can see what the external pins of the circuit are.
print([pin.name for pin in circuit.pins])

# The SPI file defines an analyzer (SweepSimulator) that we can access to run
# the simulation.
simulator = circuit[-1]
wl, t = simulator.simulate()
plt.plot(wl, t)
plt.title("MZI")
plt.xlabel("Wavelength (m)")
plt.ylabel("Transmission")
plt.tight_layout()
plt.show()
