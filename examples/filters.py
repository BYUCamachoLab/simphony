#!/usr/bin/env python3
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)
#
# File: filters.py

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from simphony.libraries import siepic, sipann
from simphony.simulators import SweepSimulator


def ring_factory(radius):
    r"""Creates a full ring (with terminator) from a half ring.

    Ports of a half ring are ordered like so:
    2           4
     |         |
      \       /
       \     /
     ---=====---
    1           3

    Resulting pins are ('pass', 'in', 'out').

    Parameters
    ----------
    radius : float
        The radius of the ring resonator, in nanometers.
    """
    # Have rings for selecting out frequencies from the data line.
    # See SiPANN's model API for argument order and units.
    halfring1 = sipann.HalfRing(500e-9, 220e-9, radius, 100e-9)
    halfring2 = sipann.HalfRing(500e-9, 220e-9, radius, 100e-9)
    terminator = siepic.Terminator()

    halfring1.rename_pins("pass", "midb", "in", "midt")
    halfring2.rename_pins("out", "midt", "term", "midb")

    # the interface method will connect all of the pins with matching names
    # between the two components together
    halfring1.interface(halfring2)
    halfring2["term"].connect(terminator)

    # bundling the circuit as a Subcircuit allows us to interact with it
    # as if it were a component
    return halfring1.circuit.to_subcircuit()


# Behold, we can run a simulation on a single ring resonator.
ring1 = ring_factory(10e-6)

simulator = SweepSimulator(1500e-9, 1600e-9)
simulator.multiconnect(ring1["in"], ring1["pass"])

f, t = simulator.simulate(mode="freq")
plt.plot(f, t)
plt.title("10-micron Ring Resonator")
plt.tight_layout()
plt.show()

simulator.disconnect()

# Now, we'll create the circuit (using several ring resonator subcircuits)
# instantiate the basic components
wg_input = siepic.Waveguide(100e-6)
wg_out1 = siepic.Waveguide(100e-6)
wg_connect1 = siepic.Waveguide(100e-6)
wg_out2 = siepic.Waveguide(100e-6)
wg_connect2 = siepic.Waveguide(100e-6)
wg_out3 = siepic.Waveguide(100e-6)
terminator = siepic.Terminator()

# instantiate the rings with varying radii
ring1 = ring_factory(10e-6)
ring2 = ring_factory(11e-6)
ring3 = ring_factory(12e-6)

# connect the circuit together
ring1.multiconnect(wg_connect1, wg_input["pin2"], wg_out1)
ring2.multiconnect(wg_connect2, wg_connect1, wg_out2)
ring3.multiconnect(terminator, wg_connect2, wg_out3)

# prepare the plots
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 3)
ax = fig.add_subplot(gs[0, :2])

# prepare the simulator
simulator = SweepSimulator(1524.5e-9, 1551.15e-9)
simulator.connect(wg_input)

# get the results for output 1
simulator.multiconnect(None, wg_out1)
wl, t = simulator.simulate()
ax.plot(wl * 1e9, t, label="Output 1", lw="0.7")

# get the results for output 2
simulator.multiconnect(None, wg_out2)
wl, t = simulator.simulate()
ax.plot(wl * 1e9, t, label="Output 2", lw="0.7")

# get the results for output 3
simulator.multiconnect(None, wg_out3)
wl, t = simulator.simulate()
ax.plot(wl * 1e9, t, label="Output 3", lw="0.7")

# finish first subplot and move to next
ax.set_ylabel("Fractional Optical Power")
ax.set_xlabel("Wavelength (nm)")
plt.legend(loc="upper right")
ax = fig.add_subplot(gs[0, 2])

# get the results for output 1
simulator.multiconnect(None, wg_out1)
wl, t = simulator.simulate()
ax.plot(wl * 1e9, t, label="Output 1", lw="0.7")

# get the results for output 2
simulator.multiconnect(None, wg_out2)
wl, t = simulator.simulate()
ax.plot(wl * 1e9, t, label="Output 2", lw="0.7")

# get the results for output 3
simulator.multiconnect(None, wg_out3)
wl, t = simulator.simulate()
ax.plot(wl * 1e9, t, label="Output 3", lw="0.7")

# plot the results
ax.set_xlim(1543, 1545)
ax.set_ylabel("Fractional Optical Power")
ax.set_xlabel("Wavelength (nm)")
fig.align_labels()
plt.show()
