# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
layout_aware.py
---------------

Author: Skandan Chandrasekar
Modified: July 15, 2022

A script that walks the user through the process of running layout-aware Monte Carlo simulations.
"""
import matplotlib.pyplot as plt

from simphony.die import Die
from simphony.libraries import siepic
from simphony.simulation import Detector, Laser, Simulation

# first we initialize all of the components in the MZI circuit
gc_input = siepic.GratingCoupler(name="gcinput")
y_splitter = siepic.YBranch(name="ysplit")
wg_long = siepic.Waveguide(length=150e-6, name="wglong")
wg_short = siepic.Waveguide(length=50e-6, name="wgshort")
y_recombiner = siepic.YBranch(name="yrecombiner")
gc_output = siepic.GratingCoupler(name="gcoutput")

# next, we instantiate a Die object
die = Die(name="die1")

# then, we throw in the components into the Die
die.add_components([gc_input, y_splitter, wg_long, gc_output, y_recombiner, wg_short])

# we then distribute the devices in the die in a grid
# we can specify the number of rows and columns using
# the `shape` argument, and the spacing between devices
die.distribute_devices(direction="grid", shape=(3, 2), spacing=(5, 10))

# we can visualize the grid arrangement
die.visualize()

# We connect the components like we would usually.
# Simphony will take care of the routing and
# device connections for us.

y_splitter["pin1"].connect(gc_input["pin1"])

y_recombiner["pin1"].connect(gc_output["pin1"])

y_splitter["pin2"].connect(wg_long)
y_recombiner["pin3"].connect(wg_long)

y_splitter["pin3"].connect(wg_short)
y_recombiner["pin2"].connect(wg_short)

# visualize after connecting
die.visualize(show_ports=False)

# Run the layout aware monte carlo computation
with Simulation() as sim:
    l = Laser(power=1)
    l.freqsweep(187370000000000.0, 199862000000000.0)
    l.connect(gc_input["pin2"])
    d = Detector()
    d.connect(gc_output["pin2"])

    results = sim.layout_aware_simulation()

# Plot the results
f = l.freqs
for run in results:
    p = []
    for sample in run:
        for data_list in sample:
            for data in data_list:
                p.append(data)
    plt.plot(f, p)

run = results[0]
p = []
for sample in run:
    for data_list in sample:
        for data in data_list:
            p.append(data)
plt.plot(f, p, "k")
plt.title("MZI Layout Aware Monte Carlo")
plt.show()
