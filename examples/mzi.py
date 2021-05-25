# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import matplotlib.pyplot as plt

from simphony.libraries import siepic
from simphony.simulators import MonteCarloSweepSimulator, SweepSimulator

# first we initialize all of the components in the MZI circuit
gc_input = siepic.GratingCoupler()
y_splitter = siepic.YBranch()
wg_long = siepic.Waveguide(length=150e-6)
wg_short = siepic.Waveguide(length=50e-6)
y_recombiner = siepic.YBranch()
gc_output = siepic.GratingCoupler()

# next we connect the components to each other
# you can connect pins directly:
y_splitter["pin1"].connect(gc_input["pin1"])

# or connect components with components:
# (when using components to make connections, their first unconnected pin will
# be used to make the connection.)
y_splitter.connect(wg_long)

# or any combination of the two:
y_splitter["pin3"].connect(wg_short)
# y_splitter.connect(wg_short["pin1"])

# when making multiple connections, it is often simpler to use `multiconnect`
# multiconnect accepts components, pins, and None
# if None is passed in, the corresponding pin is skipped
y_recombiner.multiconnect(gc_output, wg_short, wg_long)

# instantiate the sweep simulator and connect it to our circuit
# the first simulator pin connects to the input, the second to the output
simulator = SweepSimulator(1500e-9, 1600e-9)
simulator.multiconnect(gc_input, gc_output)

# simulate and plot the results
f, p = simulator.simulate()
plt.plot(f, p)
plt.title("MZI")
plt.tight_layout()
plt.show()

# run a Monte Carlo simulation by disconnecting the previous simulator and
# instantiating a MonteCarloSweepSimulator
simulator.disconnect()
simulator = MonteCarloSweepSimulator(1500e-9, 1600e-9)
simulator.multiconnect(gc_input, gc_output)

# plot the values for 10 runs
results = simulator.simulate(runs=10)
for f, p in results:
    plt.plot(f, p)

# redraw the first results since they contain the ideal values
f, p = results[0]
plt.plot(f, p, "k")
plt.title("MZI Monte Carlo")
plt.tight_layout()
plt.show()
