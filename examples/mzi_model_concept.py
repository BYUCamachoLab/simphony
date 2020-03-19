import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import simphony
import simphony.library.ebeam as ebeam
from simphony.netlist import Subcircuit
from simphony.simulation import SweepSimulation

# models
gc = ebeam.ebeam_gc_te1550()
y = ebeam.ebeam_y_1550()
wg150 = ebeam.ebeam_wg_integral_1550(length=150e-6) # can optionally include ne=10.1, ng=1.3
wg50 = ebeam.ebeam_wg_integral_1550(length=50e-6)

circuit = Subcircuit('MZI')
circuit.add([
    ('input', gc,
    ('output', gc),
    ('splitter', y),
    ('recombiner', y),
    ('wg_long', wg150),
    ('wg_short', wg50),
])

circuit['splitter'].rename_nodes(('in1', 'out1', 'out2'))
circuit['recombiner'].rename_nodes(('out1', 'in2', 'in1'))

circuit.connect_many([
    ('input', 'n1', 'splitter', 'in1'),
    ('splitter', 'out1', 'wg_long', 'n1'),
    ('splitter', 'out2', 'wg_short', 'n1'),
    ('recombiner', 'in1', 'wg_long', 'n2'),
    ('recombiner', 'in1', 'wg_short', 'n2'),
    ('output', 'n1', 'recombiner', 'out1'),
])

# Run a simulation on the netlist.
simulation = SweepSimulation(circuit, 1500e-9, 1600e-9)
simulation.simulate()

# for item in simulation.cache.keys():
#     print(type(simulation.cache[item][0]), simulation.cache[item][0].shape, item)


# # Testing interpolation
# import matplotlib.pyplot as plt
# bdc = ebeam.ebeam_bdc_te1550()
# wl, s = bdc.s_parameters(1.5e-6, 1.6e-6, 2000)
# plt.plot(wl, np.abs(s[:,0,2])**2)
# plt.scatter(bdc.s_params[0], np.abs(bdc.s_params[1][:,0,2])**2)
# plt.show()


import sys
sys.exit()

# The simulation provides us a few attributes:
#   freq_array: A frequency array, each entry corresponding to a frequency 
#       the circuit was evaluated at.
#   s_parameters(): Returns a matrix containing the s-parameters of the
#       circuit. Its indexing is [frequency, output port, input port].
one2one = abs(simu.s_parameters()[:, 1, 1])**2

# Let's view the results
import matplotlib.pyplot as plt

# We'll plot the back reflections from the input port and view it in log scale.
plt.subplot(221)
s_out = 0
s_in = 0
plt.plot(simu.freq_array, np.log10(abs(simu.s_parameters()[:, s_out, s_in])**2))

# Next, let's see the result of putting light in the input and measuring the
# output, also in log scale.
plt.subplot(222)
s_out = 0
s_in = 1
plt.plot(simu.freq_array, np.log10(abs(simu.s_parameters()[:, s_out, s_in])**2))

# We don't have to view the results in log scale. We can look at it with
# just its magnitude.
plt.subplot(223)
s_out = 1
s_in = 0
plt.plot(simu.freq_array, abs(simu.s_parameters()[:, s_out, s_in])**2)

# Back reflections from the output to the output can also be viewed.
plt.subplot(224)
plt.plot(simu.freq_array, abs(simu.s_parameters()[:, 1, 1])**2)

plt.suptitle("MZI")
plt.tight_layout()
plt.show()