import numpy as np

import simphony.core as core
from simphony.core import ComponentInstance as inst
import simphony.DeviceLibrary.ebeam as dev
import simphony.DeviceLibrary.sipann as lib
import simphony.simulation as sim

# Have a main data line where frequency multiplexed data enters the circuit.
data_line = [inst(dev.ebeam_wg_integral_1550, extras={'length':100e-6}) for _ in range(3)]
# Have different radii rings for selecting out frequencies from the data line.
radii = [10+i for i in range(3)]
selectors = []
for i in range(3):
    # Format of a ring resonator:
    #   - Each ring is made up of two half rings of the same radius.
    #   - The first ring in the tuple is the upper ring, the second the lower
    #     (see diagram).
    ring = inst(lib.sipann_dc_halfring, extras={'radius': radii[i]}), inst(lib.sipann_dc_halfring, extras={'radius': radii[i]})
    selectors.append(ring)
# The outputs are waveguides that connect to the output of each ring resonator.
outputs = [inst(dev.ebeam_wg_integral_1550, extras={'length':100e-6}) for _ in range(3)]
# Terminators dissipate any light that might by some misfortune reach them, 
# and including them helps remove "false" output ports from the simulation.
terminators = [inst(dev.ebeam_terminator_te1550) for _ in range(3)]

connections = []

for i in range(3):
    connections.append([data_line[i], 1, selectors[i][0], 2])
for i in range(1, 3):
    connections.append([data_line[i], 0, selectors[i-1][0], 0])
for i in range(3):
    connections.append([selectors[i][0], 3, selectors[i][1], 1])
    connections.append([selectors[i][0], 1, selectors[i][1], 3])
for i in range(3):
    connections.append([outputs[i], 0, selectors[i][1], 0])
    connections.append([terminators[i], 0, selectors[i][1], 2])

nl = core.Netlist()
nl.load(connections, formatter='ll')
simu = sim.Simulation(nl, num=500)


freq = simu.freq_array
# g10 = np.log10(abs(simu.s_parameters()[:, 1, 0])**2)*10
# g11 = np.log10(abs(simu.s_parameters()[:, 1, 1])**2)*10
# g12 = np.log10(abs(simu.s_parameters()[:, 1, 2])**2)*10
# g13 = np.log10(abs(simu.s_parameters()[:, 1, 3])**2)*10
g00 = (abs(simu.s_parameters()[:, 0, 0])**2)
g02 = (abs(simu.s_parameters()[:, 2, 0])**2)
g03 = (abs(simu.s_parameters()[:, 3, 0])**2)
g04 = (abs(simu.s_parameters()[:, 4, 0])**2)

import matplotlib.pyplot as plt
plt.plot(freq, g00, label="0-0")
plt.plot(freq, g02, label="0-2")
plt.plot(freq, g03, label="0-3")
plt.plot(freq, g04, label="0-4")
plt.legend()
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.title("Add/Drop Filter Simulation")
plt.show()