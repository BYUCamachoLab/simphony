import numpy as np

# from simphony.elements import ComponentInstance as inst
import simphony.library.ebeam as ebeam
# import simphony.library.sipann as sipann
import simphony.simulation as sim

# -----------------------------------------------------------------------------
#
# Some helper functions for converting between wavelength and frequency
#
c = 299792458
def freq2wl(f):
    return c/f
def wl2freq(l):
    return c/l

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
terminators = [inst(dev.ebeam_terminator_te1550) for _ in range(4)]

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
connections.append([selectors[2][0], 0, terminators[-1], 0])

nl = core.Netlist()
nl.load(connections, formatter='ll')
simu = sim.Simulation(nl, start_freq=wl2freq(1551.15e-9), stop_freq=wl2freq(1524.5e-9))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

freq, s = simu.freq_array, simu.s_parameters()

fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 3)

ax = fig.add_subplot(gs[0, :2])
for inport in range(1):
    for outport in range(1,4):
        ax.plot(freq2wl(freq)*1e9, np.abs(s[:, outport, inport])**2, label="Out {}".format(outport), lw="0.7")
        # plt.plot(freq, 10*np.log10(np.abs(s[:, outport, inport])**2), label="Port {} to {}".format(inport, outport))
ax.set_ylabel("Fractional Optical Power")
ax.set_xlabel("Wavelength (nm)")
plt.legend(loc='upper right')

ax = fig.add_subplot(gs[0, 2])
for inport in range(1):
    for outport in range(1,4):
        ax.plot(freq2wl(freq)*1e9, np.abs(s[:, outport, inport])**2, label="Output {}".format(outport), lw="0.7")
        # plt.plot(freq, 10*np.log10(np.abs(s[:, outport, inport])**2), label="Port {} to {}".format(inport, outport))
ax.set_xlim(1543,1545)
ax.set_ylabel("Fractional Optical Power")
ax.set_xlabel("Wavelength (nm)")

fig.align_labels()

plt.show()

# plt.legend(loc='upper right')
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Fractional Optical Power")
# # plt.title("Optical Filter Simulation")
# plt.show()