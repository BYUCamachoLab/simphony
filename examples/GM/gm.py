import numpy as np

import simphony.core as core
from simphony.core import ComponentInstance as inst
# import simphony.errors as errors
import simphony.DeviceLibrary.ebeam as dev
import simphony.DeviceLibrary.sipann as lib
import simphony.simulation as sim

inputs = [inst(dev.ebeam_gc_te1550) for _ in range(4)]
wg1 = [inst(dev.ebeam_wg_integral_1550, extras={'length':100e-6}) for _ in range(4)]
dc1 = [inst(dev.ebeam_bdc_te1550) for _ in range(2)]
wg_inner1 = [inst(dev.ebeam_wg_integral_1550, extras={'length':100e-6}) for _ in range(2)]
crossover = inst(lib.sipann_dc_crossover1550)
wg_inner2 = [inst(dev.ebeam_wg_integral_1550, extras={'length':100e-6}) for _ in range(2)]
wg_outer = [inst(dev.ebeam_wg_integral_1550, extras={'length':300e-6}) for _ in range(2)]
dc2 = [inst(dev.ebeam_bdc_te1550) for _ in range(2)]
wg3 = [inst(dev.ebeam_wg_integral_1550, extras={'length':100e-6}) for _ in range(4)]
outputs = [inst(dev.ebeam_gc_te1550) for _ in range(4)]

connections = []
for i in range(4):
    connections.append([inputs[i], 0, wg1[i], 0])

connections.append([wg1[0], 1, dc1[0], 1])
connections.append([wg1[1], 1, dc1[0], 0])
connections.append([wg1[2], 1, dc1[1], 1])
connections.append([wg1[3], 1, dc1[1], 0])

connections.append([wg_outer[0], 0, dc1[0], 3])
connections.append([wg_outer[1], 0, dc1[1], 2])
connections.append([wg_inner1[0], 0, dc1[0], 2])
connections.append([wg_inner1[1], 0, dc1[1], 3])

connections.append([wg_inner1[0], 1, crossover, 1])
connections.append([wg_inner1[1], 1, crossover, 0])
connections.append([crossover, 3, wg_inner2[0], 0])
connections.append([crossover, 2, wg_inner2[1], 0])

connections.append([wg_outer[0], 1, dc2[0], 1])
connections.append([wg_outer[1], 1, dc2[1], 0])
connections.append([wg_inner2[0], 1, dc2[0], 0])
connections.append([wg_inner2[1], 1, dc2[1], 1])

connections.append([dc2[0], 3, wg3[0], 0])
connections.append([dc2[0], 2, wg3[1], 0])
connections.append([dc2[1], 3, wg3[2], 0])
connections.append([dc2[1], 2, wg3[3], 0])

for i in range(4):
    connections.append([outputs[i], 0, wg3[i], 1])


nl = core.Netlist()
nl.load(connections, formatter='ll')
simu = sim.Simulation(nl, num=1)


freq = simu.freq_array
zero2zero = np.log10(abs(simu.s_parameters()[:, 0, 0])**2)
zero2one = np.log10(abs(simu.s_parameters()[:, 0, 1])**2)
one2zero = abs(simu.s_parameters()[:, 1, 0])**2
one2one = abs(simu.s_parameters()[:, 1, 1])**2

import matplotlib.pyplot as plt
plt.subplot(221)
plt.plot(freq, zero2zero)
plt.subplot(222)
plt.plot(freq, zero2one)
plt.subplot(223)
plt.plot(freq, one2zero)
plt.subplot(224)
plt.plot(freq, one2one)
plt.suptitle("MZI")
plt.tight_layout()
plt.show()