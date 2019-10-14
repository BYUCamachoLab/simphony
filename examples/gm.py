import numpy as np

import simphony.core as core
from simphony.core import ComponentInstance as inst
import simphony.DeviceLibrary.ebeam as dev
import simphony.DeviceLibrary.sipann as lib
import simphony.simulation as sim

import matplotlib.pyplot as plt

c = 299792458
def freq2wl(f):
    return c/f
def wl2freq(l):
    return c/l

# inputs = [inst(dev.ebeam_gc_te1550) for _ in range(4)]
wg1 = [inst(dev.ebeam_wg_integral_1550, extras={'length':100e-6}) for _ in range(4)]
dc1 = [inst(lib.sipann_dc_fifty) for _ in range(2)]
wg_inner1 = [inst(dev.ebeam_wg_integral_1550, extras={'length':100e-6}) for _ in range(2)]
crossover = inst(lib.sipann_dc_crossover1550)
wg_inner2 = [inst(dev.ebeam_wg_integral_1550, extras={'length':100e-6}) for _ in range(2)]
wg_outer = [inst(dev.ebeam_wg_integral_1550, extras={'length':300e-6}) for _ in range(2)]
dc2 = [inst(lib.sipann_dc_fifty) for _ in range(2)]
wg3 = [inst(dev.ebeam_wg_integral_1550, extras={'length':100e-6}) for _ in range(4)]
# outputs = [inst(dev.ebeam_gc_te1550) for _ in range(4)]

# plt.figure()
# device = inputs[0]
# f,s = device.get_s_parameters()
# outport, inport = 1, 0
# # plt.plot(f, np.real(s[:,outport,inport]))
# # plt.plot(f, np.imag(s[:,outport,inport]))
# # plt.plot(f, np.abs(s[:,outport,inport])**2)
# plt.plot(f, np.abs(s[:,inport,inport])**2)
# plt.plot(f, np.abs(s[:,outport,outport])**2)
# plt.title("Grating")
# plt.legend()
# plt.tight_layout()
# plt.show()

connections = []
# for i in range(4):
#     connections.append([inputs[i], 0, wg1[i], 0])

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

# for i in range(4):
#     connections.append([outputs[i], 0, wg3[i], 1])


nl = core.Netlist()
nl.load(connections, formatter='ll')
# simu = sim.Simulation(nl, start_freq=1.925e+14, stop_freq=1.945e+14, num=500)
simu = sim.Simulation(nl, num=500)


freq, s = simu.freq_array, simu.s_parameters()
set_wl = 1550e-9
set_freq = wl2freq(set_wl)
# set_freq = 1.93e+14

plt.figure()
for i in range(1, 2):
    for j in range(8):
        plt.plot(freq/1e12, np.abs(s[:,j,i])**2, label="Port {} to {}".format(i, j))
        # plt.plot(freq, 10*np.log10(np.abs(simu.s_parameters()[:,j,i])**2), label="Port {} to {}".format(i, j))
plt.axvline(set_freq/1e12)
plt.legend()
plt.xlabel("Frequency (THz)")
plt.ylabel("Normalized Power")

plt.figure()
for i in range(1, 2):
    for j in range(8):
        # plt.plot(freq/1e12, np.rad2deg(np.unwrap(np.angle(s[:,j,i]))), label="Port {} to {}".format(i, j))
        plt.plot(freq/1e12, np.rad2deg(np.angle(s[:,j,i])), label="Port {} to {}".format(i, j))
plt.axvline(set_freq/1e12)
plt.legend()
plt.xlabel("Frequency (THz)")
plt.ylabel("Phase")

plt.show()