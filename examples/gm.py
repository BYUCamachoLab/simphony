import numpy as np

import simphony.core as core
from simphony.core import ComponentInstance as inst
import simphony.DeviceLibrary.ebeam as dev
import simphony.DeviceLibrary.sipann as lib
import simphony.simulation as sim

import matplotlib.pyplot as plt

# Some helper functions for converting between wavelength and frequency
c = 299792458
def freq2wl(f):
    return c/f
def wl2freq(l):
    return c/l

# Define all input component instances
inputs = [inst(dev.ebeam_gc_te1550) for _ in range(4)]
wg1 = [inst(dev.ebeam_wg_integral_1550, extras={'length':100e-6}) for _ in range(4)]
dc1 = [inst(lib.sipann_dc_fifty) for _ in range(2)]
wg_inner1 = [inst(dev.ebeam_wg_integral_1550, extras={'length':100e-6}) for _ in range(2)]
crossover = inst(lib.sipann_dc_crossover1550)
wg_inner2 = [inst(dev.ebeam_wg_integral_1550, extras={'length':100e-6}) for _ in range(2)]
wg_outer = [inst(dev.ebeam_wg_integral_1550, extras={'length':300e-6}) for _ in range(2)]
dc2 = [inst(lib.sipann_dc_fifty) for _ in range(2)]
wg3 = [inst(dev.ebeam_wg_integral_1550, extras={'length':100e-6}) for _ in range(4)]
outputs = [inst(dev.ebeam_gc_te1550) for _ in range(4)]

# Define all circuit connections
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

# Run the actual simulation (over some optional frequency range)
nl = core.Netlist()
nl.load(connections, formatter='ll')
simu = sim.Simulation(nl, start_freq=1.925e+14, stop_freq=1.945e+14)
# simu = sim.Simulation(nl)

# Get the simulation results
freq, s = simu.freq_array, simu.s_parameters()

# We're interested in investigating behavior at this frequency
set_wl = 1550e-9
set_freq = wl2freq(set_wl)
# set_freq = 1.93e+14

# Plot the response of the entire green machine using input port i
i = 3
plt.figure()
# for i in range(1, 2):
for j in range(8):
    plt.plot(freq/1e12, np.abs(s[:,j,i])**2, label="Port {} to {}".format(i, j))
    # plt.plot(freq, 10*np.log10(np.abs(simu.s_parameters()[:,j,i])**2), label="Port {} to {}".format(i, j))
plt.axvline(set_freq/1e12)
plt.legend()
plt.xlabel("Frequency (THz)")
plt.ylabel("Normalized Power")

plt.figure()
# for i in range(1, 2):
for j in range(8):
    plt.plot(freq/1e12, np.rad2deg(np.unwrap(np.angle(s[:,j,i]))), label="Port {} to {}".format(i, j))
    # plt.plot(freq/1e12, np.rad2deg(np.angle(s[:,j,i])), label="Port {} to {}".format(i, j))
plt.axvline(set_freq/1e12)
plt.legend()
plt.xlabel("Frequency (THz)")
plt.ylabel("Phase")

# Response at precisely 1550nm
idx = np.argmax(freq>set_freq)
print(idx, freq2wl(freq[idx]))

# Phases of the four outputs at 1550nm
plt.figure()
circle = np.linspace(0, 2*np.pi)
plt.plot(np.cos(circle), np.sin(circle))

inputs1550 = [0] * 8
for output in range(4,8):
    rad = np.angle(s[idx,output,1])
    plt.plot(np.cos(rad), np.sin(rad), 'o')
    # inputs1550[output - 4] = np.cos(rad) + np.sin(rad) * 1j
    inputs1550[output] = np.cos(rad) + np.sin(rad) * 1j
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.axes().set_aspect('equal')

for val in inputs1550:
    print(val, np.rad2deg(np.angle(val)))

# Multiple input stuffs:
num_ports = 8
inputs1550 = np.array(inputs1550)
out = np.zeros([len(freq), num_ports], dtype='complex128')
for i in range(len(freq)):
    # out[i, :] = np.dot(np.reshape(s[i, :, :], [num_ports, num_ports]), inputs1550.T)
    out[i, :] = np.dot(s[i, :, :], inputs1550.T)

print(out.shape)

plt.figure()
for j in range(8):
    plt.subplot(4, 2, j+1)
    plt.plot(freq/1e12, np.abs(out[:,j])**2, label="Port {}".format(j))
    plt.axvline(set_freq/1e12)
    plt.legend()
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Normalized Power")

# plt.figure()
# for j in range(8):
#     plt.plot(freq/1e12, np.rad2deg(np.unwrap(np.angle(out[:,j]))), label="Port {}".format(j))
#     # plt.plot(freq/1e12, np.rad2deg(np.angle(s[:,j,i])), label="Port {} to {}".format(i, j))
# plt.axvline(set_freq/1e12)
# plt.legend()
# plt.xlabel("Frequency (THz)")
# plt.ylabel("Phase")

plt.show()
