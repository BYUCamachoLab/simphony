import numpy as np

import simphony.core as core
from simphony.core import ComponentInstance as inst
import simphony.DeviceLibrary.ebeam as dev
import simphony.DeviceLibrary.sipann as lib
import simphony.simulation as sim

import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
#
# Some helper functions for converting between wavelength and frequency
#
c = 299792458
def freq2wl(f):
    return c/f
def wl2freq(l):
    return c/l

# -----------------------------------------------------------------------------
#
# Define all input component instances
#
inputs = [inst(dev.ebeam_gc_te1550) for _ in range(4)]
wg1 = [inst(dev.ebeam_wg_integral_1550, extras={'length':100e-6}) for _ in range(4)]
dc1 = [inst(lib.sipann_dc_fifty) for _ in range(2)]
wg_inner1 = [inst(dev.ebeam_wg_integral_1550, extras={'length':100e-6}) for _ in range(2)]
crossover = inst(lib.sipann_dc_crossover1550)
wg_inner2 = [inst(dev.ebeam_wg_integral_1550, extras={'length':102.125e-6}) for _ in range(2)]
wg_outer = [inst(dev.ebeam_wg_integral_1550, extras={'length':300e-6}) for _ in range(2)]
dc2 = [inst(lib.sipann_dc_fifty) for _ in range(2)]
wg3 = [inst(dev.ebeam_wg_integral_1550, extras={'length':100e-6}) for _ in range(4)]
outputs = [inst(dev.ebeam_gc_te1550) for _ in range(4)]

# -----------------------------------------------------------------------------
#
# Define all circuit connections
#
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

def local():
    plt.figure()
    device = dc1[0]
    f,s = device.get_s_parameters()

    set_wl = 1550e-9
    set_freq = wl2freq(set_wl)
    # set_freq = 1.93e+14
    idx = np.argmax(f>set_freq)
    print(idx, freq2wl(f[idx]))

    plt.plot(f, np.abs(s[:,3,0])**2)
    plt.plot(f, np.abs(s[:,2,0])**2)
    plt.title("DC")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(f, np.rad2deg(np.unwrap(np.angle(s[:,3,0]))))
    plt.plot(f, np.rad2deg(np.unwrap(np.angle(s[:,2,0]))))
    plt.legend()
    plt.tight_layout()
    plt.show()
# local()

# -----------------------------------------------------------------------------
#
# Run the actual simulation (over some optional frequency range)
#
nl = core.Netlist()
nl.load(connections, formatter='ll')
# simu = sim.Simulation(nl, start_freq=1.925e+14, stop_freq=1.945e+14)
simu = sim.Simulation(nl, start_freq=wl2freq(1.5501e-6), stop_freq=wl2freq(1.5499e-6))
# simu = sim.Simulation(nl)

# Get the simulation results
freq, s = simu.freq_array, simu.s_parameters()


# -----------------------------------------------------------------------------
#
# We're interested in investigating behavior at this frequency
#
set_wl = 1550e-9
set_freq = wl2freq(set_wl)
# set_freq = 1.93e+14

# -----------------------------------------------------------------------------
#
# Plot the response of the entire green machine using input port i
#
# for i in range(0,4):
i = 2
# plt.figure()
# # for i in range(1, 2):
# for j in range(4,8):
#     # plt.plot(freq/1e12, np.abs(s[:,j,i])**2, label="Port {} to {}".format(i, j))
#     plt.plot(freq2wl(freq)*1e9, np.abs(s[:,j,i])**2, label="Port {}".format(j), linewidth="0.7")
# # plt.axvline(set_freq/1e12)
# # plt.axvline(1550)
# plt.legend(loc="upper right")
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Fractional Optical Power")

plt.figure()
idx = np.argmax(freq>set_freq)
print(idx, freq2wl(freq[idx]))
# for i in range(1, 2):
offsets = [0] * 4
for j in range(4,8):
    offsets[j-4] = np.angle(s[idx,j,i])

angles = [None] * 4
for j in range(4,8):
    angles[j-4] = np.unwrap(np.angle(s[:,j,i]))

print(offsets, "Min:", min(offsets))
for j in range(4):
    angles[j] -= min(offsets)
    print(angles[j][idx])

for j in range(4,8):
    # plt.plot(freq/1e12, np.rad2deg(np.unwrap(np.angle(s[:,j,i]))), label="Port {} to {}".format(i, j))
    # angles = np.rad2deg(np.unwrap(np.angle(s[:,j,i])))
    # angles = np.unwrap(np.angle(s[:,j,i]))
    # angles -= min(offsets)
    # angles = angles + (angles[idx] % 2*np.pi) - angles[idx]
    plt.plot(freq2wl(freq)*1e9, angles[j-4], linewidth='0.7')
plt.axvline(1550, color='k', linestyle='--', linewidth='0.5')
plt.legend([r'$\phi_4$',r'$\phi_5$',r'$\phi_6$',r'$\phi_7$'], loc='upper right')
plt.xlabel("Wavelength (nm)")
plt.ylabel("Phase")
plt.show()

import sys; sys.exit()

plt.figure()
idx = np.argmax(freq>set_freq)
print(idx, freq2wl(freq[idx]))
for j in range(4,8):
    # print(np.rad2deg(np.angle(s[idx,j,i])))
    angles = np.rad2deg(np.unwrap(np.angle(s[:,j,i])))
    angles = angles + (angles[idx] % 2*np.pi) - angles[idx]
    print(angles[idx], angles)
    plt.plot(freq2wl(freq)*1e9, angles, label="Port {} to {}".format(i, j))
    plt.plot(freq2wl(freq[idx])*1e9, angles[idx], 'rx')

plt.axvline(1550)
plt.legend()
plt.xlabel("Wavelength (nm)")
plt.ylabel("Phase")


# plt.axvline(set_freq/1e12)


plt.show()

# -----------------------------------------------------------------------------
#
# Response at precisely 1550nm
#
idx = np.argmax(freq>set_freq)
print(idx, freq2wl(freq[idx]))

# Phases of the four outputs at 1550nm
plt.figure()
circle = np.linspace(0, 2*np.pi)
plt.plot(np.cos(circle), np.sin(circle))

# for i in range(0,4):
inputs1550 = [0] * 8
for output in range(4,8):
    rad = np.angle(s[idx,output,i])
    plt.plot(np.cos(rad), np.sin(rad), 'o')
    inputs1550[output-4] = np.cos(rad) + np.sin(rad) * 1j
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.axes().set_aspect('equal')

# for val in inputs1550:
#     print(val, np.rad2deg(np.angle(val)))

# -----------------------------------------------------------------------------
#
# Multiple input stuffs:
#
def multi_input(num_ports, inputs, verbose=True):
    inputs = np.array(inputs, dtype=np.complex_)
    if verbose:
        angles = np.rad2deg(np.angle(inputs))
        print(angles - min(angles))
    out = np.zeros([len(freq), num_ports], dtype='complex128')
    for j in range(len(freq)):
        out[j, :] = np.dot(s[j, :, :], inputs.T)
    return out

def plot_outputs(out):
    plt.figure()
    for j in range(8):
        plt.subplot(8, 1, j+1)
        plt.plot(freq/1e12, np.abs(out[:,j])**2, label="Port {}".format(j))
        plt.axvline(set_freq/1e12)
        plt.legend()
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Normalized Power")

out = multi_input(8, inputs1550)

plt.figure()
for j in range(8):
    plt.subplot(8, 1, j+1)
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
