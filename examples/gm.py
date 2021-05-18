#!/usr/bin/env python3
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np

from simphony.library import ebeam
from simphony.netlist import Subcircuit
from simphony.simulation import SweepSimulation
from simphony.tools import freq2wl, wl2freq

from SiPANN.scee_opt import premade_coupler
from SiPANN.scee_int import SimphonyWrapper

# We can rename the pins attribute on the wg before we instantiate it;
# then we don't have to rename the pins on each element individually later.
ebeam.ebeam_wg_integral_1550.pins = ("in", "out")


# Get all the models we're going to need for the green machine circuit:
gc = ebeam.ebeam_gc_te1550()
wg100 = ebeam.ebeam_wg_integral_1550(length=100e-6)
dc = SimphonyWrapper( premade_coupler(50)[0] )
crossover = SimphonyWrapper( premade_coupler(100)[0] )
wgin2 = ebeam.ebeam_wg_integral_1550(length=102.125e-6)
wg300 = ebeam.ebeam_wg_integral_1550(length=300e-6)

#we rename the coupler pins
dc.pins = ("in1", "in2", "out1", "out2")
crossover.pins = ("in1", "in2", "out1", "out2")

# Add all the elements used in the circuit
circuit = Subcircuit("Green Machine")
e = circuit.add(
    [
        # Define the four input grating couplers
        (gc, "in1"),
        (gc, "in2"),
        (gc, "in3"),
        (gc, "in4"),
        # The grating couplers each feed into their own waveguide
        (wg100, "wg1"),
        (wg100, "wg2"),
        (wg100, "wg3"),
        (wg100, "wg4"),
        # Each pair of waveguides feeds into a 50/50 directional coupler
        (dc, "dc1"),
        (dc, "dc2"),
        # After mixing, the center pair of waveguides cross paths at a 100/0
        # crossing. The edge pair of waveguides pass uninterrupted.
        (wg300, "wg_pass1"),
        (wg100, "wg_in1"),
        (wgin2, "wg_out1"),
        (crossover, "crossing"),
        (wg100, "wg_in2"),
        (wgin2, "wg_out2"),
        (wg300, "wg_pass2"),
        # After crossing, the waveguides are mixed again.
        (dc, "dc3"),
        (dc, "dc4"),
        # The outputs are fed through waveguides.
        (wg100, "wg5"),
        (wg100, "wg6"),
        (wg100, "wg7"),
        (wg100, "wg8"),
        # We finally output the values through grating couplers.
        (gc, "out1"),
        (gc, "out2"),
        (gc, "out3"),
        (gc, "out4"),
    ]
)

# Let's rename some ports on some of our elements so that we can:
#   1) find them again later, and
#   2) make our code clearer by using plain english for the connections.
circuit.elements["in1"].pins["n1"] = "in1"
circuit.elements["in2"].pins["n1"] = "in2"
circuit.elements["in3"].pins["n1"] = "in3"
circuit.elements["in4"].pins["n1"] = "in4"

circuit.elements["out1"].pins["n2"] = "out1"
circuit.elements["out2"].pins["n2"] = "out2"
circuit.elements["out3"].pins["n2"] = "out3"
circuit.elements["out4"].pins["n2"] = "out4"

# Phew! Now that we got all those elements out of the way, we can finally
# work on the circuit connnections.
circuit.connect_many(
    [
        #connect in grating couplers with waveguides
        ("in1", "n2", "wg1", "in"),
        ("in2", "n2", "wg2", "in"),
        ("in3", "n2", "wg3", "in"),
        ("in4", "n2", "wg4", "in"),

        #connect waveguides to input of directional couplers
        ("wg1", "out", "dc1", "in1"),
        ("wg2", "out", "dc1", "in2"),
        ("wg3", "out", "dc2", "in1"),
        ("wg4", "out", "dc2", "in2"),

        #connect top dc to out waveguides
        ("dc1", "out1", "wg_pass1", "in"),
        ("dc1", "out2", "wg_in1", "in"),
        #connect waveguides to crossing
        ("wg_in1", "out", "crossing", "in1"),
        ("crossing", "out1", "wg_out1", "in"),
        #connect bottom dc to out waveguides
        ("dc2", "out1", "wg_in2", "in"),
        ("wg_in2", "out", "crossing", "in2"),
        #connect waveguides to crossing
        ("crossing", "out2", "wg_out2", "in"),
        ("dc2", "out2", "wg_pass2", "in"),

        #connect waveguides to final step of directional couplers
        ("wg_pass1", "out", "dc3", "in1"),
        ("wg_out1", "out", "dc3", "in2"),
        ("wg_out2", "out", "dc4", "in1"),
        ("wg_pass2", "out", "dc4", "in2"),

        #then directional couplers to waveguides
        ("dc3", "out1", "wg5", "in"),
        ("dc3", "out2", "wg6", "in"),
        ("dc4", "out1", "wg7", "in"),
        ("dc4", "out2", "wg8", "in"),

        #and finally waveguides to grating couplers
        ("wg5", "out", "out1", "n1"),
        ("wg6", "out", "out2", "n1"),
        ("wg7", "out", "out3", "n1"),
        ("wg8", "out", "out4", "n1"),
    ]
)

# Run a simulation on our circuit.
simulation = SweepSimulation(circuit, 1549.9e-9, 1550.1e-9)
# simulation = SweepSimulation(circuit, 1510e-9, 1590e-9)
result = simulation.simulate()

# Get the simulation results
# f, s = result.data(result.pinlist['in1'], result.pinlist['out1'])

# The Green Machine is optimized for 1550 nanometers. We'd like to investigate
# its behavior at that specific frequency:
set_freq = wl2freq(1550e-9)

in_port = "in1"
plt.figure()
plt.plot(*result.data(result.pinlist[in_port], result.pinlist["out1"]), label="1 to 5")
plt.plot(*result.data(result.pinlist[in_port], result.pinlist["out2"]), label="1 to 6")
plt.plot(*result.data(result.pinlist[in_port], result.pinlist["out3"]), label="1 to 7")
plt.plot(*result.data(result.pinlist[in_port], result.pinlist["out4"]), label="1 to 8")
plt.axvline(set_freq)
plt.legend(loc="upper right")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Fractional Optical Power")
plt.show()

# We're interested now in the phase offsets at our wavelength of interest.
plt.figure()
freq, s = result.f, result.s
idx = np.argmax(freq > set_freq)
input_pin = result.pinlist["in1"].index
outputs = [result.pinlist["out" + str(n)].index for n in range(1, 5)]
offset = min(np.angle(s[idx, outputs, input_pin]))
# angles = np.unwrap(np.angle(s[:, outputs, input_pin])).T - offset
angles = np.angle(s[:, outputs, input_pin]).T - offset

for angle in angles:
    plt.plot(freq2wl(freq) * 1e9, angle, linewidth="0.7")
plt.axvline(1550, color="k", linestyle="--", linewidth="0.5")
plt.legend([r"$\phi_4$", r"$\phi_5$", r"$\phi_6$", r"$\phi_7$"], loc="upper right")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Phase")
plt.show()

import sys

sys.exit()

plt.figure()
idx = np.argmax(freq > set_freq)
print(idx, freq2wl(freq[idx]))
angles = np.rad2deg(np.unwrap(np.angle(s[:, outputs, input_pin]))).T
angles = angles + ((angles[:, idx] % 2 * np.pi) - angles[:, idx]).reshape((4, 1))
print(angles[:, idx], angles)
for i in range(4):
    plt.plot(freq2wl(freq) * 1e9, angles[i])  # , label="Port {} to {}".format(i, j))
    plt.plot(freq2wl(freq[idx]) * 1e9, angles[i][idx], "rx")

plt.axvline(1550)
# plt.legend()
plt.xlabel("Wavelength (nm)")
plt.ylabel("Phase")
plt.show()
import sys

sys.exit()
# plt.axvline(set_freq/1e12)


plt.show()

# -----------------------------------------------------------------------------
#
# Response at precisely 1550nm
#
idx = np.argmax(freq > set_freq)
print(idx, freq2wl(freq[idx]))

# Phases of the four outputs at 1550nm
plt.figure()
circle = np.linspace(0, 2 * np.pi)
plt.plot(np.cos(circle), np.sin(circle))

# for i in range(0,4):
inputs1550 = [0] * 8
for output in range(4, 8):
    rad = np.angle(s[idx, output, i])
    plt.plot(np.cos(rad), np.sin(rad), "o")
    inputs1550[output - 4] = np.cos(rad) + np.sin(rad) * 1j
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.axes().set_aspect("equal")

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
    out = np.zeros([len(freq), num_ports], dtype="complex128")
    for j in range(len(freq)):
        out[j, :] = np.dot(s[j, :, :], inputs.T)
    return out


def plot_outputs(out):
    plt.figure()
    for j in range(8):
        plt.subplot(8, 1, j + 1)
        plt.plot(freq / 1e12, np.abs(out[:, j]) ** 2, label="Port {}".format(j))
        plt.axvline(set_freq / 1e12)
        plt.legend()
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Normalized Power")


out = multi_input(8, inputs1550)

plt.figure()
for j in range(8):
    plt.subplot(8, 1, j + 1)
    plt.plot(freq / 1e12, np.abs(out[:, j]) ** 2, label="Port {}".format(j))
    plt.axvline(set_freq / 1e12)
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
