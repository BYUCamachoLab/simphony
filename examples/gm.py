#!/usr/bin/env python3
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)
r"""
gm.py
-----

The layout for the Green Machine is as follows:

    in4____    ____________    ____out4
           \  /            \  /
            == dc2          == dc4
    in3____/  \____    ____/  \____out3
                   \  /
                    == coupling
    in2____    ____/  \____    ____out2
           \  /            \  /
            == dc1          == dc3
    in1____/  \____________/  \____out1

For more information, see: https://camacholab.byu.edu/research/quantum-photonics
"""

import matplotlib.pyplot as plt
import numpy as np

from simphony.libraries import siepic, sipann
from simphony.simulators import SweepSimulator
from simphony.tools import freq2wl, wl2freq

# Get all the components that we're going to need for the green machine circuit:
in1 = siepic.GratingCoupler(name="in1")
in2 = siepic.GratingCoupler(name="in2")
in3 = siepic.GratingCoupler(name="in3")
in4 = siepic.GratingCoupler(name="in4")
out1 = siepic.GratingCoupler(name="out1")
out2 = siepic.GratingCoupler(name="out2")
out3 = siepic.GratingCoupler(name="out3")
out4 = siepic.GratingCoupler(name="out4")
wg1 = siepic.Waveguide(length=100e-6, name="wg1")
wg2 = siepic.Waveguide(length=100e-6, name="wg2")
wg3 = siepic.Waveguide(length=100e-6, name="wg3")
wg4 = siepic.Waveguide(length=100e-6, name="wg4")
wg5 = siepic.Waveguide(length=100e-6, name="wg5")
wg6 = siepic.Waveguide(length=100e-6, name="wg6")
wg7 = siepic.Waveguide(length=100e-6, name="wg7")
wg8 = siepic.Waveguide(length=100e-6, name="wg8")
wg_in1 = siepic.Waveguide(length=100e-6, name="wg_in1")
wg_in2 = siepic.Waveguide(length=100e-6, name="wg_in2")
dc1 = sipann.PremadeCoupler(50, name="dc1")
dc2 = sipann.PremadeCoupler(50, name="dc2")
dc3 = sipann.PremadeCoupler(50, name="dc3")
dc4 = sipann.PremadeCoupler(50, name="dc4")
crossing = sipann.PremadeCoupler(100, name="crossing")
wg_out1 = siepic.Waveguide(length=102.125e-6, name="wg_out1")
wg_out2 = siepic.Waveguide(length=102.125e-6, name="wg_out2")
wg_pass1 = siepic.Waveguide(length=300e-6, name="wg_pass1")
wg_pass2 = siepic.Waveguide(length=300e-6, name="wg_pass2")

# connect input grating couplers to directional couplers
wg1.multiconnect(in1, dc1)
wg2.multiconnect(in2, dc1)
wg3.multiconnect(in3, dc2)
wg4.multiconnect(in4, dc2)

# connect directional couplers and crossing
wg_pass1.multiconnect(dc1, dc3)
wg_in1.multiconnect(dc1, crossing)
wg_in2.multiconnect(dc2, crossing)
wg_out1.multiconnect(crossing, dc3)
wg_out2.multiconnect(crossing, dc4)
wg_pass2.multiconnect(dc2, dc4)

# connect output grating couplers to directional couplers
wg5.multiconnect(dc3, out1)
wg6.multiconnect(dc3, out2)
wg7.multiconnect(dc4, out3)
wg8.multiconnect(dc4, out4)

simulator = SweepSimulator(1549.9e-9, 1550.1e-9)

# The Green Machine is optimized for 1550 nanometers. We'd like to investigate
# its behavior at that specific frequency:
set_freq = wl2freq(1550e-9)

plt.figure()

simulator.multiconnect(in1["pin2"], out1)
plt.plot(*simulator.simulate(), label="1 to 5")

simulator.multiconnect(in1["pin2"], out2)
plt.plot(*simulator.simulate(), label="1 to 6")

simulator.multiconnect(in1["pin2"], out3)
plt.plot(*simulator.simulate(), label="1 to 7")

simulator.multiconnect(in1["pin2"], out4)
plt.plot(*simulator.simulate(), label="1 to 8")

plt.axvline(set_freq)
plt.legend(loc="upper right")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Fractional Optical Power")
plt.show()

# We're interested now in the phase offsets at our wavelength of interest.
plt.figure()

# port 0 is the input port. ports 4-7 are the output ports.
freqs = np.argmax(simulator.freqs > set_freq)
input = 0
outputs = slice(4, 8)
s = simulator.circuit.s_parameters(simulator.freqs)
offset = min(np.angle(s[freqs, outputs, input]))
angles = np.angle(s[:, outputs, input]).T - offset

for angle in angles:
    plt.plot(freq2wl(simulator.freqs) * 1e9, angle, linewidth="0.7")

plt.axvline(1550, color="k", linestyle="--", linewidth="0.5")
plt.legend([r"$\phi_4$", r"$\phi_5$", r"$\phi_6$", r"$\phi_7$"], loc="upper right")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Phase")
plt.show()

plt.figure()
angles = np.rad2deg(np.unwrap(np.angle(s[:, outputs, input]))).T
angles = angles + ((angles[:, freqs] % 2 * np.pi) - angles[:, freqs]).reshape((4, 1))

for i in range(4):
    plt.plot(freq2wl(simulator.freqs) * 1e9, angles[i])
    plt.plot(freq2wl(simulator.freqs[freqs]) * 1e9, angles[i][freqs], "rx")

plt.axvline(1550)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Phase")
plt.show()

# -----------------------------------------------------------------------------
#
# Response at precisely 1550nm
#

# Phases of the four outputs at 1550nm
plt.figure()
circle = np.linspace(0, 2 * np.pi)
plt.plot(np.cos(circle), np.sin(circle))

inputs1550 = [0] * 8
for output in range(4, 8):
    rad = np.angle(s[freqs, output, i])
    plt.plot(np.cos(rad), np.sin(rad), "o")
    inputs1550[output - 4] = np.cos(rad) + np.sin(rad) * 1j

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.axes().set_aspect("equal")


# -----------------------------------------------------------------------------
#
# Multiple input stuffs:
#
def multi_input(num_ports, inputs, verbose=True):
    inputs = np.array(inputs, dtype=np.complex_)
    if verbose:
        angles = np.rad2deg(np.angle(inputs))
        print(angles - min(angles))

    out = np.zeros([len(simulator.freqs), num_ports], dtype="complex128")
    for j in range(len(simulator.freqs)):
        out[j, :] = np.dot(s[j, :, :], inputs.T)

    return out


def plot_outputs(out):
    plt.figure()
    for j in range(8):
        plt.subplot(8, 1, j + 1)
        plt.plot(simulator.freqs / 1e12, np.abs(out[:, j]) ** 2, label=f"Port {j}")
        plt.axvline(set_freq / 1e12)
        plt.legend()
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Normalized Power")


out = multi_input(8, inputs1550)

plt.figure()
for j in range(8):
    plt.subplot(8, 1, j + 1)
    plt.plot(simulator.freqs / 1e12, np.abs(out[:, j]) ** 2, label=f"Port {j}")
    plt.axvline(set_freq / 1e12)
    plt.legend()
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Normalized Power")

plt.show()
