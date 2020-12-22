#!/usr/bin/env python3
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
subnetwork_growth.py
--------------------

Author: Sequoia Ploeg
Modified: April 2, 2020

A simple, non-abstracted script that demonstrates the process of subnetwork
growth using the base functions in simphony. In this example, we construct
a 10-micron ring resonator with an input port, a through-port, and a drop-port.

The half-ring model provided by SiPANN (and available in the simphony
library) has ports that, corresponding to its s-matrix, are ordered like so
(zero-indexed):

1           3
 |         |
  \       /
   \     /
 ---=====---
0           2

A ring resonator is simply two half-rings cascaded together. Our configuration
will look like two of the above half-rings, rotated and placed next to each
other:

Ring (L)   Ring (R)
0     1    3     2
|  ---      ---  |
| /            \ |
||              ||
| \            / |
|  ---      ---  |
2     3    1     0
0
| <- Terminator

A description of subnetwork growth can be found in the paper introducing
Simphony. But briefly, cascading networks involves placing them into a large
matrix and performing operations that depend on whether its an internal
connection (same network, two ports connected) or an external connection. By
the end of the subnetwork growth process, the entire network is unified.
Connecting two independenet networks creates one new network with all the
leftover, unconnected ports of the original two networks.

One cascading order is shown below. Two cascading orders are demonstrated in
the code, with their results plotted on top of each other, showing their
equivalence.

In this example, intermediary networks are numbered 'n_i'.

n1:
Ring (L)   Ring (R)
0                5
|  ------------  |
| /            \ |
||              ||
| \            / |
|  ---      ---  |
1     2    4     3
0
| <- Terminator

n2:
Ring (L)   Ring (R)
0                3
|  ------------  |
| /            \ |
||              ||
| \            / |
|  ------------  |
1                2
0
| <- Terminator

n3:
Ring (L)   Ring (R)
0                3
|  ------------  |
| /            \ |
||              ||
| \            / |
|  ------------  |
|                2
| <- Terminator

You can see that in the final network, our input port is port 3, our through
port is port 2, and our drop port is port 0.

To get the transmission from input to output in the s-matrix, the indexing is
``s[out, in]``.
"""

import matplotlib.pyplot as plt
import numpy as np
from simphony.connect import connect_s, innerconnect_s
from simphony.library import ebeam
from simphony.tools import freq2wl, wl2freq


def coupler_ring(
    bend_radius: float = 5,
    wg_width: float = 0.5,
    thickness: float = 0.22,
    gap: float = 0.22,
    length_x: float = 4.0,
    sw_angle: float = 90.0,
):
    from SiPANN.scee_int import SimphonyWrapper

    width = wg_width * 1e3
    thickness = thickness * 1e3
    gap = gap * 1e3
    length = length_x * 1e3
    radius = bend_radius * 1e3
    # print(f'ignoring {kwargs}')

    s = HalfRacetrack(
        radius=radius,
        width=width,
        thickness=thickness,
        gap=gap,
        length=length,
        sw_angle=sw_angle,
    )
    s2 = SimphonyWrapper(s)
    s2.pins = ("W0", "N0", "E0", "N1")
    return s2


if __name__ == "__main__":
    from SiPANN.scee import HalfRacetrack

    # First, we'll set up the frequency range we wish to perform the simulation on.
    freq = np.linspace(wl2freq(1600e-9), wl2freq(1500e-9), 2000)
    wavelengths = freq2wl(freq)

    # Get the scattering parameters for each of the elements in our network.
    half_ring_left = coupler_ring(bend_radius=10).s_parameters(freq)
    half_ring_right = coupler_ring(bend_radius=10).s_parameters(freq)
    term = ebeam.ebeam_terminator_te1550().s_parameters(freq)

    ### CONFIGURATION 1 ###
    n1 = connect_s(half_ring_left, 1, half_ring_right, 3)
    n2 = innerconnect_s(n1, 2, 4)
    n3 = connect_s(n2, 1, term, 0)

    ### CONFIGURATION 2 ###
    m1 = connect_s(half_ring_right, 1, half_ring_left, 3)
    m2 = innerconnect_s(m1, 2, 4)
    m3 = connect_s(term, 0, m2, 3)

    plt.plot(wavelengths, np.abs(n3[:, 1, 2]) ** 2, "b.")
    plt.plot(wavelengths, np.abs(m3[:, 0, 1]) ** 2, "r--")
    plt.xlabel('wavelength (m)')
    plt.ylabel('Transmission')
    plt.tight_layout()
    plt.show()
