import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from simphony.library import ebeam, sipann
from simphony.netlist import Subcircuit
from simphony.simulation import SweepSimulation, freq2wl, wl2freq

# Have a main data line where frequency multiplexed data enters the circuit.
wg_data = ebeam.ebeam_wg_integral_1550(100e-6)

# A terminator for dispersing unused light
term = ebeam.ebeam_terminator_te1550()

# Have rings for selecting out frequencies from the data line.
half_ring = sipann.sipann_dc_halfring(radius=10)

start1 = wl2freq(1500e-9)
start2 = wl2freq(1520e-9)
stop1 = wl2freq(1600e-9)
stop2 = wl2freq(1580e-9)

fig, ax = plt.subplots()
f, s = wg_data.s_parameters(start1, stop1, 1000)
ax.plot(f, np.abs(s[:,0,1])**2)
f, s = wg_data.s_parameters(start2, stop2, 1000)
ax.plot(f, np.abs(s[:,0,1])**2)
plt.title('Waveguide')

fig, ax = plt.subplots()
f, s = term.s_parameters(wl2freq(1480e-9), wl2freq(1620e-9), 1000)
ax.plot(f, np.abs(s[:,0,0])**2, 'rx')
f, s = term.s_parameters(start2, stop2, 1000)
ax.plot(f, np.abs(s[:,0,0])**2, 'bx')
plt.title('Terminator')

fig, ax = plt.subplots(2, 2)
f, s = half_ring.s_parameters(start1, stop1, 1000)
ax[0,0].plot(f, np.abs(s[:,0,0])**2)
ax[0,1].plot(f, np.abs(s[:,0,1])**2)
ax[1,0].plot(f, np.abs(s[:,0,2])**2)
ax[1,1].plot(f, np.abs(s[:,0,3])**2)
f, s = half_ring.s_parameters(start2, stop2, 1000)
ax[0,0].plot(f, np.abs(s[:,0,0])**2)
ax[0,1].plot(f, np.abs(s[:,0,1])**2)
ax[1,0].plot(f, np.abs(s[:,0,2])**2)
ax[1,1].plot(f, np.abs(s[:,0,3])**2)
plt.title('Half Ring')
plt.show()
