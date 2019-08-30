import numpy as np
import simphony.core as core
import simphony.errors as errors
import simphony.DeviceLibrary.ebeam as dev
import simphony.DeviceLibrary.ann as ann
import simphony.simulation as sim

import matplotlib.pyplot as plt

radius = 10

bottom = core.ComponentInstance(dev.ebeam_dc_halfring_te1550)
#bottom = core.ComponentInstance(ann.sipann_dc_halfring, extras=
#                                {'radius': radius})

top    = core.ComponentInstance(ann.ann_wg_integral, extras=
                                {'length': np.pi*radius})

c1 = [bottom, top]
p1 = [1, 1]
c2 = [top, bottom]
p2 = [0, 3]

con = zip(c1, p1, c2, p2)

nl = core.Netlist()
nl.load(con, formatter='ll')
simu = sim.Simulation(nl)

freq = simu.freq_array
sparams = simu.s_parameters()

print(sparams.shape)

plt.plot(freq, np.abs(sparams[:,0,1]))
plt.show()