# All simphony data uses numpy arrays and matrices.
import numpy as np

# simphony.core contains the base modules we'll need for constructing a
# circuit; namely, ComponentInstance and Netlist.
import simphony.core as core
# simphony.DeviceLibrary.ebeam is a set of presimulated components from 
# the SiEPIC EBeam Process Design Kit (PDK). We'll use these in our circuit.
import simphony.DeviceLibrary.ebeam as dev
# simphony.simulation is the module that defines various simulators we can
# run our netlist through.
import simphony.simulation as sim

# Our circuit is as follows (diagram)
gc_in = core.ComponentInstance(dev.ebeam_gc_te1550)
gc_out = core.ComponentInstance(dev.ebeam_gc_te1550)
y1 = core.ComponentInstance(dev.ebeam_y_1550)
y2 = core.ComponentInstance(dev.ebeam_y_1550)
wg1 = core.ComponentInstance(dev.ebeam_wg_integral_1550, extras={'length':50e-6, 'ne':10.1, 'ng':1.3})
wg2 = core.ComponentInstance(dev.ebeam_wg_integral_1550, extras={'length':150e-6, 'ne':12.15, 'ng':-3.7})

# We define our connections (read vertically; device y1, port 1, is connected
# to device wg1, port 0, and so on).
c1 = [ y1,  y1,  y2,  y2, gc_in, gc_out]
p1 = [  1,   2,   2,   1,     0,      0]
c2 = [wg1, wg2, wg1, wg2,    y1,     y2]
p2 = [  0,   0,   1,   1,     0,      0]
con = zip(c1, p1, c2, p2)

# Create a netlist instance that we will handover the connection list to.
nl = core.Netlist()
# The formatter argument is optional.
nl.load(con, formatter='ll')
# Run a simulation on the netlist.
simu = sim.Simulation(nl)

# The simulation provides us a few attributes:
#   freq_array: A frequency array, each entry corresponding to a frequency 
#       the circuit was evaluated at.
#   s_parameters(): Returns a matrix containing the s-parameters of the
#       circuit. Its indexing is [frequency, output port, input port].
one2one = abs(simu.s_parameters()[:, 1, 1])**2

# Let's view the results
import matplotlib.pyplot as plt

# We'll plot the back reflections from the input port and view it in log scale.
plt.subplot(221)
s_out = 0
s_in = 0
plt.plot(simu.freq_array, np.log10(abs(simu.s_parameters()[:, s_out, s_in])**2))

# Next, let's see the result of putting light in the input and measuring the
# output, also in log scale.
plt.subplot(222)
s_out = 0
s_in = 1
plt.plot(simu.freq_array, np.log10(abs(simu.s_parameters()[:, s_out, s_in])**2))

# We don't have to view the results in log scale. We can look at it with
# just its magnitude.
plt.subplot(223)
s_out = 1
s_in = 0
plt.plot(simu.freq_array, abs(simu.s_parameters()[:, s_out, s_in])**2)

# Back reflections from the output to the output can also be viewed.
plt.subplot(224)
plt.plot(simu.freq_array, abs(simu.s_parameters()[:, 1, 1])**2)

plt.suptitle("MZI")
plt.tight_layout()
plt.show()