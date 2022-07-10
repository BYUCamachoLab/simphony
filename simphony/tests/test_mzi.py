from time import time

import matplotlib.pyplot as plt
import numpy as np
from numpy import log10

from simphony.libraries import siepic
from simphony.simulation import Detector, Laser, Simulation

gc_input = siepic.GratingCoupler(name="gc_input")
y_splitter = siepic.YBranch(name="y_splitter")
wg_long = siepic.Waveguide(name="wg_long", length=150e-6)
wg_short = siepic.Waveguide(name="wg_short", length=50e-6)
y_recombiner = siepic.YBranch(name="y_recombiner")
gc_output = siepic.GratingCoupler(name="gc_output")

# next we connect the components to each other
# you can connect pins directly:
y_splitter["pin1"].connect(gc_input["pin1"])

# or connect components with components:
# (when using components to make connections, their first unconnected pin will
# be used to make the connection.)
y_splitter.connect(wg_long)

# or any combination of the two:
y_splitter["pin3"].connect(wg_short)
# y_splitter.connect(wg_short["pin1"])

# when making multiple connections, it is often simpler to use `multiconnect`
# multiconnect accepts components, pins, and None
# if None is passed in, the corresponding pin is skipped
y_recombiner.multiconnect(gc_output, wg_short, wg_long)

results = []
with Simulation(fs=10e9, seed=117) as simulator:
    laser = Laser(power=1e-3, wl=1550e-9)
    laser.freqsweep(187370000000000.0, 199862000000000.0)
    laser.connect(gc_input)
    Detector().connect(gc_output)

    start = time()
    data = simulator.layout_aware_simulation(runs=10, num_samples=1)
    end = time()
    print(start - end)
    results = data

# for f, p in results:
# plt.plot(f, p)

g = []
f = np.linspace(187370000000000.0, 199862000000000.0, 500)
for p in results:
    p = 10 * np.log10(p)
    _p = []
    for val1 in p:
        for val2 in val1:
            _p.append(val2)
    plt.plot(f, _p)
    # print(np.max(p))
    g.append(_p)

plt.title("MZI Monte Carlo")
plt.tight_layout()
plt.show()

max = []
for i in range(len(g)):
    max.append(np.max(g[i][:]))
    # print(np.max(g[i][:]))
x = np.sort(max)

y = 1.0 * np.arange(len(max)) / (len(max) - 1)

plt.plot(x, y, label="Simphony")

data = np.loadtxt("C:\\Users\\12269\\Downloads\\mzi_Gain_VW.txt")
data = data[:2000000]
data = np.reshape(data, (2000, 1000))
max = []

for i in range(data.shape[1]):
    max.append(np.max(data[:][i]))
x = np.sort(max)

y = 1.0 * np.arange(len(max)) / (len(max) - 1)

plt.plot(x, y, label="Lumerical")
plt.xlabel("max. Transmission (dB)")
plt.ylabel("Cumulative Probability")
plt.legend()
plt.show()
