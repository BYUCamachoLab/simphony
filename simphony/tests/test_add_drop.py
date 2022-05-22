import matplotlib.pyplot as plt
import numpy as np
import time

from simphony.libraries import siepic
from ..simulators import LayoutAwareMonteCarloSweepSimulator


gc_input = siepic.GratingCoupler()
y_splitter = siepic.YBranch()
wg_long = siepic.Waveguide(length=150e-6)
wg_short = siepic.Waveguide(length=50e-6)
y_recombiner = siepic.YBranch()
gc_output = siepic.GratingCoupler()


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

simulator = LayoutAwareMonteCarloSweepSimulator(1500e-9, 1600e-9)
simulator.multiconnect(gc_input, gc_output)

x = [-1.736e-05, 6.54e-06, 1.8978e-05, 4.1228e-05, 5.63e-06, -1.827e-05]
y = [7e-07, 7e-07, -1.791e-05, -1.791e-05, -3.652e-05, -3.652e-05]
start = time.time()
results = simulator.simulate(x=x, y=y, runs=1000)
end = time.time()
print(f'{start-end}')

g = []
wl = np.loadtxt("wl.txt")
for f, p in results:
    p = 10*np.log10(p)
    plt.plot(f, p)
    # print(np.max(p))
    g.append(p)

f, p = results[0]
p = 10*np.log10(p)
plt.plot(f, p, "k")
plt.title("MZI Monte Carlo")
plt.tight_layout()
plt.show()

max = []
# g = np.reshape(g, (2000,10))
for i in range(len(g)):
    max.append(np.max(g[i][:]))
    # print(np.max(g[i][:]))
x = np.sort(max)

y = 1. * np.arange(len(max)) / (len(max) - 1)

plt.plot(x, y, label="Simphony")

data = np.loadtxt("C:\\Users\\12269\\Downloads\\mzi_Gain_VW.txt")
# data = data[4000:2004000]
data = np.reshape(data, (2000,1000))
max = []

for i in range(data.shape[1]):
    max.append(np.max(data[:][i]))
x = np.sort(max)

y = 1. * np.arange(len(max)) / (len(max) - 1)

plt.plot(x, y, label="Lumerical")
plt.xlabel("max. Transmission (dB)")
plt.ylabel("Cumulative Probability")
plt.legend()
plt.show()

g = np.asarray(g)
wl = np.asarray(wl)
np.savetxt("MZI_simphony_test4.txt", g)
np.savetxt("wl.txt", wl)