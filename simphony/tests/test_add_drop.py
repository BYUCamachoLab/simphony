import matplotlib.pyplot as plt
import numpy as np
import time

from simphony.libraries import siepic
from ..simulators import LayoutAwareMonteCarloSweepSimulator


gc_input = siepic.GratingCoupler(name="gc_input")
y_splitter = siepic.YBranch(name="y_splitter")
wg_long = siepic.Waveguide(width=5e-7, height=2.2e-7, length=150e-6, name="wg_long")
wg_short = siepic.Waveguide(width=5e-7, height=2.2e-7, length=50e-6, name="wg_short")
wg_short = siepic.Waveguide(width=5e-7, height=2.2e-7, length=50e-6, name="wg_short")
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

simulator = LayoutAwareMonteCarloSweepSimulator(1500e-9, 1600e-9)
simulator.multiconnect(gc_input, gc_output)

# x = [-1.736e-05, 6.54e-06, 1.8978e-05, 4.1228e-05, 5.63e-06, -1.827e-05]
# y = [7e-07, 7e-07, -1.791e-05, -1.791e-05, -3.652e-05, -3.652e-05]

coords = {
    y_splitter: {
        'x': -1.736e-05,
        'y': 7e-07
    },
    gc_input: {
        'x': 6.54e-06,
        'y': 7e-07
    },
    wg_short: {
        'x': 1.8978e-05,
        'y': -1.791e-05
    },
    wg_long: {
        'x': 4.1228e-05,
        'y': -1.791e-05
    },
    y_recombiner: {
        'x': 5.63e-06,
        'y': -3.652e-05
    },
    gc_output: {
        'x': -1.827e-05,
        'y': -3.652e-05
    }
}

start = time.time()
results = simulator.simulate(coords=coords, runs=1000)
end = time.time()
print(f'{start-end}')

g = []
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
for i in range(len(g)):
    max.append(np.max(g[i][:]))
    # print(np.max(g[i][:]))
x = np.sort(max)

y = 1. * np.arange(len(max)) / (len(max) - 1)

plt.plot(x, y, label="Simphony")

data = np.loadtxt("C:\\Users\\12269\\Downloads\\mzi_Gain_VW.txt")
data = data[:2000000]
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
np.savetxt("MZI_simphony_testsipann.txt", g)
