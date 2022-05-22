from matplotlib.pyplot import legend, plot, show, xlabel, ylabel
from numpy import log10, max
import numpy as np
import time
from ..libraries import siepic
from ..simulators import LayoutAwareMonteCarloSweepSimulator

wg1 = siepic.Waveguide(length=50e-6, width=500e-9, height=220e-9)
wg2 = siepic.Waveguide(length=50.2642e-6, width=500e-9, height=220e-9)
wg3 = siepic.Waveguide(length=50e-6, width=500e-9, height=220e-9)
wg4 = siepic.Waveguide(length=150.323e-6, width=500e-9, height=220e-9)
wg5 = siepic.Waveguide(length=50e-6, width=500e-9, height=220e-9)
wg6 = siepic.Waveguide(length=50.1057e-6, width=500e-9, height=220e-9)
wg7 = siepic.Waveguide(length=50e-6, width=500e-9, height=220e-9)
wg8 = siepic.Waveguide(length=100e-6, width=500e-9, height=220e-9)

dc1 = siepic.DirectionalCoupler()
dc2 = siepic.DirectionalCoupler()
dc3 = siepic.DirectionalCoupler()
dc4 = siepic.DirectionalCoupler()
dc5 = siepic.DirectionalCoupler()

components = (dc1, wg1, wg2, dc2, wg3, wg4, dc3, wg5, wg6, dc4, wg7, wg8, dc5)

dc1["pin3"].connect(wg1)
dc1["pin4"].connect(wg2)
dc2["pin1"].connect(wg1)
dc2["pin2"].connect(wg2)
dc2["pin3"].connect(wg3)
dc2["pin4"].connect(wg4)
dc3["pin1"].connect(wg3)
dc3["pin2"].connect(wg4)
dc3["pin3"].connect(wg5)
dc3["pin4"].connect(wg6)
dc4["pin1"].connect(wg5)
dc4["pin2"].connect(wg6)
dc4["pin3"].connect(wg7)
dc4["pin4"].connect(wg8)
dc5["pin1"].connect(wg7)
dc5["pin2"].connect(wg8)

x = np.array([-0.000214895, -0.000169565, -1.69565e-04, -1.24235e-04, -7.89050e-05, -7.89050e-05, -3.35750e-05, 1.17550e-05, 1.17550e-05, 5.70850e-05, 1.02415e-04, 1.02415e-04, 1.47745e-04])
y = np.array([-4.574e-05, -3.4405e-05, -5.7009e-05, -4.574e-05, -3.4445e-05, -5.7009e-05, -4.574e-05, -7.647e-05, -4.0092e-05, -4.574e-05, -5.7009e-05, -2.1971e-05, -4.574e-05])

simulator = LayoutAwareMonteCarloSweepSimulator()
simulator.multiconnect(dc1["pin1"], dc5["pin3"])

start = time.time()
results = simulator.simulate(x=x, y=y, sigmaw=5, sigmat=2, l=4.5e-3, runs=10)
end = time.time()
print(f'{end-start}')
g = []

for wl, s in results:
    g.append(10*log10(s))

for i, _g in enumerate(g):
    plot(wl, _g, label=f'{i}')
    print(max(_g))
legend()
show()

# g = np.asarray(g)
# wl = np.asarray(wl)
# np.savetxt("Gain_1_simphony_test2.txt", g)
# np.savetxt("wl.txt", wl)

# print(max(g))

max = []
# g = np.reshape(g, (2000,10))
for i in range(len(g)):
    max.append(np.max(g[i][:]))
    print(np.max(g[i][:]))
x = np.sort(max)

y = 1. * np.arange(len(max)) / (len(max) - 1)

plot(x, y, label="Simphony")

data = np.loadtxt("C:\\Users\\12269\\Downloads\\Gain_1_VW.txt")
data = data[4000:]
data = np.reshape(data, (2000,1000))
max = []

for i in range(data.shape[1]):
    max.append(np.max(data[:][i]))
x = np.sort(max)

y = 1. * np.arange(len(max)) / (len(max) - 1)

plot(x, y, label="Lumerical")
xlabel("max. Transmission (dB)")
ylabel("Cumulative Probability")
legend()
show()
