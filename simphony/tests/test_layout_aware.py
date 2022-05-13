from matplotlib.pyplot import legend, plot, show
from numpy import log10
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

components = (wg1, wg2, wg3, wg4, wg5, wg6, wg7, wg8, dc1, dc2, dc3, dc4, dc5)

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

x = np.array([-2.14895e-04, -1.69565e-04, -1.69565e-04, -1.24235e-04, -7.89050e-05, -7.89050e-05, -3.35750e-05, 1.17550e-05, 1.17550e-05, 5.70850e-05, 1.02415e-04, 1.02415e-04, 1.47745e-04])
y = np.array([-4.574e-05, -3.4405e-05, -5.7009e-05, -4.574e-05, -3.4445e-05, -5.7009e-05, -4.574e-05, -7.647e-05, -4.0092e-05, -4.574e-05, -5.7009e-05, -2.1971e-05, -4.574e-05])

simulator = LayoutAwareMonteCarloSweepSimulator()
simulator.circuit = dc1.circuit
simulator.multiconnect(dc1["pin1"], dc5["pin3"])

start = time.time()
results = simulator.simulate(x=x, y=y, sigmaw=5, sigmat=3, l=4.5e-3, runs=1000)
end = time.time()
print(f'{end-start}')
g = []

for wl, s in results:
    g.append(10 * log10(s))

for i, _g in enumerate(g):
    plot(wl, _g, label=f'{i}')
legend()
show()

g = np.asarray(g)
wl = np.asarray(wl)
np.savetxt("Gain_1_simphony.txt", g)
np.savetxt("wl.txt", wl)
