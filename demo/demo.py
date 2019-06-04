import simphony
import matplotlib.pyplot as plt

model = simphony.netlist.ObjectModelNetlist.load('output.json')
sim = simphony.simulation.Simulation(model)

x, y = sim.getMagnitudeByFrequencyTHz(2, 3)
plt.plot(x, y)
plt.show()

