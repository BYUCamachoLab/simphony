import simphony
import matplotlib.pyplot as plt

model = simphony.netlist.ObjectModelNetlist.load('demo/output.json')
sim = simphony.simulation.Simulation(model)

x, y = sim.getMagnitudeByFrequencyTHz(2, 3)
plt.plot(x, y)
plt.show()

multi = simphony.simulation.MultiInputSimulation(model)
multi.multi_input_simulation(inputs=[2])