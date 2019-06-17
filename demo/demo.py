# import simphony
# import matplotlib.pyplot as plt

# model = simphony.netlist.ObjectModelNetlist.load('demo/output.json')
# sim = simphony.simulation.Simulation(model)

# x, y = sim.getMagnitudeByFrequencyTHz(2, 3)
# plt.plot(x, y)
# plt.show()

# multi = simphony.simulation.MultiInputSimulation(model)
# multi.multi_input_simulation(inputs=[2])

import simphony

ring_resonator = simphony.core.Component("ring_resonator_radius_5", [1500,1550,1600], [0,0,0], True)
waveguide = simphony.core.Component("waveguide_length_10", [1500,1550,1600], [0,0,0], False)