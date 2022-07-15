# Sometimes, one might wish to run the layout aware simulation
# on a circuit from a netlist file. For this purpose,
# Simphony offers support for netlist co-ordinates.
# The following code snippet demonstrates how to use this.

import matplotlib.pyplot as plt

from simphony.die import Die
from simphony.formatters import CircuitSiEPICFormatter
from simphony.layout import Circuit
from simphony.libraries import siepic
from simphony.simulation import Detector, Laser, Simulation

# we import a circuit from a netlist file
circuit = Circuit.from_file("/path/to/spi/file.spi", formatter=CircuitSiEPICFormatter())


# we rename the components in such a way that Simphony's
# layout aware simulation will be able to recognize them
for component in circuit._get_components():
    if isinstance(component, siepic.Waveguide):
        component.device.name = f"wg_{component}"
    else:
        component.device.name = f"{component}"
    component.name = f"{component}"

# we instantiate a Die object
die = Die(name="die1")

# then, we throw in the components into the Die
die.add_components(circuit._get_components())

# Run the layout aware monte carlo computation

with Simulation() as sim:
    l = Laser(power=1)
    l.freqsweep(187370000000000.0, 199862000000000.0)
    l.connect(circuit._get_components()[1].pins[0])
    d = Detector()
    d.connect(circuit._get_components()[5].pins[0])

    results = sim.layout_aware_simulation()

# Plot the results

f = l.freqs
for run in results:
    p = []
    for sample in run:
        for data_list in sample:
            for data in data_list:
                p.append(data)
    plt.plot(f, p)

run = results[0]
p = []
for sample in run:
    for data_list in sample:
        for data in data_list:
            p.append(data)
plt.plot(f, p, "k")
plt.title("MZI Layout Aware Monte Carlo")
plt.show()
