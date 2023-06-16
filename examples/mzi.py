# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

import matplotlib.pyplot as plt

from simphony.libraries.siepic import GratingCoupler, YBranch, Waveguide
from simphony.simulation import Detector, Laser, Simulation
from simphony.circuit import Circuit

# first we initialize all of the components in the MZI circuit
with Circuit() as circuit:
    gc_input = circuit.add(GratingCoupler())
    circuit.connect(gc_input.o(1), y_splitter := YBranch())
    circuit.connect(y_splitter.o(1), wg_long := Waveguide(length=150e-6))
    circuit.connect(y_splitter.o(2), wg_short := Waveguide(length=50e-6))
    circuit.connect(wg_long.o(1), y_recombiner := YBranch())
    circuit.connect(gc_output := GratingCoupler())

with Circuit() as cir:
    gc_input = cir.add(GratingCoupler())
    y_splitter = cir.add(YBranch())
    cir.connect(gc_input.o(1), y_splitter.o(0))

    wg_long = cir.add(Waveguide(length=150e-6))
    cir.connect(y_splitter.o(1), wg_long)

    cir.connect(y_splitter.o(2), wg_short := Waveguide(length=50e-6))
    cir.connect(wg_long.o(1), y_recombiner := YBranch())
    cir.connect(gc_output := GratingCoupler())

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

# do a simple sweep simulation
theoretical = None
with Simulation() as sim:
    l = Laser(power=20e-3)
    l.wlsweep(1500e-9, 1600e-9)
    l.connect(gc_input)
    Detector().connect(gc_output)

    theoretical = sim.sample()

plt.plot(sim.freqs, theoretical[:, 0, 0])
plt.title("MZI")
plt.tight_layout()
plt.show()

# if we specify multiple samples, noise gets added to the simulation
with Simulation() as sim:
    l = Laser(power=20e-3)
    l.wlsweep(1500e-9, 1600e-9)
    l.connect(gc_input)
    Detector().connect(gc_output)

    # we get 101 samples even though we only use 3 because
    # filtering requires at least 21 samples and the results
    # get better with more samples and 101 isn't much slower
    # than 21
    noisy = sim.sample(101)

plt.plot(sim.freqs, noisy[:, 0, 0], label="Noisy 1")
plt.plot(sim.freqs, noisy[:, 0, 1], label="Noisy 2")
plt.plot(sim.freqs, noisy[:, 0, 2], label="Noisy 3")
plt.plot(sim.freqs, theoretical[:, 0, 0], "k", label="Theoretical")
plt.legend()
plt.title("MZI")
plt.tight_layout()
plt.show()

# do some monte carlo simulations
for n in range(10):
    print(f"Monte Carlo Run {n}")
    # note that after each run, we have to regenerate the MC parameters
    for component in gc_input.circuit:
        component.regenerate_monte_carlo_parameters()

    with Simulation() as sim:
        l = Laser(power=20e-3)
        l.wlsweep(1500e-9, 1600e-9)
        l.connect(gc_input)
        Detector().connect(gc_output)

        sim.monte_carlo(True)
        d = sim.sample()

        plt.plot(sim.freqs, d[:, 0, 0], label=f"Run {n}")

plt.plot(sim.freqs, theoretical[:, 0, 0], "k", label="Theoretical")
plt.title("MZI Monte Carlo")
plt.tight_layout()
plt.show()
