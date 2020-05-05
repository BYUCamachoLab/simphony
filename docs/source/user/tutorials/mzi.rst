.. _example-mzi:

Mach-Zehnder Interferometer
===========================

Code Walkthrough
----------------

.. ipython:: python

    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    #
    # Copyright © Simphony Project Contributors
    # Licensed under the terms of the MIT License
    # (see simphony/__init__.py for details)

For this tutorial, we will be using matplotlib and numpy to manipulate and
view the results of our simulation.

.. ipython:: python

    import matplotlib.pyplot as plt
    import numpy as np

We'll need the following modules and objects from simphony:

* **simphony.library.ebeam**:
  The SiEPIC EBeam PDK model library.

* **simphony.netlist.Subcircuit**:
  We use the Subcircuit object to define our photonic circuits.

.. ipython:: python
    
    from simphony.library import ebeam
    from simphony.netlist import Subcircuit
    from simphony.simulation import SweepSimulation, MonteCarloSweepSimulation

The MZI we'll create uses only a few simple models.

.. ipython:: python

    # Declare the models used in the circuit
    gc = ebeam.ebeam_gc_te1550()
    y = ebeam.ebeam_y_1550()
    wg150 = ebeam.ebeam_wg_integral_1550(length=150e-6)
    wg50 = ebeam.ebeam_wg_integral_1550(length=50e-6)

We'll add all the components into a circuit without worrying about the 
connections for now.

.. ipython:: python

    # Create the circuit, add all individual instances
    circuit = Subcircuit('MZI')
    e = circuit.add([
        (gc, 'input'),
        (gc, 'output'),
        (y, 'splitter'),
        (y, 'recombiner'),
        (wg150, 'wg_long'),
        (wg50, 'wg_short'),
    ])

For ease of making connections, we rename some of the ports. Pin names can be
individually:

.. ipython:: python

    circuit.elements['input'].pins['n2'] = 'input'
    circuit.elements['output'].pins['n2'] = 'output'

or simultaneously:

.. ipython:: python

    circuit.elements['splitter'].pins = ('in1', 'out1', 'out2')
    circuit.elements['recombiner'].pins = ('out1', 'in2', 'in1')

Next we define the circuit's connections.

.. ipython:: python

    circuit.connect_many([
        ('input', 'n1', 'splitter', 'in1'),
        ('splitter', 'out1', 'wg_long', 'n1'),
        ('splitter', 'out2', 'wg_short', 'n1'),
        ('recombiner', 'in1', 'wg_long', 'n2'),
        ('recombiner', 'in2', 'wg_short', 'n2'),
        ('output', 'n1', 'recombiner', 'out1'),
    ])

At this point, your circuit is defined. This file can serve as a description
for a subcircuit that is used in a larger circuit, and can simply be imported
using the Python import system (`from <> import circuit`).

We can run a simulation on our fully-defined circuit.

.. ipython:: python

    simulation = SweepSimulation(circuit, 1500e-9, 1600e-9)
    result = simulation.simulate()

Plot the data. We can access the pins using their string names.

.. ipython:: python

    f, s = result.data('input', 'output')
    @savefig plot_mzi.png width=6in
    plt.plot(f, s);

We can even run a monte carlo simulation.

.. ipython:: python

    simulation = MonteCarloSweepSimulation(circuit, 1500e-9, 1600e-9)
    runs = 10
    result = simulation.simulate(runs=runs)
    for i in range(1, runs + 1):
        f, s = result.data('input', 'output', i)
        plt.plot(f, s)
    
    f, s = result.data('input', 'output', 0)
    plt.plot(f, s, 'k')
    @savefig plot_mzi_mc2.png width=6in
    plt.title("MZI Monte Carlo")

The data stored at the 0th index, and plotted on top in black, is the ideal
values.


Full Code Listing
-----------------

.. literalinclude:: ../../../../examples/mzi.py