.. _example-layout_aware:

Layout-Aware Monte Carlo Simulations for Yield Estimation
=========================================================

Manufacturing variability can cause fabrication errors in waveguide width and thickness,
which can affect the device performance. Hence, incorporating manufacturing variability 
into the photonic device design process is crucial. Simphony provides the ability to run
layout-aware Monte Carlo simulations for yield estimation to aid in robust photonic device design.
This tutorial will walk you through the codes found in ``examples/layout_aware.py`` of the Simphony repository. 
We expect you to have read the previous tutorial, :doc:`mzi`.


The Workflow
------------
The workflow for running layout-aware Monte Carlo simulations is as follows:

1. instantiate the components
2. generate a layout using the components' ``component`` attributes
3. connect the components to form a circuit
4. run the simulation
5. extract and plot the results

Example - Mach-Zehnder Interferometer (MZI)
-------------------------------------------
We will run a layout-aware Monte Carlo simulation for the
Mach-Zehnder Interferometer (MZI) described in the previous
example.

First we need to import the necessary Simphony modules. We will need the ``siepic`` model library,
the ``Simulation``, ``Laser``, and ``Detector``. We will also need to import ``gdsfactory`` to generate the layout.
We will also import ``matplotlib.pyplot``, from outside of Simphony, to view the results
of our simulation.

.. code-block:: python

   import gdsfactory as gf
   import matplotlib.pyplot as plt
   import numpy as np

   from simphony.libraries import siepic
   from simphony.simulation import Detector, Laser, Simulation

We then create all the components and give them names. These
include the grating couplers, the Y-branches, and the
waveguides (which can be defined at any arbitrary length,
on the condition that the two lengths are different).

.. code-block:: python

   gc_input = siepic.GratingCoupler(name="gcinput")
   y_splitter = siepic.YBranch(name="ysplit")
   wg_long = siepic.Waveguide(length=150e-6, name="wglong")
   wg_short = siepic.Waveguide(length=50e-6, name="wgshort")
   y_recombiner = siepic.YBranch(name="y_recombiner")
   gc_output = siepic.GratingCoupler(name="gcoutput")

We then use the components' ``component`` attributes to create the layout.
The ``component`` attributes are ``gdsfactory.Component`` objects, and so can be used to create a 
layout. We will next define a Parametric Cell (PCell) for the MZI. We will connect
the components to form a circuit, route the Waveguides using ``gdsfactory's`` routing functions.

.. code-block:: python

    @gf.cell
    def mzi():
        c = gf.Component("mzi")

        ysplit = c << y_splitter.component

        gcin = c << gc_input.component

        gcout = c << gc_output.component

        yrecomb = c << y_recombiner.component

        yrecomb.move(destination=(0, -55.5))
        gcout.move(destination=(-20.4, -55.5))
        gcin.move(destination=(-20.4, 0))

        gc_input["pin1"].connect(y_splitter, gcin, ysplit)
        gc_output["pin1"].connect(y_recombiner["pin1"], gcout, yrecomb)
        y_splitter["pin2"].connect(wg_long)
        y_recombiner["pin3"].connect(wg_long)
        y_splitter["pin3"].connect(wg_short)
        y_recombiner["pin2"].connect(wg_short)

        wg_long_ref = gf.routing.get_route_from_steps(
            ysplit.ports["pin2"],
            yrecomb.ports["pin3"],
            steps=[{"dx": 91.75 / 2}, {"dy": -61}],
        )
        wg_short_ref = gf.routing.get_route_from_steps(
            ysplit.ports["pin3"],
            yrecomb.ports["pin2"],
            steps=[{"dx": 47.25 / 2}, {"dy": -50}],
        )

        wg_long.path = wg_long_ref
        wg_short.path = wg_short_ref

        c.add(wg_long_ref.references)
        c.add(wg_short_ref.references)

        c.add_port("o1", port=gcin.ports["pin2"])
        c.add_port("o2", port=gcout.ports["pin2"])

        return c
 
We can then call the function, and visualize the layout in KLayout.

.. code-block:: python
  
    c = mzi()
    c.show()

.. image:: /_static/images/mzi_layout_aware.png
    :alt: layout
    :align: center

We can also view a 3D representation of the layout.

.. code-block:: python

    c.to_3d().show("gl")


Layout-Aware Monte Carlo Simulation
-----------------------------------
We use the ``Simulation`` class to run a simulation. We attach a ``Laser`` to one of the GratingCouplers,
and a ``Detector`` to the other GratingCoupler.

.. code-block:: python

    with Simulation() as sim:
        l = Laser(power=1)
        l.freqsweep(187370000000000.0, 199862000000000.0)
        l.connect(gc_input['pin2'])
        d = Detector()
        d.connect(gc_output['pin2'])

        results = sim.layout_aware_simulation(c)

Here, we can pass in the standard deviations of the widths and thicknesses, as well as the correlation length
as arguments to the ``layout_aware_simulation`` method. For this example, we use the default values.

After the simulation is run, we can extract the results, and plot them. We will see several, slightly different curves
due to random variations incorporated into the components' widths and thicknesses.

.. code-block:: python

    f = l.freqs
    for run in results:
      p = []
      for sample in run:
          for data_list in sample:
              for data in data_list:
                  p.append(data)
      plt.plot(f, p)

    run_0 = results[0]
    p = []
    for sample in run_0:
        for data_list in sample:
            for data in data_list:
                p.append(data)
    plt.plot(f, p, 'k')
    plt.title('MZI Layout Aware Monte Carlo')
    plt.show()

You should see something similar to this graph when you run
your MZI now:

.. image:: /_static/images/layout_aware.png
    :alt: layout-aware simulation
    :align: center

From our data, we can then compute various performance markers which are sensitive
to width and thickness variations.
