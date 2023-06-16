.. _example-filters:


Add-Drop Filters
================
In this tutorial, we are designing a circuit called an
Add-Drop Filter, with a single input and multiple outputs.
We'll walk through the code in ``examples/filters.py`` in
the Simphony repo, and we expect you to have already
completed the previous tutorial: :doc:`mzi`.


Deconstruction
--------------
An add-drop filter uses rings of different radii to select
specific frequencies from an input waveguide and convey them
to an output.

.. figure:: /_static/images/filters.png
    :alt: Add-Drop Filter
    :align: center

    A sample Add-Drop Filter. The rings all have differing
    radii.

Light travels through the input waveguide, and some
frequencies carry over to the ring waveguides, depending on
the radius of the ring. These signals move along the ring
until they transfer over to the output waveguides, giving us
a reading on what frequencies of light traveled through the
input. Light is designed to travel only in one direction
after reaching the output waveguides, but we must account
for backwards scattering light. We simply add a terminator
at the other end of the output waveguides to diffuse any
such light.

Notice how the Add-Drop Filter is composed of three similar
rings, differing only by their radius:

.. figure:: /_static/images/ring.png
    :align: center

    An isolated, single ring resonator.

This single ring resonator can be defined using models from
both SiEPIC and SiPANN libraries in Simphony. Instead of
defining each model for each ring resonator sequentially,
we can use what we call the "factory" design pattern: we 
will create a method that defines a ring resonator for us.


Factory Design Pattern
----------------------
First, we need to import the libraries we need. The SiEPIC
library, the sweep simulator, and matplotlib will be used,
just as last tutorial. In addition, we need the SiPANN 
library. This library of models is not included by default
in Simphony, but it integrates well. You will need to 
install it as shown in the `SiPANN docs`_.

.. code-block:: python

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    from simphony.libraries import siepic, sipann
    from simphony.simulators import SweepSimulator

We now create the method that generates a ring resonator for
us. We pass in the radius as a parameter, and we are
returned a subcircuit, which can be used much the same way a
component can.

.. code-block:: python

    def ring_factory(radius):
      """Creates a full ring (with terminator) from a half ring.

      Resulting pins are ('pass', 'in', 'out').

      Parameters
      ----------
      radius : float
          The radius of the ring resonator, in meters.
      """
      # Have rings for selecting out frequencies from the data line.
      # See SiPANN's model API for argument order and units.
      halfring1 = sipann.HalfRing(500e-9, 220e-9, radius, 100e-9)
      halfring2 = sipann.HalfRing(500e-9, 220e-9, radius, 100e-9)
      terminator = siepic.Terminator()

      halfring1.rename_pins("pass", "midb", "in", "midt")
      halfring2.rename_pins("out", "midt", "term", "midb")

      # the interface method will connect all of the pins with matching names
      # between the two components together
      halfring1.interface(halfring2)
      halfring2["term"].connect(terminator)

      # bundling the circuit as a Subcircuit allows us to interact with it
      # as if it were a component
      return halfring1.circuit.to_subcircuit()

.. note::

    In this method, we just demonstrated two new abilities of
    Simphony that will be of interest to you. First is the
    ``interface`` method of a component, another way of
    connecting components together conveniently. Second is the
    ``to_subcircuit`` method. From one of our components, we
    can get its ``circuit``, which includes all components 
    directly or indirectly connected to that first component.
    We transform that circuit into a Simphony Subcircuit,
    which behaves similarly to a single component.

Before we construct the full Add-Drop Filter, we can run a
simulation on a single ring to make sure everything is
behaving as expected.

.. code-block:: python

    ring1 = ring_factory(10e-6)

    simulator = SweepSimulator(1500e-9, 1600e-9)
    simulator.multiconnect(ring1["in"], ring1["pass"])

    f, t = simulator.simulate(mode="freq")
    plt.plot(f, t)
    plt.title("10-micron Ring Resonator")
    plt.tight_layout()
    plt.show()

    simulator.disconnect()

When you run your python file up to this point, you should
see a graph similar to this:

.. figure:: /_static/images/10um_ring_res.png
    :align: center

    The through-port frequency response of a 10 micron ring
    resonator.

Now that we've created and tested our ``ring_factory``
method, we can use it to define the Add-Drop Filter.


Defining the Circuit
--------------------
Let's create the components we'll use in the circuit:

.. code-block:: python

    wg_input = siepic.Waveguide(100e-6)
    wg_out1 = siepic.Waveguide(100e-6)
    wg_connect1 = siepic.Waveguide(100e-6)
    wg_out2 = siepic.Waveguide(100e-6)
    wg_connect2 = siepic.Waveguide(100e-6)
    wg_out3 = siepic.Waveguide(100e-6)
    terminator = siepic.Terminator()

    ring1 = ring_factory(10e-6)
    ring2 = ring_factory(11e-6)
    ring3 = ring_factory(12e-6)

And then connect each component as seen in the diagram:

.. code-block:: python

    ring1.multiconnect(wg_connect1, wg_input["pin2"], wg_out1)
    ring2.multiconnect(wg_connect2, wg_connect1, wg_out2)
    ring3.multiconnect(terminator, wg_connect2, wg_out3)

Now we're ready to simulate.


Simulation
----------

We'll run a sweep simulation, but we're reducing the
frequency range to 1524.5-1551.15 nm, instead of a full 
1500-1600 nm sweep as we have done previously. This will 
show us a simpler graph of only a few peaks that the filter
picks out. We'll be using more advanced matplotlib features 
here, reference the `matplotlib docs`_ on these.

Let's prepare the graph and the simulator to perform
simulation:

.. code-block:: python

    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(1, 3)
    ax = fig.add_subplot(gs[0, :2])

    simulator = SweepSimulator(1524.5e-9, 1551.15e-9)
    simulator.connect(wg_input)

Next we simulate each output, and draw a curve for each.

.. code-block:: python

    # get the results for output 1
    simulator.multiconnect(None, wg_out1)
    wl, t = simulator.simulate()
    ax.plot(wl * 1e9, t, label="Output 1", lw="0.7")

    # get the results for output 2
    simulator.multiconnect(None, wg_out2)
    wl, t = simulator.simulate()
    ax.plot(wl * 1e9, t, label="Output 2", lw="0.7")

    # get the results for output 3
    simulator.multiconnect(None, wg_out3)
    wl, t = simulator.simulate()
    ax.plot(wl * 1e9, t, label="Output 3", lw="0.7")

Then we label our plot.

.. code-block:: python

    ax.set_ylabel("Fractional Optical Power")
    ax.set_xlabel("Wavelength (nm)")
    plt.legend(loc="upper right")

We could stop here and have a perfectly good plot, but you
will notice that one of the peaks will be very small and
will be hard to see clearly on this graph. To fix this,
we'll add a subplot to our graph to magnify the frequency
range of this peak, then simulate and draw each of our
outputs on this subplot again.

.. code-block:: python

    ax = fig.add_subplot(gs[0, 2])

    # get the results for output 1
    simulator.multiconnect(None, wg_out1)
    wl, t = simulator.simulate()
    ax.plot(wl * 1e9, t, label="Output 1", lw="0.7")

    # get the results for output 2
    simulator.multiconnect(None, wg_out2)
    wl, t = simulator.simulate()
    ax.plot(wl * 1e9, t, label="Output 2", lw="0.7")

    # get the results for output 3
    simulator.multiconnect(None, wg_out3)
    wl, t = simulator.simulate()
    ax.plot(wl * 1e9, t, label="Output 3", lw="0.7")

    ax.set_xlim(1543, 1545)
    ax.set_ylabel("Fractional Optical Power")
    ax.set_xlabel("Wavelength (nm)")
    fig.align_labels()

Finally, we show our plot.

.. code-block:: python

    plt.show()

What you should see when you run your Add-Drop circuit is
something like this:

.. figure:: /_static/images/add_drop_response.png
    :align: center

    The response of our designed add-drop filter.

And with that, this tutorial is concluded. For now, this is
the last tutorial in the series for learning Simphony. We
plan to write more for this series in future, but we hope
that this has sufficiently demonstrated the capabilities of
Simphony to you. If you wish, you may see the references
section to dive into the API for Simphony.

.. _SiPANN docs: https://sipann.readthedocs.io/en/latest/
.. _matplotlib docs: https://matplotlib.org/
