.. _example-filters:

Add-Drop Filters
================

Code Walkthrough
----------------

This example walks through the file "filters.py". 

.. literalinclude:: ../../../../examples/filters.py
    :lines: 1-6

For this tutorial, we will be using matplotlib and numpy to manipulate and
view the results of our simulation. 

.. literalinclude:: ../../../../examples/filters.py
    :lines: 12-14

We'll need the following modules and objects from simphony:

* **simphony.library.ebeam**:
  The SiEPIC EBeam PDK model library.

* **simphony.netlist.Subcircuit**:
  We use the Subcircuit object to define our photonic circuits.

.. literalinclude:: ../../../../examples/filters.py
    :lines: 16-19

This is a block-diagram model of our final construction.

Note the main data line and terminators. We can declare the models we'll use
for those circuit instances.

.. literalinclude:: ../../../../examples/filters.py
    :lines: 21-25

Our final product has a component that is duplicated three times with varying
parameters. This kind of redundancy makes an excellent case for the use of the
:ref:`factory method design pattern<factory-method-design-pattern>`.

.. literalinclude:: ../../../../examples/filters.py
    :lines: 27-65

Before we construct the full add-drop filter, we can run a simulation on a 
single ring to make sure our code is behaving the way we'd expect.

.. literalinclude:: ../../../../examples/filters.py
    :lines: 67-76

Now we'll cascade several of these ring resonators together to create our 
filter.

::

    # Now, we'll create the circuit (using several ring resonator subcircuits)
    # and add all individual instances.
    circuit = Subcircuit('Add-Drop Filter')
    e = circuit.add([
        (wg_data, 'input'),
        (ring_factory(10), 'ring10'),
        (wg_data, 'out1'),

        (wg_data, 'connect1'),
        (ring_factory(11), 'ring11'),
        (wg_data, 'out2'),

        (wg_data, 'connect2'),
        (ring_factory(12), 'ring12'),
        (wg_data, 'out3'),

        (term, 'terminator')
    ])

We can set the pin names individually or simulateously.

::

    # You can set pin names individually (here I'm naming all the outputs that
    # I'll want to access before they get scrambled and associated with different
    # elements):
    circuit.elements['input'].pins['n1'] = 'input'
    circuit.elements['out1'].pins['n2'] = 'out1'
    circuit.elements['out2'].pins['n2'] = 'out2'
    circuit.elements['out3'].pins['n2'] = 'out3'

Connect the circuit:

::

    circuit.connect_many([
        ('input', 'n2', 'ring10', 'in'),
        ('out1', 'n1', 'ring10', 'out'),
        ('connect1', 'n1', 'ring10', 'pass'),

        ('connect1', 'n2', 'ring11', 'in'),
        ('out2', 'n1', 'ring11', 'out'),
        ('connect2', 'n1', 'ring11', 'pass'),

        ('connect2', 'n2', 'ring12', 'in'),
        ('out3', 'n1', 'ring12', 'out'),
        ('terminator', 'n1', 'ring12', 'pass'),
    ])

Run a sweep simulation.

::

    # Run a simulation on the netlist.
    simulation = SweepSimulation(circuit, 1524.5e-9, 1551.15e-9)
    result = simulation.simulate()

    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(1, 3)

    ax = fig.add_subplot(gs[0, :2])
    f, s = result.data(result.pinlist['input'], result.pinlist['out1'])
    ax.plot(freq2wl(f)*1e9, s, label='Output 1', lw='0.7')
    f, s = result.data(result.pinlist['input'], result.pinlist['out2'])
    ax.plot(freq2wl(f)*1e9, s, label='Output 2', lw='0.7')
    f, s = result.data(result.pinlist['input'], result.pinlist['out3'])
    ax.plot(freq2wl(f)*1e9, s, label='Output 3', lw='0.7')

    ax.set_ylabel("Fractional Optical Power")
    ax.set_xlabel("Wavelength (nm)")
    plt.legend(loc='upper right')

    ax = fig.add_subplot(gs[0, 2])
    f, s = result.data(result.pinlist['input'], result.pinlist['out1'])
    ax.plot(freq2wl(f)*1e9, s, label='Output 1', lw='0.7')
    f, s = result.data(result.pinlist['input'], result.pinlist['out2'])
    ax.plot(freq2wl(f)*1e9, s, label='Output 2', lw='0.7')
    f, s = result.data(result.pinlist['input'], result.pinlist['out3'])
    ax.plot(freq2wl(f)*1e9, s, label='Output 3', lw='0.7')

    ax.set_xlim(1543,1545)
    ax.set_ylabel("Fractional Optical Power")
    ax.set_xlabel("Wavelength (nm)")

    fig.align_labels()
    plt.show()

And voila!


Full Code Listing
-----------------

.. literalinclude:: ../../../../examples/filters.py



.. Example Rendered
.. ================

.. .. ifconfig:: python_version_major < '3'

..     The example is rendered only when sphinx is run with python3 and above

.. .. automodule:: doc.example
..     :members: