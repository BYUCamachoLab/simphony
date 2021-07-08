.. _introduction-to-simphony:

Introduction to Simphony
========================
Before we start this tutorial, you should know the basics of
Python. We expect you to have an Python environment set up,
with the simphony package installed.

Our goal with this tutorial is to define and simulate a
simple circuit. In Simphony, circuits are represented all in
a single Python file. We'll go through the typical objects
found in every circuit definition, in order.

.. Note::
  Simphony uses SPICE concepts--such as components, pins,
  and nets--to define circuits. This should make Simphony
  intuitive for all those familiar with SPICE, which is
  commonly used to define electronic circuits. 


Models
------
Models are basic, presimulated devices; you will connect a
these together as components to build your circuit.

A model has a number of ports for inputs and outputs, known
as 'pins' in Simphony, and a frequency of ranges the model
is valid over. Internally, it stores a set of scattering
parameters (s-parameters) for given frequencies, which are
used when simulating your circuit.

Here's a look at the Model parent class
:py:class:`simphony.elements.Model`.

.. autoclass:: simphony.elements.Model
  :noindex:
  :members:
  :inherited-members:

.. Note::
  A basic model has no ``__init__()`` function. It is only
  required if the model takes in parameters (width or
  length, for example) that  affect the scattering
  parameters.

All models in Simphony extend this parent class, but
redefine pins, frequencies and s-parameters to match the
device they represent.


Instantiating Models
--------------------
Before we can use a model in our circuit, we need to
instantiate it. When we instantiate a model we call the
resulting object a component--the difference being that
components can have state outside of what is defined by the
model it references.

Simphony includes a default library of models from the
`SiEPIC PDK`_ (developed at the University of British
Columbia). We might define a couple of models with the
following: ::

  from simphony.libraries import siepic
  component1 = siepic.Waveguide(length=500e-9)
  component2 = siepic.Waveguide(length=1500e-9)

These are both :py:class:`Waveguide` components. The model
has two pins and a valid frequency range. We pass in
parameters when instantiating them, so that ``component1``
will have differing s-parameters, and therefore differing
simulation results, than ``component2``.

.. Note::
  All measurements in Simphony should be in base SI units: 
  this means instead of nanometer measurements, we will pass
  in meter measurements when instantiating models.


Connecting Components
---------------------
The :py:class:`simphony.netlist.Pin` class is used as an
interface to connect two components in a circuit. As an end
user, you should rarely have to interact with pins
themselves; instead, there are component methods that will
handle connecting pins for you.

Using our previous two components to demonstrate, the
simplest way to connect pins is to do so implicitly, as
follows: ::

  component1.connect(component2)

This will connect the first unconnected pin on both
components. However, if we wanted the first pin of
``component1`` to be an input, and instead connect its
second pin to ``component2``, we would have to connect the
pins explicitly: ::

  component1['pin2'].connect(component2['pin1'])

By default, a model instantiates its pins with names 'pin1', 
'pin2', etc. Here we specify 'pin2' of ``component1`` must
connect to 'pin1' of ``component2``. We can also rename pins
for semantic clarity: ::
    
  component1.rename_pins('input', 'output')
  component1['output'].connect(component2)

Here, we do the same as the previous example, except that we
rename the two pins of ``component1`` to 'input' and
'output', and then connect 'output' to ``component2``.
We do not need to explicitly specify 'pin1' for
``component2``, since that is the first unconnected pin.

With this connection, we now have a very rudimentary circuit
to run simulations on.


Simulation
----------
:py:class:`simphony.simulators` provides a collection of
simulators that connect to an input and output pin on a 
circuit, then perform a subnetwork growth algorithm (a
series of matrix operations) in order to simulate resulting 
outputs for inputs on the circuit. The simulation process
modifies pins and components, so simulators actually copy
the circuit they are passed in order to preserve the
original circuit.

Let's run a simple sweep simulation on the circuit we have
created: ::

  from simphony.simulators import SweepSimulation
  simulation = SweepSimulation(1500e-9, 1600e-9)
  simulation.multiconnect(component1['input'], component2['pin2'])
  result = simulation.simulate()

Our sweep simulation passed input light on a range of
wavelengths from 1500nm to 1600nm, and now ``result``
contains what frequencies came out of our circuit. We can
use these results however we like.

In order to view the results, we can use the ``matplotlib``
package to graph our output, but that will be demonstrated
in following tutorials. For this tutorial, we're done!

.. _SiEPIC PDK: https://github.com/SiEPIC
