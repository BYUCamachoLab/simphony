========================
Introduction to Simphony
========================

:Author: Sequoia Ploeg
:Date: |today|
:Copyright:
  This work is licensed under the `MIT License`__.

.. __: https://opensource.org/licenses/MIT


This document explains how to use the Simphony Python package for
simulation of photonic circuits.

We welcome feedback that would improve the documentation! If you have 
questions that clarification in the documentation would help resolve, please
`open an issue <https://github.com/BYUCamachoLab/simphony/issues>`_ at GitHub.

.. toctree::
    :caption: Table of Contents

    intro


Prerequisites
=============

Before reading this tutorial you should know a bit of Python. If you
would like to refresh your memory, take a look at the `Python
tutorial <https://docs.python.org/tutorial/>`__.

If you wish to work the examples in this tutorial, you must also have
some software installed on your computer. Please see :ref:`install` for instructions.


Prologue
========

Simphony aims to be Pythonic, yet pragmatic. Instead of reinventing a new
framework for everyone to learn, we build off the concepts that engineers and
scientists are already familiar with in electronics: the SPICE way of 
organizing and defining circuits and connections. In this case, we use much
of the same terminology but make it Python friendly (and, in our opinion,
improve upon its usability). As a simple example, we use the terminology for
*components*, *pins*, and *nets* in a similar manner as an electronics SPICE 
definition.

Simphony follows Python's EAFP (easier to ask forgiveness than permission) 
coding style. This contrasts with the LBYL (look before you leap) style common
to other languages. In practice, this means that if, say, a library element
component is implemented but is missing attributes, it won't be noticed until
runtime when a call to a nonexistent attribute, perhaps by a simulator, raises 
an exception.

Python often uses magic methods (also known as "dunder" methods) to implement
underlying class functionality. Simphony will sometimes use the same 
convention, but with what we'll call "sunder" methods (for single-underscore 
methods), since Python's dunder syntax is reserved for future Python features.
These methods are typically of no concern to the casual package user; they are
used for under-the-hood, behind-the-scenes operations on Simphony objects.
Since they represent "private" methods in Python, they won't appear in the 
documentation. However, if you begin developing for Simphony, or creating 
model libraries and plugins, you may need to pay a little more heed to them.

Units throughout simphony strive to all be in base SI units (unless otherwise 
specified, see the respective object's documentation) to avoid ambiguity and 
confusion. In other words, a length in nanometers will be expressed in meters, 
etc. This can sometimes lead to not-as-pretty looking values, especially 
when dealing with sub-wavelength values and frequencies in THz, as is common 
in silicon photonics. But, it is consistent.

This package initially began as an extension to `SiEPIC-Tools`_, but was 
broken off into its own independent project as its scope grew and it became 
large enough to be considered its own stand-alone project. As a result,
compatability with many SiEPIC features has been built into Simphony.

.. _SiEPIC-Tools: https://github.com/lukasc-ubc/SiEPIC-Tools

One of the main objectives of the Simphony package is to provide an 
open-source, concise, independent, and fast simulation tool.  By 
concise, we mean that scripting a simulation does not require many 
lines of code, imports, or the instantiation of many objects. 
Independent means that it has very few dependencies and it doesn't 
force the user into a specific user interface or proprietary (and 
hence incompatible) syntax in order to perform simulations. 
Fast means not recomputing what's already been simulated, which is 
accomplished by caching the calculations performed on each component.


Silicon Photonics
=================

Silicon photonics is a rapidly growing industry that uses electronic
integrated circuit fabrication technologies to produce industry-grade
photonic integrated circuits (PICs) at low cost and high volume.
Silicon photonic technologies have been largely driven by the
communications industry, but also find applications in sensing,
computing, and quantum information processing by enabling high data
transmission rates and controlled manipulation of light waves.

As the silicon photonics industry grows and the demand for PICs increases,
it is increasingly important for designers to have access to software
design tools that can accurately model and simulate PICs in a first-time-right
framework. Simulating PICs is a resource- and time-intensive process. Owing
to the long wavelengths of photons relative to electrons, photonic device
simulation requires solving Maxwell's equations with far less abstraction
than electronic circuit components. Once devices have been simulated and
bench-marked, however, compact models representing the phase and amplitude
response functions of individual components may be stitched together to form
functioning circuits. Although various commercial tools exist to perform these
functions, they are often expensive and limited in the variety and type of
photonic devices than can be simulated. Furthermore, there is often a lack of
standardization among platforms that in many cases prevents interoperability
between tools.

Here we present an open-source, software-implemented simulation tool for PICs
(documentation and downloads available at  github.com/BYUCamachoLab/simphony).
Our toolbox, which we name Simphony, provides fast simulations for PICs and
allows for the integration of device compact models that may be sourced from
a variety of platforms.  Simphony also provides the capability to add custom
components.  This interoperability is achieved by cascading device scattering
parameters, or S-parameters, for each component using sub-network growth
algorithms, a common practice in microwave/radio-frequency (RF) engineering.
Benchmark testing of Simphony against commercial software indicates a speedup
of approximately 20x.


Defining Circuits
=================

The fundamental building blocks of a simulation in most SPICE-like programs,
and therefore Simphony, are elements, pins, nets, subcircuits, and circuits. A 
regular SPICE circuit can be fully represented by what is known as a Netlist 
and stored as a single text file. Simphony circuits are defined simply as 
Python files, allowing for the persistence of defined circuits and the 
compatible sharing of designs between computers with a Simphony installation.

Let's build up our understanding by going through the typical objects that can
be found in every circuit definition, in a logical order.


Models
------

A Model is the basic representation of some designed and presimulated 
component in Simphony. A simulation library in Simphony is composed of Model
definitions, which includes what connections (inputs and outputs) to the model 
are available, what frequency range the model is valid over, and the scattering
parameters of a given model at a given frequency.

Let's take a look at the documentation for a basic 
:py:class:`simphony.elements.Model`.

.. autoclass:: simphony.elements.Model
    :noindex:
    :members:
    :inherited-members:

.. note::
    The base Model class has three default functions but no default
    ``__init__()`` function. It is only required for a model to have an
    ``__init__()`` function if it takes parameters that may affect the
    s-parameters (e.g. width, length, etc.).

Simphony includes a default model library from the 
`SiEPIC EBeam PDK <https://github.com/lukasc-ubc/SiEPIC_EBeam_PDK>`_, developed
at the University of British Columbia. Let's examine one of the models to 
learn about how they work: 

.. autoclass:: simphony.library.siepic.ebeam_y_1550
    :noindex:
    :members: pins, s_parameters

A simplified version of the code that implements this model is as follows.::

    class ebeam_y_1550(simphony.elements.Model):
        pins = ('n1', 'n2', 'n3') #: The default pin names of the device

        def __init__(self, thickness=220e-9, width=500e-9, polarization='TE'):
            # Stores values for use in calculating s-parameters later.

        def s_parameters(self, freq):
            # Calculates and returns s-parameters for the given frequency range.

Models define various *types* of devices. They are not used, however, as 
*instances* of those devices in a circuit; instead, instances, or ``Elements``,
as they are called in simphony, reference ``Models`` to know how they behave.

To use a model in a circuit, we would first instantiate a model with the 
desired parameters. If we want to use the same model a second time but with 
different parameters, we'd simply create a second object. ::

    from simphony.library import siepic
    y_te = siepic.ebeam_y_1550(thickness=220e-9, width=500e-9, polarization='TE')
    y_tm = siepic.ebeam_y_1550(thickness=220e-9, width=500e-9, polarization='TM')


Elements
--------

An ``Element`` represents some discrete component that exists in a layout.
It is the physical instantiation of some ``Model`` into a circuit.
If we consider it from the layout driven design methodology's perspective,
an Element is an object such as a y-branch or a directional coupler that
has already been designed and simulated on its own. It has a definite shape
and predefined characteristics that allow it to be simply dropped into
a layout and operate in a predictable fashion.

Just as a ``Model`` is the building block of a simulation, 
``Element`` objects, or "instances", are the building blocks of a 
circuit. Each instance must reference an object of type 
:py:class:`simphony.elements.Model` and represents
a physical manifestation of some photonic component in the circuit.
For example, a waveguide has a single :py:class:`simphony.elements.Model` which specifies its 
attributes and S-parameters. However, a circuit may have many waveguides in it, 
each of differing length. You would therefore instantiate the Waveguide model
for all the desired lengths and then associate Elements with the appropriate
model. A model can be used for several Elements; if a y-branch with the same 
parameters is used throughout the circuit, the model only needs to be instantiated
once, regardless of how many Elements reference it.

You may never directly instantiate an ``Element`` object; when you create
instances of Elements in a Subcircuit, the instantiation is handled for you.
(Subcircuits will be covered later.)
You may, however, provide a unique string name for each element (and this is
recommended). If you don't provide a name, a random one will be generated in
the background. If you do provide a name, you'll be able to retrieve that Element
from the circuit later.

Simphony is layout-agnostic; it doesn't care where components are physically 
located. However, since everything in Python is an object, other parameters may
be stored dynamically within the instance itself, even after its creation. For 
example, if you were extending Simphony, you may find it useful to include 
certain key-value pairs, including (but not limited to) layout positioning 
information. Python, as an interpreted language, contributes to
Simphony's flexibility when used in conjunction with other programs. For example, 
Lumerical's INTERCONNECT, a schematic-driven commercial simulation software, 
ignores layout and only requires components and connections when simulate a circuit.
On the other hand, KLayout with SiEPIC-Tools, created by SiEPIC and UBC, 
implements a layout-driven approach to designing photonic circuits, exporting 
locations with the components when generating a netlist.
Because Simphony can optionally store any information with its components, 
including layout information, it can act in either capacity.


Pins (and Pinlists)
-------------------

Remember the ``pins`` attribute from :py:class:`simphony.elements.Model`? 
When elements are created, the pins as defined by the Model are turned
into actual :py:class:`simphony.netlist.Pin` objects. 
:py:class:`Pin <simphony.netlist.Pin>` objects are named according to the names
defined by the Model, and are how interactions with pins are handled when
defining connections and accessing ports of the circuit once a simulation has
been run.

You should never interact with the `Pin` object itself; rather, only through
methods as exposed by Element and Subcircuit objects, typically by using
the string name of the Pin. Pins belong to "Pinlists", which are objects
handled internally by Elements. A Pin can only belong to one pinlist at a time,
and since the Simulation class manipulates pinlists, any manual intervention
can break the definition of a circuit.

You can, however, rename pins for ease of use in making connections.
Suppose for example, that you use the same y-branch from earlier to split 
incoming light between two outputs.

.. autoclass:: simphony.library.siepic.ebeam_y_1550
    :noindex:
    :members: pins

If we've created an Element from the ``ebeam_y_1550`` model, it'll have
pins as follows: ::

    >>> from simphony.netlist import Element
    >>> from simphony.library.siepic import ebeam_y_1550
    >>> y = ebeam_y_1550()
    >>> e = Element(y, name='splitter')
    >>> e.pins
    [<Pin 'n1' at <Element 'splitter' at 0x7f9e980a07b8>>, 
     <Pin 'n2' at <Element 'splitter' at 0x7f9e980a07b8>>, 
     <Pin 'n3' at <Element 'splitter' at 0x7f9e980a07b8>>]

We could rename the pins, once they belong to an element.

One at a time: ::

    >>> e.pins['n1'] = 'input'
    >>> e.pins['n2'] = 'output_top'
    >>> e.pins['n3'] = 'output_bottom'

Or, simultaneously: ::

    >>> e.pins = ('input', 'output_top', 'output_bottom')

End result: ::

    >>> e.pins
    [<Pin 'input' at <Element 'splitter' at 0x7f9e980a07b8>>, 
     <Pin 'output_top' at <Element 'splitter' at 0x7f9e980a07b8>>, 
     <Pin 'output_bottom' at <Element 'splitter' at 0x7f9e980a07b8>>]


Nets (and Netlists)
-------------------

Simphony implements a simple :py:class:`Netlist <simphony.netlist.Netlist>` 
class. A typical SPICE netlist simply contains a list of components and
assigns net numbers to each of its ports, where matching numbers identify
connections between components. Simphony alters this slightly, instead
maintaining a list of connected pins (which each belong to a specific Element) 
and dispensing with literal net ID's.

As a user, you will never interact directly with the netlist, instead defining
connections within a subcircuit using syntax explained in a later section.


Subcircuits
-----------

A Subcircuit is an example of what Simphony will call a compound structure, 
which the glossary defines as:

compound structure
   Any structure that can be broken down into smaller, simpler parts.
   A subcircuit is an example of a compound structure; it contains simpler
   elements (or other compound structures) connected internally to form 
   the overall larger structure.

First, know that a subcircuit can act as a fully-qualified circuit in its own
right. In fact, simphony doesn't even define a ``Circuit`` class (for now, at 
least). Operations are performed on subcircuits themselves. 

A circuit can often be broken up into smaller, sometimes reused "subcircuits"
that make up the whole design. Subcircuit objects allow us to create these. 

Say, for example, that you'd like to create a circuit that cascades several
sets of ring resonators of varying radius. Instead of placing lots of halfring
elements into a circuit, you can create a standalone subcircuit that represents
a ring, and then place the ring subcircuit in your circuit as many times
as necessary. This helps reduce complexity and increase abstraction as your 
circuits grow and the number of elements multiply.

A subcircuit encapsulates a series of element and connection definitions and allows 
external connections to this circuitry only through the subcircuit's nodes, 
also often referred to in Simphony as "externals" (pins not connected within
the circuit, thus available for input/output).

Because the internal circuitry is isolated from external circuitry, internal 
devices and node names with the same names as those external to the 
subcircuit are neither conflicting nor shorted. In addition, subcircuits can be 
nested within other subcircuits. 


Putting It Together
-------------------

We've now discussed separately Models, Elements, Pins, and Subcircuits.
Let's examine how to use these together. We'll make a rather useless circuit,
which simply uses two y-branches to split and input signal and then recombine 
it.

First, we declare the model we want to use for the y-branches. We'll use the
same model for both Element instances. ::

    from simphony.library.siepic import ebeam_y_1550

    y_model = ebeam_y_1550()

Next, we'll need a circuit to put the models into. Remember, we don't have to
actually interact with Element objects; when we add models and names to a 
subcircuit, creating Elements is handled in the background for us. ::

    from simphony.netlist import Subcircuit

    circuit = Subcircuit('Example Circuit')
    circuit.add([
        (y_model, 'splitter'),
        (y_model, 'recombiner'),
    ])

For ease of manipulation, we'll rename the ports to something human-readable.
The order is determined from the documentation for a y-branch, with the new pin
names corresponding to the order of the default pin names. ::

    circuit.elements['splitter'].pins = ('input', 'out1', 'out2')
    circuit.elements['recombiner'].pins = ('output', 'in1', 'in2')

We can now define the connections for our circuit. ::

    circuit.connect_many([
        ('splitter', 'out1', 'recombiner', 'in1'),
        ('splitter', 'out2', 'recombiner', 'in2'),
    ])

That's it! Our circuit is fully defined and ready for simulators to come and
analyze it.


Simulation
==========

In order to avoid repeating calculations, Simphony caches the scattering 
parameters for all unique models found in a netlist at simulation time.

The :ref:`Simulation classes <simphony-simulators>` performs the basic sub-network growth matrix 
operations required to simulate a photonic circuit. Simphony's 
simulation module contains various different kinds of simulations for swept
simulations, monte carlo simulations, and others. 
Sub-network growth then cascades all of the 
instances' matrices into one matrix.

Simulations take subcircuits as their primary argument. Since the process
of cascading the individual s-parameter matrices into a single result matrix
is a destructive process and :py:class:`Pins <simphony.netlist.Pin>` objects 
are modified in the process, simulators actually create a copy of the 
circuit given as a parameter so as to not destroy the original circuit, 
allowing it to be used for other simulations. The only objects that are not 
fully copied are the instantiated :py:class:`Model <simphony.elements.Model>` 
classes.

One result of this is that any direct object references (to pins, elements,
or subcircuits, for example) are not valid on the resulting subcircuit after
a simulation. This is why objects are retrievable by string name; you can
use the same string after a simulation to retrieve an object where the
actual Python object reference does not point to any object in the 
simulation results (such as pins).

Simple simulations could be created like so: ::

    from simphony.simulation import SweepSimulation
    simulation = SweepSimulation(circuit, 1500e-9, 1600e-9, mode='wl')
    result = simulation.simulate()

For more details on simphony simulators, see 
:ref:`Simulators <simphony-simulators>`.


Model Libraries
===============

Simphony includes two basic device libraries. 
Beyond the default device libraries, it is very easy to define your own 
collection of component models. Library authors can easily create their own 
Python modules to be used in simulation by subclassing base components 
provided by Simphony. This is another way in which Simphony's capabilities 
can be easily extended. It also allows simulation results, scripts, and 
circuits to easily be shared between collaborators or computers, since the 
entire system is cross-platform, non-proprietary, and the only libraries 
required are the modules implementing the components used in the circuit 
and the script defining the circuit.

Read more about how to create custom libraries :ref:`here <creating-model-libraries>`.

SiEPIC
------

The models for the SiEPIC library were created by 
SiEPIC and the University of British Columbia (UBC) for their Process Design Kit (PDK) used with KLayout. 
It correlates with the physical component model layouts found in 
SiEPIC-Tools for KLayout and the S-parameters are the result of FDTD 
simulations for ideal components. 

SiPANN
------

The second library, which is not preinstalled but can be obtained `here`_, includes 
components designed by the CamachoLab research group at Brigham Young University (BYU).
S-parameters are generated 
using machine learning techniques. This allows for fast iteration in designing 
new components as full FDTD simulations don't need to be run. Additionally, 
imperfections from manufacturing can be easily simulated using monte carlo techniques.

.. _here: https://sipann.readthedocs.io/
