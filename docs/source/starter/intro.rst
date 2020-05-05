========================
Introduction to Simphony
========================


.. toctree::
    :hidden:

    intro


Prerequisites
=============

Before reading this tutorial you should know a bit of Python. If you
would like to refresh your memory, take a look at the `Python
tutorial <https://docs.python.org/tutorial/>`__.

If you wish to work the examples in this tutorial, you must also have
some software installed on your computer. Please see :ref:`install` for instructions.



Understanding Circuits
======================

An explanation of photonic circuits and how to think about them.


Understanding Models
====================

An explanation of how models work in Simphony.


Understanding Pins
==================

How pins work and why they're unique and can only be attached to a single
instance.


Understanding Compound Structures
=================================

The glossary defines a compound structure as follows::

compound structure

   Any structure that can be broken down into smaller, simpler parts.
   A subcircuit is an example of a compound structure; it contains simpler
   elements (or other compound structures) connected internally to form 
   the overall larger structure.


    

About Simphony
==============

This package is still under development. It initially began as an extension to
`SiEPIC-Tools`_, but was broken off into its own independent project as its scope 
grew and it became large enough to be considered its own stand-alone project. 

NOTE: This is still under development; this next paragraph is *almost* true.
There is a repository forked from lukasc-ubc/SiEPIC-Tools, 
[SiEPIC-Tools](https://github.com/sequoiap/SiEPIC-Tools),
that integrates Simphony with SiEPIC-Tools and KLayout in order to perform 
photonic circuit simulations using a layout-driven design methodology.

.. _SiEPIC-Tools: https://github.com/lukasc-ubc/SiEPIC-Tools



.. _manual:

The Simphony User Manual
========================

:Author: Sequoia Ploeg
:Version: 0.3.0
:Date: |today|
:Copyright:
  This work is licensed under the `MIT License`__.

.. __: https://opensource.org/licenses/MIT

:Abstract:
  This document explains how to use the Simphony Python package for
  simulation of photonic circuits.

.. _intro:

Introduction
############

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

Notes about the implementation

Simphony aims to be Pythonic, yet pragmatic. Instead of reinventing a new
framework for everyone to learn, we build off the concepts that engineers and
scientists are already familiar with in electronics: the SPICE way of 
organizing and defining circuits and connections. In this case, we use much
of the same terminology but make it Python friendly (and, in our opinion,
improving upon its usability).

Simphony follows Python's EAFP (easier to ask forgiveness than permission) 
coding style. This contrasts with the LBYL (look before you leap) style common
to other languages. In practice, this means that if, say, a library element
component is implemented but is missing attributes, it won't be noticed until
runtime when a call to a nonexistent attribute throws an exception.

Python often uses magic methods (also known as "dunder" methods) to implement
underlying class functionality. Simphony uses the same convention, but with
what we'll call "sunder" methods (for single-underscore methods), since
Python's dunder syntax is reserved for future Python features.

Units throughout simphony are all SI units (unless otherwise specified) 
to avoid ambiguity and confusion. This can sometimes lead to not-as-pretty 
looking values, especially when dealing with sub-wavelength values and 
frequencies in THz. But, it is consistent.

Circuit Data Model
------------------

The fundamental building blocks of a simulation in most SPICE-like programs,
and therefore Simphony, are elements, nets, subcircuits, and circuits. A 
circuit can be fully represented by what is known as a Netlist and stored as
a single text file.

* An `Element` is ...

* A `Subcircuit` is ...

* A `Circuit` is ... and is different from a `Subcircuit` because ...




Elements can be added to circuits anonymously.

Subcircuits
-----------

Creating subcircuits is therefore just as easy. Often, a circuit can be 
broken up into smaller circuit segments that make up the whole design.
Subcircuits allow us to create these (for example, cascading a set of
ring resonators of varying radius).

A SPICE subcircuit (.subckt) wraps around a block of circuit text and allows 
external connections to this circuitry only through the subcircuit's nodes. 

Because the internal circuitry is isolated from external circuitry, internal 
devices and node names with the same names as those external to the 
subcircuit are neither conflicting nor shorted. In addition, subcircuits can 
accept circuit parameters which can be used to assign values to internal 
devices or nested subcircuits. 



Appendix: Design and Implementation of the Simphony Framework
=============================================================

Simphony source code can be found in a GitHub repository at 
github.com/BYUCamachoLab/simphony. The extended SiEPIC-Tools package 
for Klayout with Simphony integration can be found at 
github.com/BYUCamachoLab/SiEPIC-Tools. Additionally, Simphony is 
available through the Python Package Index (PyPI).

Simphony has three modules: 1) a core library containing the base 
classes for models and instances from which all user-defined photonic 
component models are derived; 2) a simulation library which handles 
operations related to sub-network growth; and 3) a device library of 
standard components with presimulated S-parameters and device 
configurations.

Simphony is implemented in Python, and aims to be very lightweight 
and fast to install. As such, it only requires dependencies needed 
to provide basic functionality. For example, our package doesn't 
provide built-in functions for plotting simulation results; instead, 
it returns the raw data as NumPy arrays, allowing the user to choose 
their own visualization package.

The Simphony Core
#################

One of the main objectives of the Simphony package is to provide an 
open-source, concise, independent, and fast simulation tool.  By 
concise, we mean that scripting a simulation does not require many 
lines of code, imports, or the instantiation of many objects. 
Independent means that it has very few dependencies and it doesn't 
force the user into a specific user interface or proprietary (and 
hence incompatible) code syntax in order to perform simulations. 
Fast means not recomputing what's already been simulated, which is 
accomplished by caching the calculations performed on each component.

The :py:mod:`simulation` library contains the base classes required for 
any simulation. The most basic building block of a Simphony simulation 
is an object called :py:class:`Model`. It is a representation 
of a type of photonic component and not of a specific component within 
a circuit. The task of tracking specific components within a circuit 
or netlist is left to :py:class:`Element`.

Component Models
################

All photonic component models, whether scripted on-the-fly or provided 
by a device library, are derived from the :py:class:`Model` class. We 
use the convention of naming each component model with a human-readable 
name prefixed with the name of the library. For example, Simphony 
includes a device library created by SiEPIC and the University of 
British Columbia (UBC)\cite{chrostowski_design_2016}. Each of the 
components in this device library are prefixed with *ebeam*. Simphony 
also includes a library created in-house by our research group, and 
its components are prefixed with *sipann*. ::

    class ComponentModel:
        @classproperty
        def component_type(cls):
            # Returns class name
            
        ports = 0
        s_parameters = None
        cachable = False

        @classmethod
        def get_s_parameters(cls, **kwargs):
            # Implementation (return preloaded or calculated values)

*The ComponentModel class.*

Simphony does not require models to be loaded before being used within 
a circuit in the program. Because Python is interpreted and not compiled, 
errors are only encountered at runtime when you try to perform an illegal 
operation. The program performs some elementary error checking at 
simulation time to verify that all Model classes contain the required 
attributes and are not malformed.

In order to avoid repeating calculations, Simphony caches the calculated 
scattering parameters for all unique models found in a netlist at 
simulation time. ::

    @register_component_model
    class YBranchCoupler(ComponentModel):
        ports = 3
        s_parameters = # Some preloaded tuple
        cachable = True

*A cachable y-branch coupler model.*::

    @register_component_model
    class Waveguide(ComponentModel):
        ports = 2
        def s_parameters(self, length, width, thickness, **kwargs):
            # return some calculation based on parameters
        cachable = False

*A non-cachable waveguide model.*

In order to create your own model based on :py:class:`Model`, there are 
some required attributes that need to be overridden. First, *ports* must be 
assigned some tuple of strings, representing the pins' names.
The *s_parameters* function must be implemented; it calculates and returns 
an matrix of S-parameters for some frequency range, passed in as an argument.
The S-parameters could be hard-coded, loaded from a file, or calculated in a
function. S-parameters can be calculated based on attributes stored at the
instance level. For example, a waveguide's S-parameters are dependent on 
its length and path. Its S-parameters should be calculated for each unique
waveguide in the circuit. In contrast, a y-branch coupler's S-parameters are 
considered unchanging since it doesn't have parameters to vary and its 
physical layout is fixed. Its S-parameters could be read in from a file once
and used repeatedly.

Component Instances
###################

::

    class ComponentInstance:
        def __init__(self, model: ComponentModel=None, nets: List=None, extras: Dict=None):
            # Store values internally

        def get_s_parameters(self):
            return self.model.get_s_parameters(**self.extras)

*A cachable RingResonator model.*

Just as a :py:class:`Model` is the building block of a simulation, 
:py:class:`Element` objects, or "instances", are the building blocks of a 
circuit. Each instance must reference a :py:class:`Model` and represents
a physical manifestation of some photonic component in the circuit.
For example, a waveguide has a single :py:class:`Model` which specifies its 
attributes and S-parameters. However, a circuit may have many waveguides in it, 
each of differing length; these are therefore instances of the waveguide.
Any number of instances for a certain photonic component will reference the same 
:py:class:`Model`, thereby obtaining its S-parameters from a "single source of truth."

One major difference between models and instances is that to build a 
simulation, instance objects need to be instantiated whereas models 
are simply defined. :py:class:`Element`s store three attributes.
The first, *model*, stores a reference to a :py:class:`Model` from which 
S-parameters can be obtained. The second, *nets*, can optionally be 
defined upon construction of an instance or it can be left for the netlist 
object (discussed later) to assign automatically. The functionality allowing 
the declaration of nets upon instantiation was built in to allow other circuit 
layout programs, which may already implement netlist generation routines, to 
be plugged in. The final attribute, *extras*, is an optional dictionary 
containing parameters that may need to be passed on to models where scattering 
parameter calculations depend on other variables (e.g. waveguide lengths and 
bend radii).

Simphony is layout-agnostic; it doesn't care where components are physically 
located. However, since everything in Python is an object, other parameters may
be stored within the instance itself; useful key-value pairs, including (but not 
limited to) layout positioning information, may be included. This contributes to
Simphony's flexibility when used in conjunction with other programs. For example, 
Lumerical's INTERCONNECT, a schematic-driven commercial simulation software, 
ignores layout and only requires components and connections when simulate a circuit.
On the other hand, KLayout with SiEPIC-Tools, created by SiEPIC and UBC, 
implements a layout-driven approach to designing photonic circuits, exporting 
locations with the components when generating a netlist \cite{chrostowski_design_2016}.
Because Simphony can optionally store any information with its components, 
including layout information, it can act in either capacity.

Netlist
#######

Simphony implements a simple *Netlist* class. A netlist simply contains a list 
of component instances and has routines used to assign net numbers to 
connections between components. However, the netlist doesn't maintain the 
list of connections internally; instead, as it parses connections, it generates 
net numbers and stores those values within each instance's *nets* attribute.
The netlist maintains a list of the components in a circuit, each of which 
independently maintain a list of their nets.

Defining circuit connections in Simphony
########################################

TODO

Simulation
##########

The Simulation class performs the basic sub-network growth matrix 
operations required to simulate a photonic circuit. While Simphony's 
simulation module contains various different kinds of simulations, 
all simulations extend this base Simulation class. Simulation performs 
interpolation on the S-parameters for all cachable components and 
caches the results. Sub-network growth then cascades all of the 
instances' matrices into one matrix.

Simphony presently includes three different simulators. The first is a 
single-mode, single-input simulation that returns transmission 
parameters from any input port to any output port (including back 
reflections). The second is a single-mode, multi-input simulation that 
allows you to sweep the wavelength of multiple input ports 
simultaneously and see the response at any output port. Finally, a monte 
carlo simulator that randomly varies waveguide widths and thicknesses, 
calculating S-parameters by interpolating the effective index of 
waveguides, allows us to mimic fabrication variation. Other simulators 
can easily be written, extending the functionality of the base *Simulation*
class.

Device Library
##############

Simphony includes two basic device libraries. The first was created by 
SiEPIC and UBC for their Process Design Kit (PDK) used with KLayout. 
It correlates with the physical component model layouts found in 
SiEPIC-Tools for KLayout and the S-parameters are the result of FDTD 
simulations for ideal components. The second library includes a few 
components designed by our research group with S-parameters generated 
using machine learning techniques. This allows for imperfections from 
manufacturing to be easily simulated or for new components to be quickly 
designed.

Beyond the default device libraries, it is very easy to define your own 
collection of component models. Each library, which is just a collection 
of class definitions, is formatted as a Python module with all the models 
it provides defined in *__init__.py*. Instead of hardcoding values, our 
cachable models simply load their S-parameters from a NumPy file (.npz) 
each time the library is imported. End-users can easily create their own 
Python modules to be used in simulation by subclassing base components 
provided by Simphony. This is another way in which Simphony's capabilities 
can be easily extended. It also allows simulation results, scripts, and 
circuits to easily be shared between collaborators or computers, since the 
entire system is cross-platform, non-proprietary, and the only libraries 
required are the modules implementing the components used in the circuit 
and the script defining the circuit.


Program Architecture
====================

Components (and Simulation Models)
##################################

Simphony attempts to be as agnostic as possible in the implementation of
optical components. This allows for greater flexibility and extensibility.
It is for this reason that Simphony separates the idea of a "component" and
a "simulation model," sometimes simply called a "model."

Components
==========

A "Component" represents some discrete component that exists in a layout.
If we consider it from the layout driven design methodology's perspective,
a component is an object such as a y-branch or a bidirectional coupler that
has already been designed and simulated on its own. It has a definite shape
and predefined characteristics that allow it to be simply dropped into
a layout and operate in a predictable fashion.

Simphony treats components as though they were physical objects stored in a
library. In other words, it is expected that they will not change shape, that
they have a pre-specified port numbering system, etc.

Simulation Models
=================

Simphony does, however,
separate the concept of a component and a simulation model. A single component
may have several simulation models associated with it; this allows a single
component to be simulated with different models; perhaps different fabs create
the same component in different ways, leading to different results. Perhaps
there exists a model with data points interpolated from test devices and another
model based on a neural net. To Simphony, it doesn't matter.

Simulation Models ought to have a nested class called Metadata. It should, at
a minimum, contain the following fields:
- simulation_models
- ports

It has the following format::

    class Metadata:
        simulation_models = [
            ('containing_module', 'simulation_class_name', 'human_readable_name'),
            ...
        ]
        ports = [INT]

Elements
########

Now that we understand what Components and Models are, "elements" are Simphony's
default implementation of some commonly used components. These models are taken
from SiEPIC's `EBeam PDK <https://github.com/lukasc-ubc/SiEPIC_EBeam_PDK>`_.

