####################
Program Architecture
####################

**********************************
Components (and Simulation Models)
**********************************

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

It has the following format:

class Metadata:
    simulation_models = [
        ('containing_module', 'simulation_class_name', 'human_readable_name'),
        ...
    ]
    ports = [INT]

********
Elements
********

Now that we understand what Components and Models are, "elements" are Simphony's
default implementation of some commonly used components. These models are taken
from SiEPIC's `EBeam PDK <https://github.com/lukasc-ubc/SiEPIC_EBeam_PDK>`_.

New elements can be created and loaded programatically into Simphony. These settings, 
however, are not persistent, and should be loaded each time Simphony is imported.
Since element names are required to be unique, you might consider creating a library
of elements with a prefix to the component name (e.g. in 'ebeam_bdc_te1550', 'ebeam'
prefixes what the component actually is).

Design Pattern
==============

Each element is contained within its own Python module (i.e. in a folder). The
folder bears the component's name. Within the folder is an empty '__init__.py'
file, a 'models.py' file where simulation models are defined, and a 'component.py'
file that describes the physical features of the component.

********
Netlists
********

**********
Simulators
**********

Single Port Simulator
=====================

Multi Port Simulator
====================

Monte Carlo Simulator
=====================