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
