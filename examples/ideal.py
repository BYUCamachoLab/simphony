"""
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

DESIGN CONSIDERATIONS
(List below as they arise.)

1. What do we do about units? Should we leave them in standard units, to avoid
confusion across the board (but leading to ugliness such as 1.88e+14 for
frequencies, 1550e-9 for wavelengths, and 50e-6 for waveguide lengths), or do
we specify units on a per-function basis?

2. Shall we consider making SParameters a dataclass of its own? That would
encapsulate the frequencies, an array, and the S-parameters, a 
multi-dimensional array, together. Additionally, we could add class methods
to do the cascading for us, instead of tracking it while calculating the
simulation results.

3. Would it be a good idea to impose a frequency restriction on Elements? As
in, should we make it a required attribute of subclasses to implement a
tuple denoting the bounds that the model's s-parameters are valid in?

"""

##################################################

# Imports are straightforward and well named.

import simphony
from simphony.netlist import Circuit, Subcircuit, Element
from simphony.simulation import SweepSimulation, SweepParams
from simphony.lib import GratingCoupler, Waveguide, YBranch

##################################################

class Element(object):
    nodes = None

    def __new__(cls):
        if not nodes:
            raise RuntimeError("Malformed class.")
        return super().__new__()

# Create your own element by subclassing `Element`.
class RingResonator(Element):
    nodes = ('n1', 'n2', 'n3', 'n4')

    def __init__(self, radius):
        """
        It is up to the library creator to determine the units they accept
        as parameters, or whether they accept other parameters/flags to 
        indicate the units of other parameters.

        By default, Simphony assumes basic SI units throughout (i.e. meters,
        Hertz, etc.).
        """
        self.radius = radius

    @property
    def s_params():
        # Note that loading .npz files allows for an approximately 
        # 3.5x speedup over parsing text files.
        # Do some calculation here that returns the s_parameters over
        # some frequency range.
        pass

##################################################

# Building a circuit is easy, and the circuit can be given a name.

circuit = Circuit('Mach-Zehnder Interferometer')

# Two methods for adding items to a circuit; direct Python variables, or string
# names that the user maintains.

##################################################

# METHOD 1: Python variables
rr1 = RingResonator(radius=10e-6)
rr2 = RingResonator(radius=11e-6)
rr3 = RingResonator(radius=12e-6)

# Elements can be added individually (though not recommended):
circuit.add(rr1)
circuit.add(rr2)
circuit.add(rr3)

# or, as a list:
circuit.add([rr1, rr2, rr3(, ...)])

##################################################

# METHOD 2: With identifying string names (not necessarily required)
circuit.add([
    ("ring 10um", RingResonator(radius=10e-6)),
    ("ring 11um", RingResonator(radius=11e-6)),
    ("ring 12um", RingResonator(radius=12e-6)),
    (None,        RingResonator(*args)),
])

# If None is specified, the element is added "anonymously," meaning a random
# unique string is generated as an identifier, based off the element's 
# __name__ attribute and random integers.

# Adding using the above method allows items to be accessed from within the
# circuit by name, later.
ring = circuit["ring 10um"]

# One restriction required by allowing this freedom, however, is that all 
# element, model, and subcircuit names must be unique from each other.

# For ease of accessing nodes/ports after running a simulation, they can be
# named with identifying strings. For now, only external ports can be labelled;
# internal nets will be automatically ignored and the names invalidated.
circuit.label("ring 10um", 'n1', 'myInput')
circuit.label("ring 12um", 'n4', 'myOutput')

##################################################

# Creating subcircuits is therefore just as easy. Often, a circuit can be 
# broken up into smaller circuit segments that make up the whole design.
# Subcircuits allow us to create these (for example, cascading a set of
# ring resonators of varying radius).

# A SPICE subcircuit (.subckt) wraps around a block of circuit text and allows 
# external connections to this circuitry only through the subcircuit's nodes. 

# Because the internal circuitry is isolated from external circuitry, internal 
# devices and node names with the same names as those external to the 
# subcircuit are neither conflicting nor shorted. In addition, subcircuits can 
# accept circuit parameters which can be used to assign values to internal 
# devices or nested subcircuits. 

"""
Create a subcircuit variable that is simply a subcircuit object. Then, add
objects like you would to a regular circuit.
Advantages: 
    - More SPICE-like.
Disadvantages:
    - Less easily parametizable for the creation of many instances. They
        could, however, conceivably be parametrized by writing a function
        that returns fully constructed subcircuit objects.
"""

# Subcircuits are written much like regular circuits.
subckt = Subcircuit('MZI', 'n1', 'n2')
subckt.add([
    ('gc in', GratingCoupler()),
    ('gc out', GratingCoupler()),
    (None, YBranch()),
    (None, YBranch()),
    ('wg short', Waveguide(length=50e-6)),
    ('wg long', Waveguide(length=65e-6)),
])

# We can create a subcircuit factory, allowing for the creation of 
# parameterized subcircuits (perhaps for stepped simulations), like so:
def MZI_Factory(l1=50e-6, l2=65e-6):
    subckt = Subcircuit(None, 'n1', 'n2')
    
    # Add parameterized components
    subckt.add([
        ('gc in', GratingCoupler()),
        ('gc out', GratingCoupler()),
        (None, YBranch()),
        (None, YBranch()),
        ('wg short', Waveguide(length=l1)),
        ('wg long', Waveguide(length=l2)),
    ])
    
    # Make connections

    #Return the subcircuit
    return subckt

subckt = MZI_Factory(50e-6, 90e-6)

##################################################

# Simulations can be run on circuit objects. Simply instantiate a simulation
# and simultaneously pass in the circuit.

params = SweepParams(start=1500e-9, end=1600e-9, num=2000)
simulation = SweepSimulation(circuit, params)

# You can then retrieve the raw data using your previously named nodes.
f, s = simulation.s_parameters[:,'myInput','myOutput']

# Or figure out how the nodes are now lined up, by index.
ext = simulation.externals
print(ext)
# [('gc in', 'n1'), ('gc out', 'n2')]
