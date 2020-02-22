"""
Simphony aims to be Pythonic, yet pragmatic. Instead of reinventing a new
framework for everyone to learn, we build off the concepts that engineers and
scientists are already familiar with in electronics: the SPICE way of 
organizing and defining circuits and connections. In this case, we use much
of the same terminology but make it Python friendly (and, in our opinion,
improving upon its usability).

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

"""

##################################################

# Imports are straightforward and well named.

import simphony
from simphony.netlist import Circuit, Subcircuit, Element
from simphony.simulation import SweepSimulation, SweepParams
from simphony.lib import GratingCoupler, Waveguide, YBranch

##################################################

# Create your own element by subclassing `Element`.
class RingResonator(Element):
    __name__ = 'Ring Resonator'
    nodes = ('n1', 'n2', 'n3', 'n4')

    def __init__(self, radius):
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

# Elements can be added individually:
circuit.add(rr1)
circuit.add(rr2)
circuit.add(rr3)

# or, as a list:
circuit.add(rr1, rr2, rr3, *additional)

##################################################

# METHOD 2: With identifying string names (not necessarily required)
circuit.add([
    ("ring 10um", RingResonator(radius=10e-6)),
    ("ring 11um", RingResonator(radius=11e-6)),
    ("ring 12um", RingResonator(radius=12e-6)),
    (None,        RingResonator(*args)),
])

# Adding using the above method allows items to be accessed from within the
# circuit by name, later.
ring = circuit["ring 10um"]

# One restriction required by allowing this freedom, however, is that all 
# element, model, and subcircuit names must be unique from each other.

# For ease of accessing nodes/ports after running a simulation, they can be
# named with identifying strings.
circuit.label("ring 10um", 'n1', 'myInput')
circuit.label("ring 12um", 'n4', 'myOutput')

##################################################

# Creating subcircuits is therefore just as easy. Often, a circuit can be 
# broken up into smaller circuit segments that make up the whole design.
# Subcircuits allow us to create these (for example, cascading a set of
# ring resonators of varying radius).

"""
TWO PROPOSED OPTIONS (or should we allow for both implementations?):
1. Create a class that inherits from `Subcircuit` and defines the subcircuit 
   within.
    Advantages:
        - The __init__ function can be written to take parameters, therefore
          these subcircuits could be parameterized, allowing for the creation
          of many with stepped parameters.
    Disadvantages:
        - It is less SPICE-like.

2. Create a subcircuit variable that is simply a subcircuit object. Then, add
   objects like you would to a regular circuit.
    Advantages: 
        - More SPICE-like.
    Disadvantages:
        - Less easily parametizable for the creation of many instances. They
          could, however, conceivably be parametrized by writing a function
          that returns fully constructed subcircuit objects.
"""

# The syntax of Proposed Method 1:
class MZI(Subcircuit):
    def __init__(self, l1=50e-6, l2=65e-6):
        # Construct the subcircuit
        self.add([
            ('gc in', GratingCoupler()),
            ('gc out', GratingCoupler()),
            (None, YBranch()),
            (None, YBranch()),
            ('wg short', Waveguide(length=l1)),
            ('wg long', Waveguide(length=l2)),
        ])

subckt = MZI(l1=50e-6, l2=65e-6)

# Alternatively, the syntax of Proposed Method 2:
subckt = Subcircuit('MZI', 'n1', 'n2')
subckt.add([
    ('gc in', GratingCoupler()),
    ('gc out', GratingCoupler()),
    (None, YBranch()),
    (None, YBranch()),
    ('wg short', Waveguide(length=50e-6)),
    ('wg long', Waveguide(length=65e-6)),
])

##################################################

# Simulations can be run on circuit objects. Simply instantiate a simulation
# and simultaneously pass in the circuit.

params = SweepParams(start=1500e-9, end=1600e-9, num=2000)
simulation = SweepSimulation(circuit, params)

# You can then retrieve the raw data using your previously named nodes.
f, s = simulation.s_parameters[:,'myInput','myOutput']
