# Simphony
[![PyPI Version](https://img.shields.io/pypi/v/simphony.svg)](https://pypi.python.org/pypi/simphony)
[![Build Status](https://travis-ci.org/BYUCamachoLab/simphony.svg?branch=master)](https://travis-ci.org/BYUCamachoLab/simphony.svg?branch=master)
[![Documentation Status](https://readthedocs.org/projects/simphonyphotonics/badge/?version=latest)](https://simphonyphotonics.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/pypi/l/simphony.svg)](https://pypi.python.org/pypi/simphony/)
[![Latest Commit](https://img.shields.io/github/last-commit/BYUCamachoLab/simphony.svg)](https://github.com/BYUCamachoLab/simphony/commits/master)

A Simulator for Photonic circuits

![](./docs/source/images/simphony_logo.jpg)

Simphony, a simulator for photonic circuits, is a fundamental package for designing and simulating photonic integrated circuits with Python.

**Key Features:**

- Free and open-source software provided under the MIT License
- Completely scriptable using Python 3.
- Cross-platform: runs on Windows, MacOS, and Linux.
- Subnetwork growth routines
- A simple, extensible framework for defining photonic component compact models.
- A SPICE-like method for defining photonic circuits.
- Complex simulation capabilities.
- Included model libraries from SiEPIC and SiPANN.

Developed by [CamachoLab](https://camacholab.byu.edu/) at 
[Brigham Young University](https://www.byu.edu/).

## Installation

Simphony can be installed via pip using Python 3:

```
python3 -m pip install simphony
```

Please note that Python 2 is not supported. With the official deprecation of
Python 2 (January 1, 2020), no future compatability is planned.

## Documentation

The documentation is hosted [online](https://simphonyphotonics.readthedocs.io/en/latest/).

Changelogs can be found in docs/changelog/. There is a changelog file for 
each released version of the software.


## Example

Simphony includes a built-in compact model library with some common components
but can easily be extended to include custom libraries.

Scripting circuits is simple and short. There are four main parts to running
a simulation in Simphony:

1. Define which component models will be used in the circuit.
1. Add instances of components into a circuit.
1. Define connection points.
1. Run a simulation.

```
# Declare the models used in the circuit
gc = siepic.ebeam_gc_te1550()
y = siepic.ebeam_y_1550()
wg150 = siepic.ebeam_wg_integral_1550(length=150e-6)
wg50 = siepic.ebeam_wg_integral_1550(length=50e-6)

# Create the circuit, add all individual instances
circuit = Subcircuit('MZI')
e = circuit.add([
    (gc, 'input'),
    (gc, 'output'),
    (y, 'splitter'),
    (y, 'recombiner'),
    (wg150, 'wg_long'),
    (wg50, 'wg_short'),
])

# You can set pin names individually:
circuit.elements['input'].pins['n2'] = 'input'
circuit.elements['output'].pins['n2'] = 'output'

# Or you can rename all the pins simultaneously:
circuit.elements['splitter'].pins = ('in1', 'out1', 'out2')
circuit.elements['recombiner'].pins = ('out1', 'in2', 'in1')

# Circuits can be connected using the elements' string names:
circuit.connect_many([
    ('input', 'n1', 'splitter', 'in1'),
    ('splitter', 'out1', 'wg_long', 'n1'),
    ('splitter', 'out2', 'wg_short', 'n1'),
    ('recombiner', 'in1', 'wg_long', 'n2'),
    ('recombiner', 'in2', 'wg_short', 'n2'),
    ('output', 'n1', 'recombiner', 'out1'),
])

# Run a simulation on the netlist.
simulation = SweepSimulation(circuit, 1500e-9, 1600e-9)
result = simulation.simulate()

f, s = result.data('input', 'output')
plt.plot(f, s)
plt.title("MZI")
plt.tight_layout()
plt.show()
```

![MZI simulation result (plot)](./docs/source/user/tutorials/images/plot_mzi.png)

More examples can be found in the 
[online documentation](https://simphonyphotonics.readthedocs.io/en/latest/user/tutorials/index.html).