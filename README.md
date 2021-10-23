# Simphony: A Simulator for Photonic Circuits

<p align="center">
  <img alt="Development version" src="https://img.shields.io/badge/master-v0.5.0-informational"/>

  <a href="https://pypi.python.org/pypi/simphony">
    <img alt="PyPI Version" src="https://img.shields.io/pypi/v/simphony.svg"/>
  </a>

  <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/simphony"/>

  <a href="https://github.com/BYUCamachoLab/simphony/actions?query=workflow%3A%22build+%28pip%29%22">
    <img alt="Build Status" src="https://github.com/BYUCamachoLab/simphony/workflows/build%20(pip)/badge.svg"/>
  </a>

  <a href="https://github.com/pre-commit/pre-commit">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white" alt="pre-commit" style="max-width:100%;"/>
  </a>

  <a href="https://simphonyphotonics.readthedocs.io/">
    <img alt="Documentation Status" src="https://readthedocs.org/projects/simphonyphotonics/badge/?version=latest"/>
  </a>

  <a href="https://pypi.python.org/pypi/simphony/">
    <img alt="License" src="https://img.shields.io/pypi/l/simphony.svg"/>
  </a>
  
  <a href="https://github.com/BYUCamachoLab/simphony/commits/master">
    <img alt="Latest Commit" src="https://img.shields.io/github/last-commit/BYUCamachoLab/simphony.svg"/>
  </a>
</p>

Simphony allows you to define photonic circuits, then run
fast simulations on them, all in Python.

- Simphony is free and open-source
- Runs on Windows, MacOS, and Linux
- Uses a SPICE-like method for defining photonic circuits
- Subnetwork growth algorithms, giving 20x speedup over
  other photonic modeling software
- Includes libraries for circuit components (known as models)
- Provides a simple framework for defining new models 

To install Simphony, simply use the following in a
Python 3 environment:

```bash
pip install simphony
```

Or use the prebuilt releases under Github's "Releases".
Changelogs are included there, and also in the source code
at `docs/changelog`.

The documentation is hosted [here](https://simphonyphotonics.readthedocs.io/en/latest/).

Simphony is primarily developed and maintained by members of
the [CamachoLab](https://camacholab.byu.edu/) at 
[Brigham Young University](https://www.byu.edu/). Feedback
is welcome: if you find errors or have suggestions for the
Simphony project, let us know by raising an issue. If you
want to contribute, see the [documentation](https://simphonyphotonics.readthedocs.io/en/latest/)
to learn more.


## Example

We can simulate this circuit, known as a Mach-Zender
Interferometer (MZI), in Simphony:

<p align="center">
<img src="https://raw.githubusercontent.com/BYUCamachoLab/simphony/master/docs/source/tutorials/images/mzi.png" width="50%">
</p>

We can define it:

```python
from simphony.libraries import siepic

input = siepic.GratingCoupler()
splitter = siepic.YBranch()
waveguide_long = siepic.Waveguide(length=150e-6)
waveguide_short = siepic.Waveguide(length=50e-6)
recombiner = siepic.YBranch()
output = siepic.GratingCoupler()

splitter.multiconnect(input, waveguide_short, waveguide_long)
recombiner.multiconnect(output, waveguide_long, waveguide_short)
```

Then simulate it:

```python
import matplotlib.pyplot as plt
from simphony.simulators import SweepSimulator

simulator = SweepSimulator(1500e-9, 1600e-9)
simulator.multiconnect(input, output)

f, p = simulator.simulate()
plt.plot(f, p)
plt.title("MZI")
plt.tight_layout()
plt.show()
```
Which shows our desired results:

<p align="center">
  <img 
    src="https://raw.githubusercontent.com/BYUCamachoLab/simphony/master/docs/source/tutorials/images/plot_mzi.png"
    width="50%">
</p>

For a deeper walkthrough of the same circuit, see the 
[doc page](https://simphonyphotonics.readthedocs.io/en/latest/tutorials/mzi.html).
