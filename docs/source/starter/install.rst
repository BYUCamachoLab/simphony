.. _install:

============
Installation
============

.. Note::
   Simphony only supports Python 3.

In most cases, the best way to install Simphony on your system is by 
installing from PyPI, or "pip". Updates are regularly pushed as "minor" 
or "micro" (patch) versions, making upgrading very easy. 
Installation is as simple as ::

    pip install simphony

or, depending on your environment setup (for MacOS and Linux), ::
    
   pip3 install simphony

If you wish to install outside of pip, you can find prebuilt wheels under
`GitHub Releases`_.

.. _Github Releases: https://github.com/BYUCamachoLab/simphony/releases


Recommended Libraries
=====================

Matplotlib
----------

The easiest way to visualize the results of simulations is using 
`matplotlib`_. While not an explicit dependency of Simphony, the tutorials
use matplotlib to display what's happening in a simulation.

.. _matplotlib: https://matplotlib.org/

.. _companion-libraries:

Companion Libraries
===================

Several libraries have been developed for use with Simphony and are listed
below. If you have developed an open library for use with Simphony, please let
us know and we can add it to this list.

SiPANN
------

`SiPANN`_ (Silicon Photonics with Artificial Neural Networks) is a library that 
leverages various machine learning techniques to simulate integrated photonic 
device circuits, meaning it can very quickly and accurately simulate devices 
with varying parameters (such as waveguide width or thickness) without
having to run a full, slow, FDTD simulation before a designed device
can be used in a photonic integrated circuit (PIC) design software
such as Simphony.

See the `SiPANN`_ documentation for installation instructions. Note that, 
due to its large number of dependncies, it is a very large package.

.. _SiPANN: https://sipann.readthedocs.io/en/latest/

