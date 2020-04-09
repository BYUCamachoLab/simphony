About Simphony
==============

Simphony is a fundamental package for designing and simulating
photonic integrated circuits with Python. 

This package is still under development. It initially began as an extension to
`SiEPIC-Tools`_, but was broken off into its own independent project as its scope 
grew and it became large enough to be considered its own stand-alone project. 

NOTE: This is still under development; this next paragraph is *almost* true.
There is a repository forked from lukasc-ubc/SiEPIC-Tools, 
[SiEPIC-Tools](https://github.com/sequoiap/SiEPIC-Tools),
that integrates Simphony with SiEPIC-Tools and KLayout in order to perform 
photonic circuit simulations using a layout-driven design methodology.

.. _SiEPIC-Tools: https://github.com/lukasc-ubc/SiEPIC-Tools

This package contains:

- :ref:`subnetwork growth <connect-documentation>` routines
- a simple framework for defining photonic component :ref:`compact models <models-documentation>`
- a SPICE-like method for defining :ref:`photonic circuits <netlist-documentation>`
- complex :ref:`simulation capabilities <simulation-documentation>`

Simphony community
------------------

Simphony was developed by CamachoLab at Brigham Young University but also
strives to be an open-source project that welcomes the efforts of volunteers. 
If there is anything you feel can be improved, functionally or in our documentation,
we welcome your feedback -- let us know what the problem is or open a pull
request with a fix!

Our main means of communication are:

- `CamchoLab website <https://camacholab.byu.edu/>`__

- `Simphony Issues <https://github.com/BYUCamachoLab/simphony/issues>`__ (bug reports and feature requests go here)

More information about the development of Simphony can be found at our 
`project webpage <https://camacholab.byu.edu/research/computational-photonics>`__.


About this documentation
========================

Conventions
-----------

Names of classes, objects, constants, etc. should typically be linked to the
more detailed documentation of the referred object.

Examples presented are prefixed with the Python prompt ``>>>``. 

.. The
.. examples assume that you have first entered::

.. >>> import simphony

.. before running the examples.
