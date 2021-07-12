Simphony Documentation
======================

Simphony allows you to define photonic circuits, then run
fast simulations on them, all in Python.

- Simphony is free and open-source
- Runs on Windows, MacOS, and Linux
- Uses a SPICE-like method for defining photonic circuits
- Subnetwork growth algorithms, giving 20x speedup over
  other photonic modeling software
- Includes libraries for circuit components (known as models)
- Provides a simple framework for defining new models 

**To install Simphony**, simply use the following in a
Python 3 environment: ::

  pip install simphony

There are also prebuilt releases available on `GitHub`_.

.. Note::
  We recommend installing two libraries, `matplotlib`_ and
  `SiPANN`_, alongside Simphony. Matplotlib provides a way 
  to visualize the results from your simulations, and SiPANN
  provides additional models for use in your circuits. View
  the links for installation instructions and find out
  more.

**To get started using Simphony**, check out
:doc:`tutorials/intro`. Tutorials and API references are
accessible through the sidebar navigation.

.. toctree::
  :hidden:
  :caption: Tutorials

  tutorials/intro
  tutorials/mzi
  tutorials/filters

.. toctree::
  :hidden:
  :caption: Reference

  reference/api
  reference/libraries

Simphony is primarily developed and maintained by members of
the `CamachoLab`_ at Brigham Young University. Feedback is
welcome: if you find errors or have suggestions for the
Simphony project, let us know by raising an issue on
`Github`_. If you want to contribute, even better! See
:doc:`dev/contributing` to learn how.

.. toctree::
  :hidden:
  :caption: Development

  dev/contributing
  dev/developing
  dev/documenting

.. _Github: https://github.com/BYUCamachoLab/simphony
.. _CamachoLab: https://camacholab.byu.edu
.. _matplotlib: https://matplotlib.org/
.. _SiPANN: https://sipann.readthedocs.io/en/latest/
