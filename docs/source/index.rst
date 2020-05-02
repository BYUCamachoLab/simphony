Simphony Documentation
======================

.. toctree::
   :hidden:

   self

.. image:: /images/simphony_logo.jpg
   :alt: Simphony (Image)

Simphony, a simulator for photonic circuits, is a fundamental package for 
designing and simulating photonic integrated circuits with Python. 

**Key Features:**

- Free and open-source software provided under the MIT License
- Completely scriptable using Python 3.
- Cross-platform: runs on Windows, MacOS, and Linux.
- Subnetwork growth routines
- A simple, extensible framework for defining photonic component compact models.
- A SPICE-like method for defining photonic circuits.
- Complex simulation capabilities.
- Included model libraries from SiEPIC and SiPANN.


Download
--------

The source repository is hosted on GitHub. Prepackaged wheels of stable 
versions are in Releases, along with the release history. An additional 
Changelog is included in the repository.

.. toctree::
   :hidden:
   :caption: Setting Up

   user/setting_up
   user/quickstart
   .. user/absolute_beginners
   user/howtos_index
   reference/index
   .. user/explanations_index


Installation
------------

Simphony only supports Python 3. Installation is really easy.::

   pip3 install simphony

If you wish to install outside of pip, you can find prebuilt wheels under
GitHub Releases.


Documentation
-------------

The documentation is hosted for free at https://simphonyphotonics.readthedocs.io/.
The source for this documentation can be found in the master branch of the repository.

**Conventions**

Names of classes, objects, constants, etc. should typically be linked to the
more detailed documentation of the referred object.

Examples presented are prefixed with the Python prompt ``>>>``. 

.. toctree::
   :hidden:
   :caption: Documentation

   docs/howto_document
   docs/howto_build_docs

Tutorials
---------

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorials

   tutorials/filters
   tutorials/gm
   tutorials/mzi


Simphony How-To's
-----------------

These documents are intended as recipes for common tasks using Simphony. 
For detailed reference documentation of the functions and classes 
contained in the package, see the :ref:`API reference <reference>`.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: How To's

   howtos/factory_subcircuits


Simphony Community
------------------

Simphony was developed by CamachoLab at Brigham Young University but also
strives to be an open-source project that welcomes the efforts of volunteers. 
If there is anything you feel can be improved, functionally or in our documentation,
we welcome your feedback -- let us know what the problem is or open a pull
request with a fix!

More information about the development of Simphony can be found at our 
`project webpage <https://camacholab.byu.edu/research/computational-photonics>`__.

.. toctree::
   :hidden:
   :caption: Contribute

   dev/index
   dev/style_guide
   .. benchmarking
   .. release

Bugs and Feature Requests
-------------------------

File bug reports or feature requests, and make contributions
(e.g. code patches), by opening a "new issue" on GitHub:

- Simphony Issues: https://github.com/BYUCamachoLab/simphony/issues

Please give as much information as you can in the ticket. It is extremely
useful if you can supply a small self-contained code snippet that reproduces
the problem. Also specify the component or module and the version you are 
using.

.. toctree::
   :hidden:
   :caption: Libraries

   libraries/index


Acknowledgements
----------------

We would like to give credit where credit is due.
Much of the documentation is based on the documentation for NumPy.
It was, in fact, used as a model and starting point for this project's
documentation. Kudos to them for such an excellent structure and
a job well done.

.. toctree::
   :hidden:
   :caption: Reference

   reference/api
   reference/license
   reference/glossary
   reference/appendix
