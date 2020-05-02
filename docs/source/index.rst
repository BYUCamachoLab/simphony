Simphony Documentation
======================

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


Getting Started
---------------

The source repository is hosted on GitHub. Prepackaged wheels of stable 
versions are in Releases, along with the release history. An additional 
Changelog is included in the repository.

.. toctree::
   :hidden:
   :caption: Getting Started

   about
   user/whatissimphony
   intro/siliconphotonics
   intro/pics


Setting Up
----------

Simphony only supports Python 3. Installation is really easy.::

   pip3 install simphony

If you wish to install outside of pip, you can find prebuilt wheels under
GitHub Releases.

.. toctree::
   :hidden:
   :caption: Setting Up

   user/setting_up
   user/quickstart
   user/install
   .. user/absolute_beginners
   .. user/explanations_index


Using Simphony
--------------

Learn the syntax and how to build useful circuits using our simple tutorials.

* **Build simple circuits**:
   :doc:`tutorials/mzi` | 
   :doc:`tutorials/gm`

* **Advanced circuits**:
   :doc:`tutorials/gm`

* **Using Simphony with SiEPIC**:
   :doc:`tutorials/siepic`

* **Useful design patterns**:
   :doc:`howtos/factory_subcircuits`

* **Use the preinstalled models**:
   :doc:`libraries/ebeam` | 
   :doc:`libraries/sipann`

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Using Simphony

   tutorials/index
   howtos/index
   libraries/index


Development
-----------

Simphony was developed by CamachoLab at Brigham Young University but also
strives to be an open-source project that welcomes the efforts of volunteers. 
If there is anything you feel can be improved, functionally or in our documentation,
we welcome your feedback -- let us know what the problem is or open a pull
request with a fix!

More information about the development of Simphony can be found at our 
`project webpage <https://camacholab.byu.edu/research/computational-photonics>`__.

The documentation is hosted for free at https://simphonyphotonics.readthedocs.io/.
The source for this documentation can be found in the master branch of the repository.

**Conventions**

Names of classes, objects, constants, etc. should typically be linked to the
more detailed documentation of the referred object.

Examples presented are prefixed with the Python prompt ``>>>``. 

* **Documenting The Simphony Project**:
   :doc:`dev/docs/howto_document` | 
   :doc:`dev/docs/howto_build_docs`

* **Contributing to Simphony**:
   :doc:`dev/index` | 
   :doc:`dev/style_guide`

* **Bugs and Feature Requests**:
   :doc:`dev/bugs`

.. toctree::
   :hidden:
   :caption: Development

   dev/docs/howto_document
   dev/docs/howto_build_docs
   dev/index
   dev/style_guide
   .. benchmarking
   .. release


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
