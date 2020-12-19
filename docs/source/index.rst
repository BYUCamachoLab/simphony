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

The source repository is hosted on `GitHub`_. Prepackaged wheels of stable 
versions are in `Releases`_, along with the release history. An additional 
`Changelog`_ is included in the repository.

.. _GitHub: https://github.com/BYUCamachoLab/simphony
.. _Releases: https://github.com/BYUCamachoLab/simphony/releases
.. _Changelog: https://github.com/BYUCamachoLab/simphony/tree/master/docs/changelog

.. we can add links to silicon photonics, pics articles, etc.

* **Get familiar with Simphony**:
  :doc:`starter/intro`

* **Installation instructions**:
  :doc:`starter/install` | 
  :ref:`companion-libraries`

.. toctree::
   :hidden:
   :caption: Getting Started

   self
   Introduction <starter/intro>
   starter/install


Using Simphony
--------------

Learn the syntax and how to build useful circuits using our simple tutorials.

* **Build simple circuits**:
  :doc:`user/tutorials/mzi` | 
  :doc:`user/tutorials/filters`

* **More advanced circuits**:
  :doc:`user/tutorials/gm`

* **Using Simphony with SiEPIC**:
  :doc:`user/tutorials/siepic`

* **Useful design patterns**:
  :doc:`user/techniques/factory_subcircuits`

* **Use models from the available libraries**:
  :doc:`user/libraries/siepic` | 
  :doc:`user/libraries/sipann`

.. toctree::
   :hidden:
   :caption: Using Simphony

   user/tutorials/index
   Techniques <user/techniques/index>
   user/libraries/index
   user/simulators/index
   Integrations <user/integrations/index>


Development
-----------

Simphony was developed by `CamachoLab at Brigham Young University <https://camacholab.byu.edu/>`_ but also
strives to be an open-source project that welcomes the efforts of volunteers. 
If there is anything you feel can be improved, functionally or in our documentation,
we welcome your feedback -- let us know what the problem is or open a pull
request with a fix!

More information about the development of Simphony can be found at our 
`project webpage <https://camacholab.byu.edu/research/computational-photonics>`__.

The documentation is hosted for free at https://simphonyphotonics.readthedocs.io/.
The source for this documentation can be found in the master branch of the repository.

* **Documenting The Simphony Project**:
  :doc:`dev/docs/howto_document` | 
  :doc:`dev/docs/howto_build_docs`

* **Contributing to Simphony**:
  :doc:`dev/index`

* **Bugs and Feature Requests**:
  :doc:`dev/bugs`

.. toctree::
   :hidden:
   :caption: Development

   dev/index


Reference
---------

* **View the API**:
  :doc:`reference/api`

* **License agreement**:
  :doc:`reference/license`

* **How we name stuff**:
  :doc:`reference/glossary`

.. toctree::
   :hidden:
   :caption: Reference

   reference/api
   reference/license
   reference/glossary
