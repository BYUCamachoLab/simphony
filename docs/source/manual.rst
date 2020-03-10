.. _manual:

========================
The Simphony User Manual
========================

:Author: Sequoia Ploeg
:Version: 0.3.0
:Date: |today|
:Copyright:
  This work is licensed under the `MIT License`__.

.. __: https://opensource.org/licenses/MIT

:Abstract:
  This document explains how to use the Simphony Python package for
  simulation of photonic circuits.

.. _intro:

Introduction
============

Silicon photonics is a rapidly growing industry that uses electronic
integrated circuit fabrication technologies to produce industry-grade
photonic integrated circuits (PICs) at low cost and high volume.
Silicon photonic technologies have been largely driven by the
communications industry, but also find applications in sensing,
computing, and quantum information processing by enabling high data
transmission rates and controlled manipulation of light waves.

As the silicon photonics industry grows and the demand for PICs increases,
it is increasingly important for designers to have access to software
design tools that can accurately model and simulate PICs in a first-time-right
framework. Simulating PICs is a resource- and time-intensive process. Owing
to the long wavelengths of photons relative to electrons, photonic device
simulation requires solving Maxwell's equations with far less abstraction
than electronic circuit components. Once devices have been simulated and
bench-marked, however, compact models representing the phase and amplitude
response functions of individual components may be stitched together to form
functioning circuits. Although various commercial tools exist to perform these
functions, they are often expensive and limited in the variety and type of
photonic devices than can be simulated. Furthermore, there is often a lack of
standardization among platforms that in many cases prevents interoperability
between tools.

Here we present an open-source, software-implemented simulation tool for PICs
(documentation and downloads available at  github.com/BYUCamachoLab/simphony).
Our toolbox, which we name Simphony, provides fast simulations for PICs and
allows for the integration of device compact models that may be sourced from
a variety of platforms.  Simphony also provides the capability to add custom
components.  This interoperability is achieved by cascading device scattering
parameters, or S-parameters, for each component using sub-network growth
algorithms, a common practice in microwave/radio-frequency (RF) engineering.
Benchmark testing of Simphony against commercial software indicates a speedup
of approximately 20x.

.. _intro-circuit-data-model

Circuit Data Model
------------------

The fundamental building blocks of a simulation in most SPICE-like programs,
and therefore Simphony, are elements, nets, subcircuits, and circuits. A 
circuit can be fully represented by what is known as a Netlist and stored as
a single text file.

* An `Element` is ...

* A `Subcircuit` is ...

* A `Circuit` is ... and is different from a `Subcircuit` because ...




Elements can be added to circuits anonymously.