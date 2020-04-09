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

Notes about the implementation

Simphony aims to be Pythonic, yet pragmatic. Instead of reinventing a new
framework for everyone to learn, we build off the concepts that engineers and
scientists are already familiar with in electronics: the SPICE way of 
organizing and defining circuits and connections. In this case, we use much
of the same terminology but make it Python friendly (and, in our opinion,
improving upon its usability).

Simphony follows Python's EAFP (easier to ask forgiveness than permission) 
coding style. This contrasts with the LBYL (look before you leap) style common
to other languages. In practice, this means that if, say, a library element
component is implemented but is missing attributes, it won't be noticed until
runtime when a call to a nonexistent attribute throws an exception.

Python often uses magic methods (also known as "dunder" methods) to implement
underlying class functionality. Simphony uses the same convention, but with
what we'll call "sunder" methods (for single-underscore methods), since
Python's dunder syntax is reserved for future Python features.

Units throughout simphony are all SI units (unless otherwise specified) 
to avoid ambiguity and confusion. This can sometimes lead to not-as-pretty 
looking values, especially when dealing with sub-wavelength values and 
frequencies in THz. But, it is consistent.

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

Subcircuits
-----------

Creating subcircuits is therefore just as easy. Often, a circuit can be 
broken up into smaller circuit segments that make up the whole design.
Subcircuits allow us to create these (for example, cascading a set of
ring resonators of varying radius).

A SPICE subcircuit (.subckt) wraps around a block of circuit text and allows 
external connections to this circuitry only through the subcircuit's nodes. 

Because the internal circuitry is isolated from external circuitry, internal 
devices and node names with the same names as those external to the 
subcircuit are neither conflicting nor shorted. In addition, subcircuits can 
accept circuit parameters which can be used to assign values to internal 
devices or nested subcircuits. 