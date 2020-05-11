.. _factory-method-design-pattern:

=================================
The Factory Method Design Pattern
=================================

.. currentmodule:: simphony

.. testsetup::

   >>> import simphony

Prerequisites
=============

Before reading this tutorial you should already be familiar with
the basic objects and models used by Simphony. Knowledge of 
how SPICE models work in many electronics simulation software packages
may also be helpful. 

**Learner profile**

This tutorial is intended as a quick overview of the factory pattern
and how it can be employed to help you create robust, easy-to-customize, 
large-scale circuits without hard-coding every detail.

**Learning Objectives**

After this tutorial, you should be able to:

- Identify situations where this design pattern should be applied to
  avoid code repetition when creating large circuits.
- Write your own factory method two produce parameterized subcircuits.
- Cascade subcircuits together in a loop to create larger circuits.

.. _howtos.factory-subcircuits:

The Factory Method
==================

What is the factory pattern?
----------------------------

The factory design pattern is a design pattern where there exists
several related types of objects and a "factory" generates desired
objects based on parameters received or attributes desired.

For our purposes, the factory pattern becomes appealing anytime we
want to create several parameterized variatons of a compound structure 
using simple elements (or even other compound elements, as they
can be nested).

What kind of structures warrant the factory pattern?
----------------------------------------------------

Here's a few examples of when a factory pattern would be useful:

- A circuit composed of MZI's of varying length.
- A circuit composed of ring resonators with radii spread over some range.
- A circuit composed of bragg grating structures with different
  periodicities.





