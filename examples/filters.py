#!/usr/bin/env python3
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)
#
# File: filters.py

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from SiPANN import scee

from simphony import Model
from simphony.libraries import siepic
from simphony.simulators import SweepSimulator
from simphony.tools import freq2wl


class SimphonyWrapper(Model):
    """Class that wraps SCEE models for use in simphony.
    Model passed into class CANNOT have varying geometries, as a device such as this
    can't be cascaded properly.
    Parameters
    -----------
    model : DC
        Chosen compact model from ``SiPANN.scee`` module. Can be any model that inherits from
        the DC abstract class
    sigmas : dict, optional
        Dictionary mapping parameters to sigma values for use in monte_carlo simulations. Note sigmas should
        be in values of nm. Defaults to an empty dictionary.
    """

    pin_count = 4
    freq_range = (
        182800279268292.0,
        205337300000000.0,
    )  #: The valid frequency range for this model.

    def __init__(self, model, sigmas=dict()):
        super().__init__()

        self.model = model
        self.sigmas = sigmas

        # save actual parameters for switching back from monte_carlo
        self.og_params = self.model.__dict__.copy()
        self.rand_params = dict()

        # make sure there's no varying geometries
        args = self.model._clean_args(None)
        if len(args[0]) != 1:
            raise ValueError(
                "You have changing geometries, use in simphony doesn't make sense!"
            )

        self.regenerate_monte_carlo_parameters()

    def s_parameters(self, freq):
        """Get the s-parameters of SCEE Model.
        Parameters
        ----------
        freq : np.ndarray
            A frequency array to calculate s-parameters over (in Hz).
        Returns
        -------
        s : np.ndarray
            Returns the calculated s-parameter matrix.
        """
        # convert wavelength to frequency
        wl = freq2wl(freq) * 1e9

        return self.model.sparams(wl)

    def monte_carlo_s_parameters(self, freq):
        """Get the s-parameters of SCEE Model with slightly changed parameters.
        Parameters
        ----------
        freq : np.ndarray
            A frequency array to calculate s-parameters over (in Hz).
        Returns
        -------
        s : np.ndarray
            Returns the calculated s-parameter matrix.
        """
        wl = freq2wl(freq) * 1e9

        # perturb params and get sparams
        self.model.update(**self.rand_params)
        sparams = self.model.sparams(wl)

        # restore parameters to originals
        self.model.update(**self.og_params)

        return sparams

    def regenerate_monte_carlo_parameters(self):
        """Varies parameters based on passed in sigma dictionary.

        Iterates through sigma dictionary to change each of those
        parameters, with the mean being the original values found in
        model.
        """
        # iterate through all params that should be tweaked
        for param, sigma in self.sigmas.items():
            self.rand_params[param] = np.random.normal(self.og_params[param], sigma)


def ring_factory(radius):
    """Creates a full ring (with terminator) from a half ring.

    Ports of a half ring are ordered like so:
    2           4
     |         |
      \       /
       \     /
     ---=====---
    1           3

    Resulting pins are ('pass', 'in', 'out').

    Parameters
    ----------
    radius : float
        The radius of the ring resonator, in nanometers.
    """
    # Have rings for selecting out frequencies from the data line.
    # See SiPANN's model API for argument order and units.
    halfring1 = SimphonyWrapper(scee.HalfRing(500, 220, radius, 100))
    halfring2 = SimphonyWrapper(scee.HalfRing(500, 220, radius, 100))
    terminator = siepic.Terminator()

    halfring1.rename_pins("pass", "midb", "in", "midt")
    halfring2.rename_pins("out", "midt", "term", "midb")

    # the interface method will connect all of the pins with matching names
    # between the two components together
    halfring1.interface(halfring2)
    halfring2["term"].connect(terminator)

    # bundling the circuit as a Subcircuit allows us to interact with it
    # as if it were a component
    return halfring1.circuit.to_subcircuit()


# Behold, we can run a simulation on a single ring resonator.
ring1 = ring_factory(10000)

simulator = SweepSimulator(1500e-9, 1600e-9)
simulator.multiconnect(ring1["in"], ring1["pass"])

f, t = simulator.simulate(mode="freq")
plt.plot(f, t)
plt.title("10-micron Ring Resonator")
plt.tight_layout()
plt.show()

simulator.disconnect()

# Now, we'll create the circuit (using several ring resonator subcircuits)
# instantiate the basic components
wg_input = siepic.Waveguide(100e-6)
wg_out1 = siepic.Waveguide(100e-6)
wg_connect1 = siepic.Waveguide(100e-6)
wg_out2 = siepic.Waveguide(100e-6)
wg_connect2 = siepic.Waveguide(100e-6)
wg_out3 = siepic.Waveguide(100e-6)
terminator = siepic.Terminator()

# instantiate the rings with varying radii
ring1 = ring_factory(10000)
ring2 = ring_factory(11000)
ring3 = ring_factory(12000)

# connect the circuit together
ring1.multiconnect(wg_connect1, wg_input["pin2"], wg_out1)
ring2.multiconnect(wg_connect2, wg_connect1, wg_out2)
ring3.multiconnect(terminator, wg_connect2, wg_out3)

# prepare the plots
fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1, 3)
ax = fig.add_subplot(gs[0, :2])

# prepare the simulator
simulator = SweepSimulator(1524.5e-9, 1551.15e-9)
simulator.connect(wg_input)

# get the results for output 1
simulator.multiconnect(None, wg_out1)
wl, t = simulator.simulate()
ax.plot(wl * 1e9, t, label="Output 1", lw="0.7")

# get the results for output 2
simulator.multiconnect(None, wg_out2)
wl, t = simulator.simulate()
ax.plot(wl * 1e9, t, label="Output 2", lw="0.7")

# get the results for output 3
simulator.multiconnect(None, wg_out3)
wl, t = simulator.simulate()
ax.plot(wl * 1e9, t, label="Output 3", lw="0.7")

# finish first subplot and move to next
ax.set_ylabel("Fractional Optical Power")
ax.set_xlabel("Wavelength (nm)")
plt.legend(loc="upper right")
ax = fig.add_subplot(gs[0, 2])

# get the results for output 1
simulator.multiconnect(None, wg_out1)
wl, t = simulator.simulate()
ax.plot(wl * 1e9, t, label="Output 1", lw="0.7")

# get the results for output 2
simulator.multiconnect(None, wg_out2)
wl, t = simulator.simulate()
ax.plot(wl * 1e9, t, label="Output 2", lw="0.7")

# get the results for output 3
simulator.multiconnect(None, wg_out3)
wl, t = simulator.simulate()
ax.plot(wl * 1e9, t, label="Output 3", lw="0.7")

# plot the results
ax.set_xlim(1543, 1545)
ax.set_ylabel("Fractional Optical Power")
ax.set_xlabel("Wavelength (nm)")
fig.align_labels()
plt.show()
