# -*- coding: utf-8 -*-
# Copyright © 2019-2020 Simphony Project Contributors and others (see AUTHORS.txt).
# The resources, libraries, and some source files under other terms (see NOTICE.txt).
#
# This file is part of Simphony.
#
# Simphony is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Simphony is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Simphony. If not, see <https://www.gnu.org/licenses/>.

"""
simphony.plugins.interconnect
=================================

This package contains handy functions for importing/exporting sparam files used by
by Lumerical Interconnect

"""

import os

import numpy as np

from simphony.elements import Model
from simphony.netlist import Subcircuit
from simphony.simulation import SweepSimulation
from simphony.tools import interpolate, wl2freq


# Helper Functions
def _parse_header(line):
    return [i.strip(' "()\n') for i in line.split(",")]


def _parse_numbers(line):
    temp = [float(i) for i in line.strip().split(" ")]
    return temp[0], temp[1] * np.exp(1j * temp[2])


def export(model, filename, wl=None, freq=None, clear=True):
    """Exports a simphony model (using pickle) for the given
    frequency/wavelength range to a '.txt' file, with suffix of your choosing.

    Must include either the wavelength or frequency argument. If both are
    included, defaults to frequency argument.

    Parameters
    -----------
    model : Model
        Any class inheriting from simphony.elements.Model
    filename : str
        The filename (may include path to directory) to save the model to.
        Make sure to include a suffix.
    wl : ndarray, optional
        Wavelengths you want to save sparameters for (in meters).
    freq : ndarray, optional
        Frequencies you want to save sparameters for (in Hz).
    clear: bool, optional
        If True, empties the file first. Defaults to True.

    Examples
    --------
    We can write the sparameters for a ``ebeam_wg_integral_1550`` instantiated with a
    length of 100 nanometers to a file  named ``wg100nm.sparams``.

    >>> import numpy as np
    >>> from simphony.library.ebeam import ebeam_wg_integral_1550
    >>> wg1 = ebeam_wg_integral_1550(100e-9)
    >>> export(wg1, 'wg100nm.sparams', wl=np.linspace(1520e-9, 1580e-9, 51))
    """
    if wl is None and freq is None:
        raise ValueError("Frequency or wavelength range not defined.")

    # Convert wavelength to frequency
    if freq is None:
        freq = wl2freq(wl)[::-1]

    # Clear out file to write to
    if clear:
        open(filename, "w").close()
    file = open(filename, "ab")

    # get sparameters and pins
    if isinstance(model, Subcircuit):
        temp = SweepSimulation(
            model, freq.min(), freq.max(), len(freq), mode="freq"
        ).simulate()
        sparams = temp.s
        pins = temp.pins
    elif isinstance(model, Model):
        sparams = model.s_parameters(freq)
        pins = model.pins
    else:
        raise ValueError("Model isn't of type model or subcircuit")

    # iterate through sparams saving
    for in_ in range(len(pins)):
        for out in range(len(pins)):
            # put things together
            sp = sparams[:, out, in_]

            # only save if it's not 0's
            if sp.any():
                temp = np.vstack((freq, np.abs(sp), np.unwrap(np.angle(sp)))).T

                # Save header
                header = f'("{pins[out]}", "TE", 1, "{pins[in_]}", 1, "transmission")\n'
                header += f"{temp.shape}"

                # save data
                np.savetxt(file, temp, header=header, comments="")

    file.close()


def load(filename):
    """Makes a model from a Lumerical Sparameter File.

    Parameters
    ----------
    filename : str
        The filename (may include path to directory) to load the model from.

    Returns
    -------
    model : class
        A class that inherits from simphony.elements.Model that is the
        reconstructed model.

    Examples
    --------
    >>> waveguide_100nm = import('wg100nano.dat')
    >>> wg = waveguide_100nm()
    >>> s = wg.s_parameters(np.linspace(wl2freq(1540e-9), wl2freq(1560e-9), 51))
    """
    # open file
    with open(filename, "r") as f:
        lines = f.readlines()

    # parse frequencies and sparameters
    freq_final = None
    freq_i = 0
    sparams_final = dict()
    i = 0
    while i < len(lines):
        # parse headers - skip everything else
        if lines[i][0] == "(":
            # get first header
            values = _parse_header(lines[i])

            # skip if it corresponds to TM modes
            if values[1] == "TM":
                i += 1
                freq_num = int(_parse_header(lines[i])[0])
                i += freq_num
                continue

            # get ports
            out = values[0]
            in_ = values[3]

            # get next header and size of sparams
            i += 1
            freq_num = int(_parse_header(lines[i])[0])
            freq = np.zeros(freq_num)
            sparams = np.zeros(freq_num, dtype="complex64")
            freq_i = 0

            # parse through all numbers
            for freq_i in range(freq_num):
                i += 1
                # parse all corresponding values
                freq[freq_i], sparams[freq_i] = _parse_numbers(lines[i])
                freq_i += 1

                # save if we've hit the last frequency point
                if freq_i == freq_num:
                    sparams_final[(out, in_)] = sparams
                    if freq_final is None:
                        freq_final = freq
                    else:
                        if not np.allclose(freq_final, freq):
                            raise ValueError(
                                "Changing frequencies on ports. Simphony doesn't support."
                            )

        i += 1

    # put into final smatrix
    pins = list(set(pin for value in sparams_final.keys() for pin in value))
    sparams = np.zeros((freq_num, len(pins), len(pins)), dtype="complex64")
    for (out, in_), values in sparams_final.items():
        sparams[:, pins.index(out), pins.index(in_)] = values

    # make class for it
    klass = type("InterconnectImport", (Model,), {"pins": pins})

    def s_parameters(self, freq):
        """
        Get the s-parameters of a parameterized waveguide.
        Parameters
        ----------
        freq : np.ndarray
            A frequency array to calculate s-parameters over (in Hz).
        Returns
        -------
        s : np.ndarray
            Returns the calculated s-parameter matrix.
        """
        return interpolate(freq, self._f, self._s)

    setattr(klass, "_f", freq_final)
    setattr(klass, "_s", sparams)
    setattr(klass, "s_parameters", s_parameters)

    return klass
