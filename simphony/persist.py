# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
simphony.persist
================

This package contains handy functions for exporting models created by other
libraries to a format that can be used by any simphony installation,
regardless of whether the creating library is installed locally.

Static models can be exported; that is to say, models that implement dynamic
functions like ``monte_carlo_s_parameters()`` will have those functions ignored,
as they are highly model-dependent. Static information, however, such as the
scattering parameters over a range of wavelengths, can be exported easily and
models recreated when imported. Other model attributes, such as pin names and
the valid frequency range, are also exported.

.. warning::
    Models are pickled files with a '.mdl' extension. Note that pickle has an
    inherent `security risk`_, so if you do not trust the source of the data, do
    not load the file!

.. _security risk: https://docs.python.org/3/library/pickle.html
"""

import inspect
import io
import os
import pickle

from simphony.elements import Model
from simphony.tools import interpolate, wl2freq
from numpy import ndarray
from py._path.local import LocalPath
from simphony.library.ebeam import ebeam_wg_integral_1550
from typing import Optional


def export_model(model: ebeam_wg_integral_1550, filename: LocalPath, wl: Optional[ndarray]=None, freq: None=None) -> None:
    """Exports a simphony model (using pickle) for the given
    frequency/wavelength range to a '.mdl' file.

    Must include either the wavelength or frequency argument. If both are
    included, defaults to frequency argument.

    Parameters
    -----------
    model : Model
        Any class inheriting from simphony.elements.Model
    filename : str
        The filename (may include path to directory) to save the model to.
        Note that the suffix '.mdl' will be appended to the filename.
    wl : ndarray, optional
        Wavelengths you want to save sparameters for (in meters).
    freq : ndarray, optional
        Frequencies you want to save sparameters for (in Hz).

    Examples
    --------
    We can write a model for a ``ebeam_wg_integral_1550`` instantiated with a
    length of 100 nanometers to a file  named ``wg100nm.mdl``.

    >>> import numpy as np
    >>> from simphony.library.ebeam import ebeam_wg_integral_1550
    >>> wg1 = ebeam_wg_integral_1550(100e-9)
    >>> export_model(wg1, 'wg100nm', wl=np.linspace(1520e-9, 1580e-9, 51))
    """
    if not issubclass(model.__class__, Model):
        raise ValueError("{} does not extend {}".format(model, Model))

    if wl is None and freq is None:
        raise ValueError("Frequency or wavelength range not defined.")

    # Convert wavelength to frequency
    if freq is None:
        freq = wl2freq(wl)[::-1]

    # Load all data into a dictionary.
    attributes = inspect.getmembers(model, lambda a: not (inspect.isroutine(a)))
    attributes = dict(
        [
            a
            for a in attributes
            if not (a[0].startswith("__") and a[0].endswith("__"))
            and not a[0].startswith("_")
        ]
    )

    params = dict()
    params["model"] = model.__class__.__name__
    params["attributes"] = attributes
    params["f"] = freq
    params["s"] = model.s_parameters(freq)

    # Dump to pickle.
    pickle.dump(
        params, io.open(filename + ".mdl", "wb"), protocol=pickle.HIGHEST_PROTOCOL
    )


def import_model(filename, force=False):
    """Imports a model from file.

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
    >>> waveguide_100nm = import_model('wg100nano.mdl')
    >>> wg = waveguide_100nm()
    >>> s = wg.s_parameters(np.linspace(wl2freq(1540e-9), wl2freq(1560e-9), 51))
    """
    path, ext = os.path.splitext(filename)
    if ext != ".mdl" and force == False:
        raise ValueError(
            "Requested file {} is not a .mdl file, to force load set parameter ``force=True``.".format(
                filename
            )
        )

    params = pickle.load(io.open(filename, "rb"))
    klass = type(params["model"], (Model,), params["attributes"])

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

    setattr(klass, "_f", params["f"])
    setattr(klass, "_s", params["s"])
    setattr(klass, "s_parameters", s_parameters)

    return klass
