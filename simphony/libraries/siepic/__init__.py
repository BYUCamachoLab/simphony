# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
simphony.libraries.siepic
=========================

This package contains parameterized models of PIC components from the SiEPIC
Electron Beam Lithography Process Development Kit (PDK), which is licensed
under the terms of the MIT License.

Terminology
-----------

argset
******

The code and documentation of this module frequently refer to something
hereafter known as an "argset". An *argset* is a dictionary of parameters
to values.

For example, let's look at the data files available for a y-branch coupler,
as available in `source_data/y_branch_source`. The filename has the format::

    Ybranch_Thickness =220 width=500.sparam

When naming argsets, we use lowercase, underescore-delimited words by
convention. In this case, our keys are `thickness` and `width and an argset
generated from this filename is::

    {'thickness': '220', 'width': '500'}

It is the responsibility of the model processing argsets to convert this to the
normalized form used by the model for comparing its parameters to available
data files. For example, the y-branch takes the following parameters::

    class YBranch(SiEPIC_PDK_Base):
        def __init__(self, thickness=220e-9, width=500e-9, polarization='TE'):
            ...

Note that `thickness` and `width` are both floats; lengths, in meters. However,
the values parsed from the datafile are strings representing lengths in
nanometers. The model therefore creates a normalized set of argsets by
converting the values to a form that can be compared with the arguments
received by `__init__()`.

Suppose we have the following filename::

    te_ebeam_dc_halfring_straight_gap=30nm_radius=3um_width=520nm_thickness=210nm_CoupleLength=0um.dat

An argset generated from this filename would be::

    {'gap': '30n', 'radius': '3u', 'width': '520n', 'thickness': '210n', 'couple_length': '0u'}

A normalized version of the above argset would be::

    {'gap': 30e-9, 'radius': 3e-6, 'width': 520e-9, 'thickness': 210e-9, 'couple_length': 0.0}


normalized
**********
A variation on argset where the values are formatted to be comparable to the
attributes stored by the model. This means that while an argset always reads
in strings from datafile names, a normalized argset converts the value to
whatever value it actually represents (usually floats).


Future Work
-----------
Perhaps we should load the .sparam data files only when `s_parameters()` is
called, instead of each time when values are changed. If 2+ attributes are
changed, that turns into a lot of extra (needless) file loading.
"""

import os
import re
import string
import warnings
from bisect import bisect_left
from collections import namedtuple

import numpy as np
from scipy.constants import c as SPEED_OF_LIGHT

from simphony import Model
from simphony.libraries.siepic import parser
from simphony.tools import interpolate, str2float


def closest(sorted_list, value):
    """Assumes `sorted_list` is sorted. Returns closest value to `value`.

    If two numbers are equally close, return the smallest number.

    Parameters
    ----------
    sorted_list : list of ints or floats
    value : int or float

    Returns
    -------
    closest : int or float

    References
    ----------
    https://stackoverflow.com/a/12141511/11530613
    """
    pos = bisect_left(sorted_list, value)
    if pos == 0:
        return sorted_list[0]
    if pos == len(sorted_list):
        return sorted_list[-1]
    before = sorted_list[pos - 1]
    after = sorted_list[pos]
    if after - value < value - before:
        return after
    else:
        return before


def get_files_from_dir(path):
    """Gets the string name of every file in a given directory.

    Parameters
    ----------
    path : str
        The absolute path to the directory where the files should be found.

    Returns
    -------
    files : list
        A list of filenames as strings.
    """
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def extract_args(strings, regex, args):
    """
    Parameters
    ----------
    strings : list of str
        A list of the strings containing parameters to be extracted.
    regex : str
        A string representing the regex used to extract the values from the
        strings.
    args : list of str
        A list of strings that will be used as keys in the dictionary of values
        returned. These must be in the same order as the parameters will be
        extracted by the regex.

    Returns
    -------
    argsets : list
        A list of all parameter combinations as dictionaries of `args` to
        values extracted from the string.
    """
    argsets = []
    for str in strings:
        matches = re.findall(regex, str)
        if len(matches):
            argsets.append(dict(zip(args, matches[0])))
    return argsets


def percent_diff(ideal, actual):
    """Calculates the percent error.

    Ideally the parameters are of a numeric nature. However, if they are of
    other types (say, string) a simple value of "1.0" is returned, representing
    the fact that the two objects are utterly, entirely different.

    Parameters
    ----------
    ideal : float, int, or object
        The verified accepted value.
    actual : float, int, or object
        The measured or desired value.

    Returns
    -------
    error : float
        The percent error of actual from ideal.

    Notes
    -----
    Percent error is calculated as (ideal - actual) / ideal
    """
    # TODO: Make actual, if 0, not zero.
    try:
        return (ideal - actual) / ideal
    except TypeError:
        return 1.0


class SiEPIC_PDK_Base(Model):
    """A base model that includes pre-implemented functions for reading,
    building, and selecting appropriate .sparam files.

    This class is a template to be subclassed and is not supposed to be
    initialized on its own. Note that the `__init__()` function of subclasses
    ought to create a normalized set of loaded available parameters that can
    be compared to the parameters passed in upon construction. Additionally,
    after finding the most matching parameter set, the parameters should be
    stored locally for future reference.

    By default, a recalculation is triggered to update the s-parameters of the
    model anytime an instance attribute that is also found in `_args_keys` is
    modified. To suspend or enable autoupdating, see the functions
    `enable_autoupdate()` and `suspend_autoupdate()`.

    To define a fully working model that subclasses `SiEPIC_PDK_Base`,
    only `__init__()`, `on_args_changed()`, `s_parameters()`, and the class
    attributes need to be redefined. A subclass' `__init__()` function should
    call `super().__init__()` and pass in all parameters that will be saved
    as attributes. The call to `super` will automatically save them as
    instance attributes. WARNING: The child class' `__init__()` should NOT
    save instance attributes itself! They should all be passed to `super`.

    Parameters
    ----------
    **kwargs : dict
        The variables that parameterize this component. All are stored as
        object instance attributes.

    Attributes
    ----------
    args
    pins : tuple of str
        The default pin names of the device.
    _base_path : str
        Path to directory containing .sparam files. This should be redefined
        in every subclass.
    _base_file : str
        A string template that can be filled with argument values to load
        the appropriate .sparam file. This should be redefined in every
        subclass.
    _args_keys : list of str
        The arguments as found in the filename of `.sparam` files. Note that
        model `kwargs` should match these names, as they are matched by string.
        The `_base_file` is also filled by matching keywords to the keys
        defined here. This attribute should be redefined by all subclasses.
    _args_trigger_update : list of str
        Other attributes that, if changed, should also trigger a callback to
        `on_args_changed()`. Can be an empty list.
    _regex : str
        The regular expression that selects available parameters from
        filenames. This attribute should be redefined by all subclasses.

    Warning
    -------
    The child class' `__init__()` should NOT save instance attributes itself!
    They should all be passed to `super`.
    """

    _base_path = os.path.join(os.path.dirname(__file__), "source_data")
    _base_file = string.Template("filepattern.sparam")
    _args_keys = []
    _args_trigger_update = []
    _regex = None

    # Do not redefine the variables between these lines in subclasses!
    # -------------------------------------------------------------------------
    _autoupdate = False
    _argset = None
    # -------------------------------------------------------------------------

    def __init__(self, **kwargs):
        model_params = ("name", "freq_range", "pins")
        model_args = {param: kwargs.get(param, None) for param in model_params}
        model_args["name"] = (
            model_args["name"] if model_args["name"] is not None else ""
        )
        super().__init__(**model_args)

        for key, value in kwargs.items():
            if key not in model_params:
                setattr(self, key, value)

        self.enable_autoupdate()
        self.on_args_changed()

    def __setattr__(self, name, value):
        """If the autoupdate mechanism is enabled, callback to
        `on_args_changed()` is performed when an attribute also found in
        `_args_keys` is set."""
        super().__setattr__(name, value)
        if self._autoupdate and (
            (name in self._args_keys) or (name in self._args_trigger_update)
        ):
            self.on_args_changed()

    @property
    def args(self):
        """A mapping of args (as found in `_args_keys`) to the stored attribute
        values.

        Returns
        -------
        args : dict
            A property that generates a dictionary of keys to instance values. Keys
            are specified by `_args_keys`.
        """
        return {k: getattr(self, k) for k in self._args_keys}

    def on_args_changed(self):
        """Callback for when model attributes are changed; updates the stored
        s-parameters based on current model attributes.

        This function is triggered any time an attribute that is in the model's
        argument list is changed. For example, if the thickness of the model's
        waveguides are changed, and thickness is a member of the class's
        `_args_keys`, this function will automatically be called.

        This function should operate only on instance attributes and should
        not accept parameters.

        In summary, ``on_args_changed()`` must do the following things:

            1. Disable autoupdate (``suspend_autoupdate()``).

            2. Normalize all argsets for comparison with model attributes.

            3. Pass the list of normalized argsets to ``_get_matched_args()``,
               which returns a single normalized argset most closely matching
               the model's attributes.

            4. Update the model's attributes with valid values; values for
               which we actually have simulation data (so, the values in the
               argset returned by ``_get_matched_args()``.

            5. Load the s-parameters from file and store them in a way they can
               later be accessed by `s_parameters()`, a function also
               implemented on a class-by-class basis.

            6. Set the instance attribute ``freq_range``; if this is not set, all
               simulations on circuits incorporating this model will fail.

            7. Enable autoupdate (``enable_autoupdate()``).

        Warning
        -------
        This function will silently change parameters to existing values if no
        matching data file can be found. For example, if some parameter
        `radius` has a requested value of 17 but there only exists data for
        values 15 and 20, the value of radius may be forced to 15 without
        raising any errors.

        A change to any attribute results in a call to this function.
        To avoid an infinite recursive loop, you should call
        `suspend_autoupdate()` before modifying any instance attributes. At
        the end of the function, re-enable autoupdate by calling
        `enable_autoupdate()`.
        """
        raise NotImplementedError

    def suspend_autoupdate(self):
        """Prevents the autoupdate of models when object attributes are
        modified."""
        self._autoupdate = False

    def enable_autoupdate(self):
        """Enables the autoupdate of models when object attributes are
        modified."""
        self._autoupdate = True

    @classmethod
    def _source_argsets(cls):
        """Generates the argsets that match .sparam filename conventions, based
        on class attributes.

        Return
        ------
        argsets : list of dicts
            A list of all available parameter combinations in the source files.
            These are not normalized.
        """
        try:
            return cls._available_argsets
        except AttributeError:
            files = get_files_from_dir(cls._base_path)
            cls._available_argsets = extract_args(files, cls._regex, cls._args_keys)
            return cls._available_argsets

    @classmethod
    def _get_file(cls, argset):
        """Given the selected argset, get the path to the appropriate data
        file.

        Parameters
        ----------
        argset : dict
            A dictionary of arguments used to load the appropriate data file.
            Keys should match names in the string template `_base_file`.

        Returns
        -------
        path : str
            A path to the data file matching the provided argument set.
        """
        return os.path.join(cls._base_path, cls._base_file.substitute(**argset))

    @classmethod
    def _get_matched_args(cls, norm_args, req_args):
        """Finds the argset from a set of normalized argsets most similar to
        the requested argset.

        Parameters
        ----------
        norm_args : list of dicts
            The available argsets formatted with values as floats instead
            of strings.
        req_args : list of dicts
            The requested "reference" argset; each `norm_args` argset is
            compared to `req_args` to find the most similar.

        Returns
        -------
        idx : int
            The index of `norm_args` that best matches `req_args`.

        Warns
        -----
        UserWarning
            Warns if exact requested parameters are not available.
        """
        try:
            return norm_args.index(req_args)
        except ValueError:
            adjusted_args = cls._find_closest(norm_args, req_args)
            msg = (
                "Exact parameters not available for '{}', ".format(cls)
                + "using closest approximation (results may not be as accurate).\n"
                + "{:<11}{}\n".format("Requested:", req_args)
                + "{:<11}{}\n".format("Selected:", adjusted_args)
                + "NOTE: Model attributes may have been automatically modified."
            )
            warnings.warn(msg, UserWarning)
            return norm_args.index(adjusted_args)

    @staticmethod
    def _find_closest(normalized, args):
        """General function for selecting a device with the most similar
        parameters.

        First, the parameter sets with the fewest mismatched parameters are
        chosen. If there are more than one, a "similarity" analysis is
        performed on each set of parameters.

        Parameters
        ----------
        normalized : list of dict
            A normalized dict of all argsets for which we have data files.
        args : dict
            The device attributes we'd ideally like to match.

        Returns
        -------
        argset : dict
            The normalized argset most closely (if not exactly) matching
            the desired arguments in *args*.
        """
        diffs = []
        Candidate = namedtuple("Candidate", ["count", "keys", "argset"])
        for argset in normalized:
            diff_keys = [k for k, v in argset.items() if v != args[k]]
            diff_count = len(diff_keys)
            diffs.append(Candidate(diff_count, diff_keys, argset))
        min_diff = min(c.count for c in diffs)
        candidates = [c for c in diffs if c.count == min_diff]

        errors = []
        for count, keys, argset in candidates:
            sum_error = sum([abs(percent_diff(argset[key], args[key])) for key in keys])
            errors.append(sum_error / count)
        idx = np.argmin(errors)
        return candidates[idx].argset


class BidirectionalCoupler(SiEPIC_PDK_Base):
    """A bidirectional coupler optimized for TE polarized light at 1550
    nanometers.

    The bidirectional coupler has 4 ports, labeled as pictured. Its efficiently
    splits light that is input from one port into the two outputs on the opposite
    side (with a corresponding pi/2 phase shift). Additionally, it efficiently
    interferes lights from two adjacent inputs, efficiently splitting the
    interfered signal between the two ports on the opposing side.

    .. image:: /reference/images/ebeam_bdc_te1550.png
        :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    thickness : float, optional
        Waveguide thickness, in meters (default 220 nanometers). Valid values
        are 210, 220, or 230 nanometers.
    width : float, optional
        Waveguide width, in meters (default 500 nanometers). Valid values are
        480, 500, or 520 nanometers.
    """

    pin_count = 4
    _base_path = os.path.join(os.path.dirname(__file__), "source_data", "bdc_TE_source")
    _base_file = string.Template("bdc_Thickness =${thickness} width=${width}.sparam")
    _args_keys = ["thickness", "width"]
    _regex = (
        r"(?:bdc_Thickness =)"
        r"([-+]?[0-9]+(?:[.][0-9]+)?[fpnumckGMT]?)"
        r"(?: width=)"
        r"([-+]?[0-9]+(?:[.][0-9]+)?[fpnumckGMT]?)"
        r"(?:\.sparam)"
    )

    def __init__(self, thickness=220e-9, width=500e-9, **kwargs):
        super().__init__(**kwargs, thickness=thickness, width=width)

    def on_args_changed(self):
        self.suspend_autoupdate()

        available = self._source_argsets()
        normalized = [
            {k: round(str2float(v) * 1e-9, 21) for k, v in d.items()} for d in available
        ]
        idx = self._get_matched_args(normalized, self.args)

        valid_args = available[idx]
        sparams = parser.read_params(self._get_file(valid_args))
        self._f, self._s = parser.build_matrix(sparams)

        self.freq_range = (self._f[0], self._f[-1])
        for key, value in normalized[idx].items():
            setattr(self, key, value)

        self.enable_autoupdate()

    def s_parameters(self, freqs):
        return interpolate(freqs, self._f, self._s)


class HalfRing(SiEPIC_PDK_Base):
    """A half-ring resonator optimized for TE polarized light at 1550
    nanometers.

    The halfring has 4 ports, labeled as pictured.

    .. image:: /reference/images/halfring.png
        :alt: halfring.png

    Parameters
    ----------
    gap : float, optional
        Coupling distance between ring and straight waveguide in meters
        (default 30 nanometers).
    radius : float, optional
        Ring radius in meters (default 10 microns).
    width : float, optional
        Waveguide width in meters (default 500 nanometers).
    thickness : float, optional
        Waveguide thickness in meters (default 220 nanometers).
    coupler_length : float, optional
        Length of the coupling edge, squares out ring; in meters (default 0).
    """

    pin_count = 4
    _base_path = os.path.join(
        os.path.dirname(__file__), "source_data", "ebeam_dc_halfring_straight"
    )
    _base_file = string.Template(
        "te_ebeam_dc_halfring_straight_gap=${gap}m_radius=${radius}m_width=${width}m_thickness=${thickness}m_CoupleLength=${couple_length}m.dat"
    )
    _args_keys = ["gap", "radius", "width", "thickness", "couple_length"]
    _regex = (
        r"(?:te_ebeam_dc_halfring_straight_gap=)"
        r"([-+]?[0-9]+(?:[.][0-9]+)?[fpnumckGMT]?)"
        r"(?:m_radius=)"
        r"([-+]?[0-9]+(?:[.][0-9]+)?[fpnumckGMT]?)"
        r"(?:m_width=)"
        r"([-+]?[0-9]+(?:[.][0-9]+)?[fpnumckGMT]?)"
        r"(?:m_thickness=)"
        r"([-+]?[0-9]+(?:[.][0-9]+)?[fpnumckGMT]?)"
        r"(?:m_CoupleLength=)"
        r"([-+]?[0-9]+(?:[.][0-9]+)?[fpnumckGMT]?)"
        r"(?:m\.dat)"
    )

    def __init__(
        self,
        gap=30e-9,
        radius=10e-6,
        width=500e-9,
        thickness=220e-9,
        couple_length=0.0,
        **kwargs
    ):
        super().__init__(
            **kwargs,
            gap=gap,
            radius=radius,
            width=width,
            thickness=thickness,
            couple_length=couple_length,
        )

    def on_args_changed(self):
        self.suspend_autoupdate()

        available = self._source_argsets()
        normalized = [
            {k: round(str2float(v), 15) for k, v in d.items()} for d in available
        ]
        idx = self._get_matched_args(normalized, self.args)

        valid_args = available[idx]
        sparams = parser.read_params(self._get_file(valid_args))
        self._f, self._s = parser.build_matrix(sparams)

        self.freq_range = (self._f[0], self._f[-1])
        for key, value in normalized[idx].items():
            setattr(self, key, value)

        self.enable_autoupdate()

    def s_parameters(self, freqs):
        return interpolate(freqs, self._f, self._s)


class DirectionalCoupler(SiEPIC_PDK_Base):
    """A directional coupler optimized for TE polarized light at 1550
    nanometers.

    The directional coupler has 4 ports, labeled as pictured. Its efficiently
    splits light that is input from one port into the two outputs on the opposite
    side (with a corresponding pi/2 phase shift). Additionally, it efficiently
    interferes lights from two adjacent inputs, efficiently splitting the
    interfered signal between the two ports on the opposing side.

    .. image:: /reference/images/ebeam_bdc_te1550.png
        :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    gap : float, optional
        Coupling gap distance, in meters (default 200 nanometers).
    Lc : float, optional
        Length of coupler, in meters (default 10 microns).
    """

    pin_count = 4
    _base_path = os.path.join(
        os.path.dirname(__file__), "source_data", "ebeam_dc_te1550"
    )
    _base_file = string.Template("dc_gap=${gap}m_Lc=${Lc}m.sparam")
    _args_keys = ["gap", "Lc"]
    _regex = (
        r"(?:dc_gap=)"
        r"([-+]?[0-9]+(?:[.][0-9]+)?[fpnumckGMT]?)"
        r"(?:m_Lc=)"
        r"([-+]?[0-9]+(?:[.][0-9]+)?[fpnumckGMT]?)"
        r"(?:m\.sparam)"
    )

    def __init__(self, gap=200e-9, Lc=10e-6, **kwargs):
        super().__init__(**kwargs, gap=gap, Lc=Lc)

    def on_args_changed(self):
        self.suspend_autoupdate()

        available = self._source_argsets()
        normalized = [
            {k: round(str2float(v), 15) for k, v in d.items()} for d in available
        ]
        idx = self._get_matched_args(normalized, self.args)

        valid_args = available[idx]
        sparams = parser.read_params(self._get_file(valid_args))
        self._f, self._s = parser.build_matrix(sparams)

        self.freq_range = (self._f[0], self._f[-1])
        for key, value in normalized[idx].items():
            setattr(self, key, value)

        self.enable_autoupdate()

    def s_parameters(self, freqs):
        return interpolate(freqs, self._f, self._s)


class Terminator(SiEPIC_PDK_Base):
    """A terminator component that dissipates light into free space optimized
    for TE polarized light at 1550 nanometers.

    The terminator dissipates excess light into free space. If you have a path
    where the light doesn't need to be measured but you don't want it reflecting
    back into the circuit, you can use a terminator to release it from the circuit.

    .. image:: /reference/images/ebeam_terminator_te1550.png
        :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    w1 : float, optional
        Width at connecting end in meters (default 500 nanometers).
    w2 : float, optional
        Width at terminating end in meters (default 60 nanometers).
    L : float, optional
        Length of terminator, in meters (default 10 microns).
    """

    pin_count = 1
    _base_path = os.path.join(
        os.path.dirname(__file__), "source_data", "ebeam_terminator_te1550"
    )
    _base_file = string.Template("nanotaper_w1=${w1},w2=${w2},L=${L}_TE.sparam")
    _args_keys = ["w1", "w2", "L"]
    _regex = (
        r"(?:nanotaper_w1=)"
        r"([-+]?[0-9]+(?:[.][0-9]+)?[fpnumckGMT]?)"
        r"(?:,w2=)"
        r"([-+]?[0-9]+(?:[.][0-9]+)?[fpnumckGMT]?)"
        r"(?:,L=)"
        r"([-+]?[0-9]+(?:[.][0-9]+)?[fpnumckGMT]?)"
        r"(?:_TE.sparam)"
    )

    def __init__(self, w1=500e-9, w2=60e-9, L=10e-6, **kwargs):
        super().__init__(**kwargs, w1=w1, w2=w2, L=L)

    def on_args_changed(self):
        self.suspend_autoupdate()

        available = self._source_argsets()
        normalized = []
        for d in available:
            w1, w2, L = [(key, d.get(key)) for key in self._args_keys]
            normalized.append(
                {
                    w1[0]: round(str2float(w1[1]) * 1e-9, 15),
                    w2[0]: round(str2float(w2[1]) * 1e-9, 15),
                    L[0]: round(str2float(L[1]) * 1e-6, 15),
                }
            )
        idx = self._get_matched_args(normalized, self.args)

        valid_args = available[idx]
        sparams = parser.read_params(self._get_file(valid_args))
        self._f, self._s = parser.build_matrix(sparams)

        self.freq_range = (self._f[0], self._f[-1])
        for key, value in normalized[idx].items():
            setattr(self, key, value)

        self.enable_autoupdate()

    def s_parameters(self, freqs):
        return interpolate(freqs, self._f, self._s)


class GratingCoupler(SiEPIC_PDK_Base):
    """A grating coupler optimized for TE polarized light at 1550 nanometers.

    The grating coupler efficiently couples light from a fiber array positioned
    above the chip into the circuit. For the TE mode, the angle is -25 degrees
    [needs citation].

    .. image:: /reference/images/ebeam_gc_te1550.png
        :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    thickness : float, optional
        The thickness of the grating coupler, in meters (default 220
        nanometers). Valid values are 210, 220, or 230 nanometers.
    deltaw : float, optional
        FIXME: unknown parameter (default 0). Valid values are -20, 0, or 20.
    polarization : str, optional
        The polarization of light in the circuit. One of 'TE' (default) or 'TM'.
    """

    pin_count = 2
    _base_path = os.path.join(os.path.dirname(__file__), "source_data", "gc_source")
    _base_file = string.Template(
        "GC_${polarization}1550_thickness=${thickness} deltaw=${deltaw}.txt"
    )
    _args_keys = ["polarization", "thickness", "deltaw"]
    _regex = (
        r"(?:GC_)"
        r"([T][emEM])"
        r"(?:1550_thickness=)"
        r"([-+]?[0-9]+(?:[.][0-9]+)?[fpnumckGMT]?)"
        r"(?: deltaw=)"
        r"([-+]?[0-9]+(?:[.][0-9]+)?[fpnumckGMT]?)"
        r"(?:\.txt)"
    )

    def __init__(self, thickness=220e-9, deltaw=0, polarization="TE", **kwargs):
        super().__init__(
            **kwargs, thickness=thickness, deltaw=deltaw, polarization=polarization
        )

    def on_args_changed(self):
        self.suspend_autoupdate()

        available = self._source_argsets()
        normalized = []
        for d in available:
            polarization, thickness, deltaw = [
                (key, d.get(key)) for key in self._args_keys
            ]
            normalized.append(
                {
                    polarization[0]: polarization[1],
                    thickness[0]: round(str2float(thickness[1]) * 1e-9, 15),
                    deltaw[0]: round(str2float(deltaw[1]) * 1e-9, 15),
                }
            )
        idx = self._get_matched_args(normalized, self.args)
        for key, value in normalized[idx].items():
            setattr(self, key, value)

        valid_args = available[idx]
        params = np.genfromtxt(self._get_file(valid_args), delimiter="\t")
        self._f = params[:, 0]
        self._s = np.zeros((len(self._f), 2, 2), dtype="complex128")
        self._s[:, 0, 0] = params[:, 1] * np.exp(1j * params[:, 2])
        self._s[:, 0, 1] = params[:, 3] * np.exp(1j * params[:, 4])
        self._s[:, 1, 0] = params[:, 5] * np.exp(1j * params[:, 6])
        self._s[:, 1, 1] = params[:, 7] * np.exp(1j * params[:, 8])

        # Arrays are from high frequency to low; reverse it,
        # for convention's sake.
        self._f = self._f[::-1]
        self._s = self._s[::-1]
        self.freq_range = (self._f[0], self._f[-1])

        self.enable_autoupdate()

    def s_parameters(self, freqs):
        return interpolate(freqs, self._f, self._s)


class Waveguide(SiEPIC_PDK_Base):
    """Model for an waveguide optimized for TE polarized light at 1550
    nanometers.

    A waveguide easily connects other optical components within a circuit.

    .. image:: /reference/images/ebeam_wg_integral_1550.png
        :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    length : float
        Waveguide length in meters (default 0.0 meters).
    width : float, optional
        Waveguide width in meters (default 500 nanometers).
    height : float, optional
        Waveguide height in meters (default 220 nanometers).
    polarization : str, optional
        Polarization of light in the waveguide; one of 'TE' (default) or 'TM'.
    sigma_ne : float, optional
        Standard deviation of the effective index for monte carlo simulations
        (default 0.05).
    sigma_ng : float, optional
        Standard deviation of the group velocity for monte carlo simulations
        (default 0.05).
    sigma_nd : float, optional
        Standard deviation of the group dispersion for monte carlo simulations
        (default 0.0001).

    Notes
    -----
    The `sigma_` values in the parameters are used for monte carlo simulations.
    """

    pin_count = 2
    freq_range = (
        187370000000000.0,
        199862000000000.0,
    )  #: The valid frequency range for this model.
    _base_path = os.path.join(
        os.path.dirname(__file__), "source_data", "wg_integral_source"
    )
    _base_file = string.Template("WaveGuideTETMStrip,w=${width},h=${height}.txt")
    _args_keys = ["width", "height"]
    _args_trigger_update = ["polarization"]
    _regex = (
        r"(?:WaveGuideTETMStrip,w=)"
        r"([-+]?[0-9]+(?:[.][0-9]+)?[fpnumckGMT]?)"
        r"(?:,h=)"
        r"([-+]?[0-9]+(?:[.][0-9]+)?[fpnumckGMT]?)"
        r"(?:\.txt)"
    )

    def __init__(
        self,
        length=0.0,
        width=500e-9,
        height=220e-9,
        polarization="TE",
        sigma_ne=0.05,
        sigma_ng=0.05,
        sigma_nd=0.0001,
        **kwargs
    ):
        if polarization not in ["TE", "TM"]:
            raise ValueError(
                "Unknown polarization value '{}', must be one of 'TE' or 'TM'".format(
                    polarization
                )
            )

        # TODO: TM calculations
        if polarization == "TM":
            raise NotImplementedError

        super().__init__(
            **kwargs,
            length=length,
            width=width,
            height=height,
            polarization=polarization,
            sigma_ne=sigma_ne,
            sigma_ng=sigma_ng,
            sigma_nd=sigma_nd,
        )

        self.regenerate_monte_carlo_parameters()

    def on_args_changed(self):
        self.suspend_autoupdate()

        available = self._source_argsets()
        normalized = [
            {k: round(str2float(v) * 1e-9, 21) for k, v in d.items()} for d in available
        ]
        idx = self._get_matched_args(normalized, self.args)

        valid_args = available[idx]
        with open(self._get_file(valid_args), "r") as f:
            params = f.read().rstrip("\n")
        if self.polarization == "TE":
            lam0, ne, _, ng, _, nd, _ = params.split(" ")
        elif self.polarization == "TM":
            lam0, _, ne, _, ng, _, nd = params.split(" ")
            raise NotImplementedError
        self.lam0 = float(lam0)
        self.ne = float(ne)
        self.ng = float(ng)
        self.nd = float(nd)

        # Updates parameters width and thickness to closest match.
        for key, value in normalized[idx].items():
            setattr(self, key, value)

        self.enable_autoupdate()

    def s_parameters(self, freqs):
        """Get the s-parameters of a waveguide.

        Parameters
        ----------
        freqs : float
            The array of frequencies to get s parameters for.

        Returns
        -------
        (freqs, s) : tuple
            Returns a tuple containing the frequency array, `freqs`,
            corresponding to the calculated s-parameter matrix, `s`.
        """
        return self.cacl_s_params(
            freqs, self.length, self.lam0, self.ne, self.ng, self.nd
        )

    def monte_carlo_s_parameters(self, freqs):
        """Returns a monte carlo (randomized) set of s-parameters.

        In this implementation of the monte carlo routine, random values
        are generated for ne, ng, and nd for each run through of the
        monte carlo simulation. This means that all waveguide elements
        throughout a single circuit will have the same (random) ne, ng,
        and nd values. Hence, there is correlated randomness in the
        monte carlo parameters but they are consistent within a single
        circuit.
        """
        return self.cacl_s_params(
            freqs, self.length, self.lam0, self.rand_ne, self.rand_ng, self.rand_nd
        )

    def regenerate_monte_carlo_parameters(self):
        self.rand_ne = np.random.normal(self.ne, self.sigma_ne)
        self.rand_ng = np.random.normal(self.ng, self.sigma_ng)
        self.rand_nd = np.random.normal(self.nd, self.sigma_nd)

    @staticmethod
    def cacl_s_params(freqs, length, lam0, ne, ng, nd):
        # Initialize array to hold s-params
        s = np.zeros((len(freqs), 2, 2), dtype=complex)

        # Loss calculation
        TE_loss = 700  # dB/m for width 500nm
        alpha = TE_loss / (20 * np.log10(np.exp(1)))

        w = np.asarray(freqs) * 2 * np.pi  # get angular freqs from freqs
        w0 = (2 * np.pi * SPEED_OF_LIGHT) / lam0  # center freqs (angular)

        # calculation of K
        K = (
            2 * np.pi * ne / lam0
            + (ng / SPEED_OF_LIGHT) * (w - w0)
            - (nd * lam0 ** 2 / (4 * np.pi * SPEED_OF_LIGHT)) * ((w - w0) ** 2)
        )

        for x in range(0, len(freqs)):  # build s-matrix from K and waveguide length
            s[x, 0, 1] = s[x, 1, 0] = np.exp(-alpha * length + (K[x] * length * 1j))

        return s


class YBranch(SiEPIC_PDK_Base):
    """A y-branch efficiently splits the input 50/50 between the two outputs.
    It can also be used as a combiner if used in the opposite direction,
    combining and interfering the light from two inputs into the one output.

    .. image:: /reference/images/ebeam_y_1550.png
        :alt: ebeam_bdc_te1550.png

    Parameters
    ----------
    thickness : float, optional
        Waveguide thickness, in meters (default 220 nanometers). Valid values
        are 210, 220, or 230 nanometers.
    width : float, optional
        Waveguide width, in meters (default 500 nanometers). Valid values are
        480, 500, or 520 nanometers.
    polarization : str, optional
        Polarization of light in the circuit, either 'TE' (default) or 'TM'.
    """

    pin_count = 3
    _base_path = os.path.join(
        os.path.dirname(__file__), "source_data", "y_branch_source"
    )
    _base_file = string.Template(
        "Ybranch_Thickness =${thickness} width=${width}.sparam"
    )
    _args_keys = ["thickness", "width"]
    _args_trigger_update = ["polarization"]
    _regex = (
        r"(?:Ybranch_Thickness =)"
        r"([-+]?[0-9]+(?:[.][0-9]+)?[fpnumckGMT]?)"
        r"(?: width=)"
        r"([-+]?[0-9]+(?:[.][0-9]+)?[fpnumckGMT]?)"
        r"(?:\.sparam)"
    )

    def __init__(self, thickness=220e-9, width=500e-9, polarization="TE", **kwargs):
        if polarization not in ["TE", "TM"]:
            raise ValueError(
                "Unknown polarization value '{}', must be one of 'TE' or 'TM'".format(
                    polarization
                )
            )
        super().__init__(
            **kwargs, thickness=thickness, width=width, polarization=polarization
        )

    def on_args_changed(self):
        self.suspend_autoupdate()

        available = self._source_argsets()
        normalized = [
            {k: round(str2float(v) * 1e-9, 21) for k, v in d.items()} for d in available
        ]
        idx = self._get_matched_args(normalized, self.args)

        valid_args = available[idx]
        sparams = parser.read_params(self._get_file(valid_args))
        sparams = list(
            filter(lambda sparams: sparams["mode"] == self.polarization, sparams)
        )

        for key, value in normalized[idx].items():
            setattr(self, key, value)
        self._f, self._s = parser.build_matrix(sparams)
        self.freq_range = (self._f[0], self._f[-1])

        self.enable_autoupdate()

    def s_parameters(self, freqs):
        """Returns scattering parameters for the y-branch based on its
        parameters.

        Parameters
        ----------
        freqs : np.ndarray
            The frequency range to get scattering parameters for.

        Returns
        -------
        s : np.ndarray
            The scattering parameters corresponding to the frequency range.
        """
        return interpolate(freqs, self._f, self._s)
