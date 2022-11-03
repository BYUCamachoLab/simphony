# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
simphony.tools
==============

This package contains handy functions useful across simphony submodules
and to the average user.
"""

from cmath import rect
import numpy as np
import re

from scipy.constants import c as SPEED_OF_LIGHT
from scipy.interpolate import interp1d

MATH_SUFFIXES = {
    "f": "e-15",
    "p": "e-12",
    "n": "e-9",
    "u": "e-6",
    "m": "e-3",
    "c": "e-2",
    "k": "e3",
    "M": "e6",
    "G": "e9",
    "T": "e12",
}


def add_polar(c1, c2):
    """Adds two polar coordinates together.

    Parameters
    ----------
    c1 : (float, float)
        First polar coordinate
    c2 : (float, float)
        Second polar coordinate

    Returns
    -------
    result : (float, float)
        The resulting polar coordinate"""
    r1, phi1 = c1
    r2, phi2 = c2

    # add the vectors in rectangular form
    sum = rect(r1, phi1) + rect(r2, phi2)
    mag = np.abs(sum)
    angle = np.angle(sum)

    # calculate how many times the original vectors wrapped around
    # then add the biggest amount back to our phase
    # this simulates the steady-state in time-domain
    wrapped1 = (phi1 // (2 * np.pi)) * (2 * np.pi)
    wrapped2 = (phi2 // (2 * np.pi)) * (2 * np.pi)
    biggest = max(wrapped1, wrapped2)

    return (mag, angle + biggest)


def mul_polar(c1, c2):
    """Multiplies two polar coordinates together.

    Parameters
    ----------
    c1 : (float, float)
        First polar coordinate
    c2 : (float, float)
        Second polar coordinate

    Returns
    -------
    result : (float, float)
        The resulting polar coordinate"""
    r1, phi1 = c1
    r2, phi2 = c2

    return (r1 * r2, phi1 + phi2)


def str2float(num):
    """Converts a number represented as a string to a float. Can include
    suffixes (such as 'u' for micro, 'k' for kilo, etc.).

    Parameters
    ----------
    num : str
        A string representing a number, optionally with a suffix.

    Returns
    -------
    float
        The string converted back to its floating point representation.

    Raises
    ------
    ValueError
        If the argument is malformed or the suffix is not recognized.

    Examples
    --------
    >>> str2float('14.5c')
    0.145

    Values without suffixes get converted to floats normally.

    >>> str2float('2.53')
    2.53

    If an unrecognized suffix is present, a ``ValueError`` is raised.

    >>> str2float('17.3o')
    ValueError: Suffix 'o' in '17.3o' not recognized.
    ([-+]?[0-9]+[.]?[0-9]*((?:[eE][-+]?[0-9]+)|[a-zA-Z])?)

    Some floats are represented in exponential notation instead of suffixes,
    and we can handle those, too:

    >>> str2float('15.2e-6')
    1.52e-7

    >>> str2float('0.4E6')
    400000.0
    """
    matches = re.findall(
        r"([-+]?[0-9]+(?:[.][0-9]+)?)((?:[eE][-+]?[0-9]+)|(?:[a-zA-Z]))?", num
    )
    if len(matches) > 1:
        raise ValueError(f"'{num}' is malformed")
    num, suffix = matches[0]
    try:
        if suffix.startswith("e") or suffix.startswith("E"):
            return float(num + suffix)
        else:
            return float(num + (MATH_SUFFIXES[suffix] if suffix != "" else ""))
    except KeyError as e:
        raise ValueError(f"Suffix {str(e)} in '{matches[0]}' not recognized.")


def freq2wl(freq):
    """Convenience function for converting from frequency to wavelength.

    Parameters
    ----------
    freq : float
        The frequency in SI units (Hz).

    Returns
    -------
    wl : float
        The wavelength in SI units (m).
    """
    return SPEED_OF_LIGHT / freq


def wl2freq(wl):
    """Convenience function for converting from wavelength to frequency.

    Parameters
    ----------
    wl : float
        The wavelength in SI units (m).

    Returns
    -------
    freq : float
        The frequency in SI units (Hz).
    """
    return SPEED_OF_LIGHT / wl


def interpolate(resampled, sampled, s_parameters):
    """Returns the result of a cubic interpolation for a given frequency range.

    Parameters
    ----------
    output_freq : np.ndarray
        The desired frequency range for a given input to be interpolated to.
    input_freq : np.ndarray
        A frequency array, indexed matching the given s_parameters.
    s_parameters : np.array
        S-parameters for each frequency given in input_freq.

    Returns
    -------
    result : np.array
        The values of the interpolated function (fitted to the input
        s-parameters) evaluated at the ``output_freq`` frequencies.
    """
    func = interp1d(sampled, s_parameters, kind="cubic", axis=0)
    return func(resampled)
