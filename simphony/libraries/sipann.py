# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)
"""SiPANN models compatible with SAX circuits.

This package contains wrappers for models defined in the SiPANN (Silicon
Photonics with Artificial Neural Networks) project, another project by
CamachoLab at BYU. It leverages machine learning to simulate photonic
devices, giving greater speed and similar accuracy to a full FDTD
simulation.
"""

from itertools import product
from typing import Callable, Union

import numpy as np
import sax
from jax.typing import ArrayLike

try:
    from SiPANN import comp, scee
    from SiPANN.scee_opt import premade_coupler as sipann_premade_coupler
except ImportError as exc:
    raise ImportError(
        "SiPANN must be installed to use the SiPANN wrappers. "
        "To install SiPANN, run `pip install SiPANN`."
    ) from exc


def _create_sdict_from_model(model, wl: Union[float, ArrayLike]) -> sax.SDict:
    """Create s-parameter dict from model.

    Parameters
    ----------
    model : SiPANN model
        The component model to call.
    wl : float or ArrayLike
        Wavelength to evaluate at in microns.

    Returns
    -------
    sdict : sax.SDict
        The s-parameter dictionary.
    """
    wl = np.array(wl).reshape(-1)  # microns
    s = model.sparams(wl * 1e3)  # convert to nanometers, s is shape f x n x n
    ports = list(range(s.shape[1]))

    sdict = {}
    for p_out, p_in in product(ports, ports):
        sdict[(f"o{p_out}", f"o{p_in}")] = s[:, p_out, p_in]

    return sdict


def gap_func_symmetric(
    wl: Union[float, ArrayLike] = 1.55,
    width: float = 500.0,
    thickness: float = 220.0,
    gap: Callable[[float], float] = 100.0,
    dgap: Callable[[float], float] = 0.0,
    zmin: float = 0.0,
    zmax: float = 10e3,
    sw_angle: float = 90.0,
) -> sax.SDict:
    r"""Symmetric directional coupler, meaning both waveguides are the same
    shape.

    A gap function must describe the shape of the two waveguides, where the
    distance between the waveguides is the return value of the gap function
    at every horizontal point from left to right. The derivative of the gap
    function is also required.

    Ports are numbered as::

        |       2---\      /---4       |
        |            ------            |
        |            ------            |
        |       1---/      \---3       |

    Parameters
    ----------
    width : float
        Width of waveguides in nanometers (valid from 400 to 600).
    thickness : float
        Thickness of waveguides in nanometers (valid from 180 to 240).
    gap : callable
        Gap function along the waveguide, values it returns must be in
        nanometers (and must always be greater than 100).
    dgap : callable
        Derivative of the gap function.
    zmin : float
        Real number at which to begin integration of gap function.
    zmax : float
        Real number at which to end integration of gap function.
    sw_angle : float, optional
        Sidewall angle of waveguide from horizontal in degrees (valid from 80
        to 90, defaults to 90).
    """
    if width < 400 or width > 600:
        raise ValueError("Width must be between 400 and 600 nm")
    if thickness < 180 or thickness > 240:
        raise ValueError("Thickness must be between 180 and 240 nm")
    if sw_angle < 80 or sw_angle > 90:
        raise ValueError("Sidewall angle must be between 80 and 90 degrees")

    model = scee.GapFuncSymmetric(width, thickness, gap, dgap, zmin, zmax, sw_angle)
    sdict = _create_sdict_from_model(model, wl)
    return sdict


def gap_func_antisymmetric(
    wl: Union[float, ArrayLike] = 1.55,
    width: float = 500.0,
    thickness: float = 220.0,
    gap: Callable[[float], float] = 100.0,
    zmin: float = 0.0,
    zmax: float = 100.0,
    arc1: float = 10e3,
    arc2: float = 10e3,
    arc3: float = 10e3,
    arc4: float = 10e3,
    sw_angle: Union[float, np.ndarray] = 90,
) -> sax.SDict:
    r"""Antisymmetric directional coupler, meaning both waveguides are
    differently shaped.

    A gap function describing the vertical distance between the two waveguides
    at any horizontal point, and arc lengths from each port to the coupling
    point, describe the shape of the device.

    Ports are numbered as::

        |       2---\      /---4       |
        |            ------            |
        |            ------            |
        |       1---/      \---3       |

    Parameters
    ----------
    width : float
        Width of waveguides in nanometers (valid from 400 to 600).
    thickness : float
        Thickness of waveguides in nanometers (valid from 180 to 240).
    gap : callable
        Gap function along the waveguide, values it returns must be in
        nanometers (and must always be greater than 100).
    zmin : float
        Real number at which to begin integration of gap function.
    zmax : float
        Real number at which to end integration of gap function.
    arc1 : float
        Arc length from port 1 to minimum coupling point in nanometers.
    arc2 : float
        Arc length from port 2 to minimum coupling point in nanometers.
    arc3 : float
        Arc length from port 3 to minimum coupling point in nanometers.
    arc4 : float
        Arc length from port 4 to minimum coupling point in nanometers.
    sw_angle : float, optional
        Sidewall angle of waveguide from horizontal in degrees (valid from 80
        to 90, defaults to 90).
    """
    if width < 400 or width > 600:
        raise ValueError("Width must be between 400 and 600 nm")
    if thickness < 180 or thickness > 240:
        raise ValueError("Thickness must be between 180 and 240 nm")
    if gap < 100:
        raise ValueError("Gap must be greater than 100 nm")
    if sw_angle < 80 or sw_angle > 90:
        raise ValueError("Sidewall angle must be between 80 and 90 degrees")

    model = scee.GapFuncAntiSymmetric(
        width, thickness, gap, zmin, zmax, arc1, arc2, arc3, arc4, sw_angle
    )
    sdict = _create_sdict_from_model(model, wl)
    return sdict


def half_ring(
    wl: Union[float, ArrayLike] = 1.55,
    width: float = 500.0,
    thickness: float = 220.0,
    radius: float = 10.0,
    gap: float = 100.0,
    sw_angle: float = 90.0,
) -> sax.SDict:
    """Half of a ring resonator.

    Uses a radius and a gap to describe the shape.

    .. image:: /_static/images/sipann_half_ring.png
        :alt: Half ring port numbering.
        :width: 400px
        :align: center

    Parameters
    ----------
    wl : float or ArrayLike
        The wavelengths to evaluate at in microns.
    width : float
        Width of waveguides in nanometers (valid from 400 to 600).
    thickness : float
        Thickness of waveguides in nanometers (valid from 180 to 240).
    radius : float
        Distance from center of ring to middle of waveguide, in microns.
    gap : float
        Minimum distance from ring waveguide edge to straight waveguide edge,
        in nanometers (must be greater than 100).
    sw_angle : float, optional
        Sidewall angle of waveguide from horizontal in degrees (valid from 80
        to 90, defaults to 90).

    Examples
    --------
    >>> s = half_ring(wl, 500, 220, 5000, 100)
    """
    if width < 400 or width > 600:
        raise ValueError("Width must be between 400 and 600 nm")
    if thickness < 180 or thickness > 240:
        raise ValueError("Thickness must be between 180 and 240 nm")
    if gap < 100:
        raise ValueError("Gap must be greater than 100 nm")
    if sw_angle < 80 or sw_angle > 90:
        raise ValueError("Sidewall angle must be between 80 and 90 degrees")

    model = scee.HalfRing(width, thickness, radius * 1000, gap, sw_angle)
    sdict = _create_sdict_from_model(model, wl)
    return sdict


def straight_coupler(
    wl: Union[float, ArrayLike] = 1.55,
    width: float = 500.0,
    thickness: float = 220.0,
    gap: float = 100.0,
    length: float = 1000.0,
    sw_angle: float = 90.0,
) -> sax.SDict:
    """Straight directional coupler, both waveguides run parallel.

    Described by a gap and a length.

    .. image:: /_static/images/sipann_straight_coupler.png
        :alt: Straight coupler port numbering.
        :width: 400px
        :align: center

    Parameters
    ----------
    width : float
        Width of waveguides in nanometers (valid from 400 to 600).
    thickness : float
        Thickness of waveguides in nanometers (valid from 180 to 240).
    gap : float
        Distance between the two waveguide edges, in nanometers (must be greater
        than 100).
    length : float
        Length of both waveguides in nanometers.
    sw_angle : float, optional
        Sidewall angle of waveguide from horizontal in degrees (Valid from 80
        to 90, defaults to 90).

    Examples
    --------
    >>> s = straight_coupler(wl, 500, 220, 100, 1000)
    """
    if width < 400 or width > 600:
        raise ValueError("Width must be between 400 and 600 nm")
    if thickness < 180 or thickness > 240:
        raise ValueError("Thickness must be between 180 and 240 nm")
    if gap < 100:
        raise ValueError("Gap must be greater than 100 nm")
    if sw_angle < 80 or sw_angle > 90:
        raise ValueError("Sidewall angle must be between 80 and 90 degrees")

    model = scee.StraightCoupler(width, thickness, gap, length, sw_angle)
    sdict = _create_sdict_from_model(model, wl)
    return sdict


def standard_coupler(
    wl: Union[float, ArrayLike] = 1.55,
    width: float = 500.0,
    thickness: float = 220.0,
    gap: float = 100.0,
    length: float = 1000.0,
    horizontal: float = 10e3,
    vertical: float = 10e3,
    sw_angle: float = 90.0,
) -> sax.SDict:
    """Standard-shaped directional coupler.

    Described by a gap, length, horizontal and vertical
    distance.

    .. image:: /_static/images/sipann_standard_coupler.png
        :alt: Standard coupler port numbering.
        :width: 400px
        :align: center

    Parameters
    ----------
    width : float
        Width of waveguides in nanometers (valid from 400 to 600).
    thickness : float
        Thickness of waveguides in nanometers (Valid from 180 to 240).
    gap : float
        Distance between the two waveguide edges, in nanometers (must be greater
        than 100).
    length : float
        Length of the straight portion of both waveguides, in nanometers.
    horizontal : float
        Horizontal distance between end of coupler and straight segment, in
        nanometers.
    vertical : float
        Vertical distance between end of coupler and straight segment, in
        nanometers.
    sw_angle : float, optional
        Sidewall angle of waveguide from horizontal in degrees (valid from 80
        to 90, defaults to 90).

    Examples
    --------
    >>> s = standard_coupler(wl, 500, 220, 100, 5000, 5e3, 2e3)
    """
    if width < 400 or width > 600:
        raise ValueError("Width must be between 400 and 600 nm")
    if thickness < 180 or thickness > 240:
        raise ValueError("Thickness must be between 180 and 240 nm")
    if gap < 100:
        raise ValueError("Gap must be greater than 100 nm")
    if sw_angle < 80 or sw_angle > 90:
        raise ValueError("Sidewall angle must be between 80 and 90 degrees")

    model = scee.Standard(width, thickness, gap, length, horizontal, vertical, sw_angle)
    sdict = _create_sdict_from_model(model, wl)
    return sdict


def double_half_ring(
    wl: Union[float, ArrayLike] = 1.55,
    width: float = 500.0,
    thickness: float = 220.0,
    radius: float = 10e3,
    gap: float = 100.0,
    sw_angle: float = 90.0,
) -> sax.SDict:
    r"""Two equally sized half-rings coupling along their edges.

    Described by a radius and a gap between the two rings.

    Ports are numbered as::

        |         2 |     | 4          |
        |            \   /             |
        |             ---              |
        |             ---              |
        |            /   \             |
        |         1 |     | 3          |

    Parameters
    ----------
    width : float
        Width of the waveguide in nanometers (valid from 400 to 600).
    thickness : float
        Thickness of waveguide in nanometers (valid for 180 to 240).
    radius : float
        Distance from center of ring to middle of waveguide, in nanometers.
    gap : float
        Minimum distance from the edges of the waveguides of the two rings, in
        nanometers (must be greater than 100).
    sw_angle : float, optional
        Sidewall angle of waveguide from horizontal in degrees (valid from 80
        to 90, defaults to 90).

    Notes
    -----
    Writing to GDS is not supported for this component.
    """
    if width < 400 or width > 600:
        raise ValueError("Width must be between 400 and 600 nm")
    if thickness < 180 or thickness > 240:
        raise ValueError("Thickness must be between 180 and 240 nm")
    if gap < 100:
        raise ValueError("Gap must be greater than 100 nm")
    if sw_angle < 80 or sw_angle > 90:
        raise ValueError("Sidewall angle must be between 80 and 90 degrees")

    model = scee.DoubleHalfRing(width, thickness, radius, gap, sw_angle)
    sdict = _create_sdict_from_model(model, wl)
    return sdict


def angled_half_ring(
    wl: Union[float, ArrayLike] = 1.55,
    width: float = 500.0,
    thickness: float = 220.0,
    radius: float = 10e3,
    gap: float = 100.0,
    sw_angle: float = 90.0,
) -> sax.SDict:
    r"""A halfring resonator, except what was the straight waveguide is now
    curved.

    Described by a radius, gap, and angle (theta) that the
    "straight" waveguide is curved by.

    Ports are numbered as::

        |      2  \        / 4       |
        |          \      /          |
        |      1--- \    / ---3      |
        |          \ \  / /          |
        |           \ -- /           |
        |            ----            |

    Parameters
    ----------
    width : float
        Width of the waveguide in nanometers (valid from 400 to 600).
    thickness : float
        Thickness of waveguide in nanometers (Valid for 180 to 240).
    radius : float
        Distance from center of ring to middle of waveguide, in nanometers.
    gap : float
        Minimum distance from ring waveguide edge to "straight" waveguide edge,
        in nanometers (must be greater than 100).
    theta : float
        Angle of the curve of the "straight" waveguide, in radians.
    sw_angle : float, optional
        Sidewall angle of waveguide from horizontal in degrees (Valid from 80
        to 90, defaults to 90).

    Notes
    -----
    Writing to GDS is not supported for this component.
    """
    width: float = (500.0,)
    thickness: float = (220.0,)
    radius: float = (10e3,)
    gap: float = (100.0,)
    theta: float = (0.0,)
    sw_angle: float = (90,)
    if width < 400 or width > 600:
        raise ValueError("Width must be between 400 and 600 nm")
    if thickness < 180 or thickness > 240:
        raise ValueError("Thickness must be between 180 and 240 nm")
    if gap < 100:
        raise ValueError("Gap must be greater than 100 nm")
    if sw_angle < 80 or sw_angle > 90:
        raise ValueError("Sidewall angle must be between 80 and 90 degrees")

    model = scee.AngledHalfRing(width, thickness, radius, gap, theta, sw_angle)
    sdict = _create_sdict_from_model(model, wl)
    return sdict


def waveguide(
    wl: Union[float, ArrayLike] = 1.55,
    width: float = 500.0,
    thickness: float = 220.0,
    length: float = 10e3,
    sw_angle: float = 90.0,
) -> sax.SDict:
    """Lossless model for a straight waveguide. Main use case is for playing
    nice with other models in SCEE.

    .. image:: /_static/images/sipann_waveguide.png
        :alt: Waveguide port numbering
        :width: 200px
        :align: center

    Parameters
    ----------
    width : float
        Width of the waveguide in nanometers (valid from 400 to 600).
    thickness : float
        Thickness of waveguide in nanometers (valid for 180 to 240).
    length : float
        Length of waveguide in nanometers.
    sw_angle : float, optional
        Sidewall angle of waveguide from horizontal in degrees (valid from 80
        to 90, defaults to 90).
    """
    if width < 400 or width > 600:
        raise ValueError("Width must be between 400 and 600 nm")
    if thickness < 180 or thickness > 240:
        raise ValueError("Thickness must be between 180 and 240 nm")
    if sw_angle < 80 or sw_angle > 90:
        raise ValueError("Sidewall angle must be between 80 and 90 degrees")

    model = scee.Waveguide(width, thickness, length, sw_angle)
    sdict = _create_sdict_from_model(model, wl)
    return sdict


def racetrack(
    wl: Union[float, ArrayLike] = 1.55,
    width: float = 500.0,
    thickness: float = 220.0,
    radius: float = 10e3,
    gap: float = 100.0,
    length: float = 10e3,
    sw_angle: float = 90.0,
) -> sax.SDict:
    """Racetrack waveguide arc, used to connect to a racetrack directional
    coupler.

    .. image:: /_static/images/sipann_racetrack.png
        :alt: Racetrack port numbering
        :width: 400px
        :align: center

    Parameters
    ----------
    width : float
        Width of the waveguide in nanometers (valid from 400 to 600).
    thickness : float
        Thickness of waveguide in nanometers (valid for 180 to 240).
    radius : float
        Distance from center of ring to middle of waveguide, in nanometers.
    gap : float
        Minimum distance from ring waveguide edge to "straight" waveguide edge,
        in nanometers (must be greater than 100).
    length : float
        Length of straight portion of ring waveguide, in nanometers.
    sw_angle : float, optional
        Sidewall angle of waveguide from horizontal in degrees (valid from 80
        to 90, defaults to 90).

    Examples
    --------
    >>> dev = Racetrack(500, 220, 5000, 200, 5000)

    Notes
    -----
    You can produce a GDS file of the model you instantiate using SiPANN (see
    more `on SiPANN's docs <https://sipann.readthedocs.io/en/latest/>`_).
    """

    if width < 400 or width > 600:
        raise ValueError("Width must be between 400 and 600 nm")
    if thickness < 180 or thickness > 240:
        raise ValueError("Thickness must be between 180 and 240 nm")
    if gap < 100:
        raise ValueError("Gap must be greater than 100 nm")
    if sw_angle < 80 or sw_angle > 90:
        raise ValueError("Sidewall angle must be between 80 and 90 degrees")

    model = comp.racetrack_sb_rr(width, thickness, radius, gap, length, sw_angle)
    sdict = _create_sdict_from_model(model, wl)
    return sdict


def premade_coupler(
    wl: Union[float, ArrayLike] = 1.55,
    split: int = 50,
) -> sax.SDict:
    r"""Loads premade couplers based on the given split value.

    Various splitting ratio couplers have been made and saved. This
    function reloads them. Note that each of their lengths are different
    and are also returned for the users info. These have all been
    designed with waveguide geometry 500nm x 220nm.

    Ports are numbered as::

        |       2---\      /---4       |
        |            ------            |
        |            ------            |
        |       1---/      \---3       |

    Parameters
    ----------
    split : int
        Percent of light coming out cross port. Valid numbers are 10, 20, 30,
        40, 50, 100. 100 is a full crossover.
    """
    model = sipann_premade_coupler(split)[0]
    sdict = _create_sdict_from_model(model, wl)
    return sdict
