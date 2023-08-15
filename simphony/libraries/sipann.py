# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)
"""This package contains wrappers for models defined in the SiPANN (Silicon
Photonics with Artificial Neural Networks) project, another project by
CamachoLab at BYU. It leverages machine learning to simulate photonic devices,
giving greater speed and similar accuracy to a full FDTD simulation.

The wrappers defined here integrate SiPANN models into Simphony for
easier use.
"""

from pathlib import Path
from typing import Callable, Dict, TypeVar, Union

import numpy as np

try:
    from SiPANN import comp, scee
    from SiPANN.scee_opt import premade_coupler
except ImportError:
    raise ImportError(
        "SiPANN must be installed to use the SiPANN wrappers. "
        "To install SiPANN, run `pip install SiPANN`."
    )

from simphony.models import Model


class SipannWrapper(Model):
    """Allows wrapping models from SCEE for use in simphony. This class should
    be extended, with each extending class wrapping one model.

    Each extending class should convert parameters passed in from meters (which
    simphony uses) to nanometers (which SiPANN uses). Each extending class
    should also define a class-wide field for 'ocount', equal to the number
    of pins the subcircuit has.

    Note that the wrapped SCEE models cannot have varying geometries; such a
    device can't be cascaded properly.

    Parameters
    ----------
    model
        Model from `SiPANN.scee` or `SiPANN.comp` modules, must have the
        'sparams' method
    sigmas : dict
        Dictionary mapping parameters to sigma values for Monte-Carlo
        simulations, values should be in meters. If Monte-Carlo simulations are not
        needed, pass in an empty dictionary.
    """

    def __init__(
        self,
        model: TypeVar("M"),
        sigmas: Dict[str, float],
        name: str = None,
    ) -> None:
        super().__init__(name=name)
        self.model = model
        self.sigmas = sigmas

        # catch varying geometries
        args = self.model._clean_args(None)
        if len(args[0]) != 1:
            raise ValueError(
                "You have changing geometries, use in simphony doesn't make sense!"
            )

        # self.params = self.model.__dict__.copy()
        # self.rand_params = dict()
        # self.regenerate_monte_carlo_parameters()

    def s_params(self, wl):
        """Get the s-parameters of the SCEE Model.

        Parameters
        ----------
        wl : float or array-like
            Wavelength array to sample s-parameters at, in microns.

        Returns
        -------
        s : array-like
            The s-parameter matrix sampled at the given wavelengths.
        """
        return self.model.sparams(wl * 1e3)

    def write_gds(self, filename: Union[Path, str]) -> None:
        """Write the model to a GDS file.

        Parameters
        ----------
        filename : str or Path
            Path to write the GDS file to.
        """
        self.model.gds(str(filename), units="microns")

    # def monte_carlo_s_parameters(self, freqs: np.ndarray) -> np.ndarray:
    #     """Get the s-parameters of the SCEE Model, influenced by noise from
    #     sigma values.

    #     Parameters
    #     ----------
    #     `freqs`
    #     Frequency array to calculate s-parameters over, in
    #     Hz

    #     Returns
    #     -------
    #     `s`
    #     The s-parameter matrix
    #     """
    #     wl = freq2wl(freqs) * 1e9

    #     # Change to noise params for monte carlo, then change back
    #     self.model.update(**self.rand_params)
    #     sparams = self.model.sparams(wl)
    #     self.model.update(**self.params)

    #     return sparams

    # def regenerate_monte_carlo_parameters(self) -> None:
    #     """For each sigma value given to the wrapper, will apply noise the
    #     matching parameter."""
    #     for param, sigma in self.sigmas.items():
    #         self.rand_params[param] = np.random.normal(self.params[param], sigma * 1e9)


class GapFuncSymmetric(SipannWrapper):
    r"""Symmetric directional coupler, meaning both waveguides are the same
    shape.

    A gap function must describe the shape of the two waveguides, where the
    distance between the waveguides is the return value of the gap function
    at every horizontal point from left to right. The derivative of the gap
    function is also required.

    Ports are numbered as:

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
    sigmas : dict, optional
        Dictionary mapping parameters to sigma values for Monte-Carlo
        simulations, values should be in meters.
    """

    ocount = 4

    def __init__(
        self,
        width: Union[float, np.ndarray],
        thickness: Union[float, np.ndarray],
        gap: Callable[[float], float],
        dgap: Callable[[float], float],
        zmin: float,
        zmax: float,
        sw_angle: Union[float, np.ndarray] = 90,
        sigmas: Dict[str, float] = dict(),
        name: str = None,
    ) -> None:
        if width < 400 or width > 600:
            raise ValueError("Width must be between 400 and 600 nm")
        if thickness < 180 or thickness > 240:
            raise ValueError("Thickness must be between 180 and 240 nm")
        if sw_angle < 80 or sw_angle > 90:
            raise ValueError("Sidewall angle must be between 80 and 90 degrees")
        super().__init__(
            scee.GapFuncSymmetric(
                width,
                thickness,
                gap,
                dgap,
                zmin,
                zmax,
                sw_angle,
            ),
            sigmas,
            name=name,
        )

    # def update_variations(self, **kwargs):
    #     self.nominal_width = self.params["width"]
    #     self.nominal_thickness = self.params["thickness"]

    #     w = self.params["width"] + kwargs.get("corr_w")
    #     h = self.params["thickness"] + kwargs.get("corr_t")

    #     self.layout_aware = True
    #     self.params["width"] = w
    #     self.params["thickness"] = h

    # def regenerate_layout_aware_monte_carlo_parameters(self):
    #     self.params["width"] = self.nominal_width
    #     self.params["thickness"] = self.nominal_thickness


class GapFuncAntiSymmetric(SipannWrapper):
    r"""Antisymmetric directional coupler, meaning both waveguides are
    differently shaped.

    A gap function describing the vertical distance between the two waveguides
    at any horizontal point, and arc lengths from each port to the coupling
    point, describe the shape of the device.

    Ports are numbered as:

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
    sigmas : dict, optional
        Dictionary mapping parameters to sigma values for Monte-Carlo
        simulations, values should be in meters.
    """

    ocount = 4

    def __init__(
        self,
        width: Union[float, np.ndarray],
        thickness: Union[float, np.ndarray],
        gap: Callable[[float], float],
        zmin: float,
        zmax: float,
        arc1: float,
        arc2: float,
        arc3: float,
        arc4: float,
        sw_angle: Union[float, np.ndarray] = 90,
        sigmas: dict = dict(),
        name: str = None,
    ) -> None:
        if width < 400 or width > 600:
            raise ValueError("Width must be between 400 and 600 nm")
        if thickness < 180 or thickness > 240:
            raise ValueError("Thickness must be between 180 and 240 nm")
        if gap < 100:
            raise ValueError("Gap must be greater than 100 nm")
        if sw_angle < 80 or sw_angle > 90:
            raise ValueError("Sidewall angle must be between 80 and 90 degrees")
        super().__init__(
            scee.GapFuncAntiSymmetric(
                width,
                thickness,
                gap,
                zmin,
                zmax,
                arc1,
                arc2,
                arc3,
                arc4,
                sw_angle,
            ),
            sigmas,
            name=name,
        )

    # def update_variations(self, **kwargs):
    #     self.nominal_width = self.params["width"]
    #     self.nominal_thickness = self.params["thickness"]

    #     w = self.params["width"] + kwargs.get("corr_w")
    #     h = self.params["thickness"] + kwargs.get("corr_t")

    #     self.layout_aware = True
    #     self.params["width"] = w
    #     self.params["thickness"] = h

    # def regenerate_layout_aware_monte_carlo_parameters(self):
    #     self.params["width"] = self.nominal_width
    #     self.params["thickness"] = self.nominal_thickness


class HalfRing(SipannWrapper):
    """Half of a ring resonator.

    Uses a radius and a gap to describe the shape.

    .. image:: /_static/images/sipann_half_ring.png
        :alt: Half ring port numbering.
        :width: 400px
        :align: center

    Parameters
    ----------
    width : float
        Width of waveguides in nanometers (valid from 400 to 600).
    thickness : float
        Thickness of waveguides in nanometers (valid from 180 to 240).
    radius : float
        Distance from center of ring to middle of waveguide, in nanometers.
    gap : float
        Minimum distance from ring waveguide edge to straight waveguide edge,
        in nanometers (must be greater than 100).
    sw_angle : float, optional
        Sidewall angle of waveguide from horizontal in degrees (valid from 80
        to 90, defaults to 90).
    sigmas : dict, optional
        Dictionary mapping parameters to sigma values for Monte-Carlo
        simulations, values should be in meters.

    Examples
    --------
    >>> dev = HalfRing(500, 220, 5000, 100)
    """

    ocount = 4

    def __init__(
        self,
        width: Union[float, np.ndarray],
        thickness: Union[float, np.ndarray],
        radius: Union[float, np.ndarray],
        gap: Union[float, np.ndarray],
        sw_angle: Union[float, np.ndarray] = 90,
        sigmas: Dict[str, float] = dict(),
        name: str = None,
    ) -> None:
        if width < 400 or width > 600:
            raise ValueError("Width must be between 400 and 600 nm")
        if thickness < 180 or thickness > 240:
            raise ValueError("Thickness must be between 180 and 240 nm")
        if gap < 100:
            raise ValueError("Gap must be greater than 100 nm")
        if sw_angle < 80 or sw_angle > 90:
            raise ValueError("Sidewall angle must be between 80 and 90 degrees")
        super().__init__(
            scee.HalfRing(
                width,
                thickness,
                radius,
                gap,
                sw_angle,
            ),
            sigmas,
            name=name,
        )

    def s_params(self, wl):
        """Get the s-parameters of the SCEE Model.

        Parameters
        ----------
        wl : float or array-like
            Wavelength array to sample s-parameters at, in microns.

        Returns
        -------
        s : array-like
            The s-parameter matrix sampled at the given wavelengths.
        """
        return self.model.sparams(wl * 1e3)

    # def update_variations(self, **kwargs):
    #     self.nominal_width = self.params["width"]
    #     self.nominal_thickness = self.params["thickness"]

    #     w = self.params["width"] + kwargs.get("corr_w")
    #     h = self.params["thickness"] + kwargs.get("corr_t")

    #     self.layout_aware = True
    #     self.params["width"] = w
    #     self.params["thickness"] = h

    # def regenerate_layout_aware_monte_carlo_parameters(self):
    #     self.params["width"] = self.nominal_width
    #     self.params["thickness"] = self.nominal_thickness


class HalfRacetrack(SipannWrapper):
    """Half of a ring resonator, similar to the HalfRing class.

    Uses a radius, gap and length to describe the shape of the device.

    .. image:: /_static/images/sipann_half_racetrack.png
        :alt: Half racetrack port numbering.
        :width: 400px
        :align: center

    Parameters
    ----------
    width : float
        Width of waveguides in nanometers (valid from 400 to 600).
    thickness : float
        Thickness of waveguides in nanometers (valid from 180 to 240).
    radius : float
        Distance from center of ring to middle of waveguide, in nanometers.
    gap : float
        Minimum distance from ring waveguide edge to straight waveguide edge,
        in nanometers (must be greater than 100).
    length : float
        Length of straight portion of ring waveguide, in nanometers.
    sw_angle : float, optional
        Sidewall angle of waveguide from horizontal in degrees (valid from 80
        to 90, defaults to 90).
    sigmas : dict, optional
        Dictionary mapping parameters to sigma values for Monte-Carlo
        simulations, values should be in meters.

    Examples
    --------
    >>> dev = HalfRacetrack(500, 220, 5000, 100, 5000)
    """

    ocount = 4

    def __init__(
        self,
        width: Union[float, np.ndarray],
        thickness: Union[float, np.ndarray],
        radius: Union[float, np.ndarray],
        gap: Union[float, np.ndarray],
        length: Union[float, np.ndarray],
        sw_angle: Union[float, np.ndarray] = 90,
        sigmas: Dict[str, float] = dict(),
        name: str = None,
    ) -> None:
        if width < 400 or width > 600:
            raise ValueError("Width must be between 400 and 600 nm")
        if thickness < 180 or thickness > 240:
            raise ValueError("Thickness must be between 180 and 240 nm")
        if gap < 100:
            raise ValueError("Gap must be greater than 100 nm")
        if sw_angle < 80 or sw_angle > 90:
            raise ValueError("Sidewall angle must be between 80 and 90 degrees")
        super().__init__(
            scee.HalfRacetrack(
                width,
                thickness,
                radius,
                gap,
                length,
                sw_angle,
            ),
            sigmas,
            name=name,
        )

    # def update_variations(self, **kwargs):
    #     self.nominal_width = self.params["width"]
    #     self.nominal_thickness = self.params["thickness"]

    #     w = self.params["width"] + kwargs.get("corr_w")
    #     h = self.params["thickness"] + kwargs.get("corr_t")

    #     self.layout_aware = True
    #     self.params["width"] = w
    #     self.params["thickness"] = h

    # def regenerate_layout_aware_monte_carlo_parameters(self):
    #     self.params["width"] = self.nominal_width
    #     self.params["thickness"] = self.nominal_thickness


class StraightCoupler(SipannWrapper):
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
    sigmas : dict, optional
        Dictionary mapping parameters to sigma values for Monte-Carlo
        simulations, values should be in meters.

    Examples
    --------
    >>> dev = StraightCoupler(500, 220, 100, 5000)
    """

    ocount = 4

    def __init__(
        self,
        width: Union[float, np.ndarray],
        thickness: Union[float, np.ndarray],
        gap: Union[float, np.ndarray],
        length: Union[float, np.ndarray],
        sw_angle: Union[float, np.ndarray] = 90,
        sigmas: Dict[str, float] = dict(),
        name: str = None,
    ) -> None:
        if width < 400 or width > 600:
            raise ValueError("Width must be between 400 and 600 nm")
        if thickness < 180 or thickness > 240:
            raise ValueError("Thickness must be between 180 and 240 nm")
        if gap < 100:
            raise ValueError("Gap must be greater than 100 nm")
        if sw_angle < 80 or sw_angle > 90:
            raise ValueError("Sidewall angle must be between 80 and 90 degrees")
        super().__init__(
            scee.StraightCoupler(
                width,
                thickness,
                gap,
                length,
                sw_angle,
            ),
            sigmas,
            name=name,
        )

    # def update_variations(self, **kwargs):
    #     self.nominal_width = self.params["width"]
    #     self.nominal_thickness = self.params["thickness"]

    #     w = self.params["width"] + kwargs.get("corr_w")
    #     h = self.params["thickness"] + kwargs.get("corr_t")

    #     self.layout_aware = True
    #     self.params["width"] = w
    #     self.params["thickness"] = h

    # def regenerate_layout_aware_monte_carlo_parameters(self):
    #     self.params["width"] = self.nominal_width
    #     self.params["thickness"] = self.nominal_thickness


class StandardCoupler(SipannWrapper):
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
    sigmas : dict, optional
        Dictionary mapping parameters to sigma values for Monte-Carlo
        simulations, values should be in meters.

    Examples
    --------
    >>> dev = StandardCoupler(500, 220, 100, 5000, 5e3, 2e3)
    """

    ocount = 4

    def __init__(
        self,
        width: Union[float, np.ndarray],
        thickness: Union[float, np.ndarray],
        gap: Union[float, np.ndarray],
        length: Union[float, np.ndarray],
        horizontal: Union[float, np.ndarray],
        vertical: Union[float, np.ndarray],
        sw_angle: Union[float, np.ndarray] = 90,
        sigmas: Dict[str, float] = dict(),
        name: str = None,
    ) -> None:
        if width < 400 or width > 600:
            raise ValueError("Width must be between 400 and 600 nm")
        if thickness < 180 or thickness > 240:
            raise ValueError("Thickness must be between 180 and 240 nm")
        if gap < 100:
            raise ValueError("Gap must be greater than 100 nm")
        if sw_angle < 80 or sw_angle > 90:
            raise ValueError("Sidewall angle must be between 80 and 90 degrees")
        super().__init__(
            scee.Standard(
                width,
                thickness,
                gap,
                length,
                horizontal,
                vertical,
                sw_angle,
            ),
            sigmas,
            name=name,
        )

    # def update_variations(self, **kwargs):
    #     self.nominal_width = self.params["width"]
    #     self.nominal_thickness = self.params["thickness"]

    #     w = self.params["width"] + kwargs.get("corr_w")
    #     h = self.params["thickness"] + kwargs.get("corr_t")

    #     self.layout_aware = True
    #     self.params["width"] = w
    #     self.params["thickness"] = h

    # def regenerate_layout_aware_monte_carlo_parameters(self):
    #     self.params["width"] = self.nominal_width
    #     self.params["thickness"] = self.nominal_thickness


class DoubleHalfRing(SipannWrapper):
    r"""Two equally sized half-rings coupling along their edges.

    Described by a radius and a gap between the two rings.

    Ports are numbered as:

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
    sigmas : dict, optional
        Dictionary mapping parameters to sigma values for Monte-Carlo
        simulations, values should be in meters.

    Notes
    -----
    Writing to GDS is not supported for this component.
    """

    ocount = 4

    def __init__(
        self,
        width: Union[float, np.ndarray],
        thickness: Union[float, np.ndarray],
        radius: Union[float, np.ndarray],
        gap: Union[float, np.ndarray],
        sw_angle: Union[float, np.ndarray] = 90,
        sigmas: Dict[str, float] = dict(),
        name: str = None,
    ) -> None:
        if width < 400 or width > 600:
            raise ValueError("Width must be between 400 and 600 nm")
        if thickness < 180 or thickness > 240:
            raise ValueError("Thickness must be between 180 and 240 nm")
        if gap < 100:
            raise ValueError("Gap must be greater than 100 nm")
        if sw_angle < 80 or sw_angle > 90:
            raise ValueError("Sidewall angle must be between 80 and 90 degrees")
        super().__init__(
            scee.DoubleHalfRing(
                width,
                thickness,
                radius,
                gap,
                sw_angle,
            ),
            sigmas,
            name=name,
        )

    # def update_variations(self, **kwargs):
    #     self.nominal_width = self.params["width"]
    #     self.nominal_thickness = self.params["thickness"]

    #     w = self.params["width"] + kwargs.get("corr_w")
    #     h = self.params["thickness"] + kwargs.get("corr_t")

    #     self.layout_aware = True
    #     self.params["width"] = w
    #     self.params["thickness"] = h

    # def regenerate_layout_aware_monte_carlo_parameters(self):
    #     self.params["width"] = self.nominal_width
    #     self.params["thickness"] = self.nominal_thickness


class AngledHalfRing(SipannWrapper):
    r"""A halfring resonator, except what was the straight waveguide is now
    curved.

    Described by a radius, gap, and angle (theta) that the
    "straight" waveguide is curved by.

    Ports are numbered as:

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
    sigmas : dict, optional
        Dictionary mapping parameters to sigma values for Monte-Carlo
        simulations, values should be in meters.

    Notes
    -----
    Writing to GDS is not supported for this component.
    """

    ocount = 4

    def __init__(
        self,
        width: Union[float, np.ndarray],
        thickness: Union[float, np.ndarray],
        radius: Union[float, np.ndarray],
        gap: Union[float, np.ndarray],
        theta: Union[float, np.ndarray],
        sw_angle: Union[float, np.ndarray] = 90,
        sigmas: Dict[str, float] = dict(),
        name: str = None,
    ) -> None:
        if width < 400 or width > 600:
            raise ValueError("Width must be between 400 and 600 nm")
        if thickness < 180 or thickness > 240:
            raise ValueError("Thickness must be between 180 and 240 nm")
        if gap < 100:
            raise ValueError("Gap must be greater than 100 nm")
        if sw_angle < 80 or sw_angle > 90:
            raise ValueError("Sidewall angle must be between 80 and 90 degrees")
        super().__init__(
            scee.AngledHalfRing(
                width,
                thickness,
                radius,
                gap,
                theta,
                sw_angle,
            ),
            sigmas,
            name=name,
        )

    # def update_variations(self, **kwargs):
    #     self.nominal_width = self.params["width"]
    #     self.nominal_thickness = self.params["thickness"]

    #     w = self.params["width"] + kwargs.get("corr_w")
    #     h = self.params["thickness"] + kwargs.get("corr_t")

    #     self.layout_aware = True
    #     self.params["width"] = w
    #     self.params["thickness"] = h

    # def regenerate_layout_aware_monte_carlo_parameters(self):
    #     self.params["width"] = self.nominal_width
    #     self.params["thickness"] = self.nominal_thickness


class Waveguide(SipannWrapper):
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
    sigmas : dict, optional
        Dictionary mapping parameters to sigma values for Monte-Carlo
        simulations, values should be in meters.
    """

    ocount = 2

    def __init__(
        self,
        width: Union[float, np.ndarray],
        thickness: Union[float, np.ndarray],
        length: Union[float, np.ndarray],
        sw_angle: Union[float, np.ndarray] = 90,
        sigmas: Dict[str, float] = dict(),
        name: str = None,
    ) -> None:
        if width < 400 or width > 600:
            raise ValueError("Width must be between 400 and 600 nm")
        if thickness < 180 or thickness > 240:
            raise ValueError("Thickness must be between 180 and 240 nm")
        if sw_angle < 80 or sw_angle > 90:
            raise ValueError("Sidewall angle must be between 80 and 90 degrees")
        super().__init__(
            scee.Waveguide(width, thickness, length, sw_angle),
            sigmas,
            name=name,
        )

    # def update_variations(self, **kwargs):
    #     self.nominal_width = self.params["width"]
    #     self.nominal_thickness = self.params["thickness"]

    #     w = self.params["width"] + kwargs.get("corr_w")
    #     h = self.params["thickness"] + kwargs.get("corr_t")

    #     self.layout_aware = True
    #     self.params["width"] = w
    #     self.params["thickness"] = h

    # def regenerate_layout_aware_monte_carlo_parameters(self):
    #     self.params["width"] = self.nominal_width
    #     self.params["thickness"] = self.nominal_thickness


class Racetrack(SipannWrapper):
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
    sigmas : dict, optional
        Dictionary mapping parameters to sigma values for Monte-Carlo
        simulations, values should be in meters.

    Examples
    --------
    >>> dev = Racetrack(500, 220, 5000, 200, 5000)
    """

    ocount = 2

    def __init__(
        self,
        width: Union[float, np.ndarray],
        thickness: Union[float, np.ndarray],
        radius: Union[float, np.ndarray],
        gap: Union[float, np.ndarray],
        length: Union[float, np.ndarray],
        sw_angle: Union[float, np.ndarray] = 90,
        sigmas: Dict[str, float] = dict(),
        name: str = None,
    ) -> None:
        if width < 400 or width > 600:
            raise ValueError("Width must be between 400 and 600 nm")
        if thickness < 180 or thickness > 240:
            raise ValueError("Thickness must be between 180 and 240 nm")
        if gap < 100:
            raise ValueError("Gap must be greater than 100 nm")
        if sw_angle < 80 or sw_angle > 90:
            raise ValueError("Sidewall angle must be between 80 and 90 degrees")
        super().__init__(
            comp.racetrack_sb_rr(
                width,
                thickness,
                radius,
                gap,
                length,
                sw_angle,
            ),
            sigmas,
            name=name,
        )

    def write_gds(self, filename: Union[Path, str]) -> None:
        """Write the model to a GDS file.

        Parameters
        ----------
        filename : str or Path
            Path to write the GDS file to.
        """
        self.model.gds(str(filename), units="nms")

    # def update_variations(self, **kwargs):
    #     self.nominal_width = self.params["width"]
    #     self.nominal_thickness = self.params["thickness"]

    #     w = self.params["width"] + kwargs.get("corr_w")
    #     h = self.params["thickness"] + kwargs.get("corr_t")

    #     self.layout_aware = True
    #     self.params["width"] = w
    #     self.params["thickness"] = h

    # def regenerate_layout_aware_monte_carlo_parameters(self):
    #     self.params["width"] = self.nominal_width
    #     self.params["thickness"] = self.nominal_thickness


class PremadeCoupler(SipannWrapper):
    r"""Loads premade couplers based on the given split value.

    Various splitting ratio couplers have been made and saved. This
    function reloads them. Note that each of their lengths are different
    and are also returned for the users info. These have all been
    designed with waveguide geometry 500nm x 220nm.

    Ports are numbered as:

    |       2---\      /---4       |
    |            ------            |
    |            ------            |
    |       1---/      \---3       |

    Parameters
    ----------
    split : int
        Percent of light coming out cross port. Valid numbers are 10, 20, 30,
        40, 50, 100. 100 is a full crossover.
    sigmas : dict, optional
        Dictionary mapping parameters to sigma values for Monte-Carlo
        simulations, values should be in meters.
    """

    ocount = 4

    def __init__(
        self, split: int, sigmas: Dict[str, float] = dict(), name: str = None, **kwargs
    ) -> None:
        super().__init__(premade_coupler(split)[0], sigmas, name=name, **kwargs)
