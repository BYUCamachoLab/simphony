# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
simphony.libraries.sipann
=========================
This package contains wrappers for models defined in the
SiPANN (Silicon Photonics with Artificial Neural Networks)
project, another project by CamachoLab at BYU. It leverages
machine learning to simulate photonic devices, giving
greater speed and similar accuracy to a full FDTD
simulation.

The wrappers defined here integrate SiPANN models into
Simphony for easier use.
"""

from typing import Callable, Dict, TypeVar, Union

import numpy as np
from SiPANN import comp, scee
from SiPANN.scee_opt import premade_coupler

from simphony import Model
from simphony.tools import freq2wl


class SipannWrapper(Model):
    """Allows wrapping models from SCEE for use in simphony. This class should
    be extended, with each extending class wrapping one model.

    Each extending class should convert parameters passed in
    from meters (which simphony uses) to nanometers (which
    SiPANN uses). Each extending class should also define a
    class-wide field for 'pin_count', equal to the number of
    pins the subcircuit has.

    Note that the wrapped SCEE models cannot have varying
    geometries; such a device can't be cascaded properly.

    Parameters
    -----------
    `model`
    Model from `SiPANN.scee` or `SiPANN.comp` modules, must
    have the 'sparams' method

    `sigmas`
    Dictionary mapping parameters to sigma values for
    Monte-Carlo simulations, values should be in meters. If
    Monte-Carlo simulations are not needed, pass in
    an empty dictionary.
    """

    freq_range = (
        182800279268292.0,
        205337300000000.0,
    )

    def __init__(self, model: TypeVar("M"), sigmas: Dict[str, float], **kwargs) -> None:
        super().__init__(**kwargs)

        self.model = model
        self.sigmas = sigmas

        # catch varying geometries
        args = self.model._clean_args(None)
        if len(args[0]) != 1:
            raise ValueError(
                "You have changing geometries, use in simphony doesn't make sense!"
            )

        self.params = self.model.__dict__.copy()
        self.rand_params = dict()
        self.regenerate_monte_carlo_parameters()

    def s_parameters(self, freqs: np.array) -> np.ndarray:
        """Get the s-parameters of the SCEE Model.

        Parameters
        ----------
        `freqs`
        Frequency array to calculate s-parameters over, in
        Hz

        Returns
        -------
        `s`
        The s-parameter matrix
        """
        wl = freq2wl(freqs) * 1e9

        return self.model.sparams(wl)

    def monte_carlo_s_parameters(self, freqs: np.array) -> np.ndarray:
        """Get the s-parameters of the SCEE Model, influenced by noise from
        sigma values.

        Parameters
        ----------
        `freqs`
        Frequency array to calculate s-parameters over, in
        Hz

        Returns
        -------
        `s`
        The s-parameter matrix
        """
        wl = freq2wl(freqs) * 1e9

        # Change to noise params for monte carlo, then change back
        self.model.update(**self.rand_params)
        sparams = self.model.sparams(wl)
        self.model.update(**self.params)

        return sparams

    def regenerate_monte_carlo_parameters(self) -> None:
        """For each sigma value given to the wrapper, will apply noise the
        matching parameter."""
        for param, sigma in self.sigmas.items():
            self.rand_params[param] = np.random.normal(self.params[param], sigma * 1e9)


# Convert gap funcs from meters to nanometers
def convert_func_to_nm(func: Callable[[float], float]) -> Callable[[float], float]:
    def converted_func(input: float) -> float:
        return func(input) * 1e9

    return converted_func


class GapFuncSymmetric(SipannWrapper):
    """Symmetric directional coupler, meaning both waveguides are the same
    shape.

    A gap function must describe the shape of the two
    waveguides, where the vertical distance between the
    waveguides is the return of the gap function at every
    horizontal point from left to right. The derivative of
    the gap function is also required.

    Ports are numbered as:

        |       2---\      /---4       |
        |            ------            |
        |            ------            |
        |       1---/      \---3       |

    Parameters
    ----------
    `width`
    Width of waveguides in meters (Valid from 400e-9 to
    600e-9)

    `thickness`
    Thickness of waveguides in meters (Valid from 180e-9 to
    240e-9)

    `gap`
    Gap function along the waveguide, returns meters (Must
    always be greater than 100e-9)

    `dgap`
    Derivative of the gap function

    `zmin`
    Real number at which to begin integration of gap
    function

    `zmax`
    Real number at which to end integration of gap function

    `sw_angle, optional`
    Sidewall angle of waveguide from horizontal in degrees
    (Valid from 80 to 90, defaults to 90)

    `sigmas, optional`
    Dictionary mapping parameters to sigma values for
    Monte-Carlo simulations, values should be in meters
    """

    pin_count = 4

    def __init__(
        self,
        width: Union[float, np.array],
        thickness: Union[float, np.array],
        gap: Callable[[float], float],
        dgap: Callable[[float], float],
        zmin: float,
        zmax: float,
        sw_angle: Union[float, np.array] = 90,
        sigmas: Dict[str, float] = dict(),
        **kwargs
    ) -> None:
        super().__init__(
            scee.GapFuncSymmetric(
                width * 1e9,
                thickness * 1e9,
                convert_func_to_nm(gap),
                convert_func_to_nm(dgap),
                zmin,
                zmax,
                sw_angle,
            ),
            sigmas,
            **kwargs
        )


class GapFuncAntiSymmetric(SipannWrapper):
    """Antisymmetric directional coupler, meaning both waveguides are
    differently shaped.

    A gap function describing the vertical distance between
    the two waveguides at any horizontal point, and arc
    lengths from each port to the coupling point, describe
    the shape of the device.

    Ports are numbered as:
    |       2---\      /---4       |
    |            ------            |
    |            ------            |
    |       1---/      \---3       |

    Parameters
    ----------
    `width`
    Width of the waveguide in meters (Valid from 400e-9 to
    600e-9)

    `thickness`
    Thickness of waveguide in meters (Valid for 180e-9 to
    240e-9)

    `gap`
    Gap function along the waveguide, returns meters (must
    always be greater than 100e-9)

    `zmin`
    Real number at which to begin integration of gap
    function

    `zmax`
    Real number at which to end integration of gap function

    `arc1`
    Arc length from port 1 to minimum coupling point

    `arc2`
    Arc length from port 2 to minimum coupling point

    `arc3`
    Arc length from port 3 to minimum coupling point

    `arc4`
    Arc length from port 4 to minimum coupling point

    `sw_angle, optional`
    Sidewall angle of waveguide from horizontal in degrees
    (Valid from 80 to 90, defaults to 90)

    `sigmas, optional`
    Dictionary mapping parameters to sigma values for
    Monte-Carlo simulations, values should be in meters
    """

    pin_count = 4

    def __init__(
        self,
        width: Union[float, np.array],
        thickness: Union[float, np.array],
        gap: Callable[[float], float],
        zmin: float,
        zmax: float,
        arc1: float,
        arc2: float,
        arc3: float,
        arc4: float,
        sw_angle: Union[float, np.array] = 90,
        sigmas: dict = dict(),
        **kwargs
    ) -> None:
        super().__init__(
            scee.GapFuncAntiSymmetric(
                width * 1e9,
                thickness * 1e9,
                convert_func_to_nm(gap),
                zmin,
                zmax,
                arc1 * 1e9,
                arc2 * 1e9,
                arc3 * 1e9,
                arc4 * 1e9,
                sw_angle,
            ),
            sigmas,
            **kwargs
        )


class HalfRing(SipannWrapper):
    """Half of a ring resonator.

    Uses a radius and a gap to describe the shape.

    Ports are numbered as:

        |         2 \     / 4          |
        |            \   /             |
        |             ---              |
        |         1---------3          |

    Parameters
    ----------
    `width`
    Width of the waveguide in meters (Valid from 400e-9 to
    600e-9)

    `thickness`
    Thickness of waveguide in meters (Valid for 180e-9 to
    240e-9)

    `radius`
    Distance from center of ring to middle of waveguide, in
    meters

    `gap`
    Minimum distance from ring waveguide edge to straight
    waveguide edge, in meters (must be greater than 100e-9)

    `sw_angle, optional`
    Sidewall angle of waveguide from horizontal in degrees
    (Valid from 80 to 90, defaults to 90)

    `sigmas, optional`
    Dictionary mapping parameters to sigma values for
    Monte-Carlo simulations, values should be in meters
    """

    pin_count = 4

    def __init__(
        self,
        width: Union[float, np.array],
        thickness: Union[float, np.array],
        radius: Union[float, np.array],
        gap: Union[float, np.array],
        sw_angle: Union[float, np.array] = 90,
        sigmas: Dict[str, float] = dict(),
        **kwargs
    ) -> None:
        super().__init__(
            scee.HalfRing(
                width * 1e9, thickness * 1e9, radius * 1e9, gap * 1e9, sw_angle
            ),
            sigmas,
            **kwargs
        )


class HalfRacetrack(SipannWrapper):
    """Half of a ring resonator, similar to the HalfRing class.

    Uses a radius, gap and length to describe the shape of
    the device.

    Ports are numbered as:

        |      2 \           / 4       |
        |         \         /          |
        |          ---------           |
        |      1---------------3       |

    Parameters
    ----------
    `width`
    Width of the waveguide in meters (Valid from 400e-9 to
    600e-9)

    `thickness`
    Thickness of waveguide in meters (Valid for 180e-9 to
    240e-9)

    `radius`
    Distance from center of ring to middle of waveguide, in
    meters

    `gap`
    Minimum distance from ring waveguide edge to straight
    waveguide edge, in meters (must be greater than 100e-9)

    `length`
    Length of straight portion of ring waveguide, in meters

    `sw_angle, optional`
    Sidewall angle of waveguide from horizontal in degrees
    (Valid from 80 to 90, defaults to 90)

    `sigmas, optional`
    Dictionary mapping parameters to sigma values for
    Monte-Carlo simulations, values should be in meters
    """

    pin_count = 4

    def __init__(
        self,
        width: Union[float, np.array],
        thickness: Union[float, np.array],
        radius: Union[float, np.array],
        gap: Union[float, np.array],
        length: Union[float, np.array],
        sw_angle: Union[float, np.array] = 90,
        sigmas: Dict[str, float] = dict(),
        **kwargs
    ) -> None:
        super().__init__(
            scee.HalfRacetrack(
                width * 1e9,
                thickness * 1e9,
                radius * 1e9,
                gap * 1e9,
                length * 1e9,
                sw_angle,
            ),
            sigmas,
            **kwargs
        )


class StraightCoupler(SipannWrapper):
    """Straight directional coupler, both waveguides run parallel.

    Described by a gap and a length.

    Ports are numbered as:

        |      2---------------4       |
        |      1---------------3       |

    Parameters
    --------
    `width`
    Width of the waveguide in meters (Valid from 400e-9 to
    600e-9)

    `thickness`
    Thickness of waveguide in meters (Valid for 180e-9 to
    240e-9)

    `gap`
    Distance between the two waveguide edges, in meters
    (must be greater than 100e-9)

    `length`
    Length of both waveguides in meters

    `sw_angle, optional`
    Sidewall angle of waveguide from horizontal in degrees
    (Valid from 80 to 90, defaults to 90)

    `sigmas, optional`
    Dictionary mapping parameters to sigma values for
    Monte-Carlo simulations, values should be in meters
    """

    pin_count = 4

    def __init__(
        self,
        width: Union[float, np.array],
        thickness: Union[float, np.array],
        gap: Union[float, np.array],
        length: Union[float, np.array],
        sw_angle: Union[float, np.array] = 90,
        sigmas: Dict[str, float] = dict(),
        **kwargs
    ) -> None:
        super().__init__(
            scee.StraightCoupler(
                width * 1e9, thickness * 1e9, gap * 1e9, length * 1e9, sw_angle
            ),
            sigmas,
            **kwargs
        )


class Standard(SipannWrapper):
    """Standard-shaped directional coupler.

    Described by a gap, length, horizontal and vertical
    distance.

    Ports are numbered as:

        |       2---\      /---4       |
        |            ------            |
        |            ------            |
        |       1---/      \---3       |

    Parameters
    ----------
    `width`
    Width of the waveguide in meters (Valid from 400e-9 to
    600e-9)

    `thickness`
    Thickness of waveguide in meters (Valid for 180e-9 to
    240e-9)

    `gap`
    Distance between the two waveguide edges, in meters
    (must be greater than 100e-9)

    `length`
    Length of the straight portion of both waveguides, in
    meters

    `horizontal`
    Horizontal distance between end of coupler and straight
    segment, in meters

    `vertical`
    Vertical distance between end of coupler and straight
    segment, in meters

    `sw_angle, optional`
    Sidewall angle of waveguide from horizontal in degrees
    (Valid from 80 to 90, defaults to 90)

    `sigmas, optional`
    Dictionary mapping parameters to sigma values for
    Monte-Carlo simulations, values should be in meters
    """

    pin_count = 4

    def __init__(
        self,
        width: Union[float, np.array],
        thickness: Union[float, np.array],
        gap: Union[float, np.array],
        length: Union[float, np.array],
        horizontal: Union[float, np.array],
        vertical: Union[float, np.array],
        sw_angle: Union[float, np.array] = 90,
        sigmas: Dict[str, float] = dict(),
        **kwargs
    ) -> None:
        super().__init__(
            scee.Standard(
                width * 1e9,
                thickness * 1e9,
                gap * 1e9,
                length * 1e9,
                horizontal * 1e9,
                vertical * 1e9,
                sw_angle,
            ),
            sigmas,
            **kwargs
        )


class DoubleHalfRing(SipannWrapper):
    """Two equally sized half-rings coupling along their edges.

    Described by a radius and a gap between the two rings.

    Ports are numbered as:

        |         2 \     / 4          |
        |            \   /             |
        |             ---              |
        |             ---              |
        |            /   \             |
        |         1 /     \ 3          |

    Parameters
    ----------
    `width`
    Width of the waveguide in meters (Valid from 400e-9 to
    600e-9)

    `thickness`
    Thickness of waveguide in meters (Valid for 180e-9 to
    240e-9)

    `radius`
    Distance from center of ring to middle of waveguide, in
    meters

    `gap`
    Minimum distance from the edges of the waveguides of the
    two rings, in meters (must be greater than 100e-9)

    `sw_angle, optional`
    Sidewall angle of waveguide from horizontal in degrees
    (Valid from 80 to 90, defaults to 90)

    `sigmas, optional`
    Dictionary mapping parameters to sigma values for
    Monte-Carlo simulations, values should be in meters
    """

    pin_count = 4

    def __init__(
        self,
        width: Union[float, np.array],
        thickness: Union[float, np.array],
        radius: Union[float, np.array],
        gap: Union[float, np.array],
        sw_angle: Union[float, np.array] = 90,
        sigmas: Dict[str, float] = dict(),
        **kwargs
    ) -> None:
        super().__init__(
            scee.DoubleHalfRing(
                width * 1e9, thickness * 1e9, radius * 1e9, gap * 1e9, sw_angle
            ),
            sigmas,
            **kwargs
        )


class AngledHalfRing(SipannWrapper):
    """A halfring resonator, except what was the straight waveguide is now
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
    `width`
    Width of the waveguide in meters (Valid from 400e-9 to
    600e-9)

    `thickness`
    Thickness of waveguide in meters (Valid for 180e-9 to
    240e-9)

    `radius`
    Distance from center of ring to middle of waveguide, in
    meters

    `gap`
    Minimum distance from ring waveguide edge to "straight"
    waveguide edge, in meters (must be greater than 100e-9)

    `theta`
    Angle of the curve of the "straight" waveguide, in
    radians

    `sw_angle, optional`
    Sidewall angle of waveguide from horizontal in degrees
    (Valid from 80 to 90, defaults to 90)

    `sigmas, optional`
    Dictionary mapping parameters to sigma values for
    Monte-Carlo simulations, values should be in meters
    """

    pin_count = 4

    def __init__(
        self,
        width: Union[float, np.array],
        thickness: Union[float, np.array],
        radius: Union[float, np.array],
        gap: Union[float, np.array],
        theta: Union[float, np.array],
        sw_angle: Union[float, np.array] = 90,
        sigmas: Dict[str, float] = dict(),
        **kwargs
    ) -> None:
        super().__init__(
            scee.AngledHalfRing(
                width * 1e9, thickness * 1e9, radius * 1e9, gap * 1e9, theta, sw_angle
            ),
            sigmas,
            **kwargs
        )


class Waveguide(SipannWrapper):
    """Lossless model for a straight waveguide. Main use case is for playing
    nice with other models in SCEE.

    Ports are numbered as:

        |  1 ----------- 2   |

    Parameters
    ----------
    `width`
    Width of the waveguide in meters (Valid from 400e-9 to
    600e-9)

    `thickness`
    Thickness of waveguide in meters (Valid for 180e-9 to
    240e-9)

    `length`
    Length of waveguide in meters

    `sw_angle, optional`
    Sidewall angle of waveguide from horizontal in degrees
    (Valid from 80 to 90, defaults to 90)

    `sigmas, optional`
    Dictionary mapping parameters to sigma values for
    Monte-Carlo simulations, values should be in meters
    """

    pin_count = 2

    def __init__(
        self,
        width: Union[float, np.array],
        thickness: Union[float, np.array],
        length: Union[float, np.array],
        sw_angle: Union[float, np.array] = 90,
        sigmas: Dict[str, float] = dict(),
        **kwargs
    ) -> None:
        super().__init__(
            scee.Waveguide(width * 1e9, thickness * 1e9, length * 1e9, sw_angle),
            sigmas,
            **kwargs
        )


class Racetrack(SipannWrapper):
    """Racetrack waveguide arc, used to connect to a racetrack directional
    coupler.

    Ports labeled as:

        |           -------         |
        |         /         \       |
        |         \         /       |
        |           -------         |
        |   1 ----------------- 2   |

    Parameters
    ----------
    `width`
    Width of the waveguide in meters (Valid from 400e-9 to
    600e-9)

    `thickness`
    Thickness of waveguide in meters (Valid for 180e-9 to
    240e-9)

    `radius`
    Distance from center of ring to middle of waveguide, in
    meters

    `gap`
    Minimum distance from ring waveguide edge to straight
    waveguide edge, in meters (must be greater than 100e-9)

    `length`
    Length of straight portion of ring waveguide, in meters

    `sw_angle, optional`
    Sidewall angle of waveguide from horizontal in degrees
    (Valid from 80 to 90, defaults to 90)

    `sigmas, optional`
    Dictionary mapping parameters to sigma values for
    Monte-Carlo simulations, values should be in meters
    """

    pin_count = 2

    def __init__(
        self,
        width: Union[float, np.array],
        thickness: Union[float, np.array],
        radius: Union[float, np.array],
        gap: Union[float, np.array],
        length: Union[float, np.array],
        sw_angle: Union[float, np.array] = 90,
        sigmas: Dict[str, float] = dict(),
        **kwargs
    ) -> None:
        super().__init__(
            comp.racetrack_sb_rr(
                width * 1e9,
                thickness * 1e9,
                radius * 1e9,
                gap * 1e9,
                length * 1e9,
                sw_angle,
            ),
            sigmas,
            **kwargs
        )


class PremadeCoupler(SipannWrapper):
    """Loads premade couplers.

    Various splitting ratio couplers have been made and saved. This function reloads them. Note that each of their
    lengths are different and are also returned for the users info. These have all been designed with waveguide
    geometry 500nm x 220nm.

    Ports are numbered as:

        |       2---\      /---4       |
        |            ------            |
        |            ------            |
        |       1---/      \---3       |
    """

    pin_count = 4

    def __init__(self, split: int, sigmas: Dict[str, float] = dict(), **kwargs) -> None:
        """Loads the premade coupler based on the given split value.

        Parameters
        ----------
        split :
            Percent of light coming out cross port. Valid numbers are 10, 20, 30, 40, 50, 100. 100 is a full crossover.
        """
        super().__init__(premade_coupler(split)[0], sigmas, **kwargs)
