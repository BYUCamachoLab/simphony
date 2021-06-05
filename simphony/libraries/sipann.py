# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

from simphony import Model
from simphony.tools import freq2wl
from SiPANN import scee#, comp


class SipannWrapper(Model):
    """Allows wrapping models from SCEE for use in simphony.
    This class should be extended, with each extending class
    wrapping one model.
    
    Note that the wrapped SCEE models cannot have varying
    geometries; such a device can't be cascaded properly.

    Parameters
    -----------
    `model`
    Model from `SiPANN.scee` or `SiPANN.comp` modules
    
    `pin_count : int`
    Number of pins for this model

    `sigmas : dict`
    Dictionary mapping parameters to sigma values for
    Monte-Carlo simulations, values should be in nanometers.
    If Monte-Carlo simulations are not needed, pass in
    an empty dictionary.
    """

    freq_range = (
        182800279268292.0,
        205337300000000.0,
    )

    def __init__(self, model, pin_count, sigmas):
        super().__init__()
        
        self.model = model
        self.pin_count = pin_count
        self.sigmas = sigmas

        # catch varying geometries
        args = self.model._clean_args(None)
        if len(args[0]) != 1:
            raise ValueError(
                "You have changing geometries, use in simphony doesn't make sense!"
            )
        
        
        self.monte_carlo_model = self.model.copy()
        self.regenerate_monte_carlo_parameters()

    def s_parameters(self, freq):
        """Get the s-parameters of the SCEE Model.

        Parameters
        ----------
        `freq : np.ndarray`
        Frequency array to calculate s-parameters over, in
        Hz

        Returns
        -------
        `s : np.ndarray`
        The s-parameter matrix
        """
        wl = freq2wl(freq) * 1e9 # conversion to nanometers

        return self.model.sparams(wl)

    def monte_carlo_s_parameters(self, freq):
        """Get the s-parameters of the SCEE Model,
        influenced by noise from sigma values.

        Parameters
        ----------
        freq : np.ndarray
        Frequency array to calculate s-parameters over, in
        Hz

        Returns
        -------
        s : np.ndarray
        The s-parameter matrix
        """
        wl = freq2wl(freq) * 1e9 # conversion to nanometers

        return self.monte_carlo_model.sparams(wl) 

    def regenerate_monte_carlo_parameters(self):
        """For each sigma value given to the wrapper, will
        apply noise the matching parameter.
        """
        noise_params = dict()
        base_params = self.model.__dict__.copy()

        for param, sigma in self.sigmas.items():
            noise_params[param] = np.random.normal(base_params[param], sigma)

        self.monte_carlo_model.update(**noise_params)


class GapFuncSymmetric(SipannWrapper):
    """This class will create arbitrarily shaped SYMMETRIC (ie both waveguides
    are same shape) directional couplers.

    It takes in a gap function that describes the gap as one progreses through the device. Note that the shape fo the waveguide
    will simply be half of gap function. Also requires the derivative of the gap function for this purpose. Ports are numbered as::

        |       2---\      /---4       |
        |            ------            |
        |            ------            |
        |       1---/      \---3       |

    Parameters
    ----------
        width : float or ndarray
            Width of the waveguide in nm (Valid for 400nm-600nm)
        thickness : float or ndarray
            Thickness of waveguide in nm (Valid for 180nm-240nm)
        gap : function
            Gap function as one progresses along the waveguide (Must always be > 100nm)
        dgap : function
            Derivative of the gap function
        zmin : float
            Where to begin integration in the gap function
        zmax : float
            Where to end integration in the gap function
        sw_angle : float or ndarray, optional
            Sidewall angle of waveguide from horizontal in degrees (Valid for 80-90 degrees). Defaults to 90.
    """
    def __init__(self, width, thickness, gap, dgap, zmin, zmax, sw_angle=90, sigmas=dict()):
        super().__init__(
            scee.GapFuncSymmetric(width, thickness, gap, dgap, zmin, zmax, sw_angle),
            4, # pin count
            sigmas
        )


class GapFuncAntiSymmetric(SipannWrapper):
    """This class will create arbitrarily shaped ANTISYMMETRIC (ie waveguides
    are different shapes) directional couplers.

    It takes in a gap function that describes the gap as one progreses through the device. Also takes in arc length
    of each port up till coupling point.
    Ports are numbered as:
    |       2---\      /---4       |
    |            ------            |
    |            ------            |
    |       1---/      \---3       |

    Parameters
    ----------
        width : float or ndarray
            Width of the waveguide in nm (Valid for 400nm-600nm)
        thickness : float or ndarray
            Thickness of waveguide in nm (Valid for 180nm-240nm)
        gap : function
            Gap function as one progresses along the waveguide (Must always be > 100nm)
        zmin : float
            Where to begin integration in the gap function
        zmax : float
            Where to end integration in the gap function
        arc1, arc2, arc3, arc4 : float
            Arclength from entrance of each port till minimum coupling point
        sw_angle : float or ndarray, optional
            Sidewall angle of waveguide from horizontal in degrees (Valid for 80-90 degrees). Defaults to 90.
    """
    def __init__(self, width, thickness, gap, zmin, zmax, arc1, arc2, arc3, arc4, sw_angle=90, sigmas=dict()):
        super().__init__(
            scee.GapFuncAntiSymmetric(width, thickness, gap, zmin, zmax, arc1, arc2, arc3, arc4, sw_angle),
            4, # pin count
            sigmas
        )


class HalfRing(SipannWrapper):
    """This class will create half of a ring resonator.

    It takes in a radius and gap along with usual waveguide parameters. Ports are numbered as::

        |         2 \     / 4          |
        |            \   /             |
        |             ---              |
        |         1---------3          |

    Parameters
    ----------
        width : float or ndarray
            Width of the waveguide in nm (Valid for 400nm-600nm)
        thickness : float or ndarray
            Thickness of waveguide in nm (Valid for 180nm-240nm)
        radius : float or ndarray
            Distance from center of ring to middle of waveguide in nm.
        gap : float or ndarray
            Minimum distance from ring waveguide edge to straight waveguide edge in nm. (Must be > 100nm)
        sw_angle : float or ndarray, optional
            Sidewall angle of waveguide from horizontal in degrees (Valid for 80-90 degrees). Defaults to 90.
    """
    def __init__(self, width, thickness, radius, gap, sw_angle=90, sigmas=dict()):
        super().__init__(
            scee.HalfRing(width, thickness, radius, gap, sw_angle),
            4, # pin count
            sigmas
        )


class HalfRacetrack(SipannWrapper):
    """This class will create half of a ring resonator.

    It takes in a radius and gap along with usual waveguide parameters. Ports are numbered as::

        |      2 \           / 4       |
        |         \         /          |
        |          ---------           |
        |      1---------------3       |

    Parameters
    ----------
        width : float or ndarray
            Width of the waveguide in nm (Valid for 400nm-600nm)
        thickness : float or ndarray
            Thickness of waveguide in nm (Valid for 180nm-240nm)
        radius : float or ndarray
            Distance from center of ring to middle of waveguide in nm.
        gap : float or ndarray
            Minimum distance from ring waveguide edge to straight waveguide edge in nm. (Must be > 100nm)
        length : float or ndarray
            Length of straight portion of ring waveguide in nm.
        sw_angle : float or ndarray, optional
            Sidewall angle of waveguide from horizontal in degrees (Valid for 80-90 degrees). Defaults to 90.
    """
    def __init__(self, width, thickness, radius, gap, length, sw_angle=90, sigmas=dict()):
        super().__init__(
            scee.HalfRacetrack(width, thickness, radius, gap, length, sw_angle),
            4, # pin count
            sigmas
        )


class StraightCoupler(SipannWrapper):
    """This class will create half of a ring resonator.

    It takes in a radius and gap along with usual waveguide parameters. Ports are numbered as::

        |      2---------------4       |
        |      1---------------3       |

    Parameters
    ----------
        width : float or ndarray
            Width of the waveguide in nm (Valid for 400nm-600nm)
        thickness : float or ndarray
            Thickness of waveguide in nm (Valid for 180nm-240nm)
        gap : float or ndarray
           Distance between the two waveguides edge in nm. (Must be > 100nm)
        length : float or ndarray
            Length of both waveguides in nm.
        sw_angle : float or ndarray, optional
            Sidewall angle of waveguide from horizontal in degrees (Valid for 80-90 degrees). Defaults to 90.
    """
    def __init__(self, width, thickness, gap, length, sw_angle=90, sigmas=dict()):
        super().__init__(
            scee.StraightCoupler(width, thickness, gap, length, sw_angle),
            4, # pin count
            sigmas
        )

    
class Standard(SipannWrapper):
    """Normal/Standard Shaped Directional Coupler.

    This is what most people think of when they think directional coupler. Ports are numbered as::

        |       2---\      /---4       |
        |            ------            |
        |            ------            |
        |       1---/      \---3       |

    Parameters
    ----------
        width : float or ndarray
            Width of the waveguide in nm (Valid for 400nm-600nm)
        thickness : float or ndarray
            Thickness of waveguide in nm (Valid for 180nm-240nm)
        gap : float or ndarray
           Minimum distance between the two waveguides edge in nm. (Must be > 100nm)
        length : float or ndarray
            Length of the straight portion of both waveguides in nm.
        H : float or ndarray
            Horizontal distance between end of coupler until straight portion in nm.
        V : float or ndarray
            Vertical distance between end of coupler until straight portion in nm.
        sw_angle : float or ndarray, optional
            Sidewall angle of waveguide from horizontal in degrees (Valid for 80-90 degrees). Defaults to 90.
    """
    def __init__(self, width, thickness, gap, length, H, V, sw_angle=90, sigmas=dict()):
        super().__init__(
            scee.Standard(width, thickness, gap, length, H, V, sw_angle),
            4, # pin count
            sigmas
        )


class DoubleHalfRing(SipannWrapper):
    """This class will create two equally sized halfrings coupling.

    It takes in a radius and gap along with usual waveguide parameters. Ports are numbered as::

        |         2 \     / 4          |
        |            \   /             |
        |             ---              |
        |             ---              |
        |            /   \             |
        |         1 /     \ 3          |

    Parameters
    ----------
        width : float or ndarray
            Width of the waveguide in nm (Valid for 400nm-600nm)
        thickness : float or ndarray
            Thickness of waveguide in nm (Valid for 180nm-240nm)
        radius : float or ndarray
            Distance from center of ring to middle of waveguide in nm.
        gap : float or ndarray
            Minimum distance from ring waveguide edge to other ring waveguide edge in nm. (Must be > 100nm)
        sw_angle : float or ndarray, optional
            Sidewall angle of waveguide from horizontal in degrees (Valid for 80-90 degrees). Defaults to 90.
    """
    def __init__(self, width, thickness, radius, gap, sw_angle=90, sigmas=dict()):
        super().__init__(
            scee.DoubleHalfRing(width, thickness, radius, gap, sw_angle),
            4, # pin count
            sigmas
        )


class AngledHalfRing(SipannWrapper):
    """This class will create a halfring resonator with a pushed side.

    It takes in a radius and gap along with usual waveguide parameters. Ports are numbered as::

        |      2  \        / 4       |
        |          \      /          |
        |      1--- \    / ---3      |
        |          \ \  / /          |
        |           \ -- /           |
        |            ----            |

    Parameters
    ----------
        width : float or ndarray
            Width of the waveguide in nm (Valid for 400nm-600nm)
        thickness : float or ndarray
            Thickness of waveguide in nm (Valid for 180nm-240nm)
        radius : float or ndarray
            Distance from center of ring to middle of waveguide in nm.
        gap : float or ndarray
            Minimum distance from ring waveguide edge to straight waveguide edge in nm.  (Must be > 100nm)
        theta : float or ndarray
            Angle that the straight waveguide is curved in radians (???).
        sw_angle : float or ndarray, optional (Valid for 80-90 degrees)
    """

    def __init__(self, width, thickness, radius, gap, theta, sw_angle=90, sigmas=dict()):
        super().__init__(
            scee.AngledHalfRing(width, thickness, radius, gap, theta, sw_angle),
            4, # pin count
            sigmas
        )


class Waveguide(SipannWrapper):
    """Lossless model for a straight waveguide.

    Simple model that makes sparameters for a straight waveguide. May not be
    the best option, but plays nice with other models in SCEE. Ports are numbered as::

        |  1 ----------- 2   |

    Parameters
    ----------
        width : float or ndarray
            Width of the waveguide in nm (Valid for 400nm-600nm)
        thickness : float or ndarray
            Thickness of waveguide in nm (Valid for 180nm-240nm)
        length : float or ndarray
            Length of waveguide in nm.
        sw_angle : float or ndarray, optional
            Sidewall angle of waveguide from horizontal in degrees (Valid for 80-90 degrees). Defaults to 90.
    """

    def __init__(self, width, thickness, length, sw_angle=90, sigmas=dict()):
        super().__init__(
            scee.Waveguide(width, thickness, length, sw_angle),
            2, # pin count
            sigmas
        )

"""
class Racetrack(SipannWrapper):
    Racetrack waveguide arc used to connect to a racetrack directional
    coupler. Ports labeled as::

        |           -------         |
        |         /         \       |
        |         \         /       |
        |           -------         |
        |   1 ----------------- 2   |

    Parameters
    ----------
        width : float or ndarray
            Width of the waveguide in nm
        thickness : float or ndarray
            Thickness of waveguide in nm
        radius : float or ndarray
            Distance from center of ring to middle of waveguide in nm.
        gap : float or ndarray
            Minimum distance from ring waveguide edge to straight waveguide edge in nm.
        length : float or ndarray
            Length of straight portion of ring waveguide in nm.
        sw_angle : float or ndarray, optional
            Sidewall angle of waveguide from horizontal in degrees. Defaults to 90.
    

    def __init__(self, width, thickness, radius, gap, length, sw_angle=90, sigmas=dict()):
        super().__init__(
            comp.racetrack_sb_rr(width, thickness, radius, gap, length, sw_angle),
            2, # pin count
            sigmas
        )
        """
